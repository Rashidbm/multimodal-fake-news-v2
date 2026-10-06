"""Aggregate run directories into mean +- std tables and group-bootstrap confidence intervals.

    python -m fnd.imagev2.aggregate --runs D:\\...\\runs --manifest-dir data/image_branch_v2 --out D:\\...\\results

Writes summary.json (everything) and tables.md (the tables used by the experiment report).
CIs: groups (manifest group_id) of the clean TEST set are resampled with replacement; every seed's predictions are
scored on the same resample and averaged, so the interval is for the seed-mean. Paired deltas use identical resamples.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .manifest import arrays, load_joint

N_BOOT = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_runs(runs):
    out = defaultdict(list)
    for d in sorted(Path(runs).iterdir()):
        if not (d / "metrics.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        key = (cfg["arm"], cfg.get("tag", ""), cfg["protocol"], cfg["lambda_aux"])
        out[key].append(dict(dir=d, cfg=cfg, metrics=json.loads((d / "metrics.json").read_text())))
    return out


def ms(values):
    values = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return (float(values.mean()), float(values.std(ddof=0))) if len(values) else (float("nan"), float("nan"))


def fmt(pair, digits=3):
    return f"{pair[0]:.{digits}f} ± {pair[1]:.{digits}f}"


def pick(run, path):
    node = run["metrics"]
    for p in path:
        node = node.get(p) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def seed_mean(runs, path):
    return ms([pick(r, path) for r in runs])


class Bootstrap:
    """Group-level resamples of one clean split, scored per seed."""

    def __init__(self, runs, split_key, arr_by_id, n_boot=N_BOOT):
        z = np.load(runs[0]["dir"] / "predictions.npz", allow_pickle=False)
        self.ids = z[f"{split_key}__ids"]
        self.groups = np.array([arr_by_id[i]["group"] for i in self.ids])
        self.y = np.array([int(arr_by_id[i]["y3"] > 0) for i in self.ids])
        self.domain = np.array([arr_by_id[i]["domain"] for i in self.ids])
        self.resamples = M.bootstrap_groups(self.groups, n_boot, seed=2026)
        self.split_key = split_key

    def probs(self, runs):
        out = []
        for r in runs:
            z = np.load(r["dir"] / "predictions.npz", allow_pickle=False)
            if not np.array_equal(z[f"{self.split_key}__ids"], self.ids):
                raise ValueError("prediction id order differs between runs")
            out.append(sigmoid(z[f"{self.split_key}__logit"]))
        return out

    def statistic(self, runs, name):
        """array [n_boot] of seed-averaged metric values."""
        probs = self.probs(runs)
        values = np.zeros((len(self.resamples), len(probs)))
        for b, idx in enumerate(self.resamples):
            for s, p in enumerate(probs):
                y, pp = self.y[idx], p[idx]
                if name == "bacc":
                    values[b, s] = M.binary_metrics(y, pp)["balanced_accuracy"]
                elif name == "auroc":
                    values[b, s] = M.auroc(y, pp)
                elif name == "worst_domain_bacc":
                    values[b, s] = M.worst_group_bacc(y, pp, self.domain[idx])
        return values.mean(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest-dir", default="data/image_branch_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    rows = load_joint(args.manifest_dir)
    a = arrays(rows)
    by_id = {i: dict(group=g, y3=y, domain=d) for i, g, y, d in zip(a["ids"], a["group"], a["y3"], a["domain"])}
    groups = load_runs(args.runs)
    out, md = {}, []
    stats = {}
    main_keys = sorted(k for k in groups if k[2] == "main")
    md.append("### Main protocol (joint_forensic test, core rows), mean ± std over seeds [95% group-bootstrap CI of the seed mean]\n")
    md.append("| Arm | tag | λ | seeds | val worst-dom bAcc | test bAcc | test AUROC | fake recall | real recall | precision | F1 | worst-domain bAcc | JPEG-75 bAcc | blur bAcc | JPEG-75 AUROC | blur AUROC |")
    md.append("|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in main_keys:
        arm, tag, protocol, lam = key
        runs = groups[key]
        boot = Bootstrap(runs, "test_clean", by_id, args.n_boot)
        stats[key] = {n: boot.statistic(runs, n) for n in ("bacc", "auroc", "worst_domain_bacc")}
        cis = {n: M.ci(v) for n, v in stats[key].items()}
        b = lambda *p: seed_mean(runs, p)
        row = dict(
            val_worst=b("selection", "best_val_worst_domain_bacc"),
            bacc=b("test_clean", "at_0.5", "binary", "balanced_accuracy"), auroc=b("test_clean", "at_0.5", "binary", "auroc"),
            fake_recall=b("test_clean", "at_0.5", "binary", "fake_recall"), real_recall=b("test_clean", "at_0.5", "binary", "real_recall"),
            precision=b("test_clean", "at_0.5", "binary", "precision"), f1=b("test_clean", "at_0.5", "binary", "f1"),
            worst=b("test_clean", "at_0.5", "worst_domain_bacc"),
            jpeg_bacc=b("test_jpeg75", "at_0.5", "binary", "balanced_accuracy"), blur_bacc=b("test_blur1", "at_0.5", "binary", "balanced_accuracy"),
            jpeg_auroc=b("test_jpeg75", "at_0.5", "binary", "auroc"), blur_auroc=b("test_blur1", "at_0.5", "binary", "auroc"))
        out[str(key)] = dict(row=row, ci=cis, seeds=[r["cfg"]["seed"] for r in runs])
        md.append(f"| {arm} | {tag} | {lam:g} | {len(runs)} | {fmt(row['val_worst'])} | {fmt(row['bacc'])} [{cis['bacc'][0]:.3f}, {cis['bacc'][1]:.3f}] | "
                  f"{fmt(row['auroc'])} [{cis['auroc'][0]:.3f}, {cis['auroc'][1]:.3f}] | {fmt(row['fake_recall'])} | {fmt(row['real_recall'])} | {fmt(row['precision'])} | "
                  f"{fmt(row['f1'])} | {fmt(row['worst'])} [{cis['worst_domain_bacc'][0]:.3f}, {cis['worst_domain_bacc'][1]:.3f}] | "
                  f"{fmt(row['jpeg_bacc'])} | {fmt(row['blur_bacc'])} | {fmt(row['jpeg_auroc'])} | {fmt(row['blur_auroc'])} |")
    md.append("\n### Per-domain test metrics (clean, threshold 0.5)\n")
    md.append("| Arm | tag | λ | domain | n | bAcc | AUROC | fake recall | real recall |")
    md.append("|---|---|---:|---|---:|---|---|---|---|")
    for key in main_keys:
        runs = groups[key]
        domains = sorted(pick(runs[0], ("test_clean", "at_0.5", "by_domain")) or {})
        for d in domains:
            g = lambda m: seed_mean(runs, ("test_clean", "at_0.5", "by_domain", d, m))
            n = pick(runs[0], ("test_clean", "at_0.5", "by_domain", d, "n"))
            md.append(f"| {key[0]} | {key[1]} | {key[3]:g} | {d} | {n} | {fmt(g('balanced_accuracy'))} | {fmt(g('auroc'))} | {fmt(g('fake_recall'))} | {fmt(g('real_recall'))} |")
    md.append("\n### Auxiliary 3-class (λ > 0), clean test\n")
    md.append("| Arm | tag | λ | macro-F1 | accuracy | MANIPULATED recall | MANIPULATED precision | AI recall | REAL recall |")
    md.append("|---|---|---:|---|---|---|---|---|---|")
    for key in main_keys:
        runs = groups[key]
        if key[3] <= 0:
            continue
        t = lambda *p: seed_mean(runs, ("test_clean", "three_class") + p)
        md.append(f"| {key[0]} | {key[1]} | {key[3]:g} | {fmt(t('macro_f1'))} | {fmt(t('accuracy'))} | {fmt(t('per_class', 'MANIPULATED', 'recall'))} | "
                  f"{fmt(t('per_class', 'MANIPULATED', 'precision'))} | {fmt(t('per_class', 'AI_GENERATED', 'recall'))} | {fmt(t('per_class', 'REAL', 'recall'))} |")
    md.append("\n### Representation diagnostics (v_imgfor, frozen)\n")
    md.append("| Arm | tag | λ | participation ratio | rank@95% | domain probe bAcc (chance 0.33) | news JPEG/PNG probe bAcc (chance 0.50) |")
    md.append("|---|---|---:|---|---|---|---|")
    for key in main_keys:
        runs = groups[key]
        r = lambda *p: seed_mean(runs, ("representation",) + p)
        md.append(f"| {key[0]} | {key[1]} | {key[3]:g} | {fmt(r('effective_rank', 'participation_ratio'), 2)} | {fmt(r('effective_rank', 'rank_95'), 1)} | "
                  f"{fmt(r('domain_probe_real', 'balanced_accuracy'))} | {fmt(r('news_format_probe_real', 'balanced_accuracy'))} |")
    md.append("\n### Source-held-out protocols (clean test)\n")
    md.append("| Arm | protocol | λ | n test | bAcc | AUROC | fake recall | real recall |")
    md.append("|---|---|---:|---:|---|---|---|---|")
    for key in sorted(k for k in groups if k[2] != "main"):
        runs = groups[key]
        h = lambda m: seed_mean(runs, ("test_clean", "at_0.5", "binary", m))
        md.append(f"| {key[0]}{('/' + key[1]) if key[1] else ''} | {key[2]} | {key[3]:g} | {pick(runs[0], ('test_clean', 'at_0.5', 'binary', 'n'))} | "
                  f"{fmt(h('balanced_accuracy'))} | {fmt(h('auroc'))} | {fmt(h('fake_recall'))} | {fmt(h('real_recall'))} |")
    # paired deltas between every pair of main configurations with the same lambda and tag
    md.append("\n### Paired differences (group bootstrap, seed-mean; Δ = first − second)\n")
    md.append("| Comparison | λ | Δ test bAcc [95% CI] | Δ test AUROC [95% CI] | Δ worst-domain bAcc [95% CI] |")
    md.append("|---|---:|---|---|---|")
    deltas = {}
    arms = sorted({(k[0], k[1]) for k in main_keys})
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            if x[1] != y[1]:
                continue
            for lam in sorted({k[3] for k in main_keys}):
                kx, ky = (x[0], x[1], "main", lam), (y[0], y[1], "main", lam)
                if kx not in stats or ky not in stats:
                    continue
                cells = []
                for n in ("bacc", "auroc", "worst_domain_bacc"):
                    d = stats[kx][n] - stats[ky][n]
                    lo, hi = M.ci(d)
                    cells.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    deltas[f"{x[0]}-{y[0]}|{x[1]}|{lam:g}|{n}"] = dict(mean=float(d.mean()), ci=M.ci(d))
                md.append(f"| {x[0]} − {y[0]} | {lam:g} | " + " | ".join(cells) + " |")
    out["deltas"] = deltas
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (out_dir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
