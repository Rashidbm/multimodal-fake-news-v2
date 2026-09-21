"""Match vs. mismatch, head to head: which frozen backbone should this branch use?

    python -m fnd.compare_match --csv data/processed/balanced_5group.csv \
        --features clip=features/v_match_clip.pt \
        --features qwenvl=features/v_match_qwenvl.pt \
        --out outputs/match_compare

The decision this script exists to settle is narrow and worth stating plainly:
the match/mismatch branch needs one encoder, CLIP and Qwen-VL are both
plausible, and "Qwen-VL is a bigger model" is not evidence.  So both are frozen,
both are extracted over the same CSV, and everything after the encoder is held
identical - same rows, same splits, same standardisation, same head, same
optimiser, same seeds.  The encoder is the only thing that varies, so the
difference in the numbers is the thing being measured.

Three defences against reading a winner into noise:

  1. **Same rows, proven.**  The id lists are compared, not assumed.  If the
     two runs covered different rows the script stops; scoring two backbones on
     two different test sets would produce a confident and meaningless answer.
  2. **Several seeds.**  One head on one seed is a sample of size one.  Each
     backbone is trained --seeds times and the table reports mean +/- spread,
     so a gap smaller than the seed-to-seed wobble is visible as such.
  3. **A paired bootstrap on the test set.**  Seed-averaged probabilities for
     the two backbones are resampled together, sample by sample, and the AUC
     difference is recomputed on each resample.  Pairing matters: both
     backbones see the same pairs, and an easy resample lifts both, so the
     paired interval is much tighter than two independent ones.

The headline target is ``mismatch_clean`` (ooc vs genuine).  ``mismatch_all``
is reported beside it because that is what the branch meets at inference.
A winner is declared only when the bootstrap interval excludes zero; otherwise
the script says the two are indistinguishable on this data and points at cost,
which is the tie-breaker that then matters.

Outputs in --out:
    comparison.json     every number, machine-readable
    comparison.md       the table, for pasting into the report
    report.txt          the printed output
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

from fnd.metrics import binary_metrics
from fnd.probe_match import TARGETS, load_pairs, zero_shot_cosine
from fnd.probe_textfor import best_threshold, split_mask, train_head, with_balanced

DEFAULT_TARGETS = ("mismatch_clean", "mismatch_all")


def parse_features(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--features must look like NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if name in out:
            raise ValueError(f"--features {name}= given twice")
        out[name] = path
    if len(out) < 2:
        raise ValueError("give --features NAME=PATH at least twice; there is nothing to compare")
    return out


def check_same_rows(runs: dict[str, dict]) -> None:
    """Every backbone must have been extracted over the same rows in the same
    order, and each row must have landed in the same split.  Checked rather
    than assumed: a --limit on one extraction and not the other is an easy
    mistake, and it would silently turn this into a comparison of two
    different test sets."""
    names = list(runs)
    ref = runs[names[0]]
    for name in names[1:]:
        other = runs[name]
        if other["ids"] != ref["ids"]:
            extra = set(other["ids"]) - set(ref["ids"])
            missing = set(ref["ids"]) - set(other["ids"])
            raise ValueError(
                f"{name} and {names[0]} cover different rows: "
                f"{len(missing)} only in {names[0]}, {len(extra)} only in {name}"
                + (f" (first missing: {sorted(missing)[:3]})" if missing else "")
                + ". Re-extract both from the same CSV with the same --limit/--split."
            )
        if other["split"] != ref["split"]:
            n = sum(1 for a, b in zip(other["split"], ref["split"]) if a != b)
            raise ValueError(f"{name} and {names[0]} disagree on the split of {n} rows")


def paired_bootstrap_auc(y_true: list[int], prob_a: list[float], prob_b: list[float],
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample test pairs with replacement; recompute AUC(a) - AUC(b) each time.

    Both backbones are resampled on the SAME indices, which is what makes the
    comparison paired: the shared difficulty of a resampled set cancels, and
    what is left is the difference between the encoders.
    """
    rng = random.Random(seed)
    n = len(y_true)
    observed = binary_metrics(y_true, prob_a)["auc"] - binary_metrics(y_true, prob_b)["auc"]
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [y_true[i] for i in idx]
        if len(set(ys)) < 2:                     # a one-class resample has no AUC
            continue
        a = binary_metrics(ys, [prob_a[i] for i in idx])["auc"]
        b = binary_metrics(ys, [prob_b[i] for i in idx])["auc"]
        diffs.append(a - b)
    if not diffs:
        return {"observed_difference": observed, "resamples": 0}
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    wins = sum(1 for d in diffs if d > 0) / len(diffs)
    return {"observed_difference": observed, "ci95_low": lo, "ci95_high": hi,
            "p_first_better": wins, "resamples": len(diffs),
            "significant": lo > 0 or hi < 0}


def spread(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare match/mismatch backbones under one protocol.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", action="append", required=True, metavar="NAME=PATH",
                    help="repeat: --features clip=... --features qwenvl=...")
    ap.add_argument("--out", default="outputs/match_compare")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help=f"comma separated, from {sorted(TARGETS)}")
    ap.add_argument("--seeds", type=int, default=5, help="heads trained per backbone per target")
    ap.add_argument("--bootstrap", type=int, default=2000, help="paired bootstrap resamples")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"unknown target(s) {unknown}; choose from {sorted(TARGETS)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    paths = parse_features(args.features)
    runs = {name: load_pairs(path, args.csv) for name, path in paths.items()}
    check_same_rows(runs)

    ref = runs[next(iter(runs))]
    tr, va, te = (split_mask(ref, s) for s in ("train", "val", "test"))

    log("=" * 74)
    log("Match/mismatch: backbone comparison under one identical protocol")
    log("=" * 74)
    log(f"  rows     {len(ref['ids'])}  (train {int(tr.sum())} / val {int(va.sum())} / "
        f"test {int(te.sum())}) - identical for every backbone, verified by id")
    log(f"  protocol {args.seeds} seeds x same head (H->1024->512->1, Tanh), "
        f"lr {args.lr}, early stop patience {args.patience}")
    log(f"  varying  only the frozen encoder")
    log()
    for name, d in runs.items():
        m = d["meta"]
        detail = (f"{m.get('clip_feature_mode', '?')} features" if m.get("backbone") == "clip"
                  else f"hidden_states[{m.get('layer_resolved', '?')}], {m.get('pooling', '?')} pooling")
        log(f"  {name:<10} {tuple(d['x'].shape)}  {m.get('model_name', '?')}  ({detail})")

    # Standardise each backbone with its own TRAIN statistics: the two feature
    # spaces are not comparable in scale, and using test rows here would leak.
    std_x = {}
    for name, d in runs.items():
        x = d["x"]
        mu, sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True).clamp(min=1e-6)
        std_x[name] = ((x - mu) / sd).to(device)

    results: dict = {
        "csv": args.csv,
        "seeds": args.seeds,
        "rows": {"total": len(ref["ids"]), "train": int(tr.sum()),
                 "val": int(va.sum()), "test": int(te.sum())},
        "backbones": {name: d["meta"] for name, d in runs.items()},
        "targets": {},
    }
    md: list[str] = ["# Match/mismatch: backbone comparison", "",
                     f"Same {len(ref['ids'])} rows, same splits, same head, "
                     f"{args.seeds} seeds each. Only the frozen encoder differs.", ""]

    for target in targets:
        log()
        log("=" * 74)
        log(f"{target}  -  {TARGETS[target]}")
        log("=" * 74)

        y_ref = ref["targets"].get(target)
        if y_ref is None:
            log("  skipped: the CSV cannot support this target")
            continue
        known = y_ref >= 0
        ktr, kva, kte = (msk & known for msk in (tr, va, te))
        if not (ktr.any() and kva.any() and kte.any()):
            log("  skipped: not enough labelled rows in every split")
            continue
        y_te = y_ref[kte].int().tolist()
        log(f"  scored on {len(y_te)} test rows, positive rate {sum(y_te)/len(y_te):.3f}"
            + (f"  ({int((~known).sum())} rows excluded: no defensible label)"
               if not known.all() else ""))

        per_backbone: dict[str, dict] = {}
        mean_prob: dict[str, list[float]] = {}

        for name, d in runs.items():
            x = std_x[name]
            y = y_ref.to(device)
            ktr_d, kva_d, kte_d = (msk.to(device) for msk in (ktr, kva, kte))
            aucs, baccs, f1s, probs = [], [], [], []
            log()
            log(f"  --- {name} ---")
            for s in range(args.seeds):
                seed = 42 + s
                head = train_head(x[ktr_d], y[ktr_d], x[kva_d], y[kva_d], 1, args.epochs,
                                  args.lr, args.patience, args.batch_size, device, seed,
                                  lambda _m: None)
                with torch.no_grad():
                    p_te = torch.sigmoid(head(x[kte_d])).squeeze(1).cpu().tolist()
                    p_va = torch.sigmoid(head(x[kva_d])).squeeze(1).cpu().tolist()
                thr = best_threshold(y[kva_d].cpu().int().tolist(), p_va)
                met = with_balanced(binary_metrics(y_te, p_te, threshold=thr), thr)
                aucs.append(met["auc"]); baccs.append(met["balanced_accuracy"]); f1s.append(met["f1"])
                probs.append(p_te)
                log(f"    seed {seed}  AUC {met['auc']:.4f}  balanced acc "
                    f"{met['balanced_accuracy']:.4f}  F1 {met['f1']:.4f}  (threshold {thr:.2f})")

            # The seed-averaged probability is one pooled detector per backbone;
            # the paired bootstrap below compares those, not one lucky seed.
            mean_prob[name] = [sum(col) / len(col) for col in zip(*probs)]
            pooled = binary_metrics(y_te, mean_prob[name])
            zs = zero_shot_cosine(d["similarity"], y_ref, te) if d["similarity"] is not None else None
            per_backbone[name] = {
                "auc_mean": statistics.fmean(aucs), "auc_std": spread(aucs), "auc_per_seed": aucs,
                "balanced_accuracy_mean": statistics.fmean(baccs),
                "balanced_accuracy_std": spread(baccs),
                "f1_mean": statistics.fmean(f1s), "f1_std": spread(f1s),
                "auc_seed_averaged": pooled["auc"], "zero_shot_cosine": zs,
            }
            log(f"    mean      AUC {statistics.fmean(aucs):.4f} +/- {spread(aucs):.4f}"
                f"   balanced acc {statistics.fmean(baccs):.4f} +/- {spread(baccs):.4f}")
            if zs:
                log(f"    raw cosine AUC {zs['auc']:.4f}  (no head trained at all)")

        # ---- the table -----------------------------------------------------
        log()
        log(f"  {'backbone':<12} {'AUC':>18} {'balanced acc':>20} {'F1':>18}")
        for name, r in per_backbone.items():
            log(f"  {name:<12} {r['auc_mean']:>8.4f} +/-{r['auc_std']:<5.4f} "
                f"{r['balanced_accuracy_mean']:>10.4f} +/-{r['balanced_accuracy_std']:<5.4f} "
                f"{r['f1_mean']:>8.4f} +/-{r['f1_std']:<5.4f}")

        md.append(f"## `{target}` - {TARGETS[target]}")
        md.append("")
        md.append("| backbone | AUC | balanced accuracy | F1 | raw cosine AUC |")
        md.append("|---|---|---|---|---|")
        for name, r in per_backbone.items():
            zs = r["zero_shot_cosine"]
            cosine = f"{zs['auc']:.4f}" if zs else "-"   # the VLM has no cosine to report
            md.append(f"| {name} "
                      f"| {r['auc_mean']:.4f} ± {r['auc_std']:.4f} "
                      f"| {r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f} "
                      f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} "
                      f"| {cosine} |")
        md.append("")

        # ---- paired bootstrap on every pairing ------------------------------
        names = list(per_backbone)
        pairwise = {}
        log()
        log("  paired bootstrap on the test set (seed-averaged probabilities):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                bs = paired_bootstrap_auc(y_te, mean_prob[a], mean_prob[b],
                                          n_boot=args.bootstrap)
                pairwise[f"{a}_vs_{b}"] = bs
                verdict = (f"{a if bs['observed_difference'] > 0 else b} is ahead"
                           if bs.get("significant") else "indistinguishable")
                log(f"    AUC({a}) - AUC({b}) = {bs['observed_difference']:+.4f}  "
                    f"95% CI [{bs.get('ci95_low', float('nan')):+.4f}, "
                    f"{bs.get('ci95_high', float('nan')):+.4f}]  -> {verdict}")
                md.append(f"- `AUC({a}) - AUC({b})` = **{bs['observed_difference']:+.4f}**, "
                          f"95% CI [{bs.get('ci95_low', float('nan')):+.4f}, "
                          f"{bs.get('ci95_high', float('nan')):+.4f}] → {verdict}")
        md.append("")

        best = max(per_backbone, key=lambda n: per_backbone[n]["auc_mean"])
        decisive = any(v.get("significant") for v in pairwise.values())
        results["targets"][target] = {"description": TARGETS[target],
                                      "test_rows": len(y_te),
                                      "positive_rate": sum(y_te) / len(y_te),
                                      "per_backbone": per_backbone,
                                      "pairwise_bootstrap": pairwise,
                                      "highest_mean_auc": best,
                                      "difference_is_significant": decisive}
        log()
        if decisive:
            log(f"  VERDICT  {best} wins on {target}: the bootstrap interval excludes zero.")
        else:
            log(f"  VERDICT  {best} has the higher mean AUC on {target}, but the bootstrap")
            log("           interval includes zero - on this data the backbones are not")
            log("           distinguishable, so pick on cost, latency and memory instead.")

    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log()
    log("=" * 74)
    log(f"written to {out_dir}/  (comparison.json, comparison.md, report.txt)")
    log("The winner is the encoder the fusion stage should cache v_match from;")
    log("nothing here is retrained later, the vectors are.")
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
