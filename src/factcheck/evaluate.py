"""Run a manifest through one mode, resumably, and compare runs.

Layout of an output directory (one per experiment):

    <out>/<mode>/config.json        configuration, manifest hash, evidence source
    <out>/<mode>/examples/<id>.json one result per example (errors included)
    <out>/<mode>/metrics.json
    <out>/<mode>/report.md
    <out>/comparison.md             written by ``compare``

Re-running the same command resumes: an example whose file already exists with
the same input hash is skipped, including recorded errors.  A run directory
whose saved configuration differs is refused; use a fresh directory.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import prompts
from .config import MODES, REPLAY_MODES, Config, sha256_file
from .metrics import compute_metrics, format_report, mcnemar_exact
from .pipeline import FactChecker, input_sha256


class RunConfigMismatch(RuntimeError):
    pass


def read_manifest(path: str | Path) -> list[dict]:
    rows, ids = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id"):
                raise ValueError(f"{path}:{n}: example has no id")
            if row["id"] in ids:
                raise ValueError(f"{path}:{n}: duplicate id {row['id']!r}")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no examples")
    tasks = {r.get("task", prompts.DEFAULT_TASK) for r in rows}
    if len(tasks) != 1:
        raise ValueError(f"{path} mixes tasks {sorted(tasks)}; use one manifest per task")
    return rows


def safe_name(example_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", example_id)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _run_config(cfg: Config, mode: str, manifest: Path, evidence_from: Path | None) -> dict:
    out = {"mode": mode, "config": cfg.identity(), "config_sha256": cfg.sha256(),
           "manifest_sha256": sha256_file(manifest), "model": cfg.model,
           "prompt_version": cfg.prompt_version}
    if evidence_from is not None:
        saved = json.loads((evidence_from / "config.json").read_text(encoding="utf-8"))
        out["evidence_from"] = {"config_sha256": saved["config_sha256"], "mode": saved["mode"],
                                "manifest_sha256": saved["manifest_sha256"]}
    return out


def run_manifest(manifest: str | Path, out_dir: str | Path, mode: str, cfg: Config,
                 evidence_from: str | Path | None = None, checker: FactChecker | None = None,
                 log=print) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    manifest = Path(manifest)
    examples = read_manifest(manifest)
    task = examples[0].get("task", prompts.DEFAULT_TASK)
    labels = prompts.labels(task)

    run_dir = Path(out_dir) / mode
    ex_dir = run_dir / "examples"
    ev_dir = None
    if mode in REPLAY_MODES:
        if evidence_from is None:
            raise ValueError(f"mode {mode} replays saved evidence: pass --evidence-from <out>/search")
        ev_dir = Path(evidence_from)
        if not (ev_dir / "config.json").is_file():
            raise FileNotFoundError(f"{ev_dir} is not a search run directory (no config.json)")
    elif evidence_from is not None:
        raise ValueError("--evidence-from is only used by the direct and assessed modes")

    run_cfg = _run_config(cfg, mode, manifest, ev_dir)
    if ev_dir is not None:
        if run_cfg["evidence_from"]["mode"] != "search":
            raise ValueError(f"{ev_dir} is a {run_cfg['evidence_from']['mode']} run, not a search run")
        if run_cfg["evidence_from"]["manifest_sha256"] != run_cfg["manifest_sha256"]:
            raise RunConfigMismatch(f"{ev_dir} was run on a different manifest")
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        if saved != run_cfg:
            changed = sorted(k for k in set(saved) | set(run_cfg) if saved.get(k) != run_cfg.get(k))
            if "config" in changed:
                changed += [f"config.{k}" for k in sorted(set(saved["config"]) | set(run_cfg["config"]))
                            if saved["config"].get(k) != run_cfg["config"].get(k)]
            raise RunConfigMismatch(f"{run_dir} was run with a different configuration ({', '.join(changed)}); "
                                    "use a fresh output directory")
    ex_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg_path, run_cfg)

    checker = checker or FactChecker(cfg)
    results, done, ran = [], 0, 0
    t0 = time.perf_counter()
    for i, ex in enumerate(examples, 1):
        path = ex_dir / f"{safe_name(ex['id'])}.json"
        in_hash = input_sha256(ex)
        if path.is_file():
            prev = json.loads(path.read_text(encoding="utf-8"))
            if prev.get("input_sha256") == in_hash and prev.get("config_sha256") == cfg.sha256():
                results.append(prev)
                done += 1
                continue
            raise RunConfigMismatch(f"{path} holds a result for different input or configuration; "
                                    "use a fresh output directory")
        saved = None
        if ev_dir is not None:
            saved_path = ev_dir / "examples" / path.name
            saved = (json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.is_file()
                     else {"mode": "search", "evidence": None, "errors": [{"stage": "missing_saved_run"}]})
        result = checker.run_example(ex, mode, saved=saved)
        result["input_sha256"] = in_hash
        result["subcategory"] = ex.get("subcategory")
        write_json(path, result)
        results.append(result)
        ran += 1
        mark = "ok " if result["correct"] else result["status"][:3] if result["status"] != "ok" else "x  "
        log(f"[{mode}] {i}/{len(examples)} {mark} {ex['id']} -> {result['prediction']} "
            f"(gold {result['label']}, {result['latency_s']}s)"
            + (f" {result['errors'][-1]['message'][:120]}" if result["status"] == "error" else ""))
    metrics = compute_metrics(results, labels)
    metrics.update(mode=mode, task=task, resumed=done, ran=ran,
                   wall_clock_s=round(time.perf_counter() - t0, 1))
    if mode in REPLAY_MODES:
        metrics["latency_s"]["note"] += "; replay latency excludes evidence collection"
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(
        format_report(mode, metrics, labels, {**run_cfg, "config_sha256": run_cfg["config_sha256"]}),
        encoding="utf-8")
    log(f"[{mode}] accuracy {metrics['accuracy']}  macro-F1 {metrics['macro_f1']}  "
        f"errors {metrics['status']['error']}  abstained {metrics['status']['abstained']}  -> {run_dir}")
    return metrics


# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------

PAIRS = [("search", "closed_book", "Does retrieval improve the complete system?"),
         ("assessed", "direct", "Does the assessment step help on identical evidence?")]


def _load_run(run_dir: Path) -> tuple[dict, dict]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    res = {}
    for p in sorted((run_dir / "examples").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        res[r["id"]] = r
    return cfg, res


def compare(out_dir: str | Path, log=print) -> str:
    out_dir = Path(out_dir)
    runs = {m: _load_run(out_dir / m) for m in MODES if (out_dir / m / "config.json").is_file()}
    if not runs:
        raise FileNotFoundError(f"no runs under {out_dir}")
    manifests = {c["manifest_sha256"] for c, _ in runs.values()}
    if len(manifests) != 1:
        raise RunConfigMismatch("runs in this directory used different manifests")
    task = next(iter(next(iter(runs.values()))[1].values()), {}).get("task", prompts.DEFAULT_TASK)
    labels = prompts.labels(task)
    matched = sorted(set.intersection(*(set(r) for _, r in runs.values())))
    lines = ["# Fact-check comparison", "",
             f"Task `{task}`. Matched examples present in every run: **{len(matched)}** "
             f"(runs: {', '.join(f'{m}={len(r)}' for m, (_, r) in runs.items())}).", "",
             "Errors and abstentions count as wrong. Replay modes (direct, assessed) reuse the search run's "
             "evidence, so their latency excludes evidence collection.", "",
             "| Run | Accuracy | Macro-F1 | Abstained | Errors | Median latency (s) | Config |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for m, (cfg, res) in runs.items():
        sub = [res[i] for i in matched]
        met = compute_metrics(sub, labels)
        summary[m] = met
        acc = "n/a" if met["accuracy"] is None else f"{100 * met['accuracy']:.1f}%"
        f1 = "n/a" if met["macro_f1"] is None else f"{100 * met['macro_f1']:.1f}%"
        lines.append(f"| {m} | {acc} | {f1} | {met['status']['abstained']} | {met['status']['error']} | "
                     f"{met['latency_s']['median']} | `{cfg['config_sha256'][:12]}` |")
    lines += ["", "## Paired comparisons", "",
              "Discordant pairs and an exact McNemar test on the matched examples. Examples are not "
              "independent when they share events or sources; treat p-values as indicative.", ""]
    for a, b, question in PAIRS:
        if a not in runs or b not in runs:
            lines.append(f"- {a} vs {b}: not available (run both modes).")
            continue
        ra, rb = runs[a][1], runs[b][1]
        both = sum(1 for i in matched if ra[i]["correct"] and rb[i]["correct"])
        only_a = sum(1 for i in matched if ra[i]["correct"] and not rb[i]["correct"])
        only_b = sum(1 for i in matched if rb[i]["correct"] and not ra[i]["correct"])
        neither = len(matched) - both - only_a - only_b
        lines += [f"### {a} vs {b}", "", question, "",
                  f"- both correct {both}, only {a} {only_a}, only {b} {only_b}, neither {neither}",
                  f"- exact McNemar p = {mcnemar_exact(only_a, only_b):.4f}", ""]
        if (a, b) == ("assessed", "direct"):
            same = sum(1 for i in matched
                       if (ra[i].get("evidence_origin") or {}).get("evidence_sha256") is not None
                       and (ra[i].get("evidence_origin") or {}).get("evidence_sha256")
                       == (rb[i].get("evidence_origin") or {}).get("evidence_sha256"))
            lines += [f"- examples where both used identical saved evidence: {same}/{len(matched)}", ""]
    text = "\n".join(lines) + "\n"
    (out_dir / "comparison.md").write_text(text, encoding="utf-8")
    write_json(out_dir / "comparison.json", {"matched": len(matched), "task": task, "runs": summary})
    log(text)
    return text
