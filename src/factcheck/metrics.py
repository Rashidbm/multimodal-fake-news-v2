"""Metrics over per-example results.

Errors and abstentions stay in the denominator: they count as wrong for
accuracy and as missed recall for their gold class in macro-F1.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def compute_metrics(results: list[dict], labels: list[str]) -> dict:
    n = len(results)
    status = Counter(r["status"] for r in results)
    scored = [r for r in results if r.get("label") is not None]
    correct = sum(1 for r in scored if r["correct"])

    per_class, f1s = {}, []
    for c in labels:
        tp = sum(1 for r in scored if r["status"] == "ok" and r["prediction"] == c and r["label"] == c)
        fp = sum(1 for r in scored if r["status"] == "ok" and r["prediction"] == c and r["label"] != c)
        support = sum(1 for r in scored if r["label"] == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                        "support": support, "predicted": tp + fp}
        if support:
            f1s.append(f1)

    columns = labels + ["abstained", "error"]
    confusion = {g: {c: 0 for c in columns} for g in labels}
    for r in scored:
        if r["label"] not in confusion:
            continue
        col = r["prediction"] if r["status"] == "ok" else r["status"]
        if col in confusion[r["label"]]:
            confusion[r["label"]][col] += 1

    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None and r["status"] != "error"]
    n_cit = [len(r.get("citations") or []) for r in results if r["status"] != "error"]
    error_stages = Counter(e.get("stage", "?") for r in results if r["status"] == "error" for e in r["errors"][-1:])
    retrieval = [r["retrieval"] for r in results if r.get("retrieval")]

    return {
        "n": n,
        "n_labelled": len(scored),
        "status": {"ok": status.get("ok", 0), "abstained": status.get("abstained", 0), "error": status.get("error", 0)},
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "accuracy_answered": (round(correct / sum(1 for r in scored if r["status"] == "ok"), 4)
                              if any(r["status"] == "ok" for r in scored) else None),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "per_class": per_class,
        "confusion": confusion,
        "latency_s": {"mean": round(statistics.fmean(latencies), 3) if latencies else None,
                      "median": _percentile(latencies, 0.5), "p90": _percentile(latencies, 0.9),
                      "max": max(latencies) if latencies else None,
                      "note": "excludes errored examples"},
        "errors_by_stage": dict(error_stages),
        "non_fatal_errors": sum(1 for r in results for e in r["errors"] if r["status"] != "error"),
        "citations": {"mean_valid": round(statistics.fmean(n_cit), 3) if n_cit else None,
                      "results_with_invalid": sum(1 for r in results if r.get("citation_errors"))},
        "retrieval": {
            "examples": len(retrieval),
            "mean_evidence_items": (round(statistics.fmean(len(r.get("evidence") or []) for r in results
                                                           if r.get("retrieval")), 3) if retrieval else None),
            "failed_queries": sum(1 for x in retrieval for q in x["queries"] if q["status"] == "error"),
            "fetch_status": dict(Counter(f["status"] for x in retrieval for f in x["fetches"])),
            "filtered": dict(sum((Counter(x["filtered"]) for x in retrieval), Counter())),
        },
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def format_report(name: str, m: dict, labels: list[str], cfg: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    lines = [f"# Fact-check run: {name}", "",
             f"Model `{cfg.get('model')}`, prompt version `{cfg.get('prompt_version')}`, "
             f"config `{cfg.get('config_sha256', '')[:12]}`.", "",
             "| Examples | Accuracy | Macro-F1 | OK | Abstained | Errors | Latency median / p90 (s) |",
             "|---|---|---|---|---|---|---|",
             f"| {m['n']} | {pct(m['accuracy'])} | {pct(m['macro_f1'])} | {m['status']['ok']} | "
             f"{m['status']['abstained']} | {m['status']['error']} | "
             f"{m['latency_s']['median']} / {m['latency_s']['p90']} |", "",
             "Errors and abstentions count as wrong in accuracy and macro-F1.", "",
             "## Per class", "", "| Class | Precision | Recall | F1 | Support | Predicted |", "|---|---|---|---|---|---|"]
    for c in labels:
        p = m["per_class"][c]
        lines.append(f"| {c} | {pct(p['precision'])} | {pct(p['recall'])} | {pct(p['f1'])} | {p['support']} | {p['predicted']} |")
    cols = labels + ["abstained", "error"]
    lines += ["", "## Confusion (rows = gold)", "", "| gold \\ predicted | " + " | ".join(cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
    for g in labels:
        lines.append(f"| {g} | " + " | ".join(str(m["confusion"][g][c]) for c in cols) + " |")
    if m["errors_by_stage"]:
        lines += ["", "Errors by stage: " + ", ".join(f"{k}={v}" for k, v in m["errors_by_stage"].items())]
    r = m["retrieval"]
    if r["examples"]:
        lines += ["", "## Retrieval", "",
                  f"- mean evidence items per example: {r['mean_evidence_items']}",
                  f"- failed queries: {r['failed_queries']}",
                  f"- page fetches: {r['fetch_status']}",
                  f"- results filtered: {r['filtered'] or 'none'}"]
    lines += ["", f"Citations: mean valid per answered example {m['citations']['mean_valid']}; "
                  f"examples with at least one rejected citation {m['citations']['results_with_invalid']}."]
    return "\n".join(lines) + "\n"
