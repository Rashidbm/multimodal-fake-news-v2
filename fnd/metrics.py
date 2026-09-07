"""Metrics without external dependencies, so results are reproducible anywhere.

Binary: accuracy, precision, recall, F1 (fake = positive class), AUC.
Multi-class: accuracy, macro F1.
Per scenario: accuracy of the binary decision inside each of the 5 scenarios,
which is the diagnostic we care most about (which kind of fake is missed).
"""
from __future__ import annotations

from collections import defaultdict


def binary_metrics(y_true: list[int], prob: list[float], threshold: float = 0.5) -> dict:
    pred = [1 if p >= threshold else 0 for p in prob]
    tp = sum(1 for t, p in zip(y_true, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, pred) if t == 1 and p == 0)
    n = max(len(y_true), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": n, "accuracy": (tp + tn) / n, "precision": precision, "recall": recall,
            "f1": f1, "auc": auc(y_true, prob), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def auc(y_true: list[int], score: list[float]) -> float:
    """Area under the ROC curve via the rank statistic (ties get half credit)."""
    pos = [s for t, s in zip(y_true, score) if t == 1]
    neg = [s for t, s in zip(y_true, score) if t == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted((s, t) for t, s in zip(y_true, score))
    # average ranks with ties
    ranks, i = [0.0] * len(ranked), 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sum_pos = sum(r for r, (_, t) in zip(ranks, ranked) if t == 1)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def multiclass_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    n = max(len(y_true), 1)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"n": n, "accuracy": acc, "f1_macro": sum(f1s) / num_classes, "f1_per_class": f1s}


def per_scenario_accuracy(scenarios: list[int], y_true: list[int], y_pred: list[int]) -> dict:
    hit, tot = defaultdict(int), defaultdict(int)
    for s, t, p in zip(scenarios, y_true, y_pred):
        tot[s] += 1
        hit[s] += int(t == p)
    return {int(s): {"n": tot[s], "accuracy": hit[s] / tot[s]} for s in sorted(tot)}
