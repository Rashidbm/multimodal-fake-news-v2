"""Metrics, representation diagnostics and group-aware bootstrap for the Image Branch v2 comparison."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y, score):
    y = np.asarray(y)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def binary_metrics(y, prob, threshold=0.5):
    y, pred = np.asarray(y), (np.asarray(prob) >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    fake_recall = tp / max(tp + fn, 1)
    real_recall = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * fake_recall / max(precision + fake_recall, 1e-12)
    return dict(n=int(len(y)), accuracy=(tp + tn) / max(len(y), 1), balanced_accuracy=(fake_recall + real_recall) / 2,
                auroc=auroc(y, prob), precision=precision, recall=fake_recall, f1=f1, fake_recall=fake_recall,
                real_recall=real_recall, tp=tp, tn=tn, fp=fp, fn=fn)


def per_group(y, prob, groups, threshold=0.5):
    """balanced accuracy / AUROC / fake recall for each label of `groups` that has both classes."""
    out = {}
    for g in sorted(set(groups)):
        m = np.asarray(groups) == g
        if len(set(np.asarray(y)[m])) < 2:
            continue
        r = binary_metrics(np.asarray(y)[m], np.asarray(prob)[m], threshold)
        out[str(g)] = dict(n=r["n"], balanced_accuracy=r["balanced_accuracy"], auroc=r["auroc"], fake_recall=r["fake_recall"], real_recall=r["real_recall"])
    return out


def worst_group_bacc(y, prob, groups, threshold=0.5):
    values = [v["balanced_accuracy"] for v in per_group(y, prob, groups, threshold).values()]
    return min(values) if values else float("nan")


def best_threshold(y, prob):
    """Threshold (midpoint between sorted scores) that maximizes balanced accuracy."""
    order = np.unique(prob)
    candidates = np.r_[0.5, (order[:-1] + order[1:]) / 2] if len(order) > 1 else np.array([0.5])
    scores = [binary_metrics(y, prob, t)["balanced_accuracy"] for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best])


def three_class(y3, pred3):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y3, pred3):
        cm[int(t), int(p)] += 1
    out = dict(confusion=cm.tolist(), per_class={})
    f1s = []
    for c, name in enumerate(("REAL", "AI_GENERATED", "MANIPULATED")):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c]
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        if cm[c].sum() > 0:
            f1s.append(f1)
        out["per_class"][name] = dict(precision=p, recall=r, f1=f1, support=int(cm[c].sum()))
    out["macro_f1"] = float(np.mean(f1s))
    out["accuracy"] = float(np.trace(cm) / max(cm.sum(), 1))
    return out


def effective_rank(v):
    """participation ratio (sum l)^2 / sum l^2 of the covariance spectrum, and #components for 95% variance."""
    v = v - v.mean(0, keepdims=True)
    s = np.linalg.svd(v, compute_uv=False) ** 2
    s = s[s > 0]
    ratio = float(s.sum() ** 2 / (s ** 2).sum())
    return dict(participation_ratio=ratio, rank_95=int(np.searchsorted(np.cumsum(s) / s.sum(), 0.95) + 1))


def linear_probe(v_train, y_train, v_test, y_test, seed=0):
    """Frozen-embedding linear probe; returns balanced accuracy and chance level."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return dict(balanced_accuracy=float("nan"), chance=float("nan"))
    scaler = StandardScaler().fit(v_train)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed).fit(scaler.transform(v_train), y_train)
    pred = clf.predict(scaler.transform(v_test))
    return dict(balanced_accuracy=float(balanced_accuracy_score(y_test, pred)), chance=1.0 / len(set(y_train)))


def bootstrap_groups(groups, n_boot, seed):
    """Index arrays of group-level resamples (with replacement), reusable across arms and seeds."""
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[j] for j in rng.integers(0, len(unique), len(unique))]) for _ in range(n_boot)]


def ci(values, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
