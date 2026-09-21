"""Score the match/mismatch stream on its own, for one backbone (ablation).

    python -m fnd.probe_match --features features/v_match_clip.pt \
                              --csv data/processed/balanced_5group.csv \
                              --out outputs/match_probe_clip

extract_match produces vectors, not predictions, so the stream has no accuracy
of its own until something classifies those vectors.  This trains a small head
on the cached features - the same head the textual-forensic probe uses, so the
two streams' numbers are comparable - and reports what the frozen backbone
alone can do before any fusion.  The head is a diagnostic; its weights are
never used again.

Use ``fnd.compare_match`` to put two backbones side by side; this script is the
single-backbone view, with the per-scenario detail the comparison table leaves
out.

Four targets, whichever the CSV can support:

    mismatch_clean  ooc vs genuine ONLY.  The honest question for this stream:
                    both halves are individually authentic in both classes, so
                    the only thing separating them is whether the caption
                    belongs to the picture.  This is the headline number.
    mismatch_all    ooc vs the other four scenarios.  What the branch actually
                    faces at inference, and harder for a reason that is not
                    the stream's fault: a fake-text pair also fails to describe
                    its picture, so pushing it to the negative class asks the
                    encoder to call a genuine mismatch a match.
    binary          is the POST fake.  Partly unanswerable from agreement
                    alone - a Photoshopped image with its own true caption
                    still matches - and reported to show the gap.
    5-class         the five scenarios, for reference against the other streams.

Quoting mismatch_all alone would understate the stream; quoting mismatch_clean
alone would overstate what it contributes to the full system.  Both are printed.

Every score sits beside a majority-class and a random baseline on the same test
split, and - for CLIP - beside the raw cosine with no head trained at all.
That last line is the one that says whether the head is doing anything.

Outputs in --out:
    metrics.json        every number below, machine-readable
    predictions.csv     one row per test sample, for error analysis
    report.txt          the printed report
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from fnd.data.records import GROUPS
from fnd.extract_textfor import find_id_column
from fnd.metrics import (
    binary_metrics,
    confusion_matrix,
    format_confusion,
    multiclass_metrics,
    per_group_accuracy,
)
from fnd.probe_textfor import (
    baselines,
    best_threshold,
    short_label,
    split_mask,
    train_head,
    with_balanced,
)

TARGETS = {
    "mismatch_clean": "does the caption belong to the picture - ooc vs genuine only",
    "mismatch_all": "is this pair out-of-context - ooc vs all four other scenarios",
    "binary": "is the post fake - not fully answerable from agreement alone",
    "multiclass": "which of the five scenarios",
}

TITLES = {
    "mismatch_clean": "MISMATCH_CLEAN  (ooc vs genuine only - the stream's own question)",
    "mismatch_all": "MISMATCH_ALL  (ooc vs the other four scenarios - the deployment view)",
    "binary": "LABEL_BINARY  (is the post fake - agreement is only part of the answer)",
}

# -1 means "this row has no defensible label for this target": excluded from
# training and from scoring, never guessed.
EXCLUDED = -1.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def group_of(row: dict) -> str:
    """The scenario name for one CSV row, from whichever column carries it."""
    if row.get("group"):
        return row["group"]
    if str(row.get("label_index", "")) != "":
        return GROUPS[int(row["label_index"])]
    if str(row.get("scenario", "")) != "":
        return GROUPS[int(row["scenario"]) - 1]
    raise KeyError("the CSV carries no group, label_index or scenario column, so the "
                   "match/mismatch target cannot be derived")


def load_pairs(features_path: str | Path, csv_path: str | Path) -> dict:
    """Join cached features to the CSV's labels by id, and build the targets.

    Joining by id rather than by row order is the whole reason extract_match
    saves them: the two files come from separate runs, and the two backbones
    from two separate runs again.
    """
    payload = torch.load(features_path, weights_only=False)
    feats = payload["features"]
    ids = payload.get("ids") or payload["sample_ids"]

    with open(csv_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        raise ValueError(f"{csv_path} is empty")
    id_col = find_id_column(all_rows[0])
    rows = {r[id_col]: r for r in all_rows}

    missing = [i for i in ids if i not in rows]
    if missing:
        raise KeyError(
            f"{len(missing)} ids in the feature file are not in the CSV "
            f"(first: {missing[:3]}). The features came from a different build."
        )

    groups = [group_of(rows[i]) for i in ids]
    unknown = sorted({g for g in groups if g not in GROUPS})
    if unknown:
        raise ValueError(f"unknown group(s) in the CSV: {unknown}")

    targets = {
        # ooc against genuine; the three tampered scenarios are excluded rather
        # than folded into either side.
        "mismatch_clean": torch.tensor(
            [1.0 if g == "ooc" else (0.0 if g == "genuine" else EXCLUDED) for g in groups]),
        "mismatch_all": torch.tensor([1.0 if g == "ooc" else 0.0 for g in groups]),
    }
    first = rows[ids[0]]
    if "label_binary" in first and str(first["label_binary"]) != "":
        targets["binary"] = torch.tensor([float(rows[i]["label_binary"]) for i in ids])

    return {
        "ids": ids,
        "id_column": id_col,
        "x": feats.float(),
        "similarity": payload.get("similarity"),
        "targets": targets,
        "group": groups,
        "y_idx": torch.tensor([GROUPS.index(g) for g in groups], dtype=torch.long),
        "subcategory": [rows[i].get("subcategory") or "?" for i in ids],
        "split": [rows[i]["split"] for i in ids],
        "meta": payload.get("meta", {}),
    }


def zero_shot_cosine(sim: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict | None:
    """What the raw CLIP cosine scores with no head trained at all.

    A mismatch should sit at a *low* cosine, so the detector score is the
    negated similarity.  If the trained head cannot beat this, the head is
    decoration and the honest summary of the stream is one number per pair.
    """
    if sim is None:
        return None
    keep = mask & (y >= 0)
    y_true = y[keep].int().tolist()
    if len(set(y_true)) < 2:
        return None
    score = (-sim[keep]).tolist()
    return {"auc": binary_metrics(y_true, score)["auc"], "n": len(y_true),
            "note": "no training: -cosine(image, text) used directly as the mismatch score"}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Probe v_match on its own and report scores.")
    ap.add_argument("--features", required=True, help="e.g. features/v_match_clip.pt")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/match_probe")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weight", action="store_true",
                    help="do not re-weight the loss by inverse class frequency")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    data = load_pairs(args.features, args.csv)
    tr, va, te = (split_mask(data, s) for s in ("train", "val", "test"))
    m = data["meta"]

    log("=" * 68)
    log(f"Match/mismatch probe: what the {m.get('backbone', '?')} stream alone can do")
    log("=" * 68)
    log(f"  features {tuple(data['x'].shape)} from {m.get('model_name', '?')}")
    if m.get("backbone") == "clip":
        log(f"  composed {m.get('clip_feature_mode', '?')} over {m.get('embed_dim', '?')}-d embeddings")
    else:
        log(f"  layer    hidden_states[{m.get('layer_resolved', '?')}] of {m.get('num_layers', '?')}, "
            f"{m.get('pooling', '?')} pooling")
    log(f"  splits   train {int(tr.sum())} / val {int(va.sum())} / test {int(te.sum())}")

    # Standardise with TRAIN statistics only: whole-set statistics would leak
    # test information into the features.
    x = data["x"]
    mu, sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True).clamp(min=1e-6)
    x = ((x - mu) / sd).to(device)
    sim = data["similarity"]

    results: dict = {"features": m, "id_column": data["id_column"],
                     "splits": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}

    def run_binary(key: str, y: torch.Tensor):
        y = y.to(device)
        log(); log("-" * 68); log(TITLES[key]); log("-" * 68)

        known = y >= 0
        if not known.all():
            log(f"    {int((~known).sum())} rows have no label for this target and are excluded")
        ktr, kva, kte = (msk.to(device) & known for msk in (tr, va, te))
        if not (ktr.any() and kva.any() and kte.any()):
            log("    skipped: not enough labelled rows in every split")
            return None
        for name, msk in (("train", ktr), ("val", kva), ("test", kte)):
            log(f"    {name:<5} n={int(msk.sum()):<6} positive rate {float(y[msk].mean()):.3f}")

        head = train_head(x[ktr], y[ktr], x[kva], y[kva], 1, args.epochs, args.lr,
                          args.patience, args.batch_size, device, args.seed, log,
                          weighted=not args.no_class_weight)
        with torch.no_grad():
            prob = torch.sigmoid(head(x[kte])).squeeze(1).cpu().tolist()
            prob_va = torch.sigmoid(head(x[kva])).squeeze(1).cpu().tolist()
        y_te, y_va = y[kte].cpu().int().tolist(), y[kva].cpu().int().tolist()

        # Choose the operating point on validation, never on test.
        thr = best_threshold(y_va, prob_va)
        met = with_balanced(binary_metrics(y_te, prob, threshold=thr), thr)
        base = baselines(y[ktr].cpu().int().tolist(), y_te, 2, args.seed)
        zs = zero_shot_cosine(sim, y.cpu(), te) if sim is not None else None
        results[key] = {**met, "threshold_selected_on": "val", "baselines": base,
                        "zero_shot_cosine": zs, "target": TARGETS[key]}

        log()
        log(f"  AUC          {met['auc']:.4f}      (chance 0.5000; prior-independent)")
        if zs:
            delta = met["auc"] - zs["auc"]
            log(f"  cosine AUC   {zs['auc']:.4f}      (no head at all; the head adds {delta:+.4f})")
        log(f"  threshold    {thr:.3f}       (chosen on val, not test)")
        log(f"  accuracy     {met['accuracy']:.4f}      (majority {base['majority_accuracy']:.4f})")
        log(f"  balanced acc {met['balanced_accuracy']:.4f}")
        log(f"  precision    {met['precision']:.4f}")
        log(f"  recall  pos  {met['recall_positive']:.4f}   neg {met['recall_negative']:.4f}")
        log(f"  F1           {met['f1']:.4f}")
        log(f"  tp {met['tp']}  tn {met['tn']}  fp {met['fp']}  fn {met['fn']}")
        return prob, y_te, kte.cpu()

    binary_out = {}
    for key in ("mismatch_clean", "mismatch_all", "binary"):
        if key in data["targets"]:
            got = run_binary(key, data["targets"][key])
            if got is not None:
                binary_out[key] = got

    # ---- 5-class reference -------------------------------------------------
    y_idx = data["y_idx"].to(device)
    log(); log("-" * 68); log("5-CLASS  (the five scenarios)"); log("-" * 68)
    head_m = train_head(x[tr], y_idx[tr], x[va], y_idx[va], len(GROUPS), args.epochs,
                        args.lr, args.patience, args.batch_size, device, args.seed, log,
                        weighted=not args.no_class_weight)
    with torch.no_grad():
        pred_te = head_m(x[te]).argmax(1).cpu().tolist()
    yi_te = y_idx[te].cpu().tolist()
    mm = multiclass_metrics(yi_te, pred_te, len(GROUPS))
    mb = baselines(y_idx[tr].cpu().tolist(), yi_te, len(GROUPS), args.seed)
    cm = confusion_matrix(yi_te, pred_te, len(GROUPS))
    recall_per_class = [cm[i][i] / sum(cm[i]) if sum(cm[i]) else 0.0 for i in range(len(GROUPS))]
    results["multiclass"] = {**mm, "baselines": mb, "confusion_matrix": cm,
                             "classes": list(GROUPS), "recall_per_class": recall_per_class,
                             "target": TARGETS["multiclass"]}
    log()
    log(f"  accuracy   {mm['accuracy']:.4f}      (majority {mb['majority_accuracy']:.4f}, "
        f"random {mb['random_expected']:.4f})")
    log(f"  macro F1   {mm['f1_macro']:.4f}")
    log()
    log("  per class:")
    log(f"    {'class':<22} {'F1':>8} {'recall':>8}")
    for g, f1, rc in zip(GROUPS, mm["f1_per_class"], recall_per_class):
        log(f"    {g:<22} {f1:>8.4f} {rc:>8.4f}")
    log()
    log("  confusion matrix (rows = true, columns = predicted):")
    log(format_confusion(cm, [short_label(g) for g in GROUPS]))

    # ---- where the mismatch score fires ------------------------------------
    # A mismatch detector is not only judged by its accuracy on ooc. If it also
    # fires on fake_text_real_image it is reading "caption does not describe the
    # picture", which is the honest description of what agreement can see, and
    # the fusion stage needs to know that rather than discover it later.
    if "mismatch_all" in binary_out:
        prob, _, kmask = binary_out["mismatch_all"]
        groups_te = [g for g, k in zip(data["group"], kmask.tolist()) if k]
        by_group: dict[str, list[float]] = {}
        for g, p in zip(groups_te, prob):
            by_group.setdefault(g, []).append(p)
        means = {g: sum(v) / len(v) for g, v in by_group.items()}
        results["mean_mismatch_probability_per_group"] = {
            g: {"mean_probability": means[g], "n": len(by_group[g])} for g in by_group}
        log()
        log("  mean predicted mismatch probability inside each scenario")
        log("  (what the stream is really reacting to, not just how often it is right):")
        for g in GROUPS:
            if g in means:
                log(f"    {g:<22} n={len(by_group[g]):<6} mean p(mismatch) {means[g]:.4f}")

    for key, (prob, y_te, kmask) in binary_out.items():
        sub_k = [c for c, k in zip(data["subcategory"], kmask.tolist()) if k]
        if len(set(sub_k)) > 1:
            pred = [1 if p >= 0.5 else 0 for p in prob]
            per_sub = per_group_accuracy(sub_k, y_te, pred)
            results[f"per_subcategory_{key}"] = per_sub
            log()
            log(f"  {key} accuracy by source sub-category:")
            for c, d in sorted(per_sub.items(), key=lambda kv: -kv[1]["accuracy"]):
                note = "   (few samples)" if d["n"] < 20 else ""
                log(f"    {c:<28} n={d['n']:<6} acc {d['accuracy']:.4f}{note}")

    # ---- write -------------------------------------------------------------
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    id_col = data["id_column"]
    ids_te = [i for i, k in zip(data["ids"], te.tolist()) if k]
    cols: dict[str, list] = {id_col: ids_te,
                             "group": [g for g, k in zip(data["group"], te.tolist()) if k]}
    if sim is not None:
        cols["cosine"] = [f"{v:.6f}" for v, k in zip(sim.tolist(), te.tolist()) if k]
    for key, (prob, y_te, kmask) in binary_out.items():
        if len(y_te) != len(ids_te):        # this head scored a subset of the test split
            continue
        cols[key] = y_te
        cols[f"prob_{key}"] = [f"{p:.6f}" for p in prob]
    cols["label_index"], cols["pred_index"] = yi_te, pred_te
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        w.writerows(zip(*cols.values()))

    log()
    log("=" * 68)
    log(f"written to {out_dir}/  (metrics.json, predictions.csv, report.txt)")
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
