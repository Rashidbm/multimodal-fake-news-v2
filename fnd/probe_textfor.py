"""Score the textual-forensic stream on its own (ablation).

    python -m fnd.probe_textfor --features features/v_textfor.pt \
                                --csv data/processed/balanced_5group.csv \
                                --out outputs/textfor_probe

extract_textfor.py produces vectors, not predictions, so the stream has no
accuracy of its own until something classifies those vectors.  This trains a
small head on the cached features and reports what the frozen LLM alone can
do, before any fusion with the image or semantic streams.  The head is a
diagnostic: its weights are never used again.

The head follows the Text Fluoroscopy paper: three fully connected layers
with Tanh (H -> 1024 -> 512 -> out).  Whichever of these the CSV carries are
trained in one run, on the same features and the same splits:

    text_fake     is the CAPTION fake or edited - rumour, word edit OR
                  generated.  This stream's own diagnostic target, and NOT
                  a measure of AI authorship: a human-written false rumour
                  has text_fake=1.
    ai_text       was the caption machine-generated, derived from the source
                  folder by scripts/enrich_provenance.py.  Blank rows (word
                  edits, ambiguous provenance) are excluded, never guessed.
    label_binary  is the POST fake.  True for out-of-context and tampered-
                  image pairs whose captions are genuine human prose, so it
                  is partly unanswerable from text alone; reported to show
                  the gap, not to grade this stream.
    5-class       the five scenarios.

Scoring this stream against label_binary alone would make working features
look broken: it asks the model to call real human writing "fake" whenever the
image or the pairing is the thing that is wrong.

Every score is printed beside a majority-class and a random baseline on the
same test split, because "68% accuracy" means nothing until you know that
guessing scores 20%.

`--where COL=VALUE` restricts the run to one slice of the CSV.  That is the
domain-matched control: `--where domain=gossip` holds the topic fixed and
varies only the authorship, so the gap against the full score is the topic
leakage.

Outputs in --out:
    metrics.json        every number below, machine-readable
    predictions.csv     one row per test sample, for error analysis
    split_ids.csv       which id landed in which split
    report.txt          the printed report
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

from fnd.data.records import GROUPS
from fnd.extract_textfor import find_id_column
from fnd.metrics import (
    binary_metrics,
    confusion_matrix,
    format_confusion,
    multiclass_metrics,
    per_group_accuracy,
)

TARGETS = {
    "text_fake": "is the caption fake or edited (rumour, edit OR generated) - not AI authorship",
    "ai_text": "was the caption machine-generated, derived from the source folder",
    "binary": "is the post fake - partly unanswerable from text alone",
    "multiclass": "which of the five scenarios",
}

TITLES = {
    "text_fake": "TEXT_FAKE  (is the caption fake or edited: rumour, edit OR generated)",
    "ai_text": "AI_TEXT  (was the caption machine-generated - provenance-derived)",
    "binary": "LABEL_BINARY  (is the post fake - not answerable from text alone)",
}


def short_label(group: str) -> str:
    """'fake_text_real_image' -> 'ft_ri', so confusion-matrix headers stay
    narrow without two classes abbreviating to the same string."""
    parts = group.split("_")
    if len(parts) < 4:
        return group
    return f"{parts[0][0]}{parts[1][0]}_{parts[2][0]}{parts[3][0]}"


class ProbeHead(nn.Module):
    """H -> 1024 -> 512 -> out, Tanh between (Yang et al., EMNLP 2024)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_aligned(features_path: str | Path, csv_path: str | Path, where: str | None = None) -> dict:
    """Join cached features to the CSV's labels by id.

    Joining by id rather than row order is the whole reason extract_textfor
    saves them: the two files are produced by separate runs.  Label columns
    are optional - whichever are present become probe targets.
    """
    payload = torch.load(features_path, weights_only=False)
    feats = payload["features"]
    ids = payload.get("ids") or payload["sample_ids"]

    with open(csv_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        raise ValueError(f"{csv_path} is empty")
    id_col = find_id_column(all_rows[0])

    if where:
        if "=" not in where:
            raise ValueError("--where must look like COL=VALUE, e.g. domain=gossip")
        wcol, wval = where.split("=", 1)
        before = len(all_rows)
        all_rows = [r for r in all_rows if str(r.get(wcol, "")) == wval]
        if not all_rows:
            raise ValueError(f"--where {where} matched no rows")
        print(f"--where {where}: {len(all_rows)} of {before} rows")

    rows = {r[id_col]: r for r in all_rows}
    if where:
        keep = [i in rows for i in ids]
        if not any(keep):
            raise ValueError(f"--where {where} left no rows that also have features")
        feats = feats[torch.tensor(keep)]
        ids = [i for i, k in zip(ids, keep) if k]
    else:
        missing = [i for i in ids if i not in rows]
        if missing:
            raise KeyError(
                f"{len(missing)} ids in the feature file are not in the CSV "
                f"(first: {missing[:3]}). The features came from a different build."
            )

    first = rows[ids[0]]

    def have(name):
        return name in first and str(first[name]) != ""

    def col(name, cast=int):
        return [cast(rows[i][name]) for i in ids]

    targets: dict[str, torch.Tensor] = {}
    if have("text_fake"):
        targets["text_fake"] = torch.tensor(col("text_fake"), dtype=torch.float32)
    if have("ai_text"):
        # blank = provenance excluded this row; -1 marks it and run_binary drops it
        targets["ai_text"] = torch.tensor(
            [int(rows[i]["ai_text"]) if str(rows[i]["ai_text"]).strip() != "" else -1
             for i in ids], dtype=torch.float32)
    if have("label_binary"):
        targets["binary"] = torch.tensor(col("label_binary"), dtype=torch.float32)

    y_idx = torch.tensor(col("label_index"), dtype=torch.long) if have("label_index") else None
    if not targets and y_idx is None:
        raise KeyError(
            "the CSV carries no label column. Expected at least one of: text_fake, "
            "ai_text, label_binary, label_index (or a 'group' column to derive them from)"
        )

    return {
        "ids": ids,
        "id_column": id_col,
        "x": feats.float(),
        "targets": targets,
        "y_idx": y_idx,
        "scenario": col("scenario") if have("scenario") else None,
        "subcategory": [rows[i].get("subcategory") or "?" for i in ids],
        "split": [rows[i]["split"] for i in ids],
        "meta": payload.get("meta", {}),
    }


def split_mask(data: dict, name: str) -> torch.Tensor:
    m = torch.tensor([s == name for s in data["split"]])
    if not m.any():
        raise ValueError(f"no rows with split={name!r}")
    return m


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def class_weights(y: torch.Tensor, out_dim: int) -> torch.Tensor | None:
    """Re-weight the loss by inverse class frequency.

    The five scenario groups are equal in size, but they do not divide evenly
    by any binary question: text_fake is 2 groups against 3 (40/60),
    label_binary is 1 against 4 (20/80).  Weighting the loss is the right
    lever here - dropping rows to force a 50/50 split would throw away real
    examples and, because the same CSV feeds all three streams, would tear the
    row set away from v_semantic and v_imgfor.
    """
    if out_dim == 1:
        pos, neg = float((y == 1).sum()), float((y == 0).sum())
        if pos == 0 or neg == 0:
            return None
        return torch.tensor([neg / pos], device=y.device)
    counts = torch.bincount(y, minlength=out_dim).float().clamp(min=1.0)
    return (counts.sum() / (out_dim * counts)).to(y.device)


def train_head(xtr, ytr, xva, yva, out_dim, epochs, lr, patience, batch_size,
               device, seed, log, weighted: bool = True) -> ProbeHead:
    """Train one head, keeping the weights from the best validation epoch."""
    torch.manual_seed(seed)
    head = ProbeHead(xtr.shape[1], out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    w = class_weights(ytr, out_dim) if weighted else None
    if out_dim == 1:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=w)
        if w is not None:
            log(f"    class balance: {int((ytr == 1).sum())} positive / "
                f"{int((ytr == 0).sum())} negative, pos_weight {w.item():.3f}")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w)

    best_score, best_state, bad = -1.0, None, 0
    n = xtr.shape[0]

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = head(xtr[idx])
            target = ytr[idx].unsqueeze(1) if out_dim == 1 else ytr[idx]
            loss = loss_fn(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            logits = head(xva)
            if out_dim == 1:
                prob = torch.sigmoid(logits).squeeze(1).cpu().tolist()
                score = binary_metrics(yva.cpu().int().tolist(), prob)["f1"]
            else:
                pred = logits.argmax(dim=1)
                score = multiclass_metrics(yva.cpu().tolist(), pred.cpu().tolist(), out_dim)["f1_macro"]

        if score > best_score:
            best_score, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1

        if epoch % 5 == 0 or epoch == 1:
            log(f"    epoch {epoch:>3}  train loss {total/n:.4f}  val {score:.4f}"
                f"{'  *' if bad == 0 else ''}")
        if bad >= patience:
            log(f"    early stop at epoch {epoch} (best val {best_score:.4f})")
            break

    head.load_state_dict(best_state)
    return head.eval()


def best_threshold(y_true: list[int], prob: list[float]) -> float:
    """The threshold maximising balanced accuracy on the validation split.

    A fixed 0.5 assumes the training prior carries over. When the positive
    rate differs between splits - as it does in this dataset - it does not,
    and 0.5 quietly costs recall on the rarer class. Chosen on val, never on
    test, so the reported test numbers stay honest.
    """
    if not y_true or len(set(y_true)) < 2:
        return 0.5
    best, best_score = 0.5, -1.0
    for t in [i / 100 for i in range(5, 100, 5)]:
        pred = [1 if p >= t else 0 for p in prob]
        tp = sum(1 for a, b in zip(y_true, pred) if a == 1 and b == 1)
        fn = sum(1 for a, b in zip(y_true, pred) if a == 1 and b == 0)
        tn = sum(1 for a, b in zip(y_true, pred) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(y_true, pred) if a == 0 and b == 1)
        rp = tp / (tp + fn) if tp + fn else 0.0
        rn = tn / (tn + fp) if tn + fp else 0.0
        if (rp + rn) / 2 > best_score:
            best, best_score = t, (rp + rn) / 2
    return best


def with_balanced(m: dict, threshold: float) -> dict:
    """binary_metrics plus the figures that survive a shifting class prior."""
    rn = m["tn"] / (m["tn"] + m["fp"]) if m["tn"] + m["fp"] else 0.0
    return {**m, "threshold": threshold, "recall_positive": m["recall"],
            "recall_negative": rn, "balanced_accuracy": (m["recall"] + rn) / 2}


def baselines(y_train: list[int], y_test: list[int], num_classes: int, seed: int) -> dict:
    """What a classifier with no access to the features would score."""
    majority = max(set(y_train), key=y_train.count)
    rng = random.Random(seed)
    rand = [rng.randrange(num_classes) for _ in y_test]
    return {"majority_class": majority,
            "majority_accuracy": sum(1 for t in y_test if t == majority) / len(y_test),
            "random_accuracy": sum(1 for t, p in zip(y_test, rand) if t == p) / len(y_test),
            "random_expected": 1.0 / num_classes}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Probe v_textfor on its own and report scores.")
    ap.add_argument("--features", default="features/v_textfor.pt")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/textfor_probe")
    ap.add_argument("--where", default=None,
                    help="restrict to one slice, e.g. domain=gossip (the domain-matched control)")
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

    data = load_aligned(args.features, args.csv, args.where)
    tr, va, te = (split_mask(data, s) for s in ("train", "val", "test"))
    m = data["meta"]

    log("=" * 68)
    log("Text Fluoroscopy probe: what this stream alone can do")
    log("=" * 68)
    log(f"  features {tuple(data['x'].shape)} from {m.get('model_name', '?')}")
    log(f"  layer    hidden_states[{m.get('layer_resolved', '?')}] of "
        f"{m.get('num_layers', '?')}, {m.get('pooling', '?')} pooling")
    log(f"  splits   train {int(tr.sum())} / val {int(va.sum())} / test {int(te.sum())}")
    if args.where:
        log(f"  slice    --where {args.where}")

    # Standardise with TRAIN statistics only: whole-set statistics would leak
    # test information into the features.
    x = data["x"]
    mu, sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True).clamp(min=1e-6)
    x = ((x - mu) / sd).to(device)

    results: dict = {"features": m, "id_column": data["id_column"], "where": args.where,
                     "splits": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}

    def run_binary(key: str, title: str, y: torch.Tensor):
        y = y.to(device)
        log(); log("-" * 68); log(title); log("-" * 68)

        rates = []
        for name, msk in (("train", tr), ("val", va), ("test", te)):
            mm = msk.to(device) & (y >= 0)
            k = int(mm.sum())
            rates.append((name, k, float(y[mm].mean()) if k else float("nan")))
        log("    positive rate: " + "  ".join(f"{n} {r:.3f} (n={k})" for n, k, r in rates))
        spread = max(r for _, _, r in rates) - min(r for _, _, r in rates)
        if spread > 0.05:
            log(f"    NOTE: the positive rate differs by {spread:.3f} across splits, so a")
            log("          fixed 0.5 threshold is miscalibrated. AUC and balanced accuracy")
            log("          are the figures to quote; the threshold below is picked on val.")

        # -1 means "no defensible label for this row": excluded, never guessed.
        known = y >= 0
        if not known.all():
            log(f"    {int((~known).sum())} rows have no label for this target and are excluded")
        ktr, kva, kte = (msk.to(device) & known for msk in (tr, va, te))
        if not (ktr.any() and kva.any() and kte.any()):
            log("    skipped: not enough labelled rows in every split")
            return None

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
        met_half = with_balanced(binary_metrics(y_te, prob), 0.5)
        base = baselines(y[ktr].cpu().int().tolist(), y_te, 2, args.seed)
        results[key] = {**met, "at_threshold_0.5": met_half,
                        "threshold_selected_on": "val", "baselines": base,
                        "positive_rate": {n: r for n, _, r in rates},
                        "target": TARGETS[key]}

        log()
        log(f"  AUC          {met['auc']:.4f}      (chance 0.5000; prior-independent)")
        log(f"  threshold    {thr:.3f}       (chosen on val, not test)")
        log(f"  accuracy     {met['accuracy']:.4f}      "
            f"(majority {base['majority_accuracy']:.4f}; at 0.5: {met_half['accuracy']:.4f})")
        log(f"  balanced acc {met['balanced_accuracy']:.4f}")
        log(f"  precision    {met['precision']:.4f}")
        log(f"  recall  pos  {met['recall_positive']:.4f}   neg {met['recall_negative']:.4f}")
        log(f"  F1           {met['f1']:.4f}")
        log(f"  tp {met['tp']}  tn {met['tn']}  fp {met['fp']}  fn {met['fn']}")
        return prob, y_te, kte.cpu()

    binary_out = {}
    for key in ("text_fake", "ai_text", "binary"):
        if key in data["targets"]:
            got = run_binary(key, TITLES[key], data["targets"][key])
            if got is not None:
                binary_out[key] = got

    # ---- 5-class scenario --------------------------------------------------
    pred_te = yi_te = None
    if data["y_idx"] is None:
        log()
        log("5-class: skipped, the CSV carries no label_index / group column")
    else:
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
                                 "target": TARGETS["multiclass"],
                                 "class_order_note": ("the project's five-scenario order; do NOT "
                                                      "renumber - the fusion class dictionary is "
                                                      "agreed separately")}
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
        log("    " + "   ".join(f"{short_label(g)}={g}" for g in GROUPS))

    # ---- per-scenario and per-source breakdowns -----------------------------
    # text_fake is not one phenomenon. MMFakeBench builds it from AI-generated
    # text (chatgpt_match, fever_AI, llm_*), human-written rumours (rumor_match,
    # politicat_match, gossipcop_match) and algorithmic word edits
    # (DGM4_text_edit_senti, coco_text_edit). Text Fluoroscopy detects machine
    # generation, so it should separate the first group and struggle on the
    # second - a human rumour carries no generation fingerprint. Breaking the
    # score down by sub-category turns "the stream scores X" into a statement
    # about what it actually detects.
    scen = data["scenario"]
    for key, (prob, y_te, kmask) in binary_out.items():
        pred = [1 if p >= 0.5 else 0 for p in prob]
        sub_k = [c for c, k in zip(data["subcategory"], kmask.tolist()) if k]
        scen_k = [s for s, k in zip(scen, kmask.tolist()) if k] if scen else None

        if scen_k:
            ps = per_group_accuracy(scen_k, y_te, pred)
            results[f"per_scenario_{key}"] = {str(k): v for k, v in ps.items()}
            log()
            log(f"  {key} accuracy inside each scenario:")
            for sc, d in ps.items():
                log(f"    {sc} {GROUPS[int(sc)-1]:<22} n={d['n']:<6} acc {d['accuracy']:.4f}")

        if len(set(sub_k)) > 1:
            per_sub = per_group_accuracy(sub_k, y_te, pred)
            results[f"per_subcategory_{key}"] = per_sub
            log()
            log(f"  {key} accuracy by source sub-category (what it really detects):")
            for c, d in sorted(per_sub.items(), key=lambda kv: -kv[1]["accuracy"]):
                note = "   (few samples)" if d["n"] < 20 else ""
                log(f"    {c:<28} n={d['n']:<6} acc {d['accuracy']:.4f}{note}")

    # ---- write -------------------------------------------------------------
    id_col = data["id_column"]
    with open(out_dir / "split_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([id_col, "split"])
        w.writerows(zip(data["ids"], data["split"]))

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    ids_te = [i for i, k in zip(data["ids"], te.tolist()) if k]
    cols: dict[str, list] = {id_col: ids_te,
                             "subcategory": [c for c, k in zip(data["subcategory"], te.tolist()) if k]}
    if scen:
        cols["scenario"] = [s for s, k in zip(scen, te.tolist()) if k]
    for key, (prob, y_te, _) in binary_out.items():
        if len(y_te) != len(ids_te):        # this head scored a subset of the test split
            continue
        cols[key] = y_te
        cols[f"prob_{key}"] = [f"{p:.6f}" for p in prob]
        cols[f"pred_{key}"] = [1 if p >= 0.5 else 0 for p in prob]
    if yi_te is not None:
        cols["label_index"], cols["pred_index"] = yi_te, pred_te
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        w.writerows(zip(*cols.values()))

    log()
    log("=" * 68)
    log(f"written to {out_dir}/  (metrics.json, predictions.csv, split_ids.csv, report.txt)")
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
