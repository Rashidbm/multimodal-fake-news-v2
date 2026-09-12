"""Score the textual-forensic stream on its own (ablation).

    python -m fnd.probe_textfor --features features/v_textfor.pt \
                                --csv data/processed/balanced_5group.csv \
                                --out outputs/textfor_probe

extract_textfor.py produces vectors, not predictions, so the stream has no
accuracy of its own until something classifies those vectors.  This trains a
small head on the cached features and reports what Qwen2 alone can do,
before any fusion with the image or semantic streams.

The head follows the Text Fluoroscopy paper: three fully connected layers
with Tanh (H -> 1024 -> 512 -> out).  Three are trained in one run, on the
same features and the same splits:

    text_fake     was the CAPTION machine-written.  This stream's own
                  question, and the number that says whether it works.
    label_binary  is the POST fake.  True for out-of-context and
                  tampered-image pairs whose captions are genuine human
                  prose, so it is partly unanswerable from text alone;
                  reported to show the gap, not to grade this stream.
    5-class       the five scenarios.

Scoring this stream against label_binary alone would make working features
look broken: it asks Qwen to call real human writing "fake" whenever the
image or the pairing is the thing that is wrong.

Every score is printed beside a majority-class and a random baseline on the
same test split, because "68% accuracy" means nothing until you know that
guessing scores 20%.

Outputs in --out:
    metrics.json        every number below, machine-readable
    predictions.csv     one row per test sample, for error analysis
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
from fnd.metrics import (
    binary_metrics,
    confusion_matrix,
    format_confusion,
    multiclass_metrics,
    per_scenario_accuracy,
)


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

def load_aligned(features_path: str | Path, csv_path: str | Path) -> dict:
    """Join cached features to the CSV's labels by sample_id.

    Joining by id rather than row order is the whole reason extract_textfor
    saves sample_ids: the two files are produced by separate runs.
    """
    payload = torch.load(features_path, weights_only=False)
    feats, ids = payload["features"], payload["sample_ids"]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = {r["sample_id"]: r for r in csv.DictReader(f)}

    missing = [i for i in ids if i not in rows]
    if missing:
        raise KeyError(
            f"{len(missing)} sample_ids in the feature file are not in the CSV "
            f"(first: {missing[:3]}). The features were extracted from a different build."
        )

    keep, y_bin, y_txt, y_idx, scen, splits = [], [], [], [], [], []
    for pos, sid in enumerate(ids):
        r = rows[sid]
        keep.append(pos)
        y_bin.append(int(r["label_binary"]))
        y_txt.append(int(r["text_fake"]))
        y_idx.append(int(r["label_index"]))
        scen.append(int(r["scenario"]))
        splits.append(r["split"])

    return {
        "ids": [ids[p] for p in keep],
        "x": feats[keep].float(),
        "y_bin": torch.tensor(y_bin, dtype=torch.float32),
        "y_txt": torch.tensor(y_txt, dtype=torch.float32),
        "y_idx": torch.tensor(y_idx, dtype=torch.long),
        "scenario": scen,
        "split": splits,
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

def train_head(xtr, ytr, xva, yva, out_dim, epochs, lr, patience, batch_size,
               device, seed, log) -> ProbeHead:
    """Train one head, keeping the weights from the best validation epoch."""
    torch.manual_seed(seed)
    head = ProbeHead(xtr.shape[1], out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if out_dim == 1 else nn.CrossEntropyLoss()

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
                pred = (torch.sigmoid(logits).squeeze(1) >= 0.5).float()
                score = binary_metrics(yva.cpu().tolist(),
                                       torch.sigmoid(logits).squeeze(1).cpu().tolist())["f1"]
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


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def baselines(y_train: list[int], y_test: list[int], num_classes: int, seed: int) -> dict:
    """What a classifier with no access to the features would score."""
    majority = max(set(y_train), key=y_train.count)
    maj_acc = sum(1 for t in y_test if t == majority) / len(y_test)

    rng = random.Random(seed)
    rand_pred = [rng.randrange(num_classes) for _ in y_test]
    rand_acc = sum(1 for t, p in zip(y_test, rand_pred) if t == p) / len(y_test)

    return {
        "majority_class": majority,
        "majority_accuracy": maj_acc,
        "random_accuracy": rand_acc,
        "random_expected": 1.0 / num_classes,
    }


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Probe v_textfor on its own and report scores.")
    ap.add_argument("--features", default="features/v_textfor.pt")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/textfor_probe")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    data = load_aligned(args.features, args.csv)
    tr, va, te = (split_mask(data, s) for s in ("train", "val", "test"))

    log("=" * 66)
    log("Text Fluoroscopy probe: what Qwen2 alone can do")
    log("=" * 66)
    m = data["meta"]
    log(f"features   {tuple(data['x'].shape)} from {m.get('model_name', '?')}, "
        f"layer {m.get('layer_resolved', '?')}, {m.get('pooling', '?')} pooling")
    log(f"splits     train {int(tr.sum())} / val {int(va.sum())} / test {int(te.sum())}")
    log(f"device     {device}")

    # Standardise with TRAIN statistics only: computing mean/std over the whole
    # set would leak test information into the features.
    x = data["x"]
    mu, sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True).clamp(min=1e-6)
    x = ((x - mu) / sd).to(device)

    y_bin, y_idx = data["y_bin"].to(device), data["y_idx"].to(device)
    scen_te = [s for s, k in zip(data["scenario"], te.tolist()) if k]
    results: dict = {"features": m, "splits": {
        "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}

    # ---- the two binary tasks ---------------------------------------------
    # text_fake is this stream's own question: was the CAPTION machine-written.
    # label_binary is the system's question: is the POST fake, which is true for
    # out-of-context and tampered-image pairs whose captions are genuine human
    # prose.  Qwen cannot see an image or a mismatched pairing, so the second
    # target is partly unanswerable from text alone - reported so the gap
    # between the two is visible, not as a measure of this stream's quality.
    def run_binary(key: str, title: str, y: torch.Tensor):
        log()
        log("-" * 66)
        log(title)
        log("-" * 66)
        head = train_head(x[tr], y[tr], x[va], y[va], 1, args.epochs, args.lr,
                          args.patience, args.batch_size, device, args.seed, log)
        with torch.no_grad():
            prob = torch.sigmoid(head(x[te])).squeeze(1).cpu().tolist()
        y_te = y[te].cpu().int().tolist()
        met = binary_metrics(y_te, prob)
        base = baselines(y[tr].cpu().int().tolist(), y_te, 2, args.seed)
        results[key] = {**met, "baselines": base}

        log()
        log(f"  accuracy   {met['accuracy']:.4f}      (majority baseline {base['majority_accuracy']:.4f})")
        log(f"  precision  {met['precision']:.4f}")
        log(f"  recall     {met['recall']:.4f}")
        log(f"  F1         {met['f1']:.4f}")
        log(f"  AUC        {met['auc']:.4f}      (chance 0.5000)")
        log(f"  tp {met['tp']}  tn {met['tn']}  fp {met['fp']}  fn {met['fn']}")
        return prob, y_te

    prob_txt, ytxt_te = run_binary(
        "text_fake", "TEXT_FAKE  (was the caption machine-written - this stream's own task)",
        data["y_txt"].to(device))

    prob_te, yb_te = run_binary(
        "binary", "LABEL_BINARY  (is the post fake - the system's task, not answerable from text alone)",
        y_bin)

    # ---- 5-class scenario --------------------------------------------------
    log()
    log("-" * 66)
    log("5-CLASS  (the five scenarios)")
    log("-" * 66)
    head_m = train_head(x[tr], y_idx[tr], x[va], y_idx[va], len(GROUPS), args.epochs, args.lr,
                        args.patience, args.batch_size, device, args.seed, log)
    with torch.no_grad():
        logits_te = head_m(x[te])
        pred_te = logits_te.argmax(1).cpu().tolist()
    yi_te = y_idx[te].cpu().tolist()
    mm = multiclass_metrics(yi_te, pred_te, len(GROUPS))
    mb = baselines(y_idx[tr].cpu().tolist(), yi_te, len(GROUPS), args.seed)
    cm = confusion_matrix(yi_te, pred_te, len(GROUPS))
    results["multiclass"] = {**mm, "baselines": mb, "confusion_matrix": cm,
                             "classes": list(GROUPS)}

    log()
    log(f"  accuracy   {mm['accuracy']:.4f}      (majority {mb['majority_accuracy']:.4f}, "
        f"random {mb['random_expected']:.4f})")
    log(f"  macro F1   {mm['f1_macro']:.4f}")
    log()
    log("  F1 per class:")
    for g, f1 in zip(GROUPS, mm["f1_per_class"]):
        log(f"    {g:<22} {f1:.4f}")

    log()
    log("  confusion matrix (rows = true, columns = predicted):")
    log(format_confusion(cm, [short_label(g) for g in GROUPS]))
    log("    " + "   ".join(f"{short_label(g)}={g}" for g in GROUPS))

    # ---- per scenario -------------------------------------------------------
    txt_pred_te = [1 if p >= 0.5 else 0 for p in prob_txt]
    ps = per_scenario_accuracy(scen_te, ytxt_te, txt_pred_te)
    results["per_scenario_text_fake"] = ps
    log()
    log("  text_fake accuracy inside each scenario (which caption type is missed):")
    for s, d in ps.items():
        log(f"    {s} {GROUPS[s-1]:<22} n={d['n']:<6} acc {d['accuracy']:.4f}")

    bin_pred_te = [1 if p >= 0.5 else 0 for p in prob_te]
    results["per_scenario_binary"] = per_scenario_accuracy(scen_te, yb_te, bin_pred_te)

    # ---- write -------------------------------------------------------------
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "scenario", "text_fake", "prob_text_fake", "pred_text_fake",
                    "label_binary", "prob_fake", "pred_binary", "label_index", "pred_index"])
        ids_te = [i for i, k in zip(data["ids"], te.tolist()) if k]
        for row in zip(ids_te, scen_te, ytxt_te, prob_txt, txt_pred_te,
                       yb_te, prob_te, bin_pred_te, yi_te, pred_te):
            sid, s, yt, pt, qt, yb, pr, pb, yi, pi = row
            w.writerow([sid, s, yt, f"{pt:.6f}", qt, yb, f"{pr:.6f}", pb, yi, pi])

    log()
    log("=" * 66)
    log(f"written to {out_dir}/  (metrics.json, predictions.csv, report.txt)")
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
