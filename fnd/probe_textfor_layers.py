"""Compare LLM text features on `text_fake` with the probe_textfor.py MLP head.

Step 1 - pick a layer on VAL only (test is never touched):

    python -m fnd.probe_textfor_layers select \
        --layers-dir D:/fnd_features/ministral3_8b \
        --csv dataset_2026-09-14/data/text_samples.csv \
        --out reports/llm_compare/ministral3_8b

Step 2 - score TEST once for the chosen features, several seeds:

    python -m fnd.probe_textfor_layers final --layers-dir D:/fnd_features/ministral3_8b --layer 30 ...
    python -m fnd.probe_textfor_layers final --features features/v_textfor.pt ...   # Qwen L30

Head, training loop (AdamW, early stop on val F1), class weighting and the
val-chosen threshold (max balanced accuracy) are imported from probe_textfor,
so every model is scored by identical code. Rows are the unique captions of
text_samples.csv, joined by text_asset_id.

`select` ranks layers by val AUC (threshold-free). Val also drives early
stopping, so the val numbers are slightly optimistic - equally for every layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from fnd.metrics import binary_metrics
from fnd.probe_textfor import best_threshold, train_head

TARGET = "text_fake"
REPORT_KEYS = ("auc", "f1", "balanced_accuracy", "accuracy", "precision", "recall")


def load_labels(csv_path: str | Path) -> dict[str, tuple[str, int]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["text_asset_id"]: (r["split"], int(r[TARGET])) for r in rows}


def align(ids: list[str], labels: dict) -> tuple[list[str], torch.Tensor]:
    """Row order of the feature file -> (splits, y). Join by id only."""
    missing = [i for i in ids if i not in labels]
    if missing:
        raise KeyError(f"{len(missing)} feature ids not in the CSV (first: {missing[:2]})")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in the feature file")
    splits = [labels[i][0] for i in ids]
    y = torch.tensor([labels[i][1] for i in ids], dtype=torch.float32)
    return splits, y


def layer_paths(layers_dir: Path) -> list[Path]:
    paths = sorted(layers_dir.glob("layer_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no layer_*.npy in {layers_dir}")
    return paths


def load_features(args) -> tuple[list[tuple[str, torch.Tensor]], list[str]]:
    """[(name, X)], ids. --features: one .pt; --layers-dir: chosen layer(s)."""
    if args.features:
        payload = torch.load(args.features, map_location="cpu", weights_only=False)
        name = Path(args.features).stem + f"_L{payload.get('meta', {}).get('layer_index', '?')}"
        return [(name, payload["features"].float())], list(payload["ids"])
    d = Path(args.layers_dir)
    ids = [str(i) for i in np.load(d / "ids.npy")]
    if getattr(args, "layer", None) is not None:
        paths = [d / f"layer_{args.layer:02d}.npy"]
    else:
        paths = layer_paths(d)
    # loaded lazily per layer by the caller to keep RAM low
    return [(p.stem, p) for p in paths], ids


def as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.from_numpy(np.load(x).astype(np.float32))


def fit_and_score(x, y, splits, seed, args, device, score_test: bool) -> dict:
    tr = torch.tensor([s == "train" for s in splits])
    va = torch.tensor([s == "val" for s in splits])
    te = torch.tensor([s == "test" for s in splits])
    xd, yd = x.to(device), y.to(device)
    head = train_head(xd[tr], yd[tr], xd[va], yd[va], 1, args.epochs, args.lr, args.patience,
                      args.batch_size, device, seed, lambda *_: None)
    with torch.no_grad():
        p_va = torch.sigmoid(head(xd[va])).squeeze(1).cpu().tolist()
        p_te = torch.sigmoid(head(xd[te])).squeeze(1).cpu().tolist() if score_test else None
    y_va = y[va].int().tolist()
    thr = best_threshold(y_va, p_va)
    out = {"seed": seed, "threshold": thr,
           "val": {k: binary_metrics(y_va, p_va, thr)[k] for k in REPORT_KEYS}}
    if score_test:
        out["test"] = {k: binary_metrics(y[te].int().tolist(), p_te, thr)[k] for k in REPORT_KEYS}
    return out


def mean_sd(vals: list[float]) -> dict:
    return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["select", "final"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--layers-dir", help="output dir of extract_textfor_layers.py")
    src.add_argument("--features", help="a single-layer .pt from extract_textfor.py")
    ap.add_argument("--layer", type=int, default=None, help="final: which layer of --layers-dir")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args(argv)

    if args.mode == "final" and args.layers_dir and args.layer is None:
        ap.error("final with --layers-dir needs --layer (chosen by `select`)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats, ids = load_features(args)
    splits, y = align(ids, load_labels(args.csv))
    print(f"rows {len(ids)}  train/val/test {splits.count('train')}/{splits.count('val')}/"
          f"{splits.count('test')}  target {TARGET}")

    if args.mode == "select":
        seed = args.seeds[0]
        table = []
        for name, src_x in feats:
            r = fit_and_score(as_tensor(src_x), y, splits, seed, args, device, score_test=False)
            table.append({"layer": name, **r["val"], "threshold": r["threshold"]})
            print(f"  {name}  val auc {r['val']['auc']:.4f}  f1 {r['val']['f1']:.4f}  "
                  f"bal-acc {r['val']['balanced_accuracy']:.4f}")
        best = max(table, key=lambda t: t["auc"])
        with open(out / "val_layers.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader(); w.writerows(table)
        (out / "selection.json").write_text(json.dumps(
            {"criterion": "max val AUC", "seed": seed, "chosen": best["layer"], "val": best,
             "test_touched": False}, indent=2), encoding="utf-8")
        print(f"chosen on val: {best['layer']} (auc {best['auc']:.4f})  -> {out / 'selection.json'}")
        return 0

    name, src_x = feats[0]
    x = as_tensor(src_x)
    runs = [fit_and_score(x, y, splits, s, args, device, score_test=True) for s in args.seeds]
    summary = {k: mean_sd([r["test"][k] for r in runs]) for k in REPORT_KEYS}
    (out / "metrics.json").write_text(json.dumps(
        {"features": args.features or str(Path(args.layers_dir) / f"{name}.npy"),
         "target": TARGET, "seeds": args.seeds, "test_summary": summary, "runs": runs},
        indent=2), encoding="utf-8")
    print(f"{name}  test over seeds {args.seeds}:")
    for k in REPORT_KEYS:
        print(f"  {k:<18} {summary[k]['mean']:.4f} +- {summary[k]['sd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
