"""Cache v_textfor for every row of the built CSV (guidelines section 4).

    python -m fnd.extract_textfor --csv data/processed/balanced_5group.csv \
                                  --out features/v_textfor.pt

Qwen2 is frozen, so a caption's pooled vector is identical in every epoch;
extracting once and caching is what keeps stage 2 minutes rather than days.

Output (torch.save):

    {
      "sample_ids": [str, ...],           # row order
      "features":   FloatTensor (N, H),   # pooled, float32
      "splits":     [str, ...],           # copied from the CSV
      "meta":       {...}                 # model, layer, dtype, pooling, n
    }

sample_ids ship with the tensor so stage 2 can join to v_semantic and
v_imgfor by id rather than by assumed row order.

Roughly 5-10 samples/s on one RTX 4090 (Qwen2-7B, bf16, batch 8,
max_len 512).  See docs/TEXT_FLUOROSCOPY.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from fnd.models.text_fluoroscopy import TextFluoroscopy, TextFluoroscopyConfig


def read_rows(csv_path: str | Path, split: str | None = None) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    for col in ("sample_id", "text"):
        if col not in rows[0]:
            raise KeyError(f"{csv_path} has no {col!r} column; found {list(rows[0])}")
    if split:
        rows = [r for r in rows if r.get("split") == split]
        if not rows:
            raise ValueError(f"no rows with split={split!r} in {csv_path}")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id values: features could not be joined unambiguously")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract v_textfor with a frozen Qwen2 (Text Fluoroscopy).")
    ap.add_argument("--csv", required=True, help="built CSV, e.g. data/processed/balanced_5group.csv")
    ap.add_argument("--out", default="features/v_textfor.pt")
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct",
                    help="use Qwen/Qwen2-0.5B-Instruct to smoke-test the pipeline on CPU")
    ap.add_argument("--layer", type=int, default=26,
                    help="index into hidden_states; 0 = embeddings, 1..num_layers = blocks")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8, help="8 fits Qwen2-7B bf16 on 24 GB; lower it if OOM")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default=None, help="only this split; default = all rows")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N rows")
    ap.add_argument("--no-truncate-layers", action="store_true",
                    help="keep the layers above --layer (slower, same output)")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    rows = read_rows(args.csv, args.split)
    if args.limit:
        rows = rows[: args.limit]

    cfg = TextFluoroscopyConfig(
        model_name=args.model, layer=args.layer, max_len=args.max_len,
        dtype=args.dtype, truncate_layers=not args.no_truncate_layers,
    )

    print("===== Text Fluoroscopy =====")
    print(f"csv     {args.csv}  ({len(rows)} rows"
          + (f", split={args.split}" if args.split else "") + ")")
    if device.type == "cuda":
        print(f"gpu     {torch.cuda.get_device_name(0)}")

    extractor = TextFluoroscopy(cfg, device=device)
    print(f"model   {extractor.describe()}")
    print("-" * 60)

    chunks: list[torch.Tensor] = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pooled = extractor.encode_texts([r["text"] for r in batch])
        # float32 on disk: written once, read every epoch, and avoids baking
        # bf16 rounding into the cached features.
        chunks.append(pooled.float().cpu())

        done = min(start + args.batch_size, len(rows))
        rate = done / max(time.time() - t0, 1e-9)
        eta = (len(rows) - done) / max(rate, 1e-9)
        print(f"  {done:>7}/{len(rows)}  {rate:5.1f} samples/s  eta {eta/60:5.1f} min", end="\r")
    print()

    features = torch.cat(chunks, dim=0)
    if features.shape[0] != len(rows):
        raise RuntimeError(f"got {features.shape[0]} vectors for {len(rows)} rows")
    if not torch.isfinite(features).all():
        raise RuntimeError("non-finite values in features: check dtype and model load")

    payload = {
        "sample_ids": [r["sample_id"] for r in rows],
        "features": features,
        "splits": [r.get("split", "") for r in rows],
        "meta": {
            "model_name": cfg.model_name,
            "hidden_size": extractor.hidden_size,
            "num_layers": extractor.num_layers,
            "layer_requested": args.layer,
            "layer_resolved": extractor.layer,
            "truncated_layers": extractor._truncated,
            "max_len": cfg.max_len,
            "dtype": str(extractor.dtype),
            "pooling": "masked_mean",
            "projected": False,
            "proj_dim_expected": cfg.proj_dim,
            "csv": str(args.csv),
            "split": args.split,
            "n": len(rows),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(payload["meta"], indent=2), encoding="utf-8")

    mb = out_path.stat().st_size / 1e6
    print("-" * 60)
    print(f"features  {tuple(features.shape)}")
    print(f"saved     {out_path}  ({mb:.1f} MB)   meta -> {meta_path}")
    print(f"elapsed   {(time.time() - t0)/60:.1f} min")
    print()
    print("Pooled hidden states, NOT projected to 768: TextForensicProjection")
    print("belongs inside the fusion module so it trains. Join on sample_ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
