"""Cache v_textfor for every row of the built CSV (guidelines section 4).

    python -m fnd.extract_textfor --csv data/processed/balanced_5group.csv \
                                  --out features/v_textfor.pt

The model is frozen, so a caption's pooled vector is identical in every epoch;
extracting once and caching is what keeps stage 2 minutes rather than days.

Writes three files beside --out:

    v_textfor.pt    {"ids", "features" (N, H) float32, "splits", "meta"}
    v_textfor.npz   the same ids + `pooled_features`, no object arrays,
                    which is what the fusion stage's load_streams() reads
    v_textfor.json  the provenance record

The ids ship with the tensor so the fusion stage joins to v_semantic and
v_imgfor by id rather than by assumed row order.

About 8 samples/s on one RTX 4080 Super (Qwen3.5-9B, bf16, batch 8,
max_len 96).  See docs/TEXT_FLUOROSCOPY.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fnd.models.text_fluoroscopy import TextFluoroscopy, TextFluoroscopyConfig

# The guidelines key rows by text_asset_id; the repo build writes sample_id.
# Accept either and carry the chosen name through to the outputs.
ID_COLUMNS = ("text_asset_id", "sample_id", "id")


def find_id_column(row: dict) -> str:
    for name in ID_COLUMNS:
        if name in row:
            return name
    raise KeyError(f"no id column: expected one of {ID_COLUMNS}, found {list(row)}")


def read_rows(csv_path: str | Path, split: str | None = None) -> tuple[list[dict], str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    id_col = find_id_column(rows[0])
    if "text" not in rows[0]:
        raise KeyError(f"{csv_path} has no 'text' column; found {list(rows[0])}")
    if split:
        rows = [r for r in rows if r.get("split") == split]
        if not rows:
            raise ValueError(f"no rows with split={split!r} in {csv_path}")
    ids = [r[id_col] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {id_col} values: features could not be joined unambiguously")
    return rows, id_col


def evenly_spaced(rows: list[dict], limit: int) -> list[dict]:
    """Take `limit` rows spread across the file, not the first `limit`.

    build.py writes the CSV grouped by scenario (all ooc, then all
    fake_text_real_image, ...), so rows[:200] would be 200 rows of one class
    from one part of the split. A stride keeps every scenario and every split
    represented, which is what makes a subset usable as a stand-in for the
    full file while the real extraction runs.
    """
    if limit >= len(rows):
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract v_textfor with a frozen Qwen (Text Fluoroscopy).")
    ap.add_argument("--csv", required=True, help="built CSV, e.g. data/processed/balanced_5group.csv")
    ap.add_argument("--out", default="features/v_textfor.pt")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="use Qwen/Qwen2-0.5B-Instruct to smoke-test the plumbing on CPU")
    ap.add_argument("--layer", type=int, default=30,
                    help="index into hidden_states; 0 = embeddings, 1..num_layers = blocks")
    ap.add_argument("--max-len", type=int, default=96,
                    help="longest caption measured on this dataset was 45 tokens")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Qwen3.5-9B in bf16 is ~18 GB; lower this if the GPU spills to system RAM")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default=None, help="only this split; default = all rows")
    ap.add_argument("--limit", type=int, default=None,
                    help="only N rows, spread evenly across the file so every "
                         "scenario and split stays represented")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    rows, id_col = read_rows(args.csv, args.split)
    if args.limit:
        rows = evenly_spaced(rows, args.limit)

    cfg = TextFluoroscopyConfig(model_name=args.model, layer=args.layer,
                                max_len=args.max_len, dtype=args.dtype)

    print("===== Text Fluoroscopy =====")
    print(f"csv     {args.csv}  ({len(rows)} rows, id column {id_col!r}"
          + (f", split={args.split}" if args.split else "") + ")")
    if device.type == "cuda":
        print(f"gpu     {torch.cuda.get_device_name(0)}")

    extractor = TextFluoroscopy(cfg, device=device)
    print(f"model   {extractor.describe()}")
    print("-" * 60)

    chunks: list[torch.Tensor] = []
    lengths: list[int] = []
    truncated: list[dict] = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pooled, lens = extractor.encode_texts([r["text"] for r in batch])
        # float32 on disk: written once, read every epoch, and avoids baking
        # bf16 rounding into the cached features.
        chunks.append(pooled.float().cpu())
        lengths.extend(lens)
        for r, n_tok in zip(batch, lens):
            if n_tok > cfg.max_len:
                truncated.append({"id": r[id_col], "tokens": n_tok})

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

    ids = [r[id_col] for r in rows]
    meta = {
        "model_name": cfg.model_name,
        "hidden_size": extractor.hidden_size,
        "num_layers": extractor.num_layers,
        "layer_requested": args.layer,
        "layer_resolved": extractor.layer,
        "layer_convention": ("hidden_states[i]: index 0 is the embedding output, so "
                             "index i is the output of the i-th transformer block"),
        "max_len": cfg.max_len,
        "dtype": str(extractor.dtype),
        "pooling": "masked_mean",
        "projected": False,
        "proj_dim_expected": cfg.proj_dim,
        "id_column": id_col,
        "csv": str(args.csv),
        "split": args.split,
        "n": len(rows),
        "tokens": {
            "max": max(lengths, default=0),
            "mean": round(sum(lengths) / max(len(lengths), 1), 2),
            "over_max_length": len(truncated),
            "truncated_fraction": round(len(truncated) / max(len(rows), 1), 4),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "features": features,
                "splits": [r.get("split", "") for r in rows], "meta": meta}, out_path)

    # The guidelines ask for an .npz keyed by the id, with no object arrays,
    # and it is what the fusion stage's load_streams() reads.
    npz_path = out_path.with_suffix(".npz")
    np.savez(npz_path, **{id_col: np.array(ids, dtype="U64"),
                          "pooled_features": features.numpy().astype("float32")})

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if truncated:
        trunc_path = out_path.with_name(out_path.stem + "_truncated.csv")
        with open(trunc_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([id_col, "tokens"])
            for t in truncated:
                w.writerow([t["id"], t["tokens"]])

    print("-" * 60)
    print(f"features  {tuple(features.shape)}   tokens max {meta['tokens']['max']} "
          f"mean {meta['tokens']['mean']}   truncated {len(truncated)} "
          f"({100 * meta['tokens']['truncated_fraction']:.1f}%)")
    print(f"saved     {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"          {npz_path}   <- keyed by {id_col}, for the fusion stage")
    print(f"          {meta_path}")
    if truncated:
        print(f"          {trunc_path}   <- {len(truncated)} truncated captions")
    print(f"elapsed   {(time.time() - t0)/60:.1f} min")
    print()
    print("Pooled hidden states, NOT projected to 768: TextForensicProjection")
    print(f"belongs inside the fusion module so it trains. Join on {id_col}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
