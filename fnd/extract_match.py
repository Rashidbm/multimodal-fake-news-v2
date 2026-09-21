"""Cache v_match for every row of the built CSV, with either backbone.

    python -m fnd.extract_match --csv data/processed/balanced_5group.csv \
                                --backbone clip   --out features/v_match_clip.pt
    python -m fnd.extract_match --csv data/processed/balanced_5group.csv \
                                --backbone qwenvl --out features/v_match_qwenvl.pt

Both backbones are frozen, so a pair's vector is the same in every epoch and
extracting once is what keeps the comparison affordable: CLIP takes minutes,
the VLM takes hours, and neither is repeated per training run.

Writes three files beside --out, exactly as extract_textfor does:

    v_match_*.pt    {"ids", "features" (N, D) float32, "splits", "similarity", "meta"}
    v_match_*.npz   the same ids + `pooled_features`, no object arrays
    v_match_*.json  the provenance record

The ids ship with the tensor so compare_match can prove the two backbones were
scored on identical rows rather than assuming it.  ``similarity`` is the raw
CLIP cosine and is absent for the VLM, which never embeds the two sides apart.

A missing or unreadable image stops the run and is printed.  Skipping it would
put the two backbones on different row sets and silently invalidate every
number downstream.
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

from fnd.extract_textfor import evenly_spaced, find_id_column
from fnd.models.match_mismatch import DEFAULT_MODELS, MatchConfig, build_encoder


def read_pair_rows(csv_path: str | Path, split: str | None = None) -> tuple[list[dict], str]:
    """Rows carrying both halves of a pair, with a unique id each."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    id_col = find_id_column(rows[0])
    for needed in ("text", "image_path"):
        if needed not in rows[0]:
            raise KeyError(f"{csv_path} has no {needed!r} column; found {list(rows[0])}")
    if split:
        rows = [r for r in rows if r.get("split") == split]
        if not rows:
            raise ValueError(f"no rows with split={split!r} in {csv_path}")
    ids = [r[id_col] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {id_col} values: features could not be joined unambiguously")
    return rows, id_col


def resolve_image_path(row: dict, image_root: str | Path | None) -> Path:
    p = Path(row["image_path"])
    return Path(image_root) / p if image_root and not p.is_absolute() else p


def load_images(rows: list[dict], id_col: str, image_root: str | Path | None) -> list:
    """Open one batch of images. Convert to RGB: a greyscale or CMYK JPEG
    otherwise reaches the processor with the wrong channel count."""
    from PIL import Image

    images = []
    for r in rows:
        path = resolve_image_path(r, image_root)
        try:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        except Exception as exc:                     # noqa: BLE001 - re-raised with the id
            raise RuntimeError(f"cannot read the image for {r[id_col]}: {path} ({exc})") from exc
    return images


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract v_match with a frozen CLIP or Qwen-VL backbone.")
    ap.add_argument("--csv", required=True, help="built CSV, e.g. data/processed/balanced_5group.csv")
    ap.add_argument("--backbone", default="clip", choices=sorted(DEFAULT_MODELS))
    ap.add_argument("--out", default=None, help="default: features/v_match_<backbone>.pt")
    ap.add_argument("--model", default=None, help=f"default per backbone: {DEFAULT_MODELS}")
    ap.add_argument("--image-root", default=None,
                    help="prefix for relative image_path values in the CSV")
    ap.add_argument("--features", default="interaction",
                    choices=["interaction", "concat", "sim"],
                    help="clip only: what to hand the head (see models/match_mismatch.py)")
    ap.add_argument("--layer", type=int, default=-1,
                    help="qwenvl only: index into hidden_states; -1 = last")
    ap.add_argument("--pooling", default="last", choices=["last", "masked_mean"],
                    help="qwenvl only: how to read one vector out of the sequence")
    ap.add_argument("--max-len", type=int, default=77, help="clip only: its text encoder caps at 77")
    ap.add_argument("--max-pixels", type=int, default=401408,
                    help="qwenvl only: caps how many tokens one image becomes")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="the VLM needs far less than CLIP; try 4 to start")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default=None, help="only this split; default = all rows")
    ap.add_argument("--limit", type=int, default=None,
                    help="only N rows, spread evenly across the file so every "
                         "scenario and split stays represented")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.out or f"features/v_match_{args.backbone}.pt")
    rows, id_col = read_pair_rows(args.csv, args.split)
    if args.limit:
        rows = evenly_spaced(rows, args.limit)

    cfg = MatchConfig(backbone=args.backbone, model_name=args.model or "", layer=args.layer,
                      max_len=args.max_len, pooling=args.pooling, features=args.features,
                      max_pixels=args.max_pixels, dtype=args.dtype)

    print(f"===== match/mismatch: {args.backbone} =====")
    print(f"csv     {args.csv}  ({len(rows)} rows, id column {id_col!r}"
          + (f", split={args.split}" if args.split else "") + ")")
    if device.type == "cuda":
        print(f"gpu     {torch.cuda.get_device_name(0)}")

    encoder = build_encoder(cfg, device=device)
    print(f"model   {encoder.describe()}")
    print("-" * 60)

    chunks: list[torch.Tensor] = []
    sims: list[torch.Tensor] = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = load_images(batch, id_col, args.image_root)
        out = encoder.encode_pairs([r["text"] for r in batch], images)
        # float32 on disk: written once, read every epoch, and it keeps bf16
        # rounding out of the cached features.
        chunks.append(out["features"].float().cpu())
        if out.get("similarity") is not None:
            sims.append(out["similarity"].float().cpu())

        done = min(start + args.batch_size, len(rows))
        rate = done / max(time.time() - t0, 1e-9)
        eta = (len(rows) - done) / max(rate, 1e-9)
        print(f"  {done:>7}/{len(rows)}  {rate:5.1f} samples/s  eta {eta/60:5.1f} min", end="\r")
    print()

    features = torch.cat(chunks, dim=0)
    if features.shape[0] != len(rows):
        raise RuntimeError(f"got {features.shape[0]} vectors for {len(rows)} rows")
    if features.shape[1] != encoder.feature_dim:
        raise RuntimeError(f"got width {features.shape[1]}, encoder declared {encoder.feature_dim}")
    if not torch.isfinite(features).all():
        raise RuntimeError("non-finite values in features: check dtype and model load")
    similarity = torch.cat(sims, dim=0) if sims else None

    ids = [r[id_col] for r in rows]
    meta = {
        "stream": "v_match",
        "backbone": cfg.backbone,
        "model_name": cfg.model_name,
        "feature_dim": encoder.feature_dim,
        "clip_feature_mode": cfg.features if cfg.backbone == "clip" else None,
        "embed_dim": getattr(encoder, "embed_dim", None),
        "hidden_size": getattr(encoder, "hidden_size", None),
        "num_layers": getattr(encoder, "num_layers", None),
        "layer_requested": args.layer if cfg.backbone == "qwenvl" else None,
        "layer_resolved": getattr(encoder, "layer", None),
        "pooling": cfg.pooling if cfg.backbone == "qwenvl" else None,
        "prompt": cfg.prompt if cfg.backbone == "qwenvl" else None,
        "max_len": cfg.max_len,
        "max_pixels": cfg.max_pixels if cfg.backbone == "qwenvl" else None,
        "dtype": str(encoder.dtype),
        "projected": False,
        "proj_dim_expected": cfg.proj_dim,
        "has_similarity": similarity is not None,
        "id_column": id_col,
        "csv": str(args.csv),
        "image_root": args.image_root,
        "split": args.split,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ids": ids, "features": features,
               "splits": [r.get("split", "") for r in rows], "meta": meta}
    if similarity is not None:
        payload["similarity"] = similarity
    torch.save(payload, out_path)

    npz_path = out_path.with_suffix(".npz")
    arrays = {id_col: np.array(ids, dtype="U64"),
              "pooled_features": features.numpy().astype("float32")}
    if similarity is not None:
        arrays["similarity"] = similarity.numpy().astype("float32")
    np.savez(npz_path, **arrays)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("-" * 60)
    print(f"features  {tuple(features.shape)}")
    if similarity is not None:
        print(f"cosine    mean {similarity.mean():.4f}  min {similarity.min():.4f}  "
              f"max {similarity.max():.4f}")
    print(f"saved     {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"          {npz_path}   <- keyed by {id_col}, for the fusion stage")
    print(f"          {meta_path}")
    print(f"elapsed   {(time.time() - t0)/60:.1f} min")
    print()
    print("Raw vectors, NOT projected to 768: the projection belongs inside the")
    print(f"fusion module so it trains. Join on {id_col}.")
    print("Score this stream with:  python -m fnd.probe_match --features "
          f"{out_path} --csv {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
