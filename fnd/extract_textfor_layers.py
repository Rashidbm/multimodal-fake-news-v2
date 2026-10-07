"""Cache pooled hidden states for EVERY layer of a frozen LLM, one pass.

    python -m fnd.extract_textfor_layers \
        --csv dataset_2026-09-14/data/text_samples.csv \
        --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
        --out D:/fnd_features/ministral3_8b --max-gpu-memory 12GiB --skip-bos

Same pipeline as extract_textfor.py (frozen forward, masked mean, max_len 64,
bf16), but keeps hidden_states[0..L] instead of one index, so the layer can be
chosen on val afterwards (fnd/probe_textfor_layers.py) without re-running the
model. Used to compare other LLM families against Qwen3.5-9B layer 30.

Writes into --out:

    layer_00.npy .. layer_LL.npy   (N, H) float16, row i = ids[i]
    ids.npy                        full ids (no fixed-width truncation)
    splits.npy                     split per row
    meta.json                      provenance + token stats + per-layer max |x|

float16 is exact for bf16 values inside fp16's range; any overflow to inf is
caught and fails the run rather than writing a silently broken layer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from fnd.extract_textfor import evenly_spaced, read_rows
from fnd.models.text_fluoroscopy import TextFluoroscopy, TextFluoroscopyConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract masked-mean features for every layer.")
    ap.add_argument("--csv", required=True, help="e.g. dataset_2026-09-14/data/text_samples.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-len", type=int, default=64, help="64 = what the fusion features used")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-gpu-memory", default=None,
                    help='e.g. "12GiB": layers beyond this stay in CPU RAM')
    ap.add_argument("--skip-bos", action="store_true",
                    help="leave BOS out of the mean (Mistral/Llama; Qwen adds no BOS)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N evenly spaced rows")
    args = ap.parse_args(argv)

    rows, id_col = read_rows(args.csv)
    if args.limit:
        rows = evenly_spaced(rows, args.limit)
    n = len(rows)

    cfg = TextFluoroscopyConfig(model_name=args.model, layer=-1, max_len=args.max_len,
                                dtype=args.dtype, max_gpu_memory=args.max_gpu_memory,
                                pool_skip_bos=args.skip_bos)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== Text Fluoroscopy, all layers =====")
    print(f"csv     {args.csv}  ({n} rows, id column {id_col!r})")
    extractor = TextFluoroscopy(cfg, device=device)
    print(f"model   {extractor.describe()}")
    if getattr(extractor.model, "hf_device_map", None):
        placed = sorted(set(map(str, extractor.model.hf_device_map.values())))
        n_cpu = sum(1 for v in extractor.model.hf_device_map.values() if str(v) == "cpu")
        print(f"placed  {placed}  ({n_cpu} modules in CPU RAM)")
    tok = extractor.tokenizer
    print(f"tokens  bos={tok.bos_token!r} (skipped in pool: {extractor.bos_id is not None})  "
          f"pad={tok.pad_token!r}")
    if args.skip_bos and extractor.bos_id is None:
        raise RuntimeError("--skip-bos given but the tokenizer has no bos_token_id")
    print("-" * 60)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_states, h = extractor.num_layers + 1, extractor.hidden_size
    mmaps = [open_memmap(out / f"layer_{i:02d}.npy", mode="w+", dtype=np.float16, shape=(n, h))
             for i in range(n_states)]
    max_abs = [0.0] * n_states

    lengths: list[int] = []
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        batch = rows[start:start + args.batch_size]
        pooled, lens = extractor.encode_texts_all_layers([r["text"] for r in batch])
        pooled = pooled.float()
        if not torch.isfinite(pooled).all():
            raise RuntimeError(f"non-finite pooled values in batch starting at row {start}")
        for i in range(n_states):
            max_abs[i] = max(max_abs[i], pooled[i].abs().max().item())
        arr = pooled.to(torch.float16).cpu().numpy()
        if not np.isfinite(arr).all():
            raise RuntimeError(f"float16 overflow in batch starting at row {start} "
                               f"(max |x| {max(max_abs):.0f}); rerun storing float32")
        for i in range(n_states):
            mmaps[i][start:start + len(batch)] = arr[i]
        lengths.extend(lens)
        done = start + len(batch)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done:>7}/{n}  {rate:6.1f} samples/s  eta {(n - done) / max(rate, 1e-9) / 60:5.1f} min",
              end="\r")
    print()
    for m in mmaps:
        m.flush()
    del mmaps

    ids = [r[id_col] for r in rows]
    np.save(out / "ids.npy", np.array(ids))
    np.save(out / "splits.npy", np.array([r.get("split", "") for r in rows]))
    over = sum(1 for x in lengths if x > args.max_len)
    meta = {
        "model_name": args.model,
        "architecture": type(extractor.model).__name__,
        "hidden_size": h,
        "num_layers": extractor.num_layers,
        "layer_convention": "layer_i = hidden_states[i]; 0 = embedding output",
        "dtype": str(extractor.dtype),
        "stored_dtype": "float16",
        "pooling": "masked_mean" + (" (BOS excluded)" if extractor.bos_id is not None else ""),
        "max_length": args.max_len,
        "max_gpu_memory": args.max_gpu_memory,
        "chat_template": False,
        "id_column": id_col,
        "csv": str(args.csv),
        "n": n,
        "tokens": {"max": max(lengths), "mean": round(sum(lengths) / n, 2),
                   "over_max_length": over, "truncated_fraction": round(over / n, 4),
                   "note": "token counts include special tokens the tokenizer adds"},
        "max_abs_per_layer": [round(x, 2) for x in max_abs],
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("-" * 60)
    print(f"saved   {n_states} layers x ({n}, {h}) float16 -> {out}")
    print(f"tokens  max {meta['tokens']['max']}  mean {meta['tokens']['mean']}  truncated {over}")
    print(f"max|x|  {max(max_abs):.1f} (fp16 limit 65504)")
    print(f"elapsed {meta['minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
