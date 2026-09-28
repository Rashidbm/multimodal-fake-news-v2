"""Cache frozen Qwen3-VL-Embedding joint image+caption embeddings.

The model is used through the official ``Qwen3VLEmbedder`` shipped with the
checkpoint (``<model>/scripts/qwen3_vl_embedding.py``): its default instruction,
chat template, dynamic-resolution image processing, last-token pooling and L2
normalization are applied unchanged. The local model directory comes from
``--model`` or ``QWEN_MODEL_PATH``; nothing is downloaded.

    python -m fnd.cache_qwen_embedding --csv train_dev.csv --out features/qwen_train_dev.pt
    python -m fnd.cache_qwen_embedding --csv train_dev.csv --out shard0.pt --shard 0/2
    python -m fnd.cache_qwen_embedding --csv train_dev.csv --out merged.pt --merge shard0.pt shard1.pt
    python -m fnd.cache_qwen_embedding --csv train_dev.csv --out smoke.pt --smoke 4

Pick the GPU with CUDA_VISIBLE_DEVICES; the official embedder uses the first visible CUDA device.
The cache matches the other semantic caches, so ``fit_semantic_fusion.load_cache`` reads it.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

from .train_clip_matcher import digest, read_rows

FEATURE_KIND = 'qwen3_vl_embedding'
# Settings that must agree between shards, the training cache and an inference bundle.
PROVENANCE_KEYS = ['model_config_sha256', 'embedder_script_sha256', 'instruction', 'min_pixels', 'max_pixels', 'dtype']


def model_directory(path=None):
    path = path or os.environ.get('QWEN_MODEL_PATH')
    if not path:
        raise ValueError('Set QWEN_MODEL_PATH (or pass --model) to the local Qwen3-VL-Embedding directory')
    path = Path(path)
    if not (path/'config.json').is_file():
        raise FileNotFoundError(f'No config.json in Qwen model directory {path}')
    return path


def load_embedder(path=None, min_pixels=None, max_pixels=None, device=None):
    """Official embedder from the local checkpoint, frozen, offline, BF16 on CUDA."""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    path = model_directory(path)
    script = path/'scripts'/'qwen3_vl_embedding.py'
    if not script.is_file():
        raise FileNotFoundError(f'Official embedder script missing: {script}')
    spec = importlib.util.spec_from_file_location('qwen3_vl_embedding', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if min_pixels:
        kwargs['min_pixels'] = min_pixels
    if max_pixels:
        kwargs['max_pixels'] = max_pixels
    embedder = module.Qwen3VLEmbedder(str(path), **kwargs)
    embedder.model.eval().requires_grad_(False)
    if device:
        embedder.model.to(device)
    embedder.provenance = dict(
        feature_kind=FEATURE_KIND, model_name=path.name, model_config_sha256=digest(path/'config.json'),
        embedder_script_sha256=digest(script), instruction=embedder.default_instruction,
        min_pixels=embedder.min_pixels, max_pixels=embedder.max_pixels,
        dtype=str(embedder.model.dtype).removeprefix('torch.'), normalized=True)
    return embedder


@torch.inference_mode()
def encode(embedder, images, texts):
    """One official joint embedding per (RGB PIL image, caption) -> CPU float32 [B, D]."""
    if len(images) != len(texts) or not images:
        raise ValueError('Provide one caption for each image')
    output = embedder.process([dict(image=image, text=text) for image, text in zip(images, texts)], normalize=True)
    output = output.float().cpu()
    if output.ndim != 2 or len(output) != len(texts) or not torch.isfinite(output).all():
        raise ValueError(f'Invalid Qwen embeddings: shape {tuple(output.shape)}')
    return output


def image_path(row, prefix_map=None):
    path = row['image_path']
    if prefix_map:
        old, new = prefix_map
        if path.startswith(old):
            path = new+path[len(old):]
    return Path(path)


def select_rows(csv_path, splits, shard=None):
    rows = [r for r in read_rows(csv_path) if r['split'] in splits]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique rows')
    if shard:
        index, count = shard
        bounds = [len(rows)*i//count for i in range(count+1)]
        rows = rows[bounds[index]:bounds[index+1]]
    return rows


def embed_rows(embedder, rows, batch_size, prefix_map=None, log=True):
    features, start_time = [], time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        images = []
        for row in batch:
            with Image.open(image_path(row, prefix_map)) as image:
                images.append(image.convert('RGB'))
        features.append(encode(embedder, images, [r['text'] for r in batch]))
        done = start+len(batch)
        if log and (done == len(rows) or (start//batch_size) % 25 == 0):
            rate = done/max(time.time()-start_time, 1e-9)
            print(f'Qwen embeddings {done}/{len(rows)}  {rate:.2f}/s  eta {(len(rows)-done)/rate/60:.1f} min', flush=True)
    matrix = torch.cat(features)
    if len({f.shape[1] for f in features}) != 1:
        raise ValueError('Embedding dimension changed between batches')
    return matrix


def payload_for(rows, csv_path, splits, embedder, features, batch_size, shard=None):
    return dict(sample_ids=[r['sample_id'] for r in rows], csv_sha256=digest(csv_path), splits=list(splits),
                features=features, embedding_dim=int(features.shape[1]), batch_size=batch_size,
                shard=f'{shard[0]}/{shard[1]}' if shard else None, **embedder.provenance)


def save(payload, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out.with_suffix('.json').write_text(json.dumps(
        {k: v for k, v in payload.items() if k not in ['features', 'sample_ids']}, indent=2))


def merge(parts, csv_path, splits, out):
    """Concatenate contiguous shards in index order; every row exactly once, same settings."""
    loaded = [torch.load(p, map_location='cpu', weights_only=True) for p in parts]
    if not all(part.get('shard') for part in loaded):
        raise ValueError('Every file to merge must be a shard cache')
    shards = [parse_shard(part['shard']) for part in loaded]
    if len({count for _, count in shards}) != 1 or sorted(i for i, _ in shards) != list(range(shards[0][1])):
        raise ValueError(f'Need exactly one file per shard 0..N-1, got {[p["shard"] for p in loaded]}')
    loaded = [part for _, part in sorted(zip(shards, loaded), key=lambda item: item[0][0])]
    first = loaded[0]
    for part in loaded:
        if part['csv_sha256'] != digest(csv_path) or part['splits'] != list(splits):
            raise ValueError('Shard was extracted from a different CSV or split selection')
        if any(part[k] != first[k] for k in PROVENANCE_KEYS+['embedding_dim']):
            raise ValueError('Shards disagree on model or preprocessing settings')
    ids = [i for part in loaded for i in part['sample_ids']]
    if ids != [r['sample_id'] for r in select_rows(csv_path, splits)]:
        raise ValueError('Merged sample_ids do not reproduce the CSV row order')
    payload = {**first, 'sample_ids': ids, 'features': torch.cat([p['features'] for p in loaded]),
               'shard': None, 'merged_from': [digest(p) for p in parts]}
    save(payload, out)
    return payload


def smoke(embedder, rows, count, batch_size, prefix_map, allow_cpu):
    """Encode a few pairs one at a time and batched; check dimension, norms, batch invariance and writing."""
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('CUDA is not available')
    if torch.cuda.is_available():
        print(f'GPU {torch.cuda.get_device_name(0)}  (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "all")})', flush=True)
        torch.cuda.reset_peak_memory_stats()
    sample = rows[:count]
    started = time.time()
    single = embed_rows(embedder, sample, 1, prefix_map, log=False)
    seconds = (time.time()-started)/len(sample)
    batched = embed_rows(embedder, sample, max(batch_size, len(sample)), prefix_map, log=False)
    cosine = torch.nn.functional.cosine_similarity(single, batched, dim=1).min().item()
    norms = single.norm(dim=1)
    report = dict(rows=len(sample), embedding_dim=int(single.shape[1]), min_batch_cosine=cosine,
                  norm_range=[norms.min().item(), norms.max().item()], seconds_per_pair=seconds,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
                  **embedder.provenance)
    if cosine < 0.999 or (norms-1).abs().max() > 1e-3:
        raise ValueError(f'Smoke check failed: {report}')
    return report, single


def parse_shard(value):
    index, count = (int(v) for v in value.split('/'))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError('shard must be K/N with 0 <= K < N')
    return index, count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--model', help='Local model directory; default QWEN_MODEL_PATH')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--min-pixels', type=int, help='Default: the official embedder value')
    parser.add_argument('--max-pixels', type=int, help='Default: the official embedder value')
    parser.add_argument('--shard', type=parse_shard, help='K/N: extract the K-th of N contiguous row blocks')
    parser.add_argument('--merge', nargs='+', metavar='SHARD', help='Merge shard caches into --out; no model is loaded')
    parser.add_argument('--path-prefix', help='OLD=NEW rewrite of image_path prefixes (CSV and hashes unchanged)')
    parser.add_argument('--smoke', type=int, metavar='N', help='Validate on N rows and write a small cache')
    parser.add_argument('--allow-cpu', action='store_true', help='Tests only: permit --smoke without CUDA')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if args.merge:
        payload = merge(args.merge, args.csv, args.splits, out)
        print(f'COMPLETE {out}  {tuple(payload["features"].shape)}', flush=True)
        return
    prefix_map = tuple(args.path_prefix.split('=', 1)) if args.path_prefix else None
    rows = select_rows(args.csv, args.splits, args.shard)
    missing = [str(image_path(r, prefix_map)) for r in rows if not image_path(r, prefix_map).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images not found, first: {missing[:3]}')
    print(f'{len(rows)} rows, all images found', flush=True)
    embedder = load_embedder(args.model, args.min_pixels, args.max_pixels)
    print(json.dumps(embedder.provenance, indent=2), flush=True)
    if args.smoke:
        report, features = smoke(embedder, rows, args.smoke, args.batch_size, prefix_map, args.allow_cpu)
        payload = payload_for(rows[:args.smoke], args.csv, args.splits, embedder, features, 1, args.shard)
        payload['smoke'] = True
        save(payload, out)
        reloaded = torch.load(out, map_location='cpu', weights_only=True)
        assert torch.equal(reloaded['features'], features)
        report['estimated_hours_for_selected_rows'] = report['seconds_per_pair']*len(rows)/3600
        print(json.dumps(report, indent=2), flush=True)
        print(f'SMOKE PASSED {out}', flush=True)
        return
    features = embed_rows(embedder, rows, args.batch_size, prefix_map)
    save(payload_for(rows, args.csv, args.splits, embedder, features, args.batch_size, args.shard), out)
    print(f'COMPLETE {out}  {tuple(features.shape)}', flush=True)


if __name__ == '__main__':
    main()
