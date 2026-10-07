"""Copy a fusion feature npz with its `text` column replaced by another LLM's features.

    python scripts/swap_text_features.py \
        --fusion D:/fnd_features/fusion/fusion-training-features.npz \
        --layers-dir D:/fnd_features/ministral3_8b_instruct --layer 24 \
        --out D:/fnd_features/fusion/fusion-training-features_ministral_L24.npz

Rows are pairs (sample_id); text features are per caption (text_asset_id). The two are joined
through dataset_2026-09-14/data/master.csv - by id, never by row order. Before writing, the
same join is used to rebuild the existing Qwen column from --check-features and compared to
what the npz holds, so a wrong join fails loudly instead of training on shuffled text.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def caption_ids(master_csv, sample_ids):
    with open(master_csv, newline='', encoding='utf-8') as f:
        to_text = {r['sample_id']: r['text_asset_id'] for r in csv.DictReader(f)}
    missing = [s for s in sample_ids if s not in to_text]
    if missing:
        raise KeyError(f'{len(missing)} sample_ids not in {master_csv} (first: {missing[:3]})')
    return [to_text[s] for s in sample_ids]


def gather(ids, feats, wanted):
    index = {i: n for n, i in enumerate(ids)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise KeyError(f'{len(missing)} captions have no features (first: {missing[:3]})')
    return feats[np.array([index[w] for w in wanted])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fusion', required=True)
    ap.add_argument('--layers-dir', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--master', default='dataset_2026-09-14/data/master.csv')
    ap.add_argument('--check-features', default='features/v_textfor.pt',
                    help='the features the npz text column was built from (Qwen L30)')
    args = ap.parse_args()

    src = dict(np.load(args.fusion, allow_pickle=False))
    wanted = caption_ids(args.master, src['sample_ids'].astype(str).tolist())

    old = torch.load(args.check_features, map_location='cpu', weights_only=False)
    rebuilt = gather(list(old['ids']), old['features'].numpy(), wanted)
    a, b = rebuilt.astype(np.float64), src['text'].astype(np.float64)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f'join check vs existing text column: cosine min {cos.min():.6f}  mean {cos.mean():.6f}')
    if cos.min() < 0.999:
        raise SystemExit('join check failed: the existing text column is not reproduced')

    d = Path(args.layers_dir)
    ids = [str(i) for i in np.load(d / 'ids.npy')]
    new = gather(ids, np.load(d / f'layer_{args.layer:02d}.npy'), wanted).astype(np.float32)
    if new.shape != src['text'].shape or not np.isfinite(new).all():
        raise SystemExit(f'bad replacement: shape {new.shape} vs {src["text"].shape}')

    src['text'] = new
    np.savez(args.out, **src)
    print(f'wrote {args.out}  text {new.shape} from {d.name} layer {args.layer}')


if __name__ == '__main__':
    main()
