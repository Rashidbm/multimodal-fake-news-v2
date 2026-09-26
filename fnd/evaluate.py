"""Evaluate one binary V1 checkpoint; select thresholds exclusively on validation.

Both fixed-0.5 and validation-selected operating points are saved. Test examples
never participate in checkpoint or threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.torch_dataset import FakeNewsDataset, collate
from .metrics import binary_metrics, per_scenario_accuracy
from .models.fnd_clip import FNDCLIP, FNDCLIPConfig
from .train import load_tokenizers, pick_device, run_epoch, write_predictions


def score_rows(rows, threshold=0.5):
    labels = [int(r['label_binary']) for r in rows]
    probabilities = [float(r['prob']) for r in rows]
    result = binary_metrics(labels, probabilities, threshold)
    result['threshold'] = threshold
    result['per_scenario'] = per_scenario_accuracy(
        [int(r['scenario']) for r in rows], labels, [int(p >= threshold) for p in probabilities])
    genuine_ooc = [r for r in rows if int(r['scenario']) in (1, 4)]
    result['ooc_genuine_auc'] = binary_metrics([int(r['label_binary']) for r in genuine_ooc],
                                             [float(r['prob']) for r in genuine_ooc])['auc']
    return result


def choose_threshold(validation_rows, metric='balanced_accuracy'):
    """Sweep all distinct validation decisions in O(n log n); prefer 0.5 on ties."""
    if metric not in ('balanced_accuracy', 'f1_macro'):
        raise ValueError('threshold metric must account for both binary classes')
    pairs = sorted((float(r['prob']), int(r['label_binary'])) for r in validation_rows)
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError('threshold selection requires both real and fake validation examples')
    tp, fp, tn, fn = positives, negatives, 0, 0

    def value():
        if metric == 'balanced_accuracy':
            return (tp / positives + tn / negatives) / 2
        return (2 * tp / max(2 * tp + fp + fn, 1) + 2 * tn / max(2 * tn + fp + fn, 1)) / 2

    fixed = score_rows(validation_rows, 0.5)
    best = (fixed[metric], 0.0, 0.5)
    candidate = (value(), -0.5, 0.0)
    best = max(best, candidate)
    i = 0
    while i < len(pairs):
        p = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == p:
            if pairs[i][1]:
                tp -= 1
                fn += 1
            else:
                fp -= 1
                tn += 1
            i += 1
        threshold = (p + pairs[i][0]) / 2 if i < len(pairs) else math.nextafter(p, math.inf)
        best = max(best, (value(), -abs(threshold - 0.5), threshold))
    return best[2]


def validate_rows(rows, csv_path, split):
    with open(csv_path, newline='') as f:
        expected = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == split}
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f'{split} predictions do not exactly match split membership')
    for r in rows:
        original = expected[r['sample_id']]
        if int(r['label_binary']) != int(original['label_binary']) or int(r['scenario']) != int(original['scenario']):
            raise ValueError(f'prediction label mismatch for {r["sample_id"]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--threshold-metric', choices=['balanced_accuracy', 'f1_macro'], default='balanced_accuracy')
    ap.add_argument('--reuse-test', help='existing predictions from this exact checkpoint (baseline only)')
    args = ap.parse_args(argv)
    out = Path(args.out)
    if (out / 'evaluation.json').exists():
        raise FileExistsError(f'refusing to overwrite evaluation: {out}')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = FNDCLIPConfig(**checkpoint['config'])
    if cfg.num_outputs != 1:
        raise ValueError('this evaluator is for the binary V1 specification')
    device = pick_device(args.device)
    model = FNDCLIP(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    del checkpoint['model']
    bert_tok, clip_tok = load_tokenizers(cfg)

    def infer(split):
        ds = FakeNewsDataset(args.csv, split, bert_tok, clip_tok, train=False, clip_preprocess=cfg.clip_preprocess)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
        result = run_epoch(model, loader, device, 'binary')
        print(f'{split}: n={len(ds)} loss={result["loss"]:.4f}', flush=True)
        return result['_rows']

    val = infer('val')
    validate_rows(val, args.csv, 'val')
    threshold = choose_threshold(val, args.threshold_metric)
    write_predictions(out / 'val_predictions.csv', val)
    if args.reuse_test:
        with open(args.reuse_test, newline='') as f:
            test = list(csv.DictReader(f))
    else:
        test = infer('test')
    validate_rows(test, args.csv, 'test')
    write_predictions(out / 'test_predictions.csv', test)
    result = {'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch': checkpoint['epoch'],
              'csv_sha256': hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              'threshold_selected_on': 'val', 'threshold_metric': args.threshold_metric,
              'threshold': threshold, 'val_fixed': score_rows(val), 'val_selected': score_rows(val, threshold),
              'test_fixed': score_rows(test), 'test_selected': score_rows(test, threshold)}
    with open(out / 'evaluation.json', 'w') as f:
        json.dump(result, f, indent=2)
    for name in ('val_fixed', 'val_selected', 'test_fixed', 'test_selected'):
        m = result[name]
        print(f'{name}: accuracy={m["accuracy"]:.4f} balanced={m["balanced_accuracy"]:.4f} '
              f'macro_f1={m["f1_macro"]:.4f} genuine_recall={m["real_recall"]:.4f} '
              f'ooc_recall={m["per_scenario"][1]["accuracy"]:.4f} threshold={m["threshold"]:.6f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
