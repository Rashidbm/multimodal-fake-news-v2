"""Fit regularized binary V1 alternatives; selection uses development data only."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from .evaluate import score_rows
from .evaluate_alignment import paired_summary
from .train_clip_matcher import digest, read_rows


def load_cache(path, csv_path, rows):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    if cache['csv_sha256'] != digest(csv_path) or cache['sample_ids'] != [r['sample_id'] for r in rows]:
        raise ValueError('Feature cache provenance/row order mismatch')
    return cache


def training_weights(rows):
    counts = Counter(int(r['scenario']) for r in rows)
    if set(counts) != {1, 2, 3, 4, 5}:
        raise ValueError('All five training scenarios are required')
    return np.array([len(rows)*(0.5 if int(r['scenario']) == 4 else 0.125)/counts[int(r['scenario'])] for r in rows])


def validation_weights(rows):
    counts = Counter((r['evaluation_group'], int(r['label_binary'])) for r in rows)
    if len(counts) != 4:
        raise ValueError('Need both binary classes in each validation domain')
    return np.array([0.25/counts[(r['evaluation_group'], int(r['label_binary']))] for r in rows])


def choose_operating_point(rows, probabilities):
    y = [int(r['label_binary']) for r in rows]
    fpr, tpr, thresholds = roc_curve(y, probabilities, sample_weight=validation_weights(rows), drop_intermediate=False)
    valid = np.flatnonzero(np.isfinite(thresholds))
    best = max(valid, key=lambda i: (round(float(tpr[i]-fpr[i]), 12),
                                    min(float(tpr[i]), 1-float(fpr[i])), -abs(float(thresholds[i])-0.5)))
    return float(thresholds[best])


def probability(model, x, task):
    probabilities = model.predict_proba(x)
    classes = model[-1].classes_.tolist()
    if task == 'binary':
        return probabilities[:, classes.index(1)]
    return 1-probabilities[:, classes.index(0)]  # 0 genuine, 1 OOC, 2 manipulated


def metrics(rows, probabilities, threshold):
    predictions = [{**{k: r[k] for k in ['sample_id', 'scenario', 'label_binary']}, 'prob': float(p)}
                   for r, p in zip(rows, probabilities)]
    domains = {}
    for group in ['v1_five_scenarios', 'paired_news']:
        subset = [p for r, p in zip(rows, predictions) if r['evaluation_group'] == group]
        domains[group] = score_rows(subset, threshold)
    metadata = [r for r in rows if r['evaluation_group'] == 'paired_news']
    pred_pair = [p for r, p in zip(rows, predictions) if r['evaluation_group'] == 'paired_news']
    domains['paired_caption_metrics'] = paired_summary(pred_pair, metadata, threshold)
    domains['selection_utility'] = sum(domains[k]['balanced_accuracy'] for k in ['v1_five_scenarios', 'paired_news'])/2
    domains['threshold'] = threshold
    return domains, predictions


def feature_matrix(blip, v1, kind, clip_large=None, qwen=None):
    if kind == 'blip_v1_qwen_embedding':
        # Frozen Qwen3-VL-Embedding joint image+caption vector replaces the CLIP-L block.
        if qwen is None:
            raise ValueError('This feature kind needs Qwen3-VL-Embedding features')
        base = feature_matrix(blip, v1, 'blip_v1_features')
        extra = np.asarray(qwen['features'], dtype=base.dtype)
        if extra.ndim != 2 or len(extra) != len(base):
            raise ValueError('Qwen features must be [N, D] and aligned with the BLIP/V1 rows')
        return np.concatenate([base, extra], axis=-1)
    if kind == 'v1_only':
        return np.concatenate([v1['hidden'].numpy(), v1['logit'].numpy()[:, None]], -1)
    if kind in ['clip_large_only', 'blip_v1_clip_large', 'blip_v1_clip_large_match', 'blip_v1_clip_large_score']:
        if clip_large is None:
            raise ValueError('This feature kind needs CLIP-L/14 embeddings')
        image, text = clip_large['image'].numpy(), clip_large['text'].numpy()
        if image.shape != text.shape or image.shape[1] != 768:
            raise ValueError('Expected aligned 768-dimensional CLIP-L/14 embeddings')
        product, difference = image*text, np.abs(image-text)
        cosine = product.sum(axis=-1, keepdims=True)
        pair = np.concatenate([image, text, product, difference, cosine], axis=-1)
        if kind == 'clip_large_only':
            return pair
        base = feature_matrix(blip, v1, 'blip_v1_features')
        extra = (cosine if kind.endswith('_score') else
                 np.concatenate([product, difference, cosine], axis=-1) if kind.endswith('_match') else pair)
        return np.concatenate([base, extra], axis=-1)
    if kind == 'blip':
        return blip['hidden'].numpy()
    if kind == 'blip_with_score':
        return np.concatenate([blip['hidden'].numpy(), blip['logit'].numpy()[:, None]], -1)
    if kind == 'scores':
        return np.stack([blip['logit'].numpy(), v1['logit'].numpy()], -1)
    if kind == 'blip_v1_score':
        return np.concatenate([blip['hidden'].numpy(), v1['logit'].numpy()[:, None]], -1)
    if kind == 'blip_v1_features':
        return np.concatenate([blip['hidden'].numpy(), v1['hidden'].numpy(),
                               blip['logit'].numpy()[:, None], v1['logit'].numpy()[:, None]], -1)
    raise ValueError(kind)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--blip', required=True)
    parser.add_argument('--v1')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rows = read_rows(args.csv)
    if any(r['split'] not in ['train', 'val'] for r in rows):
        raise ValueError('Training/selection CSV must not contain test examples')
    blip = load_cache(args.blip, args.csv, rows)
    v1 = load_cache(args.v1, args.csv, rows) if args.v1 else None
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    train_rows, val_rows = [rows[i] for i in train], [rows[i] for i in val]
    weights = training_weights(train_rows)
    out.mkdir(parents=True)
    manifest = dict(args=vars(args), inputs_sha256={p: digest(p) for p in [args.csv, args.blip, args.v1] if p},
                    selection='Mean of V1 binary balanced accuracy and caption-paired news balanced accuracy; validation-only threshold',
                    training='Unique original genuine expansion plus caption-paired training; sample weights give genuine 50% and each fake scenario 12.5%',
                    frozen_encoders=True, test_used=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    kinds = ['blip', 'scores', 'blip_v1_score', 'blip_v1_features'] if v1 else ['blip']
    for kind in kinds:
        x = feature_matrix(blip, v1, kind)
        for task in ['binary', 'three_class']:
            y = np.array([int(r['label_binary']) if task == 'binary' else
                          (0 if int(r['scenario']) == 4 else 1 if int(r['scenario']) == 1 else 2) for r in rows])
            for c in [0.001, 0.01, 0.1, 1.0]:
                name = f'{kind}_{task}_c{c:g}'
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=42))
                with threadpool_limits(limits=2):
                    model.fit(x[train], y[train], logisticregression__sample_weight=weights)
                    probabilities = probability(model, x[val], task)
                if model[-1].n_iter_.max() >= 1000:
                    raise RuntimeError(f'{name} failed to converge')
                threshold = choose_operating_point(val_rows, probabilities)
                selected, predictions = metrics(val_rows, probabilities, threshold)
                fixed, _ = metrics(val_rows, probabilities, 0.5)
                result = dict(name=name, kind=kind, task=task, C=c, threshold=threshold, selected=selected, fixed=fixed)
                results.append(result)
                joblib.dump(dict(model=model, task=task, kind=kind, threshold=threshold), out/f'{name}.joblib')
                with (out/f'{name}_val.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
                    writer.writeheader(); writer.writerows(predictions)
                (out/'results.json').write_text(json.dumps(results, indent=2))
                print(f'{name}: utility={selected["selection_utility"]:.4f} V1={selected["v1_five_scenarios"]["balanced_accuracy"]:.4f} '
                      f'paired={selected["paired_news"]["balanced_accuracy"]:.4f} '
                      f'real={selected["v1_five_scenarios"]["real_recall"]:.4f} '
                      f'OOC={selected["v1_five_scenarios"]["per_scenario"][1]["accuracy"]:.4f}', flush=True)
    best = max(results, key=lambda r: r['selected']['selection_utility'])
    (out/'best.json').write_text(json.dumps(best, indent=2))
    print('SELECTED '+best['name'], flush=True)


if __name__ == '__main__':
    main()
