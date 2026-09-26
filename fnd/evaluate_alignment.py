"""Evaluate a locked alignment-round selection; never fit or select on test data."""
import argparse
import csv
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.torch_dataset import FakeNewsDataset, collate
from .evaluate import score_rows, validate_rows
from .models.fnd_clip import FNDCLIP, FNDCLIPConfig
from .probe_clip_alignment import features, prediction_rows
from .train import load_tokenizers, pick_device, run_epoch, write_predictions


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def pair_values(predictions, metadata, threshold):
    """One row per caption; retain the paired sampling unit for uncertainty."""
    by_id = {r['sample_id']: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['sample_id'] for r in metadata}:
        raise ValueError('paired predictions must match metadata exactly')
    groups = defaultdict(dict)
    for row in metadata:
        scenario = int(row['scenario'])
        if scenario in groups[row['caption_id']]:
            raise ValueError('duplicate scenario in a caption pair')
        prediction = by_id[row['sample_id']]
        groups[row['caption_id']][scenario] = (float(prediction['prob']),
                                              float(prediction.get('ranking_score', prediction['prob'])))
    if any(set(group) != {1, 4} for group in groups.values()):
        raise ValueError('each caption needs exactly one genuine and one OOC example')
    values = []
    for caption in sorted(groups):
        (genuine, genuine_rank), (ooc, ooc_rank) = groups[caption][4], groups[caption][1]
        real_correct, ooc_correct = float(genuine < threshold), float(ooc >= threshold)
        values.append([real_correct, ooc_correct, (real_correct+ooc_correct)/2,
                       real_correct*ooc_correct, float(ooc_rank > genuine_rank)+0.5*float(ooc_rank == genuine_rank),
                       ooc-genuine])
    return np.asarray(values, dtype=np.float64)


PAIR_METRICS = ['genuine_recall', 'ooc_recall', 'balanced_accuracy', 'both_correct',
                'within_caption_ranking', 'mean_probability_gap']


def paired_summary(predictions, metadata, threshold):
    values = pair_values(predictions, metadata, threshold)
    return {'caption_pairs': len(values), **dict(zip(PAIR_METRICS, values.mean(axis=0).tolist()))}


def paired_difference(baseline, candidate, metadata, baseline_threshold, candidate_threshold,
                      repetitions=5000, seed=20260908):
    """Percentile CI for candidate minus baseline, resampling caption pairs."""
    difference = (pair_values(candidate, metadata, candidate_threshold)
                  - pair_values(baseline, metadata, baseline_threshold))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((repetitions, len(PAIR_METRICS)))
    for start in range(0, repetitions, 100):
        indices = rng.integers(0, len(difference), size=(min(100, repetitions-start), len(difference)))
        bootstrap[start:start+len(indices)] = difference[indices].mean(axis=1)
    bounds = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {'sampling_unit': 'caption and its genuine/OOC images together', 'repetitions': repetitions,
            'seed': seed, 'metrics': {name: {'difference': float(difference[:, i].mean()),
                'ci95': bounds[:, i].tolist()} for i, name in enumerate(PAIR_METRICS)}}


def source_scores(predictions, metadata, threshold):
    original = {r['sample_id']: r for r in metadata}
    groups = defaultdict(list)
    for row in predictions:
        if int(row['scenario']) == 4:
            source = original[row['sample_id']].get('subcategory') or original[row['sample_id']]['source']
            groups[source].append(float(row['prob']) < threshold)
    return {source: {'n': len(correct), 'genuine_recall': sum(correct)/len(correct)}
            for source, correct in sorted(groups.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args(argv)
    selection = json.loads(Path(args.selection).read_text())
    if selection.get('test_used_for_selection') is not False:
        raise ValueError('a selection lock declaring validation-only selection is required')
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    selection_hash = digest(args.selection)
    # Hash all inputs before starting: do not accidentally evaluate a changed artifact.
    for spec in selection['datasets'].values():
        if digest(spec['csv']) != spec['sha256']:
            raise ValueError('locked dataset changed')
    for spec in selection['models']:
        if digest(spec['artifact']) != spec['sha256']:
            raise ValueError('locked model changed')
    device = pick_device(args.device)
    all_results, predictions_by_model = {}, {}
    for spec in selection['models']:
        directory = destination/spec['name']
        directory.mkdir(exist_ok=True)
        if (directory/'evaluation.json').exists():
            existing = json.loads((directory/'evaluation.json').read_text())
            if existing['selection_sha256'] != selection_hash:
                raise ValueError('refusing to reuse results from another selection')
            all_results[spec['name']] = existing
            predictions_by_model[spec['name']] = {name: read_rows(directory/f'{name}_predictions.csv')
                                                 for name in selection['datasets']}
            continue
        print('EVALUATE', spec['name'], flush=True)
        if spec['kind'] == 'neural':
            checkpoint = torch.load(spec['artifact'], map_location='cpu', weights_only=False)
            cfg = FNDCLIPConfig(**checkpoint['config'])
            model = FNDCLIP(cfg).to(device)
            model.load_state_dict(checkpoint['model'])
            del checkpoint
            bert_tok, clip_tok = load_tokenizers(cfg)
        elif spec['kind'] == 'probe':
            model = joblib.load(spec['artifact'])
        else:
            raise ValueError('unsupported model type')
        result = {'selection_sha256': selection_hash, 'model': spec, 'datasets': {}}
        predictions_by_model[spec['name']] = {}
        for name, data in selection['datasets'].items():
            metadata = [r for r in read_rows(data['csv']) if r['split'] == 'test']
            if spec['kind'] == 'neural':
                ds = FakeNewsDataset(data['csv'], 'test', bert_tok, clip_tok, train=False)
                loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
                rows = run_epoch(model, loader, device, 'binary')['_rows']
            else:
                payload = torch.load(data['clip_cache'], map_location='cpu', weights_only=True)
                original = read_rows(data['csv'])
                if payload['csv_sha256'] != data['sha256'] or payload['sample_ids'] != [r['sample_id'] for r in original]:
                    raise ValueError('probe cache does not match locked dataset')
                indices = [i for i, r in enumerate(original) if r['split'] == 'test']
                rows = prediction_rows(model, features(payload)[indices], metadata, spec['task'])
            validate_rows(rows, data['csv'], 'test')
            write_predictions(directory/f'{name}_predictions.csv', rows)
            predictions_by_model[spec['name']][name] = rows
            scores = {'fixed': score_rows(rows), 'selected': score_rows(rows, spec['threshold']),
                      'genuine_by_source': source_scores(rows, metadata, spec['threshold'])}
            if data.get('paired'):
                scores['paired_fixed'] = paired_summary(rows, metadata, 0.5)
                scores['paired_selected'] = paired_summary(rows, metadata, spec['threshold'])
            result['datasets'][name] = scores
            m = scores['selected']
            print(f"{name}: genuine={m['real_recall']:.4f} OOC={m['per_scenario'][1]['accuracy']:.4f} "
                  f"accuracy={m['accuracy']:.4f} pair_AUC={m['ooc_genuine_auc']:.4f}", flush=True)
        (directory/'evaluation.json').write_text(json.dumps(result, indent=2))
        all_results[spec['name']] = result
        del model
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
    baseline_name = selection['baseline']
    specs = {s['name']: s for s in selection['models']}
    comparisons = {}
    for name, data in selection['datasets'].items():
        if not data.get('paired'):
            continue
        metadata = [r for r in read_rows(data['csv']) if r['split'] == 'test']
        comparisons[name] = {candidate: paired_difference(predictions_by_model[baseline_name][name],
            predictions_by_model[candidate][name], metadata, specs[baseline_name]['threshold'], specs[candidate]['threshold'])
            for candidate in specs if candidate != baseline_name}
    (destination/'comparison.json').write_text(json.dumps({'selection_sha256': selection_hash,
        'results': all_results, 'paired_differences_from_baseline': comparisons}, indent=2))


if __name__ == '__main__':
    main()
