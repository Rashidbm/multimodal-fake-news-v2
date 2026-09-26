"""Evaluate a locked V1 candidate once using independently cached test features."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import torch
from scipy.special import logsumexp

from .evaluate import score_rows
from .evaluate_alignment import paired_summary, paired_difference
from .fit_semantic_fusion import feature_matrix, probability, load_cache
from .train_clip_matcher import digest, read_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--blip', required=True)
    parser.add_argument('--v1', required=True)
    parser.add_argument('--expanded-v1')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    lock_path, out = Path(args.selection), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    lock = json.loads(lock_path.read_text())
    if lock['test_used_for_selection'] is not False:
        raise ValueError('Need a validation-only selection lock')
    bundle_path = lock_path.parent/'bundle.json'
    if digest(bundle_path) != lock['candidate_bundle_sha256']:
        raise ValueError('Candidate bundle changed')
    bundle = json.loads(bundle_path.read_text())
    classifier_path = bundle['classifier']['path']
    if digest(classifier_path) != lock['candidate_classifier_sha256']:
        raise ValueError('Candidate classifier changed')
    csv_path = lock['evaluation_csv']
    if digest(csv_path) != lock['evaluation_csv_sha256']:
        raise ValueError('Evaluation data changed')
    rows = read_rows(csv_path)
    if any(r['split'] != 'test' for r in rows):
        raise ValueError('Expected test rows only')
    blip, v1 = load_cache(args.blip, csv_path, rows), load_cache(args.v1, csv_path, rows)
    if v1['checkpoint_sha256'] != lock['baseline_v1_sha256'] or blip['model_revision'] != bundle['blip_revision']:
        raise ValueError('Wrong feature encoder checkpoint')
    if blip.get('checkpoint_sha256') is not None:
        raise ValueError('This locked V1 candidate uses the official pretrained BLIP encoder')
    model = joblib.load(classifier_path)
    probabilities = probability(model['model'], feature_matrix(blip, v1, model['kind']), model['task'])
    decisions = model['model'].decision_function(feature_matrix(blip, v1, model['kind']))
    candidate_ranking = (decisions if model['task'] == 'binary' else logsumexp(decisions[:, 1:], axis=1)-decisions[:, 0])
    scores = dict(candidate=probabilities, standardized_v1=v1['logit'].sigmoid().numpy())
    ranking_scores = dict(candidate=candidate_ranking, standardized_v1=v1['logit'].numpy())
    thresholds = dict(candidate=lock['candidate_threshold'], standardized_v1=lock['baseline_threshold'])
    if args.expanded_v1:
        expanded = load_cache(args.expanded_v1, csv_path, rows)
        if expanded['checkpoint_sha256'] != lock['expanded_v1_sha256']:
            raise ValueError('Wrong expanded-data checkpoint')
        scores['expanded_v1'] = expanded['logit'].sigmoid().numpy()
        ranking_scores['expanded_v1'] = expanded['logit'].numpy()
        thresholds['expanded_v1'] = lock['expanded_v1_threshold']
    out.mkdir(parents=True)
    results, predictions_by_model = {}, {}
    for name, values in scores.items():
        predictions = [{**{k: r[k] for k in ['sample_id', 'scenario', 'label_binary']},
                        'prob': float(p), 'ranking_score': float(rank)}
                       for r, p, rank in zip(rows, values, ranking_scores[name])]
        predictions_by_model[name] = predictions
        with (out/f'{name}_predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        result = {}
        for group in ['v1_five_scenarios', 'paired_news']:
            subset = [p for r, p in zip(rows, predictions) if r['evaluation_group'] == group]
            result[group] = dict(fixed=score_rows(subset, 0.5), selected=score_rows(subset, thresholds[name]))
            if group == 'paired_news':
                metadata = [r for r in rows if r['evaluation_group'] == group]
                result[group]['paired_fixed'] = paired_summary(subset, metadata, 0.5)
                result[group]['paired_selected'] = paired_summary(subset, metadata, thresholds[name])
        results[name] = result
    paired_rows = [r for r in rows if r['evaluation_group'] == 'paired_news']
    paired_predictions = {name: [p for r, p in zip(rows, predictions) if r['evaluation_group'] == 'paired_news']
                          for name, predictions in predictions_by_model.items()}
    comparisons = {}
    for name in scores:
        if name == 'candidate':
            continue
        comparisons[name+'_selected'] = paired_difference(paired_predictions[name], paired_predictions['candidate'],
                                                        paired_rows, thresholds[name], thresholds['candidate'])
        comparisons[name+'_fixed'] = paired_difference(paired_predictions[name], paired_predictions['candidate'],
                                                     paired_rows, 0.5, thresholds['candidate'])
    raw_blip = [{**{k: r[k] for k in ['sample_id', 'scenario', 'label_binary']}, 'prob': float(p), 'ranking_score': float(rank)}
                for r, p, rank in zip(rows, blip['logit'].sigmoid().tolist(), blip['logit'].tolist()) if r['evaluation_group'] == 'paired_news']
    results['pretrained_blip_matching_only'] = dict(fixed=paired_summary(raw_blip, paired_rows, 0.5),
        selected=paired_summary(raw_blip, paired_rows, lock['blip_pretrained_matching_threshold']))
    report = dict(selection_sha256=digest(lock_path), results=results, paired_bootstrap_differences=comparisons,
                  ranking_definition='Raw binary log odds; avoids sigmoid saturation creating artificial ranking ties. Accuracy/recall use saved probabilities and locked thresholds.',
                  files_sha256={p: digest(p) for p in [args.blip, args.v1, args.expanded_v1] if p})
    (out/'evaluation.json').write_text(json.dumps(report, indent=2))
    for name in scores:
        for group in ['v1_five_scenarios', 'paired_news']:
            m = results[name][group]['selected']
            print(f'{name} {group}: accuracy={m["accuracy"]:.4f} balanced={m["balanced_accuracy"]:.4f} '
                  f'genuine={m["real_recall"]:.4f} OOC={m["per_scenario"][1]["accuracy"]:.4f}', flush=True)
    print('COMPLETE '+str(out/'evaluation.json'), flush=True)


if __name__ == '__main__':
    main()
