"""Train finite frozen-feature alternatives; protect both genuine and OOC recall."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .fit_semantic_fusion import load_cache, feature_matrix, training_weights, metrics
from .train_clip_matcher import digest, read_rows


class MarginClassifier(ClassifierMixin, BaseEstimator):
    """Expose monotonic margin scores, not claimed calibrated probabilities."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y, sample_weight=None):
        self.estimator.fit(x, y, sample_weight=sample_weight)
        self.classes_ = self.estimator.classes_
        return self

    def decision_function(self, x):
        return self.estimator.decision_function(x)

    def predict_proba(self, x):
        margin = self.decision_function(x)
        if margin.ndim == 1:
            positive = expit(margin)
            return np.stack([1-positive, positive], axis=1)
        return softmax(margin, axis=1)


def fake_score(model, x, task):
    values = model.predict_proba(x)
    genuine_class = 0
    return 1-values[:, list(model[-1].classes_).index(genuine_class)]


def recall_operating_point(rows, values, utility_floor):
    """Exact finite thresholds. Select on validation only, never test labels."""
    values = np.asarray(values, dtype=float)
    if len(rows) != len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid scores or row alignment')
    thresholds = np.unique(np.r_[values, np.nextafter(values.max(), np.inf)])
    groups = [('v1_five_scenarios', s) for s in [1, 2, 3, 4, 5]] + [('paired_news', s) for s in [1, 4]]
    recalls = []
    for group, scenario in groups:
        selected = np.array([r['evaluation_group'] == group and int(r['scenario']) == scenario for r in rows])
        if not selected.any():
            raise ValueError(f'Missing validation group {group}/{scenario}')
        scores = np.sort(values[selected])
        below = np.searchsorted(scores, thresholds, side='left') / len(scores)
        recalls.append(below if scenario == 4 else 1-below)
    recalls = np.stack(recalls, axis=1)
    utility = .25*recalls[:, 3] + .0625*recalls[:, [0, 1, 2, 4]].sum(axis=1) + .25*recalls[:, [5, 6]].sum(axis=1)
    minimum = recalls.min(axis=1)
    eligible = np.flatnonzero(utility >= utility_floor-1e-12)
    passed = bool(len(eligible))
    if not passed:
        eligible = np.arange(len(thresholds))
    best = max(eligible, key=lambda i: ((minimum[i], utility[i]) if passed else (utility[i], minimum[i])) + (-abs(thresholds[i]-.5),))
    return dict(threshold=float(thresholds[best]), eligible=passed, minimum_recall=float(minimum[best]),
                utility=float(utility[best]), recalls={f'{g}/s{s}': float(v) for (g, s), v in zip(groups, recalls[best])})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--blip', required=True)
    parser.add_argument('--v1', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rows = read_rows(args.csv)
    if any(r['split'] not in ['train', 'val'] for r in rows):
        raise ValueError('Only train/development rows are allowed')
    blip, v1 = load_cache(args.blip, args.csv, rows), load_cache(args.v1, args.csv, rows)
    train = np.flatnonzero([r['split'] == 'train' for r in rows])
    val = np.flatnonzero([r['split'] == 'val' for r in rows])
    train_rows, val_rows = [rows[i] for i in train], [rows[i] for i in val]
    weights = training_weights(train_rows)
    x = feature_matrix(blip, v1, 'blip_v1_features')
    ref = joblib.load(args.reference)
    reference_values = fake_score(ref['model'], x[val], ref['task'])
    reference_metrics, _ = metrics(val_rows, reference_values, ref['threshold'])
    floor = reference_metrics['selection_utility']-.005
    out.mkdir(parents=True)
    manifest = dict(args=vars(args), hashes={p: digest(p) for p in [args.csv, args.blip, args.v1, args.reference]},
                    utility_floor=floor, test_used=False,
                    criterion='Maximize minimum recall across seven validation domain/scenario groups subject to the utility floor',
                    preprocessing='All learned scaling/PCA parameters use training rows only',
                    svm_scores='Softmax/expit of decision margins; not calibrated probabilities')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []

    def save(name, model, kind, task, values):
        point = recall_operating_point(val_rows, values, floor)
        report, predictions = metrics(val_rows, values, point['threshold'])
        result = dict(name=name, kind=kind, task=task, **point, metrics=report)
        results.append(result)
        joblib.dump(dict(model=model, kind=kind, task=task, threshold=point['threshold']), out/f'{name}.joblib')
        with (out/f'{name}_val.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        (out/'results.json').write_text(json.dumps(results, indent=2))
        print(f'{name}: eligible={point["eligible"]} min_recall={point["minimum_recall"]:.4f} utility={point["utility"]:.4f} '
              f'V1 real/OOC={point["recalls"]["v1_five_scenarios/s4"]:.4f}/{point["recalls"]["v1_five_scenarios/s1"]:.4f} '
              f'paired={point["recalls"]["paired_news/s4"]:.4f}/{point["recalls"]["paired_news/s1"]:.4f}', flush=True)

    save('previous_head_recall_threshold', ref['model'], ref['kind'], ref['task'], reference_values)
    with threadpool_limits(limits=2):
        for kind in ['v1_only', 'blip_v1_features']:
            features = np.c_[v1['hidden'].numpy(), v1['logit'].numpy()] if kind == 'v1_only' else x
            for c in [.01, .1, 1.]:
                # Keep class 0 genuine; other four labels are scenario-specific targets.
                target = np.array([{4:0, 1:1, 2:2, 3:3, 5:4}[int(r['scenario'])] for r in rows])
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=42))
                model.fit(features[train], target[train], logisticregression__sample_weight=weights)
                if model[-1].n_iter_.max() >= 1500:
                    raise RuntimeError('Linear head failed to converge')
                save(f'{kind}_linear5_c{c:g}', model, kind, 'five_class', fake_score(model, features[val], 'five_class'))
            scaler = StandardScaler().fit(features[train])
            projection = PCA(n_components=128, whiten=True, svd_solver='randomized', random_state=42)
            z_train = projection.fit_transform(scaler.transform(features[train]))
            z_val = projection.transform(scaler.transform(features[val]))
            target = np.array([0 if int(r['scenario']) == 4 else 1 if int(r['scenario']) == 1 else 2 for r in rows])
            for c in [.3, 3., 30.]:
                for gamma_scale in [.25, 1.]:
                    head = MarginClassifier(SVC(C=c, gamma=gamma_scale/128, cache_size=1024, decision_function_shape='ovr', random_state=42))
                    head.fit(z_train, target[train], sample_weight=weights)
                    model = make_pipeline(scaler, projection, head)
                    save(f'{kind}_rbf3_c{c:g}_g{gamma_scale:g}', model, kind, 'three_class',
                         1-head.predict_proba(z_val)[:, list(head.classes_).index(0)])
    eligible = [r for r in results if r['eligible']]
    best = max(eligible, key=lambda r: (r['minimum_recall'], r['utility']))
    (out/'best.json').write_text(json.dumps(best, indent=2))
    print('SELECTED '+best['name'], flush=True)


if __name__ == '__main__':
    from .recall_search import main as run
    run()
