"""Test a stronger frozen CLIP encoder on fixed train/development splits."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .fit_semantic_fusion import feature_matrix, load_cache, training_weights, metrics
from .recall_search import fake_score, recall_operating_point
from .train_clip_matcher import digest, read_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['csv', 'blip', 'v1', 'clip-large', 'reference', 'out']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--ooc-mass', type=float, choices=[.125, .25], default=.125)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rows = read_rows(args.csv)
    if any(r['split'] not in ['train', 'val'] for r in rows):
        raise ValueError('Only train/development rows allowed')
    blip, v1, large = [load_cache(path, args.csv, rows) for path in [args.blip, args.v1, args.clip_large]]
    train = np.flatnonzero([r['split'] == 'train' for r in rows]); val = np.flatnonzero([r['split'] == 'val' for r in rows])
    val_rows = [rows[i] for i in val]
    weights = training_weights([rows[i] for i in train])
    if args.ooc_mass != .125:
        counts=Counter(int(rows[i]['scenario']) for i in train)
        mass={4:.5,1:args.ooc_mass,2:(.5-args.ooc_mass)/3,3:(.5-args.ooc_mass)/3,5:(.5-args.ooc_mass)/3}
        weights=np.array([len(train)*mass[int(rows[i]['scenario'])]/counts[int(rows[i]['scenario'])] for i in train])
    ref = joblib.load(args.reference)
    reference_score = fake_score(ref['model'], feature_matrix(blip, v1, ref['kind'])[val], ref['task'])
    floor = metrics(val_rows, reference_score, ref['threshold'])[0]['selection_utility']-.005
    out.mkdir(parents=True)
    manifest = dict(args=vars(args), hashes={path:digest(path) for path in [args.csv, args.blip, args.v1, args.clip_large, args.reference]},
                    utility_floor=floor, test_used=False, selection='same recall criterion as recall_search', frozen_encoders=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    with threadpool_limits(limits=2):
        for kind in ['clip_large_only', 'blip_v1_clip_large_score', 'blip_v1_clip_large_match', 'blip_v1_clip_large']:
            x = feature_matrix(blip, v1, kind, large)
            for task in ['three_class', 'five_class']:
                y = np.array([(0 if int(r['scenario'])==4 else 1 if int(r['scenario'])==1 else 2)
                              if task == 'three_class' else {4:0, 1:1, 2:2, 3:3, 5:4}[int(r['scenario'])] for r in rows])
                for c in [.001, .01, .1]:
                    name = f'{kind}_{task}_c{c:g}'
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=42))
                    model.fit(x[train], y[train], logisticregression__sample_weight=weights)
                    if model[-1].n_iter_.max() >= 1500:
                        raise RuntimeError('Non-converged classifier')
                    scores = fake_score(model, x[val], task)
                    point = recall_operating_point(val_rows, scores, floor)
                    selected, predictions = metrics(val_rows, scores, point['threshold'])
                    result = dict(name=name, kind=kind, task=task, **point, metrics=selected)
                    results.append(result)
                    joblib.dump(dict(model=model, kind=kind, task=task, threshold=point['threshold']), out/f'{name}.joblib')
                    with (out/f'{name}_val.csv').open('w', newline='') as stream:
                        w = csv.DictWriter(stream, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
                    (out/'results.json').write_text(json.dumps(results, indent=2))
                    print(f'{name}: eligible={point["eligible"]} min_recall={point["minimum_recall"]:.4f} utility={point["utility"]:.4f} '
                          f'V1 real/OOC={point["recalls"]["v1_five_scenarios/s4"]:.4f}/{point["recalls"]["v1_five_scenarios/s1"]:.4f} '
                          f'paired={point["recalls"]["paired_news/s4"]:.4f}/{point["recalls"]["paired_news/s1"]:.4f}', flush=True)
    eligible = [r for r in results if r['eligible']]
    best = max(eligible, key=lambda r:(r['minimum_recall'],r['utility'])) if eligible else None
    (out/'best.json').write_text(json.dumps(best, indent=2))
    print('COMPLETE '+str(out), flush=True)


if __name__ == '__main__':
    main()
