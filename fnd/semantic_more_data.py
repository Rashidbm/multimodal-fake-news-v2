"""Increase unique paired training volume; keep development and encoders fixed."""
import csv
import json
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .fit_semantic_fusion import feature_matrix, load_cache
from .semantic_resolution import Experiment, ROOT, OUT, CSV
from .train_clip_matcher import digest, read_rows


def main():
    with threadpool_limits(limits=2):
        exp = Experiment('data_volume')
        extra_csv = ROOT/'data/processed/semantic_resolution_extra_train.csv'
        extra = read_rows(extra_csv)
        if any(r['split'] != 'train' for r in extra):
            raise ValueError('Extra data must be training-only')
        old_rows = exp.rows
        rows = old_rows+extra
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Overlapping sample IDs')
        combined_csv = ROOT/'data/processed/semantic_resolution_train_dev.csv'
        if combined_csv.exists():
            raise FileExistsError(combined_csv)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with combined_csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        caches = []
        old_paths = ['outputs/clip_adaptation/blip_v1_train_dev.pt',
                     'outputs/clip_adaptation/standardized_v1_train_dev.pt',
                     'outputs/recall_improvement/clip_large_train_dev.pt']
        for name, old_path in zip(['blip','v1','clip'], old_paths):
            old, new = load_cache(ROOT/old_path, CSV, old_rows), load_cache(OUT/f'extra_{name}.pt', extra_csv, extra)
            for key in ['model_name','model_revision','checkpoint_sha256','preprocessing']:
                if old.get(key) != new.get(key):
                    raise ValueError(f'Encoder changed: {name}/{key}')
            keys = ['image','text'] if name == 'clip' else ['hidden','logit']
            payload = {**old, **{key:torch.cat([old[key],new[key]]) for key in keys},
                'csv_sha256':digest(combined_csv), 'sample_ids':[r['sample_id'] for r in rows],
                'reuse':dict(old_sha256=digest(ROOT/old_path), new_sha256=digest(OUT/f'extra_{name}.pt'))}
            torch.save(payload, OUT/f'expanded_{name}.pt')
            caches.append(payload)
        exp.x = feature_matrix(caches[0], caches[1], 'blip_v1_clip_large', caches[2])
        exp.rows = rows
        exp.y = np.array([0 if int(r['scenario']) == 4 else 1 if int(r['scenario']) == 1 else 2 for r in rows])
        extra_captions = list(dict.fromkeys(r['caption_id'] for r in extra))
        if len(extra_captions)*2 != len(extra):
            raise ValueError('Extra training examples are not complete caption pairs')
        for fraction in [.5, 1.]:
            selected_captions = set(extra_captions[:int(len(extra_captions)*fraction)])
            train = np.array([i for i, row in enumerate(rows) if row['split'] == 'train'
                              and (i < len(old_rows) or row['caption_id'] in selected_captions)])
            counts = Counter(int(rows[i]['scenario']) for i in train)
            for mass in [.125, .25]:
                masses = {4:.5, 1:mass, 2:(.5-mass)/3, 3:(.5-mass)/3, 5:(.5-mass)/3}
                weights = np.array([len(train)*masses[int(rows[i]['scenario'])]/counts[int(rows[i]['scenario'])] for i in train])
                for c in [.001, .01]:
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=42))
                    model.fit(exp.x[train], exp.y[train], logisticregression__sample_weight=weights)
                    if model[-1].n_iter_.max() >= 1500:
                        raise RuntimeError('Data volume classifier did not converge')
                    exp.save(f'extra{fraction:g}_ooc{mass:g}_c{c:g}', model,
                        dict(training_rows=len(train), added_caption_pairs=len(selected_captions),
                             scenario_counts=dict(counts), csv_sha256=digest(combined_csv)))
        (exp.out/'data_manifest.json').write_text(json.dumps(dict(
            csv=str(combined_csv), csv_sha256=digest(combined_csv), extra_csv_sha256=digest(extra_csv),
            extra_caption_pairs=len(extra_captions), development_unchanged=True,
            test_used=False, cache_provenance_verified=True, source_sha256=digest(__file__)), indent=2))


if __name__ == '__main__':
    main()
