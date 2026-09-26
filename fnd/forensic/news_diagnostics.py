"""Validation-only shortcut and source-generalization diagnostics, not release models."""
from collections import Counter
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .metrics import binary_metrics
from .news import CSV, OUT
from .preprocess import read_rows,sha256


def main():
    torch.set_num_threads(2)
    rows={r['sample_id']:r for r in read_rows(CSV)}
    data=torch.load(OUT/'clip_development.pt',map_location='cpu',weights_only=True)
    if data['csv_sha256']!=sha256(CSV):raise ValueError('Wrong data cache')
    x=data['features'].numpy();y=np.array(data['labels']);splits=np.array(data['splits']);ids=np.array(data['sample_ids'])
    train=splits=='train';val=splits=='val';categories=np.array([rows[i]['categories'] for i in ids])
    selection=json.loads((OUT/'clip_probe_clean/selected.json').read_text())
    selected=joblib.load(OUT/'clip_probe_clean/classifier.joblib');threshold=selection['threshold']
    p=selected.predict_proba(x[val])[:,1]
    group_metrics={}
    for group in sorted(set(categories[val])):
        mask=categories[val]==group;labels=y[val][mask];pred=p[mask]>=threshold
        group_metrics[group or 'supplemental_NewsCLIPpings']=dict(n=int(mask.sum()),label_counts=dict(Counter(map(str,labels))),accuracy=float((pred==labels).mean()),
                                                               mean_fake_probability=float(p[mask].mean()))
    source_holdouts={}
    for group in ('Fakeddit_photo_edit','coco_image_edit','antifact_image_generation'):
        allowed=train&(categories!=group)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=2000,random_state=2026091409))
        model.fit(x[allowed],y[allowed]);scores=model.predict_proba(x[val])[:,1]
        target=(categories[val]==group)|(y[val]==0)
        source_holdouts[group]=dict(training_excluded=int((train&~allowed).sum()),
                                   threshold_policy='Fixed .5; no threshold fitting for held-out source',
                                   unseen_source_and_validation_reals=binary_metrics(y[val][target],scores[target]),
                                   unseen_source_recall=float((scores[categories[val]==group]>=.5).mean()))
        print('SOURCE HOLDOUT',group,source_holdouts[group],flush=True)
    # This deliberately uses file properties to test whether the dataset can be solved by shortcuts.
    # The production classifier receives none of these metadata fields.
    fingerprints={r['sample_id']:r for r in map(json.loads,(OUT/'fresh_fingerprints.jsonl').read_text().splitlines())}
    meta=np.asarray([[np.log1p(fingerprints[i]['width']),np.log1p(fingerprints[i]['height']),
                      float(fingerprints[i]['format']=='JPEG'),np.log1p(__import__('os').path.getsize(rows[i]['image_path']))] for i in ids])
    control=RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=10,class_weight='balanced',n_jobs=2,random_state=2026091410)
    control.fit(meta[train],y[train]);scores=control.predict_proba(meta[val])[:,1]
    metadata_control=dict(inputs=['log width','log height','JPEG format indicator','log file bytes'],
                          purpose='Shortcut diagnostic only; forbidden as release model inputs',validation=binary_metrics(y[val],scores))
    corrupted={}
    for corruption in ('jpeg75','blur1'):
        cache=OUT/f'clip_val_{corruption}.pt'
        if not cache.exists():continue
        payload=torch.load(cache,map_location='cpu',weights_only=True)
        if payload['sample_ids']!=ids[val].tolist():raise ValueError('Corrupted validation image identities changed')
        corrupted[corruption]=binary_metrics(y[val],selected.predict_proba(payload['features'].numpy())[:,1],threshold)
    report=dict(per_category=group_metrics,held_out_source=source_holdouts,metadata_only_control=metadata_control,
                corrupted_validation=corrupted,test_used=False)
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2));print('METADATA CONTROL',metadata_control,flush=True)


if __name__=='__main__':main()
