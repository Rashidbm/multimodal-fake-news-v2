"""Bounded image-feature probe search; fit training only, select validation only."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .metrics import binary_metrics
from .preprocess import sha256


def choose_threshold(y,p):
    # Use only validation predictions; put both class recalls ahead of majority accuracy.
    values=np.unique(p)
    thresholds=np.unique(np.r_[.5,(values[:-1]+values[1:])/2])
    if len(thresholds)>2001:thresholds=np.unique(np.r_[.5,np.quantile(thresholds,np.linspace(0,1,2001))])
    ordered=np.argsort(p);sorted_p=p[ordered];sorted_y=y[ordered]
    positives=int(y.sum());negatives=len(y)-positives
    cumulative=np.r_[0,np.cumsum(sorted_y)]
    positions=np.searchsorted(sorted_p,thresholds,side='left')
    fn=cumulative[positions];tn=positions-fn
    fake=(positives-fn)/positives;real=tn/negatives
    scores=list(zip(np.minimum(fake,real),(fake+real)/2,-np.abs(thresholds-.5)))
    return float(thresholds[max(range(len(scores)),key=lambda i:scores[i])])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'selected.json').exists():raise FileExistsError(out/'selected.json')
    data=torch.load(args.cache,map_location='cpu',weights_only=True)
    if 'test' in data['splits']:raise ValueError('This tool accepts development data only')
    x=data['features'].numpy();y=np.asarray(data['labels']);splits=np.asarray(data['splits'])
    train=splits=='train';val=splits=='val'
    if not train.any() or not val.any():raise ValueError('Both development splits required')
    records=[];best_key=None;best=None
    for c in (.01,.1,1.,10.):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000,solver='lbfgs',random_state=2026091409))
        model.fit(x[train],y[train]);p=model.predict_proba(x[val])[:,1]
        threshold=choose_threshold(y[val],p);metric=binary_metrics(y[val],p,threshold)
        record=dict(C=c,threshold=threshold,validation=metric,validation_at_half=binary_metrics(y[val],p))
        records.append(record);key=(min(metric['real_recall'],metric['fake_recall']),metric['balanced_accuracy'],metric['AUC'])
        print(json.dumps(record),flush=True)
        if best_key is None or key>best_key:best_key=key;best=(model,record,p)
    model,record,p=best
    joblib.dump(model,out/'classifier.joblib')
    selection=dict(**record,candidates=records,selection_rule='Highest minimum class recall, then balanced accuracy, then AUC on validation',
                   cache_sha256=sha256(args.cache),classifier_sha256=sha256(out/'classifier.joblib'),
                   csv_sha256=data['csv_sha256'],input_dimension=x.shape[1],training_rows=int(train.sum()),validation_rows=int(val.sum()),
                   target='image label; no captions or pair labels enter classifier',test_used=False)
    (out/'selected.json').write_text(json.dumps(selection,indent=2))
    ids=np.asarray(data['sample_ids'])[val]
    with (out/'val_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','label','probability']);w.writeheader()
        w.writerows(dict(sample_id=i,label=int(label),probability=float(score)) for i,label,score in zip(ids,y[val],p))


if __name__=='__main__':main()
