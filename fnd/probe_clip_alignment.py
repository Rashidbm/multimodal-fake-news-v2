"""Train small binary/scenario models on frozen CLIP interaction features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alignment_selection import constrained_score
from .evaluate import score_rows
from .train import write_predictions


def features(payload):
    image,text=payload['image'].numpy(),payload['text'].numpy()
    product=image*text
    return np.concatenate([image,text,product,np.abs(image-text),product.sum(-1,keepdims=True)],axis=1)


def prediction_rows(model,x,rows,task):
    probability=model.predict_proba(x)
    fake=probability[:,1] if task=='binary' else 1-probability[:,3]
    return [{'sample_id':r['sample_id'],'scenario':int(r['scenario']),
             'label_binary':int(r['label_binary']),'label_index':int(r['label_index']),
             'prob':float(p),'pred':int(p>=0.5)} for r,p in zip(rows,fake)]


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',required=True)
    ap.add_argument('--cache',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--plan',default='outputs/v1_alignment_round/selection_plan.json')
    args=ap.parse_args(argv)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    with open(args.csv,newline='') as f:
        rows=list(csv.DictReader(f))
    payload=torch.load(args.cache,map_location='cpu',weights_only=True)
    if payload['csv_sha256']!=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest():
        raise ValueError('cache/dataset mismatch')
    if payload['sample_ids']!=[r['sample_id'] for r in rows]:
        raise ValueError('cache row order differs')
    x=features(payload)
    partitions={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','val']}
    minimum=json.loads(Path(args.plan).read_text())['min_recall']
    for task in ['binary','scenario']:
        target=np.array([int(r['label_binary'] if task=='binary' else r['label_index']) for r in rows])
        for c in [0.01,0.1,1.0]:
            directory=out/f'{task}_c{c:g}'
            if directory.exists():
                raise FileExistsError(directory)
            directory.mkdir()
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,
                                class_weight='balanced' if task=='binary' else None,random_state=42))
            with threadpool_limits(limits=2):
                model.fit(x[partitions['train']],target[partitions['train']])
            if model[1].n_iter_.max()>=model[1].max_iter:
                raise RuntimeError('linear model failed to converge')
            joblib.dump(model,directory/'model.joblib')
            for split in ['train','val']:
                indices=partitions[split]
                predictions=prediction_rows(model,x[indices],[rows[i] for i in indices],task)
                write_predictions(directory/f'{split}_predictions.csv',predictions)
                result={'fixed':score_rows(predictions),'constrained':constrained_score(predictions,minimum)}
                (directory/f'{split}_metrics.json').write_text(json.dumps(result,indent=2))
            (directory/'run_manifest.json').write_text(json.dumps({'args':vars(args),'task':task,'C':c,
                'csv_sha256':payload['csv_sha256'],'feature_definition':'image, text, elementwise product, absolute difference, cosine',
                'scaler_fitted_on':'train','test_evaluated':False},indent=2))
            m=result['constrained']
            print(directory,f"val_real={m['real_recall']:.3f} OOC={m['per_scenario'][1]['accuracy']:.3f} macro={m['f1_macro']:.3f}",flush=True)


if __name__=='__main__':
    main()
