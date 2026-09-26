"""Train image-only CLIP heads with explicit image augmentation and domain weighting."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .metrics import binary_metrics
from .news import OUT
from .preprocess import sha256


def robust_threshold(conditions):
    thresholds=np.unique(np.r_[.5,*[np.quantile(p,np.linspace(0,1,1001)) for y,p in conditions.values()]])
    recalls=[]
    for y,p in conditions.values():
        order=np.argsort(p);ys=y[order];ps=p[order];positions=np.searchsorted(ps,thresholds,side='left')
        fn=np.r_[0,np.cumsum(ys)][positions];tn=positions-fn
        recalls.extend([(y.sum()-fn)/y.sum(),tn/(len(y)-y.sum())])
    matrix=np.stack(recalls);worst=matrix.min(axis=0);mean=matrix.mean(axis=0)
    index=max(range(len(thresholds)),key=lambda i:(worst[i],mean[i],-abs(thresholds[i]-.5)))
    return float(thresholds[index])


def load(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if 'test' in data['splits']:raise ValueError('Test features are forbidden during selection')
    return data,data['features'].numpy(),np.array(data['labels']),np.array(data['splits'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--include-dgm4',action='store_true')
    ap.add_argument('--inputs');ap.add_argument('--Cs',nargs='+',type=float,default=[.01,.1,1.,10.])
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    if (out/'selected.json').exists():raise FileExistsError(out/'selected.json')
    inputs=json.loads(Path(args.inputs).read_text()) if args.inputs else dict(clean=str(OUT/'clip_development.pt'),blur_train=str(OUT/'clip_blur_training.pt'),
             val_jpeg75=str(OUT/'clip_val_jpeg75.pt'),val_blur1=str(OUT/'clip_val_blur1.pt'),dgm4=str(OUT.parent/'dgm4_clip_development.pt'))
    source=Path(inputs['clean']);data,x,y,split=load(source)
    train=split=='train';val=split=='val';groups=[(x[train],y[train],'news_clean')]
    validation={'news_clean':(x[val],y[val])};source_files=[source]
    blur=Path(inputs['blur_train']);blur_data,blur_x,blur_y,blur_s=load(blur)
    if set(blur_s)!={'train'}:raise ValueError('Augmentation must be training-only')
    source_ids={i:label for i,label,s in zip(data['sample_ids'],y,split) if s=='train'}
    if any(source_ids.get(i)!=label for i,label in zip(blur_data['sample_ids'],blur_y)):raise ValueError('Blur image identity/label mismatch')
    groups.append((blur_x,blur_y,'news_blur'));source_files.append(blur)
    for corruption in ('jpeg75','blur1'):
        path=Path(inputs[f'val_{corruption}']);payload,cx,cy,cs=load(path)
        if payload['sample_ids']!=np.array(data['sample_ids'])[val].tolist():raise ValueError('Validation identity mismatch')
        validation['news_'+corruption]=(cx,cy);source_files.append(path)
    if args.include_dgm4:
        path=Path(inputs['dgm4']);_,dx,dy,ds=load(path)
        groups.append((dx[ds=='train'],dy[ds=='train'],'dgm4_clean'));validation['dgm4_clean']=(dx[ds=='val'],dy[ds=='val']);source_files.append(path)
    train_x=np.concatenate([a for a,b,c in groups]);train_y=np.concatenate([b for a,b,c in groups]);weights=[]
    for gx,gy,name in groups:
        counts=np.bincount(gy,minlength=2)
        weights.append(len(train_x)/(len(groups)*2*counts[gy]))
    weights=np.concatenate(weights);records=[];best_key=None;best=None
    for c in args.Cs:
        scaler=StandardScaler().fit(train_x,sample_weight=weights)
        classifier=LogisticRegression(C=c,max_iter=2000,random_state=2026091413)
        classifier.fit(scaler.transform(train_x),train_y,sample_weight=weights)
        model=Pipeline([('scale',scaler),('classifier',classifier)])
        probabilities={name:(vy,model.predict_proba(vx)[:,1]) for name,(vx,vy) in validation.items()}
        threshold=robust_threshold(probabilities)
        metrics={name:binary_metrics(vy,p,threshold) for name,(vy,p) in probabilities.items()}
        key=(min(min(m['real_recall'],m['fake_recall']) for m in metrics.values()),np.mean([m['balanced_accuracy'] for m in metrics.values()]))
        record=dict(C=c,threshold=threshold,validation=metrics,worst_class_recall=float(key[0]),mean_balanced_accuracy=float(key[1]))
        records.append(record);print(json.dumps(record),flush=True)
        if best_key is None or key>best_key:best_key=key;best=(model,record)
    model,record=best;joblib.dump(model,out/'classifier.joblib')
    result=dict(**record,candidates=records,training_groups={name:len(gx) for gx,gy,name in groups},
                sample_weighting='Each training domain/view group has equal total weight; each class has half its group weight',
                selection_rule='Maximize worst class recall across predefined validation conditions, then mean balanced accuracy',
                source_sha256={str(path):sha256(path) for path in source_files},classifier_sha256=sha256(out/'classifier.joblib'),
                input_dimension=x.shape[1],target='image authenticity; image embeddings only',test_used=False)
    (out/'selected.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
