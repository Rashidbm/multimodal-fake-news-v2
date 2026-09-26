"""Final bounded CNN+CLIP image-only head on all three training domains."""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .development import OUT,DATASETS,CASES,extract,features,validation_rows,operating_threshold
from .metrics import binary_metrics
from .preprocess import read_rows,sha256


def main():
    torch.set_num_threads(2);out=OUT/'broad_feature_head';out.mkdir(exist_ok=True)
    if (out/'selected.json').exists():raise FileExistsError(out/'selected.json')
    groups=[];ids=[];all_splits=[]
    for domain,path in DATASETS.items():
        rows=[r for r in read_rows(path) if r['split']=='train'];x=features(['clip','rgb_broad'],domain,'clean',rows)
        y=np.array([int(r['label']) for r in rows]);groups.append((domain,'clean',x,y));ids.extend(r['sample_id'] for r in rows);all_splits.extend(['train']*len(rows))
    from .news import CSV
    blur_rows=read_rows(CSV.with_name('image_branch_news_blur_train.csv'))
    blur_x=np.concatenate([extract(OUT/'news/clip_blur_training.pt',blur_rows),extract(OUT/'rgb_broad/news_blur_training.pt',blur_rows)],axis=1)
    groups.append(('news','blur',blur_x,np.array([int(r['label']) for r in blur_rows])))
    x=np.concatenate([g[2] for g in groups]);y=np.concatenate([g[3] for g in groups]);weights=[]
    for domain,view,gx,gy in groups:
        views=2 if domain=='news' else 1;counts=np.bincount(gy,minlength=2)
        weights.append(len(x)/(3*views*2*counts[gy]))
    weights=np.concatenate(weights)
    rows=validation_rows();val={f'{domain}/{condition}':(features(['clip','rgb_broad'],domain,condition,rows[domain]),np.array([int(r['label']) for r in rows[domain]])) for domain,condition in CASES}
    records=[];best=None;best_key=None
    for c in (.001,.01,.1):
        scaler=StandardScaler().fit(x,sample_weight=weights);classifier=LogisticRegression(C=c,max_iter=2000,random_state=2026091416)
        classifier.fit(scaler.transform(x),y,sample_weight=weights);model=Pipeline([('scale',scaler),('classifier',classifier)])
        probabilities={key:(vy,model.predict_proba(vx)[:,1]) for key,(vx,vy) in val.items()}
        threshold=operating_threshold(probabilities)
        metrics={key:binary_metrics(vy,p,threshold) for key,(vy,p) in probabilities.items()}
        key=(min(min(v['real_recall'],v['fake_recall']) for v in metrics.values()),np.mean([v['balanced_accuracy'] for v in metrics.values()]))
        record=dict(C=c,threshold=threshold,validation=metrics,worst_class_recall=float(key[0]),mean_balanced_accuracy=float(key[1]))
        records.append(record);print(json.dumps(record),flush=True)
        if best_key is None or key>best_key:best_key=key;best=(model,record)
    model,record=best;joblib.dump(model,out/'classifier.joblib')
    result=dict(**record,candidates=records,input_dimension=x.shape[1],encoders=['clip','rgb_broad'],
                training_groups={domain+'/'+view:len(gx) for domain,view,gx,gy in groups},
                weighting='Equal total domain weight, balanced labels inside each domain, news weight split equally between original and blurred views',
                selection_rule='Maximize worst class recall across predefined seven validation cases, then mean balanced accuracy',
                classifier_sha256=sha256(out/'classifier.joblib'),target='0 authentic, 1 local manipulation or AI generation',test_used=False)
    (out/'selected.json').write_text(json.dumps(result,indent=2))
    # Keep one clean representation per physical training image for a later train-only projection.
    clean=np.concatenate([g[2] for g in groups if g[1]=='clean'])
    clean_y=np.concatenate([g[3] for g in groups if g[1]=='clean'])
    torch.save(dict(features=torch.from_numpy(clean),sample_ids=ids,splits=all_splits,labels=clean_y.tolist(),
                    csv_sha256=sha256(DATASETS['news'].parent/'image_branch_mixed/manifest.csv'),dataset_sha256={k:sha256(v) for k,v in DATASETS.items()},test_used=False),out/'projection_training.pt')
    (out/'complete.json').write_text(json.dumps(dict(test_used=False,classifier_sha256=sha256(out/'classifier.joblib'),projection_cache_sha256=sha256(out/'projection_training.pt')),indent=2))


if __name__=='__main__':main()
