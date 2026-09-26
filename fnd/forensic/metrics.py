"""Generator-level evaluation with explicit positive=generated label semantics."""
import numpy as np
from sklearn.metrics import (accuracy_score,average_precision_score,confusion_matrix,
                             f1_score,precision_score,recall_score,roc_auc_score)


def binary_metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=(p>=threshold).astype(int)
    if set(y)!={0,1}:raise ValueError('Both binary labels are needed for evaluation')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),AP=float(average_precision_score(y,p)),AUC=float(roc_auc_score(y,p)),
                accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
                fake_precision=float(precision_score(y,pred,zero_division=0)),fake_recall=float(tp/(tp+fn)),
                real_recall=float(tn/(tn+fp)),balanced_accuracy=float(.5*(tp/(tp+fn)+tn/(tn+fp))),
                threshold=float(threshold),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def report(rows,p,threshold=.5):
    y=np.array([int(r['label']) for r in rows]);p=np.asarray(p)
    overall=binary_metrics(y,p,threshold)
    groups={}
    for g in sorted({r.get('generator','unknown') for r in rows}):
        mask=np.array([r.get('generator','unknown')==g for r in rows])
        if len(set(y[mask]))==2:groups[g]=binary_metrics(y[mask],p[mask],threshold)
    averages={}
    for name,names in {'all_generators':set(groups),'GAN':{'biggan'},'diffusion':{'adm','midjourney','vqdm','glide','sdv1_4','sdv1_5','wukong'}}.items():
        selected=[m for g,m in groups.items() if g in names]
        if selected:averages[name]={metric:float(np.mean([m[metric] for m in selected])) for metric in ('AP','accuracy','AUC','real_recall','fake_recall','balanced_accuracy')}
    return dict(overall=overall,per_generator=groups,generator_averages=averages,
                generator_mean_AP=float(np.mean([m['AP'] for m in groups.values()])) if groups else None,
                generator_std_AP=float(np.std([m['AP'] for m in groups.values()])) if groups else None)
