"""Reporting-only provenance groups; none of these annotations are model inputs."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .metrics import binary_metrics


@lru_cache(maxsize=3)
def annotations(path):
    records=json.loads(Path(path).read_text())
    return {r['image']:r for r in records if r['fake_cls'] in ('orig','face_swap','face_attribute')}


def dgm4_groups(rows,p,threshold):
    y=np.array([int(r['label']) for r in rows]);p=np.asarray(p);pred=p>=threshold
    methods=[];areas=[]
    for row in rows:
        path=Path(row['image_path']);methods.append(path.parent.name if int(row['label']) else 'authentic')
        key='DGM4/'+str(path).split('/DGM4/',1)[1]
        meta=annotations(row['source_metadata'])[key];box=meta['fake_image_box']
        areas.append((box[2]-box[0])*(box[3]-box[1])/(int(row['width'])*int(row['height'])) if box else 0.)
    methods=np.array(methods);areas=np.array(areas);report={}
    for method in sorted(set(methods)):
        mask=methods==method
        report[method]=dict(n=int(mask.sum()),accuracy=float((pred[mask]==y[mask]).mean()),mean_score=float(p[mask].mean()))
        if method!='authentic':report[method]['with_authentic_images']=binary_metrics(y[mask|(y==0)],p[mask|(y==0)],threshold)
    for lower,upper in [(0,.01),(.01,.05),(.05,.15),(.15,1.01)]:
        mask=(y==1)&(areas>=lower)&(areas<upper)
        if mask.any():report[f'manipulated_face_area_{lower:g}_{upper:g}']=dict(n=int(mask.sum()),fake_recall=float(pred[mask].mean()))
    return report


def news_groups(rows,p,threshold):
    p=np.asarray(p);groups={}
    for i,row in enumerate(rows):
        label=int(row['label']);correct=int((p[i]>=threshold)==label)
        for category in row.get('categories','').split('|'):
            name=category or 'supplemental_NewsCLIPpings';group=groups.setdefault(name,dict(n=0,correct=0,labels=set()))
            group['n']+=1;group['correct']+=correct;group['labels'].add(label)
    return {k:dict(n=v['n'],accuracy=v['correct']/v['n'],labels=sorted(v['labels'])) for k,v in groups.items()}


def news_scenarios(rows,p,threshold):
    p=np.asarray(p);groups={}
    for scenario in range(1,6):
        indices=[i for i,row in enumerate(rows) if str(scenario) in row.get('scenarios','').split('|')]
        if not indices:continue
        expected=int(scenario in (3,5))
        if any(int(rows[i]['label'])!=expected for i in indices):raise ValueError('Scenario/image-target inconsistency')
        groups[str(scenario)]=dict(n=len(indices),image_label=expected,
            image_accuracy=float(((p[indices]>=threshold)==expected).mean()))
    return dict(groups=groups,note='A physical image can belong to multiple pair scenarios; groups need not sum to the number of unique test images.')
