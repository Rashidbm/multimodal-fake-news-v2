"""One final evaluation of predeclared models and operating points; no fitting."""
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
import torch

from .development import OUT, DATASETS, CASES
from .diagnostic_groups import dgm4_groups, news_groups, news_scenarios
from .metrics import binary_metrics, report
from .preprocess import read_rows, sha256


def test_cache_path(encoder, dataset, condition):
    return OUT/'final_test/features'/encoder/f'{dataset}_{condition}.pt'


@lru_cache(maxsize=10)
def checked_cache(encoder, dataset, condition, lock_path):
    lock=json.loads(Path(lock_path).read_text())
    value=torch.load(test_cache_path(encoder,dataset,condition),map_location='cpu',weights_only=True)
    if not lock['selection_complete'] or not value['test_used'] or set(value['splits'])!={'test'}:
        raise ValueError('Expected locked test-only features')
    if value['csv_sha256']!=lock['dataset_csv_sha256'][dataset]:raise ValueError('Test dataset changed')
    if encoder=='clip':
        if value['model_revision']!=lock['allowed_clip_revision']:raise ValueError('CLIP revision changed')
    elif value['checkpoint_sha256']!=lock['checkpoint_sha256'][encoder]:
        raise ValueError('Checkpoint changed')
    expected=None if condition=='clean' else condition
    if value['corruption']!=expected:raise ValueError('Wrong image condition')
    lookup={key:i for i,key in enumerate(value['sample_ids'])}
    if len(lookup)!=len(value['sample_ids']):raise ValueError('Duplicate test image identities')
    return value,lookup


def aligned_test(encoder,dataset,condition,rows,lock_path,field='features'):
    value,lookup=checked_cache(encoder,dataset,condition,str(lock_path))
    if set(lookup)!={r['sample_id'] for r in rows}:raise ValueError('Unexpected test membership')
    indices=[lookup[r['sample_id']] for r in rows]
    if any(int(r['label'])!=int(value['labels'][i]) for r,i in zip(rows,indices)):
        raise ValueError('Test label alignment failed')
    return value[field][indices].numpy()


def recall_interval(successes,n):
    # Wilson 95% interval, conditional on the fixed test reservation.
    z=1.959963984540054;p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(max(0,center-half)),float(min(1,center+half))]


def enriched_metrics(rows,p,threshold,dataset):
    y=np.array([int(r['label']) for r in rows]);metrics=binary_metrics(y,p,threshold)
    metrics['real_recall_95CI']=recall_interval(metrics['tn'],metrics['tn']+metrics['fp'])
    metrics['fake_recall_95CI']=recall_interval(metrics['tp'],metrics['tp']+metrics['fn'])
    result={'overall':metrics}
    if dataset=='genimage':result.update(report(rows,p,threshold));result['overall']=metrics
    elif dataset=='news':
        result['by_source']=news_groups(rows,p,threshold)
        result['by_pair_scenario']=news_scenarios(rows,p,threshold)
    else:result['by_manipulation']=dgm4_groups(rows,p,threshold)
    return result


def main():
    torch.set_num_threads(2)
    lock_path=OUT/'selection_lock.json';lock=json.loads(lock_path.read_text())
    if not lock['selection_complete']:raise ValueError('Models and thresholds must be locked before testing')
    if lock['final_test_cases']!=['/'.join(case) for case in CASES]:raise ValueError('Test case plan changed')
    out=OUT/'final_test';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Final evaluation already exists; do not overwrite')
    rows={name:[r for r in read_rows(path) if r['split']=='test'] for name,path in DATASETS.items()}
    for name,path in DATASETS.items():
        if sha256(path)!=lock['dataset_csv_sha256'][name]:raise ValueError('Locked data changed')
    results={};prediction_dir=out/'predictions';prediction_dir.mkdir(exist_ok=True)
    for name,candidate in lock['candidates'].items():
        classifier=None
        if candidate['kind']=='linear':
            if sha256(candidate['classifier'])!=lock['classifier_sha256'][name]:raise ValueError('Locked classifier changed')
            classifier=joblib.load(candidate['classifier'])
        results[name]={}
        for dataset,condition in CASES:
            reference=rows[dataset]
            if classifier is None:
                scores=aligned_test(candidate['encoders'][0],dataset,condition,reference,lock_path,'logits')
                p=expit(scores.astype(np.float64))
            else:
                x=np.concatenate([aligned_test(e,dataset,condition,reference,lock_path) for e in candidate['encoders']],axis=1)
                p=classifier.predict_proba(x)[:,1]
            if not np.isfinite(p).all():raise ValueError('Non-finite test predictions')
            key=dataset+'/'+condition
            # The same frozen probabilities are reported at three PRELOCKED purpose thresholds.
            results[name][key]={'at_half':enriched_metrics(reference,p,.5,dataset)}
            for purpose in ('ai','news','broad'):
                threshold=lock['results'][name][purpose]['threshold']
                results[name][key][purpose]=enriched_metrics(reference,p,threshold,dataset)
            with (prediction_dir/f'{name}_{dataset}_{condition}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['sample_id','label','probability_fake','prediction_at_half','prediction_ai','prediction_news','prediction_broad'])
                writer.writeheader()
                for row,prob in zip(reference,p):
                    value=dict(sample_id=row['sample_id'],label=row['label'],probability_fake=float(prob),prediction_at_half=int(prob>=.5))
                    value.update({f'prediction_{purpose}':int(prob>=lock['results'][name][purpose]['threshold']) for purpose in ('ai','news','broad')})
                    writer.writerow(value)
        print('EVALUATED',name,flush=True)
    output=dict(evaluated_at_utc=datetime.now(timezone.utc).isoformat(),selection_lock_sha256=sha256(lock_path),selected=lock['selected'],results=results,
                model_selection_used_test=False,threshold_selection_used_test=False,
                interpretation='AI, news and broad recommendations were fixed before test inference. Other model scores are comparisons, not a new test-based selection.',
                intervals='Wilson 95% recall intervals; do not measure training-seed uncertainty or dataset shift')
    (out/'results.json').write_text(json.dumps(output,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(results_sha256=sha256(out/'results.json'),selection_lock_sha256=sha256(lock_path)),indent=2))
    print('FINAL TEST COMPLETE',lock['selected'],flush=True)


if __name__=='__main__':main()
