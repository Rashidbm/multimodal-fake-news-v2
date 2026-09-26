"""Lock model choices and global thresholds using development data only."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

from .development import OUT,DATASETS,CASES,features,cnn_probabilities,validation_rows,operating_threshold
from .metrics import binary_metrics
from .preprocess import sha256


def candidates():
    values={name:dict(kind='cnn',encoders=[name],checkpoint=str(OUT/name/'best.pt')) for name in ('rgb_guide','dct_guide','rgb_robust','rgb_broad')}
    for name,directory,encoders in [
        ('clip_news_clean',OUT/'news/clip_probe_clean',['clip']),
        ('clip_news_robust',OUT/'news/clip_probe_robust',['clip']),
        ('clip_genimage',OUT/'clip_genimage/probe',['clip']),
        ('clip_rgb_news',OUT/'news_fusion/clip_rgb/news',['clip','rgb_robust']),
        ('clip_rgb_dgm4',OUT/'news_fusion/clip_rgb/with_dgm4',['clip','rgb_robust']),
        ('clip_rgb_dct_news',OUT/'news_fusion/clip_rgb_dct/news',['clip','rgb_robust','dct_guide']),
        ('clip_rgb_dct_dgm4',OUT/'news_fusion/clip_rgb_dct/with_dgm4',['clip','rgb_robust','dct_guide']),
        ('broad_features',OUT/'broad_feature_head',['clip','rgb_broad'])]:
        values[name]=dict(kind='linear',encoders=encoders,classifier=str(directory/'classifier.joblib'),selection=str(directory/'selected.json'))
    # The provisional CLIP+DGM4 diagnostic is excluded: it predates the final
    # cross-dataset quarantine. The clean feature-fusion candidates replace it.
    return values


def main():
    torch.set_num_threads(2)
    required=[OUT/'FINAL_DEVELOPMENT_CACHES_COMPLETE',OUT/'FEATURE_COMPARISONS_COMPLETE',OUT/'broad_feature_head/complete.json']
    if any(not p.exists() for p in required):raise ValueError('Development comparisons are not finished')
    lock_path=OUT/'selection_lock.json'
    if lock_path.exists():raise FileExistsError('Selection is already locked; do not overwrite after testing')
    rows=validation_rows();choices=candidates();predictions={};results={}
    purposes={'ai':[case for case in CASES if case[0]=='genimage'],
              'news':[case for case in CASES if case[0]=='news'],'broad':CASES}
    best={};best_keys={}
    for name,candidate in choices.items():
        p={};model=joblib.load(candidate['classifier']) if candidate['kind']=='linear' else None
        if model is not None:
            selected=json.loads(Path(candidate['selection']).read_text())
            if selected['classifier_sha256']!=sha256(candidate['classifier']):raise ValueError('Selected classifier changed')
        for dataset,condition in CASES:
            reference=rows[dataset]
            score=cnn_probabilities(name,dataset,condition,reference) if model is None else model.predict_proba(features(candidate['encoders'],dataset,condition,reference))[:,1]
            p[(dataset,condition)]=(np.array([int(r['label']) for r in reference]),score)
        predictions[name]=p;results[name]={}
        for purpose,cases in purposes.items():
            threshold=operating_threshold({case:p[case] for case in cases})
            metrics={'/'.join(case):binary_metrics(*p[case],threshold) for case in cases}
            key=(min(min(m['real_recall'],m['fake_recall']) for m in metrics.values()),np.mean([m['balanced_accuracy'] for m in metrics.values()]),-len(candidate['encoders']))
            results[name][purpose]=dict(threshold=threshold,worst_class_recall=float(key[0]),mean_balanced_accuracy=float(key[1]),
                                       validation=metrics,at_half={'/'.join(case):binary_metrics(*p[case],.5) for case in cases})
            if purpose not in best_keys or key>best_keys[purpose]:best_keys[purpose]=key;best[purpose]=name
        print(name,{key:{k:round(v[k],5) for k in ('worst_class_recall','mean_balanced_accuracy','threshold')} for key,v in results[name].items()},flush=True)
    checkpoint_hashes={name:sha256(OUT/name/'best.pt') for name in ('rgb_guide','dct_guide','rgb_robust','rgb_broad')}
    classifier_hashes={name:sha256(value['classifier']) for name,value in choices.items() if value['kind']=='linear'}
    lock=dict(selection_complete=True,locked_at_utc=datetime.now(timezone.utc).isoformat(),selection_rule='For each declared purpose: maximize worst class recall across predefined validation cases, then mean balanced accuracy, then fewer encoders',
              selected=best,candidates=choices,results=results,allowed_checkpoint_sha256=list(checkpoint_hashes.values()),
              checkpoint_sha256=checkpoint_hashes,classifier_sha256=classifier_hashes,
              allowed_csv_sha256=[sha256(path) for path in DATASETS.values()],dataset_csv_sha256={k:sha256(v) for k,v in DATASETS.items()},
              allowed_clip_revision='32bd64288804d66eefd0ccbe215aa642df71cc41',
              final_test_cases=['/'.join(case) for case in CASES],final_test_predictions_used=False,
              interpretation='Separate recommendations for AI-only GenImage, project news images, and the broader three-domain image task. No metadata/source identity is supplied at inference.')
    lock_path.write_text(json.dumps(lock,indent=2))
    (OUT/'validation_comparison.json').write_text(json.dumps(results,indent=2));print('LOCKED',best,flush=True)


if __name__=='__main__':main()
