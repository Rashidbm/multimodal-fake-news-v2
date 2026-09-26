"""Shared, identity-checked development features and operating-point selection."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from .preprocess import read_rows,sha256

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/image_branch_2026_09_14'
DATASETS={'genimage':ROOT/'data/raw/GenImage_subset/manifest.csv',
          'news':ROOT/'data/processed/image_branch_news.csv',
          'dgm4':ROOT/'data/processed/image_branch_dgm4/clean_manifest.csv'}
CASES=[('genimage','clean'),('genimage','jpeg75'),('genimage','blur1'),
       ('news','clean'),('news','jpeg75'),('news','blur1'),('dgm4','clean')]


@lru_cache(maxsize=24)
def payload(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if 'test' in value['splits']:raise ValueError('Test features are forbidden during development')
    lookup={key:i for i,key in enumerate(value['sample_ids'])}
    if len(lookup)!=len(value['sample_ids']):raise ValueError('Duplicate feature identities')
    return value,lookup


def extract(path,rows,field='features'):
    value,lookup=payload(str(path));indices=[lookup[r['sample_id']] for r in rows]
    if any(r['split']=='test' for r in rows):raise ValueError('Final-test rows are forbidden during selection')
    for r,i in zip(rows,indices):
        if int(r['label'])!=int(value['labels'][i]) or r['split']!=value['splits'][i]:raise ValueError('Feature label/split mismatch')
    return value[field][indices].numpy()


def cache_path(encoder,dataset,condition):
    if encoder=='clip':
        if dataset=='genimage':return OUT/'clip_genimage'/('development.pt' if condition=='clean' else f'val_{condition}.pt')
        if dataset=='news':return OUT/'news'/('clip_development.pt' if condition=='clean' else f'clip_val_{condition}.pt')
        if dataset=='dgm4' and condition=='clean':return OUT/'dgm4_clip_clean_development.pt'
    if encoder=='rgb_broad':return OUT/encoder/('mixed_development.pt' if condition=='clean' else f'mixed_val_{condition}.pt')
    if dataset=='genimage':return OUT/encoder/f'genimage_val_{condition}.pt'
    if dataset=='news':return OUT/encoder/('news_development.pt' if condition=='clean' else f'news_val_{condition}.pt')
    if dataset=='dgm4' and condition=='clean':return OUT/encoder/'dgm4_development.pt'
    raise ValueError((encoder,dataset,condition))


def features(encoders,dataset,condition,rows):
    arrays=[]
    for name in encoders:
        path=cache_path(name,dataset,condition);value,_=payload(str(path))
        if name=='clip':
            if value.get('model_revision')!='32bd64288804d66eefd0ccbe215aa642df71cc41':raise ValueError('CLIP feature revision mismatch')
        elif value.get('checkpoint_sha256')!=sha256(OUT/name/'best.pt'):
            raise ValueError('CNN checkpoint changed after feature extraction')
        arrays.append(extract(path,rows))
    return np.concatenate(arrays,axis=1)


def cnn_probabilities(encoder,dataset,condition,rows):
    path=cache_path(encoder,dataset,condition);value,_=payload(str(path))
    if value['checkpoint_sha256']!=sha256(OUT/encoder/'best.pt'):raise ValueError('CNN checkpoint changed after feature extraction')
    return expit(extract(path,rows,'logits').astype(np.float64))


def validation_rows():return {name:[r for r in read_rows(path) if r['split']=='val'] for name,path in DATASETS.items()}


def operating_threshold(conditions):
    # Midpoints give the same empirical decisions with a margin against floating-point noise.
    values=np.unique(np.concatenate([p for y,p in conditions.values()]))
    thresholds=np.unique(np.r_[.5,(values[:-1]+values[1:])/2])
    recalls=[]
    for y,p in conditions.values():
        order=np.argsort(p);ys=y[order];positions=np.searchsorted(p[order],thresholds,side='left')
        fn=np.r_[0,np.cumsum(ys)][positions];tn=positions-fn
        recalls.extend([(y.sum()-fn)/y.sum(),tn/(len(y)-y.sum())])
    matrix=np.stack(recalls);minimum=matrix.min(axis=0);mean=matrix.mean(axis=0)
    i=max(range(len(thresholds)),key=lambda i:(minimum[i],mean[i],-abs(thresholds[i]-.5)))
    return float(thresholds[i])
