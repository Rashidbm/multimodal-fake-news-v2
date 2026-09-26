"""Specified forensic preprocessing, including train-only scalar DCT normalization."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
import torch

from .remote_genimage import OUT


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def read_rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))


def dct_map(image):
    image=image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR).convert('YCbCr')
    y=np.asarray(image,dtype=np.float32)[:,:,0]
    outputs=[]
    for p in (8,16):
        n=224//p
        blocks=y.reshape(n,p,n,p).transpose(0,2,1,3)
        transformed=dct(dct(blocks,type=2,axis=2,norm='ortho'),type=2,axis=3,norm='ortho')
        logs=np.log(np.abs(transformed)+np.float32(1e-8))
        outputs.append(logs.transpose(0,2,1,3).reshape(224,224))
    return ((outputs[0]+outputs[1])*np.float32(.5))[None].astype(np.float32)


def raw_map_job(item):
    row,raw_dir=item
    path=Path(raw_dir)/(row['sample_id']+'.npy')
    if not path.exists():
        with Image.open(row['image_path']) as image:values=dct_map(image)
        np.save(path,values)
    values=np.load(path)
    if values.shape!=(1,224,224) or not np.isfinite(values).all():raise ValueError('Invalid DCT cache')
    return row['split'],float(values.sum(dtype=np.float64)),float(np.square(values,dtype=np.float64).sum()),values.size


def normalize_job(item):
    row,raw_dir,normalized_dir,mean,std=item
    path=Path(normalized_dir)/(row['sample_id']+'.pt')
    if path.exists():return
    values=np.load(Path(raw_dir)/(row['sample_id']+'.npy'))
    values=((values-mean)/(std+1e-8)).astype(np.float32)
    torch.save(torch.from_numpy(values),path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',default=str(OUT/'manifest.csv'))
    ap.add_argument('--out',default=str(OUT/'dct'));ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args();torch.set_num_threads(1)
    out=Path(args.out);raw=out/'raw';normalized=out/'normalized'
    raw.mkdir(parents=True,exist_ok=True);normalized.mkdir(exist_ok=True)
    rows=read_rows(args.csv);manifest_hash=sha256(args.csv)
    total=squares=count=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(split,s,ss,n) in enumerate(pool.map(raw_map_job,[(r,str(raw)) for r in rows],chunksize=32)):
            if split=='train':total+=s;squares+=ss;count+=n
            if (i+1)%1000==0:print('DCT raw',i+1,'/',len(rows),flush=True)
    if not count:raise ValueError('No training pixels for statistics')
    mean=total/count;std=max(0,squares/count-mean*mean)**.5
    stats=dict(mean=mean,std=std,fit_split='train',training_pixels=count,csv_sha256=manifest_hash,
               definition='RGB bilinear224 -> Y [0,255] -> orthonormal type-II block DCT 8/16 -> log(abs+1e-8) -> spatial reassembly -> mean -> global scalar zscore',
               source_sha256=sha256(__file__))
    dest=out/'dct_stats.json'
    if dest.exists() and json.loads(dest.read_text())!=stats:raise ValueError('DCT statistics changed; use a new cache directory')
    dest.write_text(json.dumps(stats,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(r,str(raw),str(normalized),mean,std) for r in rows]
        for i,_ in enumerate(pool.map(normalize_job,jobs,chunksize=32)):
            if (i+1)%1000==0:print('DCT normalized',i+1,'/',len(rows),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(rows=len(rows),stats_sha256=sha256(dest),csv_sha256=manifest_hash),indent=2))
    print('COMPLETE DCT',stats,flush=True)


if __name__=='__main__':main()
