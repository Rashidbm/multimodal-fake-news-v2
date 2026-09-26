"""Align image-only encoder features by identity before training any fusion head."""
import argparse
import json
from pathlib import Path

import torch

from .preprocess import sha256

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/image_branch_2026_09_14'


def aligned_concatenation(paths,dest):
    sources=[torch.load(path,map_location='cpu',weights_only=True) for path in paths]
    reference=sources[0]
    if 'test' in reference['splits']:raise ValueError('Final tests are not development features')
    joined=[]
    for source in sources:
        lookup={key:i for i,key in enumerate(source['sample_ids'])}
        if len(lookup)!=len(source['sample_ids']):raise ValueError('Duplicate feature identities')
        idx=[lookup[key] for key in reference['sample_ids']]
        for i,j in enumerate(idx):
            if reference['labels'][i]!=source['labels'][j] or reference['splits'][i]!=source['splits'][j]:raise ValueError('Cross-encoder target or split mismatch')
        joined.append(source['features'][idx])
    payload={key:reference[key] for key in ('sample_ids','splits','labels','csv_sha256')}
    payload.update(features=torch.cat(joined,dim=1),source_sha256={str(p):sha256(p) for p in paths},
                   dimensions=[x.shape[1] for x in joined],input='Image features only',test_used=False)
    torch.save(payload,dest)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoders',nargs='+',required=True,choices=['rgb_guide','rgb_robust','dct_guide'])
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    inputs={}
    for key,clip,cnn in [('clean','news/clip_development.pt','news_development.pt'),
                         ('blur_train','news/clip_blur_training.pt','news_blur_training.pt'),
                         ('val_jpeg75','news/clip_val_jpeg75.pt','news_val_jpeg75.pt'),
                         ('val_blur1','news/clip_val_blur1.pt','news_val_blur1.pt'),
                         ('dgm4','dgm4_clip_clean_development.pt','dgm4_development.pt')]:
        path=out/f'{key}.pt'
        if not path.exists():aligned_concatenation([OUT/clip,*[OUT/name/cnn for name in args.encoders]],path)
        inputs[key]=str(path)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2))
    (out/'encoders.json').write_text(json.dumps(dict(encoders=['clip',*args.encoders],feature_order='CLIP first, then requested CNNs in argument order'),indent=2))


if __name__=='__main__':main()
