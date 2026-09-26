"""Reserve a bounded external face-manipulation dataset with image-only targets."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from .build_genimage import fingerprints,HashTree
from .news import OUT as NEWS_OUT
from .preprocess import sha256

ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/Users/rashid/multimodaldetection/data/raw')
OUT=ROOT/'data/processed/image_branch_dgm4'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-root',type=Path,default=RAW,help='Directory containing DGM4/metadata and DGM4 image folders')
    args=parser.parse_args();raw_root=args.raw_root.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists():raise FileExistsError('DGM4 subset already reserved')
    if not (ROOT/'data/raw/GenImage_subset/verification.json').exists():
        raise ValueError('Complete GenImage reservation first so protection does not depend on download timing')
    protected=HashTree()
    for line in (NEWS_OUT/'fresh_fingerprints.jsonl').read_text().splitlines():
        row=json.loads(line);protected.add(int(row['dhash'],16),'news')
    # New reservations require the complete protected set. The original experiment
    # was built during download and received a final quarantine before final mixed-model fitting.
    genimage=ROOT/'data/raw/GenImage_subset/selected.jsonl'
    for line in genimage.read_text().splitlines():
        row=json.loads(line);protected.add(int(row['dhash'],16),'genimage')
    split_tree=HashTree();source_splits={};seen=set();selected=[];rejected=Counter();source_hashes={}
    for split,real_n,fake_n in [('test',500,250),('val',500,250),('train',3000,1500)]:
        file=raw_root/'DGM4/metadata'/f'{split}.json';source_hashes[split]=sha256(file)
        rows=json.loads(file.read_text());rng=random.Random(2026091411+['train','val','test'].index(split));rng.shuffle(rows)
        for category,target in [('orig',real_n),('face_swap',fake_n),('face_attribute',fake_n)]:
            n=0
            for row in (r for r in rows if r['fake_cls']==category):
                if n>=target:break
                path=raw_root/row['image'];identity=str(row['id'])
                if not path.is_file():rejected['missing_source_image']+=1;continue
                if identity in source_splits and source_splits[identity]!=split:rejected['cross_split_source_id']+=1;continue
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest() in seen:rejected['duplicate_bytes']+=1;continue
                try:info=fingerprints(raw)
                except (OSError,ValueError):rejected['unreadable']+=1;continue
                value=int(info['dhash'],16)
                if protected.cross_split_near(value,'dgm4'):rejected['near_other_dataset']+=1;continue
                if split_tree.cross_split_near(value,split):rejected['cross_split_near']+=1;continue
                label=int(category!='orig')
                if bool(row['fake_image_box'])!=bool(label):raise ValueError('Image manipulation annotation conflict')
                selected.append(dict(sample_id='dgm4_'+info['sha256'],split=split,label=label,image_path=str(path),
                                     generator=category,source_id=identity,source_metadata=str(file),**info))
                seen.add(info['sha256']);source_splits[identity]=split;split_tree.add(value,split);n+=1
            if n!=target:raise ValueError(f'Insufficient unique readable {split}/{category}: {n}/{target}')
            print(split,category,n,'rejected',dict(rejected),flush=True)
    csv_path=OUT/'manifest.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    audit=dict(rows=len(selected),counts=dict(Counter(f"{r['split']}/{r['generator']}" for r in selected)),rejected=dict(rejected),
               metadata_sha256=source_hashes,csv_sha256=sha256(csv_path),
               protected_genimage_sha256=sha256(genimage),protected_news_sha256=sha256(NEWS_OUT/'fresh_fingerprints.jsonl'),
               target='0 orig image, 1 face_swap or face_attribute; only pure categories sampled to avoid repeated image records',
               source='https://github.com/rshaojimmy/MultiModal-DeepFake',subset=True,test_predictions_used=False,
               warning='Locally available DGM4 is incomplete; missing files explicitly excluded, never replaced with zero tensors')
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2));print('COMPLETE',len(selected),flush=True)


if __name__=='__main__':main()
