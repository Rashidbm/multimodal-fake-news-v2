"""Build a deterministic, globally deduplicated subset from original GenImage archives."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from .remote_genimage import OUT, GENERATORS, extract_member


class HashTree:
    def __init__(self): self.root=None
    def add(self, value, split):
        node=[value, {split}, {}]
        if self.root is None: self.root=node;return
        cur=self.root
        while True:
            d=(value^cur[0]).bit_count()
            if not d: cur[1].add(split);return
            if d not in cur[2]: cur[2][d]=node;return
            cur=cur[2][d]
    def cross_split_near(self, value, split, radius=4):
        pending=[self.root] if self.root else []
        while pending:
            cur=pending.pop();d=(value^cur[0]).bit_count()
            if d<=radius and any(s!=split for s in cur[1]): return True
            pending.extend(n for k,n in cur[2].items() if d-radius<=k<=d+radius)
        return False


def fingerprints(raw):
    with Image.open(BytesIO(raw)) as original:
        fmt=original.format;image=original.convert('RGB');image.load()
    pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    gray=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits=(gray[:,1:]>gray[:,:-1]).ravel()
    dhash=int.from_bytes(np.packbits(bits).tobytes(),'big')
    return dict(sha256=hashlib.sha256(raw).hexdigest(),rgb_sha256=pixels,dhash=f'{dhash:016x}',
                width=image.width,height=image.height,format=fmt)


def cached_fetch(index, member):
    key=hashlib.sha256((index['generator']+'/'+member['name']).encode()).hexdigest()
    path=OUT/'objects'/key[:2]/(key+Path(member['name']).suffix.lower())
    meta=path.with_suffix(path.suffix+'.json')
    if path.exists() and meta.exists(): return path,json.loads(meta.read_text())
    raw=extract_member(index,member)
    info=fingerprints(raw)
    info.update(source_member=member['name'],source_split=member['split'],generator=index['generator'],
                label=member['label'],archive_crc=member['crc'],source_revision=index['revision'])
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part');partial.write_bytes(raw);partial.replace(path)
    meta.write_text(json.dumps(info,sort_keys=True))
    return path,info


def round_robin_members(members, seed):
    """Shuffle across ImageNet synsets when filenames expose them; otherwise shuffle all."""
    import re
    groups={}
    for m in members:
        match=re.search(r'n\d{8}',m['name'])
        groups.setdefault(match.group() if match else '_unknown',[]).append(m)
    rng=random.Random(seed);keys=sorted(groups);rng.shuffle(keys)
    for key in keys: rng.shuffle(groups[key])
    while keys:
        next_keys=[]
        for key in keys:
            yield groups[key].pop()
            if groups[key]: next_keys.append(key)
        keys=next_keys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-per-class',type=int,default=2000)
    ap.add_argument('--val-per-class',type=int,default=250)
    ap.add_argument('--test-per-class',type=int,default=500)
    ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('--seed',type=int,default=2026091407)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'subset_plan.json'
    plan={**vars(args),'generators':list(GENERATORS),'test_source':'original val',
          'train_val_source':'original train','sampling':'shuffled synset round-robin when available',
          'deduplication':'global bytes, decoded pixels and real source basename; cross-split dHash radius 4'}
    if lock.exists() and json.loads(lock.read_text())!=plan: raise ValueError('Subset plan differs from reserved plan')
    lock.write_text(json.dumps(plan,indent=2))
    indexes={}
    for g in GENERATORS:
        path=OUT/'indexes'/f'{g}.json.gz'
        if not path.exists(): raise FileNotFoundError(path)
        with gzip.open(path,'rt') as f:indexes[g]=json.load(f)
    selected=OUT/'selected.jsonl'
    rows=[json.loads(s) for s in selected.read_text().splitlines()] if selected.exists() else []
    # Protect the independent news dataset before any model sees GenImage.
    # This is a conservative perceptual screen; some low-texture matches can be false positives.
    protected_file=OUT.parents[2]/'outputs/image_branch_2026_09_14/news/fresh_fingerprints.jsonl'
    protected_rows=[json.loads(s) for s in protected_file.read_text().splitlines()]
    protected=HashTree()
    for r in protected_rows:protected.add(int(r['dhash'],16),'protected_news')
    protected_sha256=hashlib.sha256(protected_file.read_bytes()).hexdigest()
    quarantine=[r for r in rows if protected.cross_split_near(int(r['dhash'],16),'genimage')]
    if quarantine:
        if (OUT/'verification.json').exists():raise ValueError('Cannot revise a completed dataset reservation')
        with (OUT/'pretraining_cross_dataset_quarantine.jsonl').open('a') as f:
            for r in quarantine:f.write(json.dumps(dict(**r,reason='Near protected news image before GenImage training'))+'\n')
        excluded={r['sample_id'] for r in quarantine};rows=[r for r in rows if r['sample_id'] not in excluded]
        for r in quarantine:Path(r['image_path']).unlink()
        partial=selected.with_suffix('.partial');partial.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));partial.replace(selected)
        print('Pretraining cross-dataset quarantine',len(quarantine),flush=True)
    seen_bytes={r['sha256']:r for r in rows};seen_pixels={r['rgb_sha256']:r for r in rows}
    seen_sources={r['source_identity']:r for r in rows};tree=HashTree()
    for r in rows:tree.add(int(r['dhash'],16),r['split'])
    counts=Counter((r['split'],r['generator'],r['label']) for r in rows); rejected=Counter()
    # Reserve all final test images first, then internal validation, then training.
    with selected.open('a',buffering=1) as record, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split,target in [('test',args.test_per_class),('val',args.val_per_class),('train',args.train_per_class)]:
            for gi,g in enumerate(GENERATORS):
                index=indexes[g]
                for label in [0,1]:
                    key=(split,g,label)
                    if counts[key]>=target:continue
                    original='val' if split=='test' else 'train'
                    available=[m for m in index['members'] if m['split']==original and m['label']==label]
                    candidates=iter(round_robin_members(available,args.seed+gi*100+label+(10000 if split=='test' else 0)))
                    while counts[key]<target:
                        todo=[]
                        for _ in range(min(128,max(24,target-counts[key]))):
                            try:m=next(candidates)
                            except StopIteration:break
                            identity=Path(m['name']).name.lower() if label==0 else g+'/'+m['name']
                            if identity in seen_sources:continue
                            todo.append((m,identity))
                        if not todo:
                            # An exhausted list is distinct from one batch of already selected identities.
                            try:m=next(candidates)
                            except StopIteration:raise RuntimeError(f'Insufficient unique {key}: {counts[key]}/{target}')
                            identity=Path(m['name']).name.lower() if label==0 else g+'/'+m['name']
                            if identity in seen_sources:continue
                            todo=[(m,identity)]
                        futures=[pool.submit(cached_fetch,index,m) for m,_ in todo]
                        for (member,identity),future in zip(todo,futures):
                            try:path,info=future.result()
                            except (OSError,ValueError,RuntimeError) as exc:
                                rejected[type(exc).__name__]+=1
                                with (OUT/'fetch_errors.jsonl').open('a') as err:
                                    err.write(json.dumps(dict(generator=g,member=member['name'],error=repr(exc)))+'\n')
                                continue
                            if counts[key]>=target:continue
                            if protected.cross_split_near(int(info['dhash'],16),'genimage'):
                                rejected['near_protected_news']+=1;continue
                            previous=seen_pixels.get(info['rgb_sha256']) or seen_bytes.get(info['sha256'])
                            if previous:
                                if previous['label']!=label:raise ValueError('Identical image has conflicting source labels')
                                rejected['duplicate']+=1;continue
                            if tree.cross_split_near(int(info['dhash'],16),split):
                                rejected['cross_split_near']+=1;continue
                            prefix=(Path('genimage_test')/g if split=='test' else Path('genimage_train')/split)
                            relative=prefix/('0_real' if label==0 else '1_fake')/path.name
                            link=OUT/relative;link.parent.mkdir(parents=True,exist_ok=True)
                            if not link.exists():
                                import os
                                os.link(path,link)
                            row={**info,'split':split,'source_identity':identity,'image_path':str(link.resolve()),
                                 'sample_id':path.stem}
                            record.write(json.dumps(row,sort_keys=True)+'\n');rows.append(row)
                            seen_sources[identity]=row;seen_bytes[row['sha256']]=row;seen_pixels[row['rgb_sha256']]=row
                            tree.add(int(row['dhash'],16),split);counts[key]+=1
                        print(split,g,label,counts[key], '/',target,'rejected',dict(rejected),flush=True)
    fields=list(rows[0])
    with (OUT/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report=dict(rows=len(rows),counts={'/'.join(map(str,k)):v for k,v in counts.items()},
                rejected=dict(rejected),manifest_sha256=hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest(),
                source=json.loads((OUT/'source.json').read_text()),subset=True,test_predictions_used=False,
                protected_news_fingerprints_sha256=protected_sha256,
                protected_news_policy='Exclude dHash distance <=4 to all protected news images before any GenImage training')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['rows'],flush=True)


if __name__=='__main__':main()
