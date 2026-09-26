"""Recompute fingerprints and quarantine development images near protected splits."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json

from .build_genimage import fingerprints
from .news import PACKAGE, OUT
from .preprocess import read_rows, sha256


class IdentityTree:
    def __init__(self):self.root=None
    def add(self,value,item):
        node=[value,[item],{}]
        if self.root is None:self.root=node;return
        cur=self.root
        while True:
            d=(value^cur[0]).bit_count()
            if not d:cur[1].append(item);return
            if d not in cur[2]:cur[2][d]=node;return
            cur=cur[2][d]
    def near(self,value,radius=4):
        pending=[self.root] if self.root else [];result=[]
        while pending:
            cur=pending.pop();d=(value^cur[0]).bit_count()
            if d<=radius:result.extend((d,item) for item in cur[1])
            pending.extend(n for k,n in cur[2].items() if d-radius<=k<=d+radius)
        return result


def fingerprint_row(row):
    raw=(PACKAGE/row['image_path']).read_bytes();info=fingerprints(raw)
    if info['sha256']!=row['sha256']:raise ValueError('Exported image bytes changed')
    return dict(**info,sample_id=row['image_asset_id'],split=row['split'],label=int(row['image_fake']))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read_rows(PACKAGE/'data/image_samples.csv');cache=OUT/'fresh_fingerprints.jsonl'
    if cache.exists():fingerprinted=[json.loads(line) for line in cache.read_text().splitlines()]
    else:
        with ProcessPoolExecutor(max_workers=6) as pool:fingerprinted=list(pool.map(fingerprint_row,rows,chunksize=32))
        cache.write_text(''.join(json.dumps(r)+'\n' for r in fingerprinted))
    if [r['sample_id'] for r in fingerprinted]!=[r['image_asset_id'] for r in rows]:raise ValueError('Fingerprint identities changed')
    tree=IdentityTree();quarantined=[];retained=[];matches=[]
    for split in ('test','val','train'):
        for row in (r for r in fingerprinted if r['split']==split):
            near=[(distance,prior) for distance,prior in tree.near(int(row['dhash'],16)) if prior['split']!=split]
            if near:
                quarantined.append(row['sample_id'])
                for distance,prior in near:
                    matches.append(dict(excluded_id=row['sample_id'],excluded_split=split,excluded_label=row['label'],
                                        protected_id=prior['sample_id'],protected_split=prior['split'],protected_label=prior['label'],
                                        dhash_distance=distance,byte_equal=row['sha256']==prior['sha256'],pixel_equal=row['rgb_sha256']==prior['rgb_sha256']))
            else:tree.add(int(row['dhash'],16),row);retained.append(row)
    result=dict(radius=4,fingerprint='RGB, L grayscale, Lanczos 9x8, right > left',
                source_sha256=sha256(PACKAGE/'data/image_samples.csv'),fingerprints_sha256=sha256(cache),
                quarantined_ids=quarantined,matches=matches,
                retained_counts=dict(Counter(f"{r['split']}/{r['label']}" for r in retained)),
                policy='Keep test fixed; exclude validation near test; exclude training near retained validation or test',
                limitation='A 64-bit perceptual hash is a conservative screen, not proof that all content overlap is absent')
    (OUT/'near_duplicate_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('matches','quarantined_ids')},indent=2))
    print('quarantined',len(quarantined),'cross_split_matches',len(matches),flush=True)


if __name__=='__main__':main()
