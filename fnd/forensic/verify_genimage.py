"""Independent manifest/split checks and sampled byte/pixel verification."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from .build_genimage import HashTree,fingerprints
from .preprocess import read_rows,sha256
from .remote_genimage import OUT,GENERATORS


def check_pixels(row):
    info=fingerprints(Path(row['image_path']).read_bytes())
    for key in ('sha256','rgb_sha256','dhash'):
        if info[key]!=row[key]:raise ValueError(f'{key} mismatch: {row["sample_id"]}')
    return row['sample_id']


def main():
    rows=read_rows(OUT/'manifest.csv');counts=Counter((r['split'],r['generator'],r['label']) for r in rows)
    for split,n in [('train',2000),('val',250),('test',500)]:
        for generator in GENERATORS:
            for label in ('0','1'):
                if counts[split,generator,label]!=n:raise ValueError('Scenario/source balance failed')
    for key in ('sample_id','sha256','rgb_sha256','source_identity'):
        if len({r[key] for r in rows})!=len(rows):raise ValueError(f'Duplicate {key}')
    for row in rows:
        expected='val' if row['split']=='test' else 'train'
        if row['source_split']!=expected or f'/{expected}/' not in row['source_member']:raise ValueError('Official split source mismatch')
        category='/nature/' if row['label']=='0' else '/ai/'
        if category not in row['source_member']:raise ValueError('Source image label mismatch')
        if not Path(row['image_path']).is_file():raise FileNotFoundError(row['image_path'])
    tree=HashTree()
    for split in ('test','val','train'):
        for row in (r for r in rows if r['split']==split):
            value=int(row['dhash'],16)
            if tree.cross_split_near(value,split):raise ValueError('Cross-split near duplicate')
            tree.add(value,split)
    rng=random.Random(2026091415);sample=rng.sample(rows,1024)
    with ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check_pixels,sample))
    report=dict(rows=len(rows),exact_duplicate_groups=0,cross_split_dhash_radius4_matches=0,
                sampled_byte_pixel_hashes_verified=len(checked),source_split_and_label_checks='all rows',
                counts={'/'.join(k):v for k,v in counts.items()},manifest_sha256=sha256(OUT/'manifest.csv'),
                generator_source_policy='Official training for train/internal validation; official validation for final test',
                final_test_predictions_used=False)
    (OUT/'independent_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
