"""Build image-only news targets and reuse verified frozen image embeddings."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import torch

from .preprocess import read_rows, sha256

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'exports/fnd_team_dataset_2026-09-14'
CSV=ROOT/'data/processed/image_branch_news.csv'
OUT=ROOT/'outputs/image_branch_2026_09_14/news'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    near_audit=json.loads((OUT/'near_duplicate_audit.json').read_text())
    if near_audit['source_sha256']!=sha256(PACKAGE/'data/image_samples.csv'):raise ValueError('Near-duplicate audit is stale')
    quarantined=set(near_audit['quarantined_ids'])
    images=read_rows(PACKAGE/'data/image_samples.csv')
    pairs=read_rows(PACKAGE/'data/master.csv')
    by_image=defaultdict(list)
    for row in pairs:
        expected=int(int(row['scenario']) in (3,5))
        if int(row['image_fake'])!=expected:raise ValueError('Scenario/image target mismatch')
        by_image[row['image_asset_id']].append(row)
    rows=[]
    for item in images:
        if item['image_asset_id'] in quarantined:continue
        associated=by_image[item['image_asset_id']]
        if {p['image_fake'] for p in associated}!={item['image_fake']}:raise ValueError('Conflicting image labels')
        if {p['split'] for p in associated}!={item['split']}:raise ValueError('Cross-split image identity')
        path=PACKAGE/item['image_path']
        if not path.is_file():raise FileNotFoundError(path)
        rows.append(dict(sample_id=item['image_asset_id'],split=item['split'],label=item['image_fake'],
                         image_path=str(path),generator='news',sha256=item['sha256'],dhash=item['image_dhash'],
                         sources='|'.join(sorted({p['source'] for p in associated})),
                         categories='|'.join(sorted({p['subcategory'] for p in associated})),
                         scenarios='|'.join(sorted({p['scenario'] for p in associated}))))
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Repeated image files')
    with CSV.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source=ROOT/'outputs/semantic_resolution/expanded_clip.pt'
    cached=torch.load(source,map_location='cpu',weights_only=True)
    original_csv=ROOT/'data/processed/semantic_resolution_train_dev.csv'
    if cached['csv_sha256']!=sha256(original_csv):raise ValueError('Embedding source CSV changed')
    original=read_rows(original_csv)
    if cached['sample_ids']!=[r['sample_id'] for r in original]:raise ValueError('Embedding row order mismatch')
    master={r['sample_id']:r for r in pairs}
    embeddings={};duplicate_differences=[]
    for i,pair_id in enumerate(cached['sample_ids']):
        pair=master[pair_id];asset=pair['image_asset_id']
        if pair['split']=='test':raise ValueError('Unexpected test features in development cache')
        if asset in embeddings:
            difference=(embeddings[asset]-cached['image'][i]).abs().max().item()
            duplicate_differences.append(difference)
            if difference>1e-4:raise ValueError('Same physical image has different embeddings')
        else:embeddings[asset]=cached['image'][i]
    development=[r for r in rows if r['split'] in ('train','val')]
    missing={r['sample_id'] for r in development}-set(embeddings)
    if missing:raise ValueError(f'Missing {len(missing)} image features')
    payload=dict(sample_ids=[r['sample_id'] for r in development],
                 features=torch.stack([embeddings[r['sample_id']] for r in development]),
                 splits=[r['split'] for r in development],labels=[int(r['label']) for r in development],
                 csv_sha256=sha256(CSV),source_cache_sha256=sha256(source),
                 model_name=cached['model_name'],model_revision=cached['model_revision'],
                 input='Image embedding only; cached text features excluded',test_used=False)
    torch.save(payload,OUT/'clip_development.pt')
    audit=dict(rows=len(rows),counts=dict(Counter(f"{r['split']}/{r['label']}" for r in rows)),
               target='0 authentic image; 1 edited/manipulated or AI-generated image',
               source_images_sha256=sha256(PACKAGE/'data/image_samples.csv'),
               source_master_sha256=sha256(PACKAGE/'data/master.csv'),csv_sha256=sha256(CSV),
               clip_features=len(development),max_duplicate_embedding_difference=max(duplicate_differences,default=0),
               near_duplicate_quarantine=len(quarantined),near_duplicate_audit_sha256=sha256(OUT/'near_duplicate_audit.json'),tests_evaluated=False)
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
