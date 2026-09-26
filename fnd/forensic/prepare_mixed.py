"""Lock a mixed broad-manipulation extension after cross-dataset screening."""
from collections import Counter
import csv
import json
from pathlib import Path

import torch

from .build_genimage import HashTree
from .news import CSV as NEWS_CSV
from .preprocess import read_rows,sha256
from .remote_genimage import OUT as GENIMAGE

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/image_branch_mixed'


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    if not (GENIMAGE/'verification.json').exists():raise ValueError('GenImage reservation must finish first')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists():raise FileExistsError('Mixed extension is already locked')
    genimage=read_rows(GENIMAGE/'manifest.csv');news=read_rows(NEWS_CSV)
    news_fingerprints={r['sample_id']:r for r in map(json.loads,(ROOT/'outputs/image_branch_2026_09_14/news/fresh_fingerprints.jsonl').read_text().splitlines())}
    for row in news:row['dhash']=news_fingerprints[row['sample_id']]['dhash']
    original_dgm4=ROOT/'data/processed/image_branch_dgm4/manifest.csv';dgm4=read_rows(original_dgm4)
    protected=HashTree()
    for r in genimage:protected.add(int(r['dhash'],16),'genimage')
    excluded=[r for r in dgm4 if protected.cross_split_near(int(r['dhash'],16),'dgm4')]
    excluded_ids={r['sample_id'] for r in excluded};dgm4=[r for r in dgm4 if r['sample_id'] not in excluded_ids]
    cleaned=ROOT/'data/processed/image_branch_dgm4/clean_manifest.csv';write_csv(cleaned,dgm4)
    # Reuse pixels-only embeddings by verified identity, preserving the final clean reservation.
    source_cache=ROOT/'outputs/image_branch_2026_09_14/dgm4_clip_development.pt'
    payload=torch.load(source_cache,map_location='cpu',weights_only=True)
    if payload['csv_sha256']!=sha256(original_dgm4):raise ValueError('DGM4 feature source changed')
    keep=[i for i,key in enumerate(payload['sample_ids']) if key not in excluded_ids]
    payload['features']=payload['features'][keep]
    for key in ('sample_ids','splits','labels'):payload[key]=[payload[key][i] for i in keep]
    payload['source_cache_sha256']=sha256(source_cache);payload['csv_sha256']=sha256(cleaned)
    torch.save(payload,ROOT/'outputs/image_branch_2026_09_14/dgm4_clip_clean_development.pt')
    mixed=[]
    for domain,rows in [('genimage',genimage),('news',news),('dgm4',dgm4)]:
        mixed.extend(dict(sample_id=r['sample_id'],split=r['split'],label=r['label'],image_path=r['image_path'],
                          generator=domain,domain=domain,source_generator=r['generator'],sha256=r['sha256'],dhash=r['dhash']) for r in rows)
    if len({r['sample_id'] for r in mixed})!=len(mixed) or len({r['sha256'] for r in mixed})!=len(mixed):raise ValueError('Duplicate physical images in mixed data')
    csv_path=OUT/'manifest.csv';write_csv(csv_path,mixed)
    report=dict(rows=len(mixed),counts=dict(Counter(f"{r['split']}/{r['domain']}/{r['label']}" for r in mixed)),
                excluded_dgm4_near_genimage=[{k:r[k] for k in ('sample_id','split','generator')} for r in excluded],
                source_sha256={str(p):sha256(p) for p in (GENIMAGE/'manifest.csv',NEWS_CSV,original_dgm4)},
                csv_sha256=sha256(csv_path),clean_dgm4_csv_sha256=sha256(cleaned),
                purpose='Explicit broad image manipulation extension, not the GenImage-only guide experiment',test_predictions_used=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
