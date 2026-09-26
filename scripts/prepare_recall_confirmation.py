"""Reserve a second confirmation set without consulting any model predictions."""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from fnd.data.build_unique_genuine import ExclusionIndex, Fingerprints, sha256
from fnd.data.hashing import normalize_text, text_key


ROOT = Path(__file__).resolve().parents[1]
BANK = Path('/Users/rashid/multimodaldetection/data/raw/NewsCLIPpings/test_dataset/visual_news_test.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT/'data/processed/recall_confirmation.csv'))
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026091402)
    parser.add_argument('--protect', nargs='*', default=[])
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--min-pairs', type=int)
    args = parser.parse_args(argv)
    if args.pairs < 1:
        raise ValueError('Need at least one caption pair')
    minimum = args.pairs if args.min_pairs is None else args.min_pairs
    if not 1 <= minimum <= args.pairs:
        raise ValueError('min-pairs must be between 1 and pairs')
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    paths = [ROOT/'data/processed'/name for name in [
        'v1_unique_genuine.csv', 'v1_extra_genuine.csv', 'v2_3class.csv', 'v2_3class_unionexcluded.csv',
        'v2_mmfb_heldout.csv', 'v1_alignment_holdout.csv', 'v1_matching_pretrain.csv',
        'clip_adaptation/v1_blip_train_dev.csv', 'clip_adaptation/confirmation.csv']]
    paths.extend(Path(path).resolve() for path in args.protect)
    bank = json.loads(BANK.read_text())
    fingerprints, index = Fingerprints(), ExclusionIndex(radius=4)
    source_ids, reasons, unique_paths = set(), Counter(), defaultdict(list)
    for path in paths:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                unique_paths[str(Path(row['image_path']).resolve())].append(row)
                source_ids.update(str(row[k]) for k in ('caption_id', 'image_id') if row.get(k))
    for n, (path, rows) in enumerate(unique_paths.items(), 1):
        info = fingerprints.get(path)
        for row in rows:
            index.add(row, info)
        if n % 3000 == 0:
            print(f'Protected images {n}/{len(unique_paths)}', flush=True)
    source_ids.update(k for k, record in bank.items() if normalize_text(record['caption']) in index.texts)
    genuine, negative = set(), defaultdict(set)
    annotations = sorted((ROOT/'data/raw/NewsCLIPpings/additional_test_annotations').glob('*.json'))
    for path in annotations:
        for ann in json.loads(path.read_text())['annotations']:
            if type(ann['falsified']) is not bool:
                raise ValueError('Expected official boolean label')
            caption, image = str(ann['id']), str(ann['image_id'])
            if ann['falsified']:
                negative[caption].add((image, path.stem))
            else:
                if caption != image:
                    raise ValueError('Invalid pristine pair')
                genuine.add(caption)
    rng = random.Random(args.seed)
    candidates = sorted(genuine & negative.keys()); rng.shuffle(candidates)
    accepted = []
    for caption in candidates:
        if caption not in bank or caption in source_ids or normalize_text(bank[caption]['caption']) in index.texts:
            continue
        choices = sorted(negative[caption]); rng.shuffle(choices)
        for image_id, method in choices:
            if image_id not in bank or image_id in source_ids:
                continue
            rows, infos = [], []
            for scenario, image_id_, method_ in [(4, caption, 'pristine'), (1, image_id, method)]:
                row = dict(sample_id=f'nc_{out.stem}_{caption}_{image_id_}', source='NewsCLIPpings', source_split='official_test',
                           split=args.split, scenario=scenario, label_binary=int(scenario==1), label_index=scenario-1,
                           text=bank[caption]['caption'], text_key=text_key(bank[caption]['caption']),
                           image_path=str(BANK.parent/bank[image_id_]['image_path']), caption_id=caption, image_id=image_id_,
                           mismatch_method=method_, publisher=bank[caption].get('source','unknown'), evaluation_group='paired_news')
                try:
                    info = fingerprints.get(row['image_path'])
                except (OSError, ValueError):
                    reasons['unreadable'] += 1; break
                reason = index.reason(row, info)
                if reason:
                    reasons[reason] += 1; break
                rows.append(row); infos.append(info)
            if len(rows) != 2:
                continue
            if ((int(infos[0]['image_dhash'],16)^int(infos[1]['image_dhash'],16)).bit_count()<=4
                    or infos[0]['pixel_sha256']==infos[1]['pixel_sha256']):
                reasons['near_inside_pair'] += 1; continue
            for row, info in zip(rows, infos):
                row.update(info); index.add(row,info)
                source_ids.update([row['caption_id'],row['image_id']])
            accepted.extend(rows)
            break
        if len(accepted)>=2*args.pairs:
            break
    if len(accepted)<2*minimum:
        raise ValueError(f'Insufficient fresh captions: {len(accepted)//2}')
    with out.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(accepted[0])); writer.writeheader(); writer.writerows(accepted)
    manifest = dict(seed=args.seed, captions=len(accepted)//2, rows=len(accepted), predictions_used=False,
                    requested_pairs=args.pairs, minimum_pairs=minimum, split=args.split,
                    protected_images=len(unique_paths), exclusions=dict(reasons),
                    inputs_sha256={str(p):sha256(p) for p in [*paths,BANK,*annotations]}, output_sha256=sha256(out),
                    isolation='Normalized captions, source IDs, byte/RGB hashes and dHash distance >4; caption partners kept together. No event-disjointness claim.',
                    protocol='Custom split of NewsCLIPpings official test-source material; not the official benchmark.')
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='inputs_sha256'},indent=2),flush=True)


if __name__=='__main__':
    main()
