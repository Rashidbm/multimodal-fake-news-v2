"""Independent cross-split and pair audit for this round's two new datasets."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fnd.data.build_unique_genuine import Fingerprints, sha256
from fnd.data.hashing import normalize_text

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    fingerprints = Fingerprints()
    reports = {}
    for name in ['semantic_resolution_confirmation', 'semantic_resolution_extra_train']:
        path = ROOT/f'data/processed/{name}.csv'
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        assert sha256(path) == manifest['output_sha256']
        rows = read(path)
        groups = defaultdict(list)
        for row in rows:
            groups[row['caption_id']].append(row)
            actual = fingerprints.get(row['image_path'])
            assert all(row[key] == actual[key] for key in actual)
        assert len({(normalize_text(r['text']), r['image_sha1']) for r in rows}) == len(rows)
        for group in groups.values():
            assert len(group) == 2 and {r['scenario'] for r in group} == {'1','4'}
            assert len({normalize_text(r['text']) for r in group}) == 1
            assert all((r['caption_id'] == r['image_id']) == (r['scenario'] == '4') for r in group)
        protected = []
        for old_path, expected in manifest['inputs_sha256'].items():
            assert sha256(old_path) == expected
            if old_path.endswith('.csv'):
                protected.extend(read(old_path))
        text_overlap = {normalize_text(r['text']) for r in rows} & {normalize_text(r['text']) for r in protected}
        ids = lambda records: {r[key] for r in records for key in ['caption_id','image_id'] if r.get(key)}
        id_overlap = ids(rows) & ids(protected)
        assert not text_overlap and not id_overlap
        # Recompute protected image fingerprints from bytes, independently of
        # the builder's BK-tree and the stored CSV hashes.
        protected_info = [fingerprints.get(p) for p in dict.fromkeys(r['image_path'] for r in protected)]
        assert not ({r['image_sha1'] for r in rows} & {p['image_sha1'] for p in protected_info})
        assert not ({r['pixel_sha256'] for r in rows} & {p['pixel_sha256'] for p in protected_info})
        current_hash = np.array([int(r['image_dhash'],16) for r in rows], dtype=np.uint64)
        protected_hash = np.array(list({int(p['image_dhash'],16) for p in protected_info}), dtype=np.uint64)
        minimum = 64
        for start in range(0, len(rows), 64):
            distance = np.bitwise_count(current_hash[start:start+64,None]^protected_hash[None,:])
            minimum = min(minimum, int(distance.min()))
        assert minimum > 4
        within_minimum = 64
        for start in range(0, len(rows), 64):
            distance = np.bitwise_count(current_hash[start:start+64,None]^current_hash[None,:])
            distance[np.arange(len(distance)), np.arange(start,start+len(distance))] = 64
            within_minimum = min(within_minimum, int(distance.min()))
        assert within_minimum > 4
        reports[name] = dict(rows=len(rows), caption_pairs=len(groups), scenario_counts=dict(Counter(r['scenario'] for r in rows)),
            csv_sha256=sha256(path), source_id_overlap=0, normalized_caption_overlap=0,
            byte_and_pixel_overlap=0, minimum_dhash_distance_to_protected=minimum,
            minimum_dhash_distance_within_dataset=within_minimum, all_image_bytes_recomputed=True,
            duplicate_image_caption_samples=0, predictions_used=False)
        print(name+': '+json.dumps(reports[name]), flush=True)
    (ROOT/'outputs/semantic_resolution/data_verification.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
