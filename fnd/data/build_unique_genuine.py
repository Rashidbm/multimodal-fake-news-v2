"""Expand only V1 training with unique genuine records; no model is loaded.

Preserve the original baseline and evaluation rows. Protect all supplied exclusion
CSVs, even their training rows, so existing downstream experiments stay isolated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .hashing import dhash_image, normalize_text, text_key
from .mmfakebench import load_mmfakebench


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fingerprints:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        path = Path(path).resolve()
        if path not in self.cache:
            raw = path.read_bytes()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                pixels = str(rgb.size).encode() + rgb.tobytes()
                self.cache[path] = {"image_sha1": hashlib.sha1(raw).hexdigest(),
                                    "image_dhash": dhash_image(rgb),
                                    "pixel_sha256": hashlib.sha256(pixels).hexdigest()}
        return self.cache[path]


class HammingIndex:
    """BK-tree for conservative near-image exclusion on 64-bit dHash."""
    def __init__(self):
        self.root = None

    def add(self, value):
        value = int(value, 16)
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def contains_near(self, value, radius):
        value = int(value, 16)
        pending = [self.root] if self.root else []
        while pending:
            current, children = pending.pop()
            distance = (current ^ value).bit_count()
            if distance <= radius:
                return True
            pending.extend(node for edge, node in children.items()
                           if distance - radius <= edge <= distance + radius)
        return False


class ExclusionIndex:
    def __init__(self, radius):
        self.radius = radius
        self.texts, self.files, self.pixels, self.ids = set(), set(), set(), set()
        self.images = HammingIndex()

    def add(self, row, info):
        self.texts.add(normalize_text(row["text"]))
        self.files.add(info["image_sha1"])
        self.pixels.add(info["pixel_sha256"])
        self.ids.add(row["sample_id"])
        self.images.add(info["image_dhash"])

    def reason(self, row, info=None):
        if row["sample_id"] in self.ids:
            return "sample_id"
        if normalize_text(row["text"]) in self.texts:
            return "caption"
        if info is not None:
            if info["image_sha1"] in self.files or info["pixel_sha256"] in self.pixels:
                return "exact_image"
            if self.images.contains_near(info["image_dhash"], self.radius):
                return "perceptual_image"
        return None


def matched_candidates(bank_path, annotation_paths):
    bank_path = Path(bank_path)
    bank = json.loads(bank_path.read_text())
    provenance = defaultdict(list)
    for path in annotation_paths:
        for ann in json.loads(Path(path).read_text())["annotations"]:
            if type(ann["falsified"]) is not bool:
                raise ValueError("NewsCLIPpings falsified must be a JSON boolean")
            if ann["falsified"]:
                continue
            if str(ann["id"]) != str(ann["image_id"]):
                raise ValueError("pristine annotation has different caption/image IDs")
            provenance[str(ann["id"])].append(str(Path(path).resolve()))
    rows = []
    for identifier in sorted(provenance):
        if identifier not in bank:
            continue
        record = bank[identifier]
        rows.append({"sample_id": f"nc_genuine_{identifier}", "source": "newsclippings",
                     "source_split": "official_test", "group": "genuine", "scenario": "4",
                     "label_index": "3", "label_binary": "0", "text_fake": "0",
                     "image_fake": "0", "ooc": "0", "text": record["caption"],
                     "image_path": str((bank_path.parent / record["image_path"]).resolve()),
                     "raw_label": "falsified=false", "subcategory": record.get("source", "unknown"),
                     "text_source": "VisualNews", "image_source": "VisualNews",
                     "rule": "official_pristine_annotation:id=image_id", "split": "train",
                     "caption_id": identifier, "image_id": identifier,
                     "annotation_files": sorted(set(provenance[identifier]))})
    return rows, bank


def source_round_robin(rows, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("subcategory", "unknown")].append(row)
    for group in groups.values():
        rng.shuffle(group)
    while any(groups.values()):
        for source in sorted(groups):
            if groups[source]:
                yield groups[source].pop()


def build(base_rows, fields, preferred, news, reserved, bank, radius=4, seed=20260912):
    training = [r for r in base_rows if r["split"] == "train"]
    fake_counts = Counter(int(r["scenario"]) for r in training if int(r["label_binary"]) == 1)
    if set(fake_counts) != {1, 2, 3, 5} or len(set(fake_counts.values())) != 1:
        raise ValueError("base must have four equal fake training scenarios")
    target = sum(fake_counts.values())
    needed = target - sum(int(r["label_binary"]) == 0 for r in training)
    if needed < 0:
        raise ValueError("genuine class already exceeds fake; this builder never removes rows")
    cache, index = Fingerprints(), ExclusionIndex(radius)
    source_ids = set()
    unique_paths = {}
    for row in [*base_rows, *reserved]:
        path = str(Path(row["image_path"]).resolve())
        unique_paths.setdefault(path, []).append(row)
        for key in ("caption_id", "image_id"):
            if row.get(key):
                source_ids.add(str(row[key]))
    for position, (path, rows) in enumerate(unique_paths.items(), 1):
        info = cache.get(path)  # Fail on missing protected images: never silently skip exclusions.
        for row in rows:
            index.add(row, info)
        if position % 2000 == 0:
            print(f"Fingerprinting protected images: {position}/{len(unique_paths)}", flush=True)
    # A reserved OOC caption also excludes its original matched pair, even when
    # the original photo differs from the reused photo in that OOC record.
    for identifier, record in bank.items():
        if normalize_text(record["caption"]) in index.texts:
            source_ids.add(str(identifier))
    rng, added, ledger, excluded = random.Random(seed), [], [], Counter()
    for candidates in (preferred, news):
        for row in source_round_robin(candidates, rng):
            if len(added) == needed:
                break
            reason = index.reason(row)
            if not normalize_text(row["text"]):
                reason = "empty_caption"
            if row.get("caption_id") in source_ids or row.get("image_id") in source_ids:
                reason = "source_id"
            if reason:
                excluded[f"{row['source']}:{reason}"] += 1
                continue
            try:
                info = cache.get(row["image_path"])
            except (OSError, ValueError):
                excluded[f"{row['source']}:unreadable_image"] += 1
                continue
            reason = index.reason(row, info)
            if reason:
                excluded[f"{row['source']}:{reason}"] += 1
                continue
            if row.get("group") != "genuine" or int(row["label_binary"]) != 0:
                raise ValueError("only genuine additions are permitted")
            row = {**row, **info, "text_key": text_key(row["text"]),
                   "image_key": f"sha1:{info['image_sha1']}", "image_ok": "1",
                   "label_index": "3", "split": "train", "cluster": f"extra_{row['sample_id']}"}
            added.append({key: str(row.get(key, "")) for key in fields})
            ledger.append(row)
            index.add(row, info)
            source_ids.update(str(row[k]) for k in ("caption_id", "image_id") if row.get(k))
            if len(added) % 500 == 0:
                print(f"Unique genuine additions: {len(added)}/{needed}", flush=True)
    if len(added) != needed:
        raise ValueError(f"insufficient unique genuine samples: need {needed}, found {len(added)}; "
                         f"exclusions={dict(excluded)}")
    output = base_rows + added
    counts = Counter(int(r["label_binary"]) for r in output if r["split"] == "train")
    assert counts[0] == counts[1] == target
    assert len({r["sample_id"] for r in output}) == len(output)
    assert all([r for r in output if r["split"] == s] == [r for r in base_rows if r["split"] == s]
               for s in ("val", "test"))
    return output, ledger, {"added": len(added), "added_by_source": dict(Counter(r["source"] for r in added)),
                           "added_by_subcategory": dict(Counter(f"{r['source']}:{r['subcategory']}" for r in added)),
                           "excluded_candidates": dict(excluded), "train_binary_counts": dict(counts),
                           "train_scenario_counts": dict(Counter(r["scenario"] for r in output if r["split"] == "train")),
                           "protected_unique_image_paths": len(unique_paths), "val_test_unchanged": True,
                           "perceptual_exclusion_max_hamming_distance": radius}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--mmfakebench", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--exclude", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--near-radius", type=int, default=4)
    args = parser.parse_args(argv)
    if not 0 <= args.near_radius <= 8:
        parser.error("near-radius must be 0..8")
    out = Path(args.out)
    sidecars = [out, out.with_suffix(".manifest.json"), out.with_suffix(".additions.json")]
    if any(p.exists() for p in sidecars):
        raise FileExistsError("destination or sidecar already exists")
    base, fields = read_csv(args.base)
    reserved = [r for path in args.exclude for r in read_csv(path)[0]]
    samples, _ = load_mmfakebench(args.mmfakebench, require_images=True)
    preferred = [sample.to_row() for sample in samples if sample.group == "genuine"]
    news, bank = matched_candidates(args.bank, args.annotations)
    print(f"Candidate pools: MMFakeBench genuine={len(preferred)}, NewsCLIPpings matched={len(news)}", flush=True)
    result, ledger, report = build(base, fields, preferred, news, reserved, bank,
                                   radius=args.near_radius, seed=args.seed)
    input_files = [args.base, args.bank, *args.annotations, *args.exclude,
                   *map(str, sorted(Path(args.mmfakebench).glob("MMFakeBench_*.json")))]
    report.update(args=vars(args), total_rows=len(result),
                  split_counts=dict(Counter(r["split"] for r in result)),
                  input_sha256={str(Path(p).resolve()): sha256(p) for p in input_files},
                  builder_sha256=sha256(__file__),
                  baseline_rows_preserved=True, training_started=False,
                  interpretation="Custom V1 train expansion. Official NewsCLIPpings test records are reassigned "
                  "only after exclusion against existing custom experiments; no official NewsCLIPpings test "
                  "score may be claimed from this training setup. Upstream matched labels are preserved; "
                  "they are not independent factual or forensic verification. Existing baseline label issues "
                  "are not relabeled. Perceptual exclusion is conservative, not proof against every crop or edit.",
                  required_training_option="--no-oversample")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result)
        reloaded, _ = read_csv(temporary)
        if reloaded != result:
            raise AssertionError("CSV round trip changed records")
        report["csv_sha256"] = sha256(temporary)
        out.with_suffix(".additions.json").write_text(json.dumps(ledger, indent=2) + "\n")
        out.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("input_sha256", "args")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
