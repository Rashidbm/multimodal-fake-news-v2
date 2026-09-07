"""MMFakeBench loader: raw JSON records -> Sample with a group.

Layout expected on disk (what ``huggingface-cli download`` produces):

    <root>/MMFakeBench_val.json          1,000 records
    <root>/MMFakeBench_test.json        10,000 records
    <root>/MMFakeBench_val/<image_path>  extracted images (optional here)
    <root>/MMFakeBench_test/<image_path>

Each record: text, image_path, text_source, image_source, gt_answers, fake_cls.

THE MAPPING: one table, folder name -> scenario (FOLDER_TO_GROUP below).
The folder is the paper's sub-category, and section 3 of the paper says what
each sub-category is made of.  A folder not in the table stops the program.
After mapping, the loader checks that the four fake_cls totals equal the
paper's (3300/3300/1100/3300), proving the whole benchmark was read.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .records import GROUPS, MappingError, Sample

SOURCE = "mmfakebench"

# One line per MMFakeBench sub-category (= folder name without the _val_N /
# _test_N suffix), as defined in the paper, arXiv 2406.08772 section 3:
#   3.2   real data: six sources                           -> genuine
#   3.1.1 rumour text + repurposed real photo              -> fake_text_real_image
#   3.1.1 rumour text + AI-generated image                 -> fake_text_fake_image
#   3.1.2 real text + Photoshop / AI-generated image       -> real_text_fake_image
#   3.1.3 repurposed inconsistency (NewsCLIPpings)         -> ooc
#   3.1.3 edited text (DGM4 antonym swap, COCO edit)       -> fake_text_real_image / fake_text_fake_image
#   3.1.3 edited image (COCO-Counterfactuals)              -> real_text_fake_image
FOLDER_TO_GROUP: dict[str, str] = {
    "bbc": "genuine", "guardian": "genuine", "usa_today": "genuine", "wash": "genuine",
    "coco": "genuine", "fakeddit": "genuine",
    "Newsclipings_person": "ooc", "Newsclipings_scene": "ooc", "Newsclipings_semantic": "ooc",
    "rumor_match": "fake_text_real_image", "politicat_match": "fake_text_real_image",
    "gossipcop_match": "fake_text_real_image", "chatgpt_match": "fake_text_real_image",
    "DGM4_text_edit_senti": "fake_text_real_image",
    "Fakeddit_photo_edit": "real_text_fake_image", "antifact_image_generation": "real_text_fake_image",
    "coco_image_edit": "real_text_fake_image",
    "fever_AI": "fake_text_fake_image", "llm_rewrite": "fake_text_fake_image",
    "llm_gossip_md_generation": "fake_text_fake_image", "llm_science_md_generation": "fake_text_fake_image",
    "gossipcop_midjourney": "fake_text_fake_image", "coco_text_edit": "fake_text_fake_image",
}

# Paper section 3.3: 30% textual, 10% visual, 30% cross-modal, 30% real of 11,000.
PAPER_TOTALS = {
    "original": 3300,
    "textual_veracity_distortion": 3300,
    "visual_veracity_distortion": 1100,
    "mismatch": 3300,
}

_SUFFIX = re.compile(r"_(val|test)_\d+$")


def subcategory(image_path: str) -> str:
    """'/fake/fever_AI_val_100/x.png' -> 'fever_AI'."""
    parts = image_path.strip("/").split("/")
    if len(parts) < 2:
        raise MappingError(f"image_path has no sub-category folder: {image_path!r}")
    return _SUFFIX.sub("", parts[1])


def classify(rec: dict) -> tuple[str, str]:
    """Return (group, folder) for one raw record.  Unknown folder -> MappingError."""
    sub = subcategory(rec["image_path"])
    try:
        return FOLDER_TO_GROUP[sub], sub
    except KeyError:
        raise MappingError(f"folder {sub!r} is not in FOLDER_TO_GROUP; add it after reading the paper\n"
                           f"  record: {json.dumps(rec, ensure_ascii=False)[:300]}") from None


@dataclass
class LoaderReport:
    records: int = 0
    per_split: Counter = field(default_factory=Counter)
    per_group: Counter = field(default_factory=Counter)
    per_rule: Counter = field(default_factory=Counter)
    per_fake_cls: Counter = field(default_factory=Counter)
    folder_to_groups: dict = field(default_factory=lambda: defaultdict(Counter))
    blank_source_records: int = 0
    missing_images: Counter = field(default_factory=Counter)
    totals_check: str = "skipped (need both splits)"

    def format(self) -> str:
        lines = [f"MMFakeBench: {self.records} records  " + ", ".join(
            f"{s}={n}" for s, n in sorted(self.per_split.items()))]
        lines.append("\nfolder -> group (each folder must map to exactly one group):")
        for folder in sorted(self.folder_to_groups):
            groups = self.folder_to_groups[folder]
            flag = "" if len(groups) == 1 else "   <-- INCONSISTENT"
            lines.append(f"  {folder:28} " + ", ".join(f"{g} ({n})" for g, n in groups.items()) + flag)
        lines.append("\nrecords per group:")
        for g in GROUPS:
            lines.append(f"  {self.per_group[g]:6d}  {g}")
        lines.append(f"\npaper totals check (original/textual/visual/mismatch = 3300/3300/1100/3300): {self.totals_check}")
        if self.missing_images:
            lines.append("images not found on disk: " + ", ".join(
                f"{s}={n}" for s, n in sorted(self.missing_images.items())))
        else:
            lines.append("images: all found on disk")
        return "\n".join(lines)


def _resolve_image(root: Path, split: str, rel: str) -> tuple[Path, bool]:
    """Try the known layouts; return (path, exists)."""
    rel = rel.lstrip("/")
    candidates = [
        root / f"MMFakeBench_{split}" / rel,
        root / f"MMFakeBench_{split}" / f"MMFakeBench_{split}" / rel,
        root / rel,
    ]
    for c in candidates:
        if c.is_file():
            return c, True
    return candidates[0], False


def load_mmfakebench(root: str | Path, splits=("val", "test"),
                     require_images: bool = False) -> tuple[list[Sample], LoaderReport]:
    root = Path(root)
    samples: list[Sample] = []
    rep = LoaderReport()

    for split in splits:
        json_path = root / f"MMFakeBench_{split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"missing {json_path}")
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise MappingError(f"{json_path} is not a JSON list")

        for idx, rec in enumerate(records):
            group, sub = classify(rec)
            img, exists = _resolve_image(root, split, rec["image_path"])
            if not exists:
                rep.missing_images[split] += 1

            samples.append(Sample(
                sample_id=f"mmfb_{split}_{idx}",
                source=SOURCE,
                source_split=split,
                text=rec["text"],
                image_path=str(img),
                group=group,
                raw_label=rec["fake_cls"],
                subcategory=sub,
                text_source=rec.get("text_source", ""),
                image_source=rec.get("image_source", ""),
                rule=f"folder:{sub}",
                extra={"gt_answers": rec.get("gt_answers", "")},
            ))
            rep.records += 1
            rep.per_split[split] += 1
            rep.per_group[group] += 1
            rep.per_fake_cls[rec["fake_cls"]] += 1
            rep.folder_to_groups[sub][group] += 1

    # Check: reproduce the paper's four totals when the full benchmark is loaded.
    if set(splits) >= {"val", "test"}:
        got = {k: rep.per_fake_cls[k] for k in PAPER_TOTALS}
        if got != PAPER_TOTALS:
            raise MappingError(f"fake_cls totals {got} differ from the paper's {PAPER_TOTALS}")
        rep.totals_check = "PASS"

    if require_images and rep.missing_images:
        raise FileNotFoundError(f"images missing: {dict(rep.missing_images)}")
    return samples, rep


def write_csv(samples: list[Sample], path: str | Path) -> None:
    rows = [s.to_row() for s in samples]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect MMFakeBench and map every record to a group.")
    ap.add_argument("--root", required=True, help="folder containing MMFakeBench_val.json / _test.json")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--require-images", action="store_true", help="fail if any image file is missing")
    ap.add_argument("--csv", help="optional: write the mapped records to this CSV for eyeballing")
    args = ap.parse_args(argv)

    samples, rep = load_mmfakebench(args.root, tuple(args.splits), args.require_images)
    print(rep.format())
    if args.csv:
        write_csv(samples, args.csv)
        print(f"\nwrote {len(samples)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
