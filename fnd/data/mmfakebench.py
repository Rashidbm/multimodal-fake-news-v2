"""MMFakeBench loader: raw JSON records -> Sample with a group.

Layout expected on disk (what ``huggingface-cli download`` produces):

    <root>/MMFakeBench_val.json          1,000 records
    <root>/MMFakeBench_test.json        10,000 records
    <root>/MMFakeBench_val/<image_path>  extracted images (optional here)
    <root>/MMFakeBench_test/<image_path>

Each record: text, image_path, text_source, image_source, gt_answers, fake_cls.

THE MAPPING RULE (agreed with the paper, arXiv 2406.08772 section 3):

    fake_cls                     image_source            -> group
    ---------------------------  ----------------------  --------------------
    original                     (same as text_source)   genuine
    mismatch                     Newsclipings            ooc
    textual_veracity_distortion  Repurposed Image        fake_text_real_image
    textual_veracity_distortion  AI-generated Image      fake_text_fake_image
    visual_veracity_distortion   (any)                   real_text_fake_image
    mismatch                     DGM4                    fake_text_real_image
    mismatch                     COCO-Counterfactuals    coco_image_edit -> real_text_fake_image
                                                         coco_text_edit  -> fake_text_fake_image

Every rule ALSO checks that text_source is on the expected side (rumour
sources for fake text, caption sources for real text).  A record that
matches no rule, or whose text_source contradicts the rule, raises
MappingError with the offending record.  Nothing is dropped silently.

The folder name (second path component, e.g. 'fever_AI_val_100') is used
for exactly two things: telling the two COCO-Counterfactuals folders apart,
and accepting the 100 antifact records whose source fields are blank.
Afterwards the loader checks that every folder mapped to ONE group only,
and that the four fake_cls totals equal the paper's (3300/3300/1100/3300).
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

# Where the TEXT came from, according to the paper.
REAL_TEXT_SOURCES = {"VisualNews", "Newsclipings", "Fakeddit", "MS-COCO"}
RUMOR_TEXT_SOURCES = {"Fever", "GPT-generated Rumor", "Fakenewsnet", "Gossipcop"}
EDITED_TEXT_SOURCES = {"DGM4", "COCO-Counterfactuals"}

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


def _fail(rec: dict, why: str) -> MappingError:
    return MappingError(f"{why}\n  record: {json.dumps(rec, ensure_ascii=False)[:400]}")


def classify(rec: dict) -> tuple[str, str]:
    """Return (group, rule_name) for one raw record, or raise MappingError."""
    fc = rec.get("fake_cls", "")
    ts = rec.get("text_source", "")
    im = rec.get("image_source", "")
    sub = subcategory(rec["image_path"])

    if fc == "original":
        if ts not in REAL_TEXT_SOURCES or im != ts:
            raise _fail(rec, "original record whose sources are not a real-caption source")
        return "genuine", "original"

    if fc == "visual_veracity_distortion":
        # Paper 3.1.2: "the text is real and the misinformation exists in the image".
        blank_ok = (ts == "" and im == "" and sub == "antifact_image_generation")
        if ts not in REAL_TEXT_SOURCES and not blank_ok:
            raise _fail(rec, "visual distortion record whose text_source is not a real-caption source")
        return "real_text_fake_image", "visual+blank" if blank_ok else "visual"

    if fc == "textual_veracity_distortion":
        # Paper 3.1.1: rumour text + supporting image that is either
        # "AI-generated" (fake image) or "Repurposed" (real VisualNews photo).
        if ts not in RUMOR_TEXT_SOURCES:
            raise _fail(rec, "textual distortion record whose text_source is not a rumour source")
        if im == "Repurposed Image":
            return "fake_text_real_image", "textual+repurposed"
        if im == "AI-generated Image":
            return "fake_text_fake_image", "textual+ai_image"
        raise _fail(rec, f"textual distortion record with unexpected image_source {im!r}")

    if fc == "mismatch":
        # Paper 3.1.3: repurposed (NewsCLIPpings) or edited (DGM4 text, COCO-Counterfactuals).
        if ts == im == "Newsclipings":
            return "ooc", "mismatch+newsclippings"
        if ts == im == "DGM4":
            return "fake_text_real_image", "mismatch+dgm4_text_edit"
        if ts == im == "COCO-Counterfactuals":
            if sub == "coco_image_edit":
                return "real_text_fake_image", "mismatch+coco_image_edit"
            if sub == "coco_text_edit":
                return "fake_text_fake_image", "mismatch+coco_text_edit"
            raise _fail(rec, f"COCO-Counterfactuals record in unexpected folder {sub!r}")
        raise _fail(rec, f"mismatch record with unexpected sources text={ts!r} image={im!r}")

    raise _fail(rec, f"unknown fake_cls {fc!r}")


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
        lines.append("\nrecords per rule:")
        for r, n in self.per_rule.most_common():
            lines.append(f"  {n:6d}  {r}")
        lines.append(f"\nrecords with blank text/image source (accepted by folder): {self.blank_source_records}")
        lines.append(f"paper totals check (original/textual/visual/mismatch = 3300/3300/1100/3300): {self.totals_check}")
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
            group, rule = classify(rec)
            sub = subcategory(rec["image_path"])
            img, exists = _resolve_image(root, split, rec["image_path"])
            if not exists:
                rep.missing_images[split] += 1
            if rec.get("text_source", "") == "" or rec.get("image_source", "") == "":
                rep.blank_source_records += 1

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
                rule=rule,
                extra={"gt_answers": rec.get("gt_answers", "")},
            ))
            rep.records += 1
            rep.per_split[split] += 1
            rep.per_group[group] += 1
            rep.per_rule[rule] += 1
            rep.per_fake_cls[rec["fake_cls"]] += 1
            rep.folder_to_groups[sub][group] += 1

    # Check 1: fields and folder must agree -> one group per folder.
    bad = {f: dict(g) for f, g in rep.folder_to_groups.items() if len(g) != 1}
    if bad:
        raise MappingError(f"folders mapped to more than one group: {bad}")

    # Check 2: reproduce the paper's four totals when the full benchmark is loaded.
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
