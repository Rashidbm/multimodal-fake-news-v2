"""Build fact-check manifests (JSONL) from MMFakeBench.

Reads the MMFakeBench JSON files (``MMFakeBench_<split>.json``) with the
project's loader, or, when those are absent, Arrow files from a Hugging Face
datasets cache with the same fields.  Images are resolved to absolute paths on
*this* machine, so prepare manifests on the PC that holds the images.

Labels are MMFakeBench's own four-way ``fake_cls``: original,
textual_veracity_distortion, visual_veracity_distortion, mismatch.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fnd.data.mmfakebench import _resolve_image, classify

from .config import sha256_file
from .prompts import MMFAKE_LABELS

DEFAULT_ROOT = "data/raw/MMFakeBench"


def _records_from_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a JSON list")
    return records


def _records_from_arrow(files: list[Path]) -> list[dict]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as e:
        raise ImportError("reading Arrow files needs pyarrow (pip install pyarrow)") from e
    rows: list[dict] = []
    for p in files:
        with pa.memory_map(str(p)) as source:
            try:
                table = ipc.open_stream(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                table = ipc.open_file(source).read_all()
        rows.extend(table.to_pylist())
    return rows


def load_split(root: str | Path, split: str) -> tuple[list[dict], dict]:
    """Return (records, provenance) for one split."""
    root = Path(root)
    json_path = root / f"MMFakeBench_{split}.json"
    if json_path.is_file():
        return _records_from_json(json_path), {"format": "json", "files": {str(json_path): sha256_file(json_path)}}
    arrow = sorted(p for p in root.rglob("*.arrow") if split in p.name.lower() or split in p.parent.name.lower())
    if arrow:
        return _records_from_arrow(arrow), {"format": "arrow", "files": {str(p): sha256_file(p) for p in arrow}}
    raise FileNotFoundError(f"neither {json_path} nor Arrow files for split {split!r} under {root}")


def build_manifest(root: str | Path, split: str, limit: int, balanced: bool, seed: int = 0) -> tuple[list[dict], dict]:
    root = Path(root).resolve()
    records, provenance = load_split(root, split)
    examples, missing = [], 0
    for idx, rec in enumerate(records):
        label = rec.get("fake_cls")
        if label not in MMFAKE_LABELS:
            raise ValueError(f"record {idx}: unknown fake_cls {label!r}")
        group, sub = classify(rec)
        image, exists = _resolve_image(root, split, rec["image_path"])
        if not exists:
            missing += 1
            continue
        examples.append({
            "id": f"mmfb_{split}_{idx}", "task": "mmfakebench", "text": rec["text"],
            "image": str(image.resolve()), "label": label, "split": split, "subcategory": sub,
            "group": group, "text_source": rec.get("text_source", ""),
            "image_source": rec.get("image_source", ""), "excluded_urls": [], "cutoff_date": None,
        })
    if missing:
        raise FileNotFoundError(f"{missing} of {len(records)} {split} images are missing under {root}; "
                                "extract the image archives first")

    rng = random.Random(seed)
    if balanced:
        by_label = {c: [e for e in examples if e["label"] == c] for c in MMFAKE_LABELS}
        for items in by_label.values():
            rng.shuffle(items)
        per = min(len(v) for v in by_label.values())
        if limit:
            if limit % len(MMFAKE_LABELS):
                raise ValueError(f"--balanced needs --limit divisible by {len(MMFAKE_LABELS)}")
            if limit // len(MMFAKE_LABELS) > per:
                raise ValueError(f"only {per} examples in the smallest class; --limit {limit} is too large")
            per = limit // len(MMFAKE_LABELS)
        chosen = [x for c in MMFAKE_LABELS for x in by_label[c][:per]]
        chosen.sort(key=lambda e: int(e["id"].rsplit("_", 1)[1]))
    else:
        chosen = list(examples)
        if limit:
            rng.shuffle(chosen)
            chosen = sorted(chosen[:limit], key=lambda e: int(e["id"].rsplit("_", 1)[1]))

    meta = {"root": str(root), "split": split, "balanced": balanced, "limit": limit, "seed": seed,
            "n": len(chosen), "labels": dict(Counter(e["label"] for e in chosen)),
            "source": provenance, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": ("balanced subset: not the official test distribution" if balanced
                     else "full split" if not limit else "random subset")}
    return chosen, meta


def write_manifest(rows: list[dict], meta: dict, output: str | Path, force: bool = False) -> Path:
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; reuse it, or pass --force to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {**meta, "manifest_sha256": sha256_file(output)}
    output.with_name(output.name + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output
