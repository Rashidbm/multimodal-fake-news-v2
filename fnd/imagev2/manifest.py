"""Image Branch v2 manifests (data/image_branch_v2) as arrays aligned to image IDs."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

TRUTH_LABEL = {"REAL": 0, "AI_GENERATED": 1, "MANIPULATED": 2}
DOMAINS = ("news", "coco", "fakeddit")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_joint(manifest_dir):
    """All rows of joint_forensic_all.csv (core + reserve), validated."""
    rows = read_csv(Path(manifest_dir) / "joint_forensic_all.csv")
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image ids in joint manifest")
    for r in rows:
        if r["original_project_split"] != "train":
            raise ValueError(f"{r['image_id']} is not original project train")
        if TRUTH_LABEL[r["image_truth"]] != int(r["label"]):
            raise ValueError(f"label disagrees with truth for {r['image_id']}")
        if r["split"] not in ("train", "val", "test"):
            raise ValueError(f"bad split for {r['image_id']}")
    return rows


def arrays(rows):
    """Aligned numpy arrays for a list of manifest rows."""
    return dict(
        ids=np.array([r["image_id"] for r in rows]),
        y3=np.array([int(r["label"]) for r in rows]),
        split=np.array([r["split"] for r in rows]),
        role=np.array([r["balance_role"] for r in rows]),
        domain=np.array([r["domain"] for r in rows]),
        subsource=np.array([r["subsource"] for r in rows]),
        group=np.array([r["group_id"] for r in rows]),
        family=np.array([r["ai_family"] for r in rows]))


def holdout_columns(manifest_dir):
    """image_id -> (split_h_mj, split_h_cf) from the AI-detector manifest."""
    rows = read_csv(Path(manifest_dir) / "ai_detector_all.csv")
    return {r["image_id"]: (r["split_h_mj"], r["split_h_cf"]) for r in rows}
