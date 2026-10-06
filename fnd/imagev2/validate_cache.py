r"""Validate a feature cache directory against the manifest (counts, ids, splits, shapes, NaN/Inf).

    python -m fnd.imagev2.validate_cache --features D:\...\features\A1 --manifest-dir data/image_branch_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .extract import view_plan
from .manifest import load_joint, sha256_file


def validate(features, manifest_dir):
    features = Path(features)
    meta = json.loads((features / "meta.json").read_text())
    rows = load_joint(manifest_dir)
    if meta["manifest_sha256"] != sha256_file(Path(manifest_dir) / "joint_forensic_all.csv"):
        raise ValueError("cache was built from a different manifest")
    plan = view_plan(rows)
    if meta["arm"] == "A1s":
        plan = {"clean": plan["clean"]}
    report = {}
    for view, indices in plan.items():
        x, ids = np.load(features / f"{view}.npy"), np.load(features / f"{view}_ids.npy")
        expected = [rows[i]["image_id"] for i in indices]
        assert list(ids) == expected, f"{view}: ids differ from the manifest plan"
        assert len(set(ids)) == len(ids), f"{view}: duplicate ids"
        assert x.shape == (len(expected), meta["dim"]), f"{view}: shape {x.shape}"
        assert np.isfinite(x).all(), f"{view}: NaN/Inf"
        splits = {rows[i]["split"] for i in indices}
        if view in ("aug1", "aug2"):
            assert splits == {"train"}, f"{view}: non-train rows"
        if view in ("jpeg75", "blur1"):
            assert splits <= {"val", "test"} and all(rows[i]["balance_role"] == "core" for i in indices), f"{view}: unexpected rows"
        report[view] = dict(shape=list(x.shape), splits=sorted(splits), finite=True, mean=float(x.mean()), std=float(x.std()))
    return dict(arm=meta["arm"], dim=meta["dim"], blocks=meta["blocks"], views=report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest-dir", default="data/image_branch_v2")
    a = ap.parse_args()
    print(json.dumps(validate(a.features, a.manifest_dir), indent=2))
