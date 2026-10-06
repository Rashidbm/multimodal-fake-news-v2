import csv
import hashlib
import random

import pytest
from PIL import Image

from fnd.data import build_image_branch_v2 as b


def make_dataset(root, spec, seed=0):
    """spec: list of (source, subcategory, size, split, count). Returns the dataset root."""
    (root / "data").mkdir(parents=True)
    (root / "images").mkdir()
    rng, rows, index = random.Random(seed), [], 0
    for source, sub, size, split, count in spec:
        for _ in range(count):
            index += 1
            path = root / "images" / f"{index}.png"
            Image.new("RGB", size, (index % 256, (index // 256) % 256, 7)).save(path)
            raw = path.read_bytes()
            rows.append(dict(sample_id=f"s{index}", split=split, scenario="1", image_fake="0", image_asset_id=f"img_{index:04d}",
                             image_path=f"images/{index}.png", image_sha256=hashlib.sha256(raw).hexdigest(), image_sha1=hashlib.sha1(raw).hexdigest(),
                             image_dhash=f"{rng.getrandbits(64):016x}", source=source, subcategory=sub, text=f"unique caption number {index} here"))
    with (root / "data/master.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (root / "data/image_samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_asset_id", "split"]); w.writeheader()
        w.writerows({"image_asset_id": r["image_asset_id"], "split": r["split"]} for r in rows)
    with (root / "data/protected_external_evaluations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_sha1", "image_dhash"]); w.writeheader()
    return root


SPEC = [("mmfakebench", "bbc", (32, 24), "train", 30), ("NewsCLIPpings", "", (32, 24), "train", 30),
        ("mmfakebench", "fever_AI", (1024, 1024), "train", 20), ("mmfakebench", "fever_AI", (512, 320), "train", 2),
        ("mmfakebench", "Fakeddit_photo_edit", (40, 30), "train", 12), ("mmfakebench", "fakeddit", (32, 24), "train", 14),
        ("mmfakebench", "fever_AI", (1024, 1024), "val", 5), ("mmfakebench", "bbc", (32, 24), "test", 5)]


@pytest.fixture()
def result(tmp_path):
    return b.build(make_dataset(tmp_path / "ds", SPEC), verify_hashes=True, workers=2), tmp_path / "ds"


def test_truth_follows_rules_not_labels(result):
    built, _ = result
    truth = {}
    for a in built["assets"]:
        truth.setdefault((next(iter(a["subcats"])), a["image_truth"]), 0)
        truth[(next(iter(a["subcats"])), a["image_truth"])] += 1
    assert truth[("bbc", "REAL")] == 30 and truth[("fakeddit", "REAL")] == 14
    assert truth[("fever_AI", "AI_GENERATED")] == 20 and truth[("fever_AI", "EXCLUDE")] == 2   # wrong size is excluded
    assert truth[("Fakeddit_photo_edit", "MANIPULATED")] == 12


def test_only_original_train_is_used(result):
    built, root = result
    assert len(built["assets"]) == 108          # 10 val/test images in the fixture are never loaded
    original = {r["image_asset_id"] for r in b.read_csv(root / "data/master.csv") if r["split"] == "train"}
    assert {a["image_id"] for a in built["assets"]} == original


def test_invariants_and_task_contents(result):
    built, root = result
    assert b.validate(built, root)["groups"] > 0
    assert {a["image_truth"] for a in built["tables"]["ai_detector"]} == {"REAL", "AI_GENERATED"}
    assert {a["image_truth"] for a in built["tables"]["manip_detector"]} == {"REAL", "MANIPULATED"}
    assert {a["image_truth"] for a in built["tables"]["joint_forensic"]} == set(b.TRUTHS)
    assert all(a["image_truth"] != "EXCLUDE" for t in built["tables"].values() for a in t)
    files, _ = b.task_files("joint_forensic", built["tables"]["joint_forensic"])
    assert {r["label"] for r in files["joint_forensic_all.csv"]} == {0, 1, 2}
    assert len(files["joint_forensic_all.csv"]) == len(built["tables"]["joint_forensic"])    # natural counts, no duplication


def test_unknown_subcategory_fails_loudly(tmp_path):
    root = make_dataset(tmp_path / "ds", [("mmfakebench", "brand_new_folder", (8, 8), "train", 3)])
    with pytest.raises(KeyError):
        b.build(root, verify_hashes=False, workers=1)


def test_rebuild_is_deterministic(tmp_path):
    first = b.build(make_dataset(tmp_path / "a", SPEC), False, 1)
    second = b.build(make_dataset(tmp_path / "b", SPEC), False, 1)
    for task, items in first["tables"].items():
        assert b.task_files(task, items)[0] == b.task_files(task, second["tables"][task])[0]
