"""Tests for balancing, de-duplication, leak-free splitting and verification.

A tiny fake dataset is generated on the fly: real PNG files (so hashing is
exercised) and hand-made Samples.  No external data needed.
"""
import csv
import shutil

import pytest

from fnd.data.build import assign_splits, select_balanced, write_outputs
from fnd.data.records import GROUPS, Sample
from fnd.data.verify import verify_csv

PIL = pytest.importorskip("PIL.Image")


def make_png(path, color):
    img = PIL.new("RGB", (32, 24), color)
    # a little structure so the difference hash is not all zeros
    for x in range(0, 32, 4):
        img.putpixel((x, 5), (255 - color[0], 0, 0))
    img.save(path)


def sample(i, group, text, image_path, source="mmfakebench"):
    return Sample(sample_id=f"{source}_{group}_{i}", source=source, source_split="val",
                  text=text, image_path=str(image_path), group=group, raw_label="x")


@pytest.fixture
def dataset(tmp_path):
    """5 groups x 12 unique records, plus planted duplicates and shared content."""
    samples = []
    k = 0
    for gi, g in enumerate(GROUPS):
        for i in range(12):
            p = tmp_path / f"{g}_{i}.png"
            make_png(p, (10 * gi + i * 3, 50 + i, 200 - i))
            samples.append(sample(i, g, f"caption {g} {i}", p))
            k += 1
    # planted: exact duplicate pair in 'genuine' (same caption, same bytes, new file name)
    shutil.copy(tmp_path / "genuine_0.png", tmp_path / "genuine_copy.png")
    samples.append(sample(99, "genuine", "Caption genuine 0!", tmp_path / "genuine_copy.png"))
    # planted: same IMAGE reused by an ooc record with a different caption
    samples.append(sample(98, "ooc", "totally different caption", tmp_path / "genuine_1.png"))
    # planted: same CAPTION reused by a fake_text_real_image record with a different image
    p = tmp_path / "shared_caption.png"
    make_png(p, (123, 45, 67))
    samples.append(sample(97, "fake_text_real_image", "caption genuine 2", p))
    return samples


def test_balanced(dataset):
    selected, rep = select_balanced(dataset, seed=1, image_mode="required", balance="equal_scenarios")
    assert rep.target == 12                      # smallest group has 12 unique records
    assert all(rep.selected[g] == 12 for g in GROUPS)
    assert rep.images_hashed == 60 and rep.images_unverified == 0
    assert len(selected) == 60


def test_duplicate_pair_is_caught_by_content(tmp_path):
    """genuine has 5 unique records + 1 copy (same caption modulo punctuation,
    same bytes under a new file name); the other groups have 6.  The goal is
    therefore 6, so all six genuine candidates are walked, the copy is
    skipped, genuine ends with 5, and every group is truncated to 5."""
    samples = []
    for gi, g in enumerate(GROUPS):
        n = 5 if g == "genuine" else 6
        for i in range(n):
            p = tmp_path / f"{g}_{i}.png"
            make_png(p, (10 * gi + i * 3, 50 + i, 200 - i))
            samples.append(sample(i, g, f"caption {g} {i}", p))
    shutil.copy(tmp_path / "genuine_0.png", tmp_path / "genuine_copy.png")
    samples.append(sample(99, "genuine", "Caption, GENUINE 0!", tmp_path / "genuine_copy.png"))

    selected, rep = select_balanced(samples, seed=3, image_mode="required", balance="equal_scenarios")
    assert rep.available["genuine"] == 6
    assert rep.skipped_duplicate["genuine"] == 1
    assert rep.target == 5 and all(rep.selected[g] == 5 for g in GROUPS)
    kept = {s.sample_id for s in selected if s.group == "genuine"}
    assert len(kept) == 5 and len({"mmfakebench_genuine_0", "mmfakebench_genuine_99"} & kept) == 1
    assert rep.duplicate_examples and rep.duplicate_examples[0][1] in {"mmfakebench_genuine_0", "mmfakebench_genuine_99"}


def test_target_too_high_is_an_error(dataset):
    with pytest.raises(ValueError, match="only 12 available"):
        select_balanced(dataset, target=13, seed=1, balance="equal_scenarios")


def test_shared_content_lands_in_one_split(dataset):
    selected, _ = select_balanced(dataset, seed=1, balance="equal_scenarios")
    split_of, rep = assign_splits(selected, seed=1)
    by_id = {s.sample_id: s for s in selected}
    # same image bytes -> same split
    if "mmfakebench_ooc_98" in split_of:
        assert split_of["mmfakebench_ooc_98"] == split_of["mmfakebench_genuine_1"]
    # same caption -> same split
    if "mmfakebench_fake_text_real_image_97" in split_of:
        assert split_of["mmfakebench_fake_text_real_image_97"] == split_of["mmfakebench_genuine_2"]
    assert rep.clusters_multi >= 1
    for g in GROUPS:
        assert sum(rep.counts[sp][g] for sp in rep.counts) == 12


def test_missing_image_required_vs_optional(tmp_path, dataset):
    dataset.append(sample(50, "ooc", "ghost", tmp_path / "does_not_exist.png"))
    _, rep = select_balanced(dataset, seed=1, image_mode="required", balance="equal_scenarios")
    assert rep.skipped_missing_image["ooc"] == 1
    _, rep2 = select_balanced(dataset, seed=1, image_mode="optional", balance="equal_scenarios")
    assert rep2.skipped_missing_image["ooc"] == 0 and rep2.images_unverified >= 1


def test_outputs_pass_verify_and_tampering_fails(tmp_path, dataset):
    selected, sel_rep = select_balanced(dataset, seed=1, balance="equal_scenarios")
    split_of, split_rep = assign_splits(selected, seed=1)
    csv_path, manifest = write_outputs(selected, split_of, tmp_path / "out", "t", sel_rep, split_rep, {})
    assert verify_csv(csv_path, check_files=True, balance="equal_scenarios") == []
    assert manifest.is_file()

    # tamper: move one record of a shared-image cluster to another split -> leak must be detected
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    victim = next(r for r in rows if r["sample_id"] == "mmfakebench_genuine_1")
    victim["split"] = "test" if victim["split"] != "test" else "train"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    fails = verify_csv(csv_path, balance="equal_scenarios")
    assert any("more than one split" in f for f in fails)


def test_deterministic(dataset):
    a, _ = select_balanced(list(dataset), seed=7, balance="equal_scenarios")
    b, _ = select_balanced(list(dataset), seed=7, balance="equal_scenarios")
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    sa, _ = assign_splits(a, seed=7)
    sb, _ = assign_splits(b, seed=7)
    assert sa == sb


def test_real_fake_balance(tmp_path):
    """genuine gets 4 x N_fake so real == fake in total; N_fake limited by genuine // 4."""
    samples = []
    for gi, g in enumerate(GROUPS):
        n = 14 if g == "genuine" else 6         # genuine // 4 = 3 < 6 -> N_fake = 3, genuine = 12
        for i in range(n):
            p = tmp_path / f"{g}_{i}.png"
            make_png(p, (10 * gi + i * 3, 50 + i, 200 - i))
            samples.append(sample(i, g, f"caption {g} {i}", p))
    selected, rep = select_balanced(samples, seed=1, balance="real_fake")
    assert rep.target == 3
    assert rep.selected["genuine"] == 12 and all(rep.selected[g] == 3 for g in GROUPS if g != "genuine")
    real = sum(1 for s in selected if s.group == "genuine")
    fake = sum(1 for s in selected if s.group != "genuine")
    assert real == fake == 12
    split_of, split_rep = assign_splits(selected, seed=1)
    csv_path, _ = write_outputs(selected, split_of, tmp_path / "out", "rf", rep, split_rep, {})
    assert verify_csv(csv_path, check_files=True, balance="real_fake") == []
    assert verify_csv(csv_path, balance="equal_scenarios") != []   # the other mode must reject it
