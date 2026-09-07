"""Unit tests for the MMFakeBench mapping rule.

Each test builds a tiny raw record by hand and checks which group it lands
in, or that it is rejected.  No files or images are needed.
"""
import json

import pytest

from fnd.data.mmfakebench import FOLDER_TO_GROUP, classify, load_mmfakebench, subcategory
from fnd.data.records import MappingError


def rec(folder, fake_cls, text_source, image_source, split="val"):
    return {
        "text": "some caption",
        "image_path": f"/x/{folder}_{split}_50/img_0.png",
        "text_source": text_source,
        "image_source": image_source,
        "gt_answers": "True" if fake_cls == "original" else "Fake",
        "fake_cls": fake_cls,
    }


def test_subcategory_strips_split_suffix():
    assert subcategory("/fake/fever_AI_val_100/fever_dalle_val_1.png") == "fever_AI"
    assert subcategory("/real/bbc_test_500/bbc_test_0.png") == "bbc"


@pytest.mark.parametrize("folder,expected", sorted(FOLDER_TO_GROUP.items()))
def test_every_folder_maps(folder, expected):
    group, sub = classify(rec(folder, "x", "", ""))
    assert group == expected and sub == folder


def test_unknown_folder_is_rejected():
    with pytest.raises(MappingError, match="not in FOLDER_TO_GROUP"):
        classify(rec("some_new_folder", "mismatch", "", ""))


def test_table_covers_five_groups():
    from fnd.data.records import GROUPS
    assert set(FOLDER_TO_GROUP.values()) == set(GROUPS)


def test_paper_totals_check(tmp_path):
    # 1 val + 1 test record only -> totals differ from the paper -> loader refuses.
    (tmp_path / "MMFakeBench_val.json").write_text(json.dumps([rec("bbc", "original", "VisualNews", "VisualNews")]))
    (tmp_path / "MMFakeBench_test.json").write_text(json.dumps([rec("fever_AI", "textual_veracity_distortion", "Fever", "AI-generated Image", "test")]))
    with pytest.raises(MappingError, match="differ from the paper"):
        load_mmfakebench(tmp_path)


def test_loader_reports_missing_images(tmp_path):
    records = [rec("bbc", "original", "VisualNews", "VisualNews")]
    (tmp_path / "MMFakeBench_val.json").write_text(json.dumps(records))
    samples, rep = load_mmfakebench(tmp_path, splits=("val",))
    assert len(samples) == 1 and samples[0].group == "genuine"
    assert rep.missing_images["val"] == 1
    with pytest.raises(FileNotFoundError):
        load_mmfakebench(tmp_path, splits=("val",), require_images=True)
