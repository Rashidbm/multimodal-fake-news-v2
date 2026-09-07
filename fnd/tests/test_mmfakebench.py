"""Unit tests for the MMFakeBench mapping rule.

Each test builds a tiny raw record by hand and checks which group it lands
in, or that it is rejected.  No files or images are needed.
"""
import json

import pytest

from fnd.data.mmfakebench import classify, load_mmfakebench, subcategory
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


@pytest.mark.parametrize("folder,fc,ts,im,expected", [
    ("bbc", "original", "VisualNews", "VisualNews", "genuine"),
    ("coco", "original", "MS-COCO", "MS-COCO", "genuine"),
    ("Newsclipings_person", "mismatch", "Newsclipings", "Newsclipings", "ooc"),
    ("rumor_match", "textual_veracity_distortion", "Fakenewsnet", "Repurposed Image", "fake_text_real_image"),
    ("chatgpt_match", "textual_veracity_distortion", "GPT-generated Rumor", "Repurposed Image", "fake_text_real_image"),
    ("fever_AI", "textual_veracity_distortion", "Fever", "AI-generated Image", "fake_text_fake_image"),
    ("llm_rewrite", "textual_veracity_distortion", "GPT-generated Rumor", "AI-generated Image", "fake_text_fake_image"),
    ("Fakeddit_photo_edit", "visual_veracity_distortion", "Fakeddit", "Fakeddit", "real_text_fake_image"),
    ("antifact_image_generation", "visual_veracity_distortion", "MS-COCO", "AI-generated Image", "real_text_fake_image"),
    ("antifact_image_generation", "visual_veracity_distortion", "", "", "real_text_fake_image"),
    ("DGM4_text_edit_senti", "mismatch", "DGM4", "DGM4", "fake_text_real_image"),
    ("coco_image_edit", "mismatch", "COCO-Counterfactuals", "COCO-Counterfactuals", "real_text_fake_image"),
    ("coco_text_edit", "mismatch", "COCO-Counterfactuals", "COCO-Counterfactuals", "fake_text_fake_image"),
])
def test_every_rule(folder, fc, ts, im, expected):
    group, _rule = classify(rec(folder, fc, ts, im))
    assert group == expected


@pytest.mark.parametrize("folder,fc,ts,im", [
    ("x", "something_new", "VisualNews", "VisualNews"),                       # unknown fake_cls
    ("Fakeddit_photo_edit", "visual_veracity_distortion", "Fever", "Fakeddit"),  # visual but text is a rumour
    ("bbc", "original", "Fever", "Fever"),                                    # original but rumour source
    ("fever_AI", "textual_veracity_distortion", "VisualNews", "AI-generated Image"),  # textual but real source
    ("fever_AI", "textual_veracity_distortion", "Fever", "Something Else"),   # unexpected image source
    ("weird", "mismatch", "COCO-Counterfactuals", "COCO-Counterfactuals"),    # COCO folder we don't know
    ("other", "visual_veracity_distortion", "", ""),                          # blank sources outside antifact
])
def test_unexpected_records_are_rejected(folder, fc, ts, im):
    with pytest.raises(MappingError):
        classify(rec(folder, fc, ts, im))


def test_folder_consistency_check(tmp_path):
    # Two records in the same folder that map to different groups -> loader must refuse.
    records = [
        rec("fever_AI", "textual_veracity_distortion", "Fever", "AI-generated Image"),
        rec("fever_AI", "textual_veracity_distortion", "Fever", "Repurposed Image"),
    ]
    (tmp_path / "MMFakeBench_val.json").write_text(json.dumps(records))
    with pytest.raises(MappingError, match="more than one group"):
        load_mmfakebench(tmp_path, splits=("val",))


def test_loader_reports_missing_images(tmp_path):
    records = [rec("bbc", "original", "VisualNews", "VisualNews")]
    (tmp_path / "MMFakeBench_val.json").write_text(json.dumps(records))
    samples, rep = load_mmfakebench(tmp_path, splits=("val",))
    assert len(samples) == 1 and samples[0].group == "genuine"
    assert rep.missing_images["val"] == 1
    with pytest.raises(FileNotFoundError):
        load_mmfakebench(tmp_path, splits=("val",), require_images=True)
