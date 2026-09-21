"""Probe and comparison tests on synthetic features: no encoders, no downloads.

A comparison script is only worth its output if it cannot quietly compare two
different things, so these tests spend most of their effort on the guards:
the label derivation, the id join, the same-rows check, and whether the paired
bootstrap calls a real gap real and a coin flip a coin flip.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")

from fnd.compare_match import (  # noqa: E402
    check_same_rows,
    main as compare_main,
    paired_bootstrap_auc,
    parse_features,
)
from fnd.data.records import GROUPS  # noqa: E402
from fnd.probe_match import group_of, load_pairs, main as probe_main, zero_shot_cosine  # noqa: E402


def _make_csv(tmp_path, n_per_group=40, name="rows.csv"):
    rows = []
    for gi, group in enumerate(GROUPS):
        for k in range(n_per_group):
            rows.append({
                "sample_id": f"{group}_{k:03d}",
                "group": group,
                "label_index": gi,
                "scenario": gi + 1,
                "label_binary": 0 if group == "genuine" else 1,
                "subcategory": f"src_{k % 2}",
                "split": "train" if k % 5 < 3 else ("val" if k % 5 == 3 else "test"),
            })
    path = tmp_path / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path, rows


def _make_features(tmp_path, rows, name, dim=24, informative=True, strength=6.0, seed=0,
                   with_similarity=False):
    """Features where, when informative, the ooc rows sit in their own corner -
    which is exactly the signal a working match/mismatch encoder would give."""
    torch.manual_seed(seed)
    x = torch.randn(len(rows), dim)
    if informative:
        for i, r in enumerate(rows):
            if r["group"] == "ooc":
                x[i, 0] += strength
    payload = {"ids": [r["sample_id"] for r in rows], "features": x,
               "splits": [r["split"] for r in rows],
               "meta": {"backbone": name, "model_name": f"synthetic-{name}",
                        "clip_feature_mode": "interaction", "embed_dim": dim}}
    if with_similarity:
        # a mismatch should sit at a LOW cosine
        payload["similarity"] = torch.tensor(
            [-1.0 if r["group"] == "ooc" else 1.0 for r in rows]) + torch.randn(len(rows)) * 0.05
    path = tmp_path / f"v_match_{name}.pt"
    torch.save(payload, path)
    return path


# ---------------------------------------------------------------------------
# label derivation
# ---------------------------------------------------------------------------

def test_group_of_falls_back_through_the_columns():
    assert group_of({"group": "ooc"}) == "ooc"
    assert group_of({"label_index": "3"}) == GROUPS[3]
    assert group_of({"scenario": "1"}) == GROUPS[0]
    with pytest.raises(KeyError):
        group_of({"text": "no label anywhere"})


def test_mismatch_clean_excludes_the_three_tampered_scenarios(tmp_path):
    csv_path, rows = _make_csv(tmp_path, n_per_group=10)
    f = _make_features(tmp_path, rows, "clip")
    d = load_pairs(f, csv_path)

    clean, allv = d["targets"]["mismatch_clean"], d["targets"]["mismatch_all"]
    for g, c, a in zip(d["group"], clean.tolist(), allv.tolist()):
        if g == "ooc":
            assert (c, a) == (1.0, 1.0)
        elif g == "genuine":
            assert (c, a) == (0.0, 0.0)
        else:
            assert c == -1.0, "a tampered pair has no defensible match/mismatch label"
            assert a == 0.0, "but mismatch_all still has to place it somewhere"
    assert int((clean >= 0).sum()) == 20


def test_load_pairs_matches_labels_by_id_not_row_order(tmp_path):
    csv_path, rows = _make_csv(tmp_path, n_per_group=8)
    f = _make_features(tmp_path, rows, "clip")
    payload = torch.load(f, weights_only=False)
    perm = torch.randperm(len(payload["ids"]))
    payload["ids"] = [payload["ids"][i] for i in perm.tolist()]
    payload["features"] = payload["features"][perm]
    torch.save(payload, f)

    d = load_pairs(f, csv_path)
    for sid, g in zip(d["ids"], d["group"]):
        assert sid.startswith(g)


def test_load_pairs_rejects_features_from_another_build(tmp_path):
    csv_path, rows = _make_csv(tmp_path, n_per_group=4)
    f = _make_features(tmp_path, rows, "clip")
    payload = torch.load(f, weights_only=False)
    payload["ids"] = [s + "_stale" for s in payload["ids"]]
    torch.save(payload, f)
    with pytest.raises(KeyError, match="not in the CSV"):
        load_pairs(f, csv_path)


def test_zero_shot_cosine_scores_the_negated_similarity():
    sim = torch.tensor([-1.0, -0.9, 0.9, 1.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])            # the low-cosine rows are the mismatches
    mask = torch.ones(4, dtype=torch.bool)
    assert zero_shot_cosine(sim, y, mask)["auc"] == pytest.approx(1.0)
    # one class only: no AUC exists, and None is the honest answer
    assert zero_shot_cosine(sim, torch.ones(4), mask) is None
    assert zero_shot_cosine(None, y, mask) is None


# ---------------------------------------------------------------------------
# the single-backbone probe
# ---------------------------------------------------------------------------

def test_probe_finds_a_planted_mismatch_signal(tmp_path):
    csv_path, rows = _make_csv(tmp_path)
    f = _make_features(tmp_path, rows, "clip", with_similarity=True)
    out = tmp_path / "probe"
    assert probe_main(["--features", str(f), "--csv", str(csv_path), "--out", str(out),
                       "--epochs", "30", "--device", "cpu"]) == 0

    m = json.loads((out / "metrics.json").read_text())
    assert m["mismatch_clean"]["auc"] > 0.9
    assert m["mismatch_clean"]["zero_shot_cosine"]["auc"] > 0.9
    assert m["mismatch_all"]["auc"] > 0.9
    # the stream must also report where its score fires, not only how often it is right
    means = m["mean_mismatch_probability_per_group"]
    assert means["ooc"]["mean_probability"] > means["genuine"]["mean_probability"]
    assert (out / "predictions.csv").exists() and (out / "report.txt").exists()

    with open(out / "predictions.csv", newline="", encoding="utf-8") as fh:
        pred_rows = list(csv.DictReader(fh))
    assert {"group", "cosine", "prob_mismatch_all"} <= set(pred_rows[0])


def test_probe_reports_chance_on_noise(tmp_path):
    """Noise must look like noise. A probe that scores well here would mean the
    pipeline is leaking labels, and every later number would be worthless."""
    csv_path, rows = _make_csv(tmp_path)
    f = _make_features(tmp_path, rows, "clip", informative=False)
    out = tmp_path / "probe_noise"
    assert probe_main(["--features", str(f), "--csv", str(csv_path), "--out", str(out),
                       "--epochs", "20", "--device", "cpu"]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert 0.3 < m["mismatch_clean"]["auc"] < 0.7


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def test_parse_features_needs_two_named_paths():
    assert parse_features(["clip=a.pt", "qwenvl=b.pt"]) == {"clip": "a.pt", "qwenvl": "b.pt"}
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_features(["a.pt", "b.pt"])
    with pytest.raises(ValueError, match="at least twice"):
        parse_features(["clip=a.pt"])
    with pytest.raises(ValueError, match="twice"):
        parse_features(["clip=a.pt", "clip=b.pt"])


def test_comparison_refuses_two_different_row_sets():
    a = {"ids": ["x", "y", "z"], "split": ["train", "val", "test"]}
    b = {"ids": ["x", "y"], "split": ["train", "val"]}
    with pytest.raises(ValueError, match="different rows"):
        check_same_rows({"clip": a, "qwenvl": b})


def test_comparison_refuses_disagreeing_splits():
    a = {"ids": ["x", "y"], "split": ["train", "test"]}
    b = {"ids": ["x", "y"], "split": ["train", "val"]}
    with pytest.raises(ValueError, match="disagree on the split"):
        check_same_rows({"clip": a, "qwenvl": b})


def test_paired_bootstrap_calls_a_tie_a_tie():
    y = [1, 0] * 40
    p = [0.9 if t else 0.1 for t in y]
    bs = paired_bootstrap_auc(y, p, list(p), n_boot=300)
    assert bs["observed_difference"] == pytest.approx(0.0)
    assert not bs["significant"], "identical detectors cannot have a significant gap"


def test_paired_bootstrap_detects_a_real_gap():
    torch.manual_seed(0)
    y = [1] * 60 + [0] * 60
    good = [0.9 if t else 0.1 for t in y]
    coin = torch.rand(120).tolist()
    bs = paired_bootstrap_auc(y, good, coin, n_boot=400)
    assert bs["observed_difference"] > 0.2
    assert bs["significant"] and bs["ci95_low"] > 0
    assert bs["p_first_better"] > 0.95


def test_comparison_picks_the_backbone_that_can_see_the_signal(tmp_path):
    """The whole point of the experiment, in miniature: one encoder's features
    carry the mismatch and the other's do not, and the script has to say so
    rather than average them into a shrug."""
    csv_path, rows = _make_csv(tmp_path)
    good = _make_features(tmp_path, rows, "clip", with_similarity=True)
    blind = _make_features(tmp_path, rows, "qwenvl", informative=False, seed=1)
    out = tmp_path / "compare"

    assert compare_main(["--csv", str(csv_path), "--features", f"clip={good}",
                         "--features", f"qwenvl={blind}", "--out", str(out),
                         "--seeds", "2", "--epochs", "25", "--bootstrap", "200",
                         "--device", "cpu"]) == 0

    res = json.loads((out / "comparison.json").read_text())
    clean = res["targets"]["mismatch_clean"]
    assert clean["highest_mean_auc"] == "clip"
    assert clean["difference_is_significant"]
    assert clean["per_backbone"]["clip"]["auc_mean"] > 0.9
    assert clean["per_backbone"]["qwenvl"]["auc_mean"] < 0.8
    assert len(clean["per_backbone"]["clip"]["auc_per_seed"]) == 2
    # the VLM side has no cosine to report, and must not invent one
    assert clean["per_backbone"]["qwenvl"]["zero_shot_cosine"] is None
    assert clean["per_backbone"]["clip"]["zero_shot_cosine"]["auc"] > 0.9

    md = (out / "comparison.md").read_text()
    assert "| clip |" in md and "| qwenvl |" in md and "95% CI" in md
    assert res["rows"]["total"] == len(rows)


def test_comparison_stops_when_one_side_was_extracted_with_a_limit(tmp_path):
    csv_path, rows = _make_csv(tmp_path, n_per_group=10)
    full = _make_features(tmp_path, rows, "clip")
    part = _make_features(tmp_path, rows[: len(rows) // 2], "qwenvl")
    with pytest.raises(ValueError, match="different rows"):
        compare_main(["--csv", str(csv_path), "--features", f"clip={full}",
                      "--features", f"qwenvl={part}", "--out", str(tmp_path / "c"),
                      "--seeds", "1", "--epochs", "2", "--device", "cpu"])


def test_comparison_rejects_an_unknown_target(tmp_path):
    csv_path, rows = _make_csv(tmp_path, n_per_group=6)
    a = _make_features(tmp_path, rows, "clip")
    b = _make_features(tmp_path, rows, "qwenvl", seed=2)
    with pytest.raises(ValueError, match="unknown target"):
        compare_main(["--csv", str(csv_path), "--features", f"clip={a}",
                      "--features", f"qwenvl={b}", "--out", str(tmp_path / "c"),
                      "--targets", "vibes", "--device", "cpu"])
