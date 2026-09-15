"""Probe tests on synthetic features: no Qwen2, no GPU, no downloads.

The probe is what turns v_textfor into numbers a supervisor can read, so the
things worth pinning down are that it joins features to labels by id rather
than row order, that it refuses a mismatched pair instead of scoring
nonsense, and that it actually learns when the signal is there.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")

from fnd.data.records import GROUP_FLAGS, GROUPS
from fnd.metrics import confusion_matrix, format_confusion
from fnd.probe_textfor import ProbeHead, baselines, load_aligned, main as probe_main


def _make_dataset(tmp_path, n_per_class=60, dim=32, separable=True, shuffle_features=False):
    """A CSV plus a matching feature file, with one cluster per scenario."""
    torch.manual_seed(0)
    rows, feats, ids = [], [], []
    for gi, group in enumerate(GROUPS):
        centre = torch.zeros(dim)
        centre[gi] = 6.0 if separable else 0.0
        for k in range(n_per_class):
            sid = f"s_{gi}_{k:03d}"
            ids.append(sid)
            feats.append(centre + torch.randn(dim) * 0.5)
            text_fake, image_fake, ooc = GROUP_FLAGS[group]
            rows.append({
                "sample_id": sid,
                "subcategory": f"src_{gi}_{'a' if k % 2 else 'b'}",
                "scenario": gi + 1,
                "label_index": gi,
                "label_binary": 0 if group == "genuine" else 1,
                "text_fake": text_fake,
                "image_fake": image_fake,
                "ooc": ooc,
                "split": "train" if k % 5 < 3 else ("val" if k % 5 == 3 else "test"),
            })

    csv_path = tmp_path / "rows.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    x = torch.stack(feats)
    if shuffle_features:
        # feature rows in a different order than the CSV: the join must still
        # line each vector up with its own label
        perm = torch.randperm(len(ids))
        x, ids = x[perm], [ids[i] for i in perm.tolist()]

    fpath = tmp_path / "v_textfor.pt"
    torch.save({"sample_ids": ids, "features": x,
                "splits": ["" for _ in ids], "meta": {"model_name": "synthetic"}}, fpath)
    return fpath, csv_path


def test_head_shapes():
    assert ProbeHead(4096, 1)(torch.randn(4, 4096)).shape == (4, 1)
    assert ProbeHead(4096, 5)(torch.randn(4, 4096)).shape == (4, 5)


def test_load_aligned_matches_labels_by_id(tmp_path):
    """Feature rows shuffled relative to the CSV must still get their own
    labels - the failure this guards against is silent, not an exception."""
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=4, dim=8, shuffle_features=True)
    d = load_aligned(fpath, csv_path)
    for sid, yi in zip(d["ids"], d["y_idx"].tolist()):
        assert sid.startswith(f"s_{yi}_")


def test_load_aligned_rejects_a_mismatched_pair(tmp_path):
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=3, dim=8)
    payload = torch.load(fpath, weights_only=False)
    payload["sample_ids"] = [s + "_stale" for s in payload["sample_ids"]]
    torch.save(payload, fpath)
    with pytest.raises(KeyError, match="not in the CSV"):
        load_aligned(fpath, csv_path)


def test_text_fake_and_label_binary_differ(tmp_path):
    """The two binary targets are not the same question: an out-of-context
    pair is label_binary=1 with genuinely human text_fake=0.  Scoring this
    stream against label_binary alone would ask it to call real writing fake."""
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=5, dim=8)
    d = load_aligned(fpath, csv_path)
    y_bin, y_txt = d["targets"]["binary"], d["targets"]["text_fake"]
    pairs = {(sid.split("_")[1], int(b), int(t))
             for sid, b, t in zip(d["ids"], y_bin.tolist(), y_txt.tolist())}
    ooc = GROUPS.index("ooc")
    assert (str(ooc), 1, 0) in pairs                    # fake post, human caption
    rtfi = GROUPS.index("real_text_fake_image")
    assert (str(rtfi), 1, 0) in pairs                   # fake post, human caption
    ftri = GROUPS.index("fake_text_real_image")
    assert (str(ftri), 1, 1) in pairs                   # fake post, machine caption
    assert not y_bin.equal(y_txt)


def test_label_columns_are_optional(tmp_path):
    """The delivered dataset carries only text_asset_id, text, split, text_fake.

    The probe must run on that rather than demanding label_binary, scenario
    and label_index, which only the repo's own build writes.
    """
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=5, dim=8)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    slim = tmp_path / "slim.csv"
    with open(slim, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text_asset_id", "split", "text_fake"])
        w.writeheader()
        for r in rows:
            w.writerow({"text_asset_id": r["sample_id"], "split": r["split"],
                        "text_fake": r["text_fake"]})

    payload = torch.load(fpath, weights_only=False)
    payload["ids"] = payload.pop("sample_ids")
    torch.save(payload, fpath)

    d = load_aligned(fpath, slim)
    assert d["id_column"] == "text_asset_id"
    assert set(d["targets"]) == {"text_fake"}
    assert d["y_idx"] is None and d["scenario"] is None


def test_where_restricts_to_one_slice(tmp_path):
    """--where is the domain-matched control: same topic, different authorship."""
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=5, dim=8)
    d_all = load_aligned(fpath, csv_path)
    d_one = load_aligned(fpath, csv_path, where="scenario=1")
    assert 0 < len(d_one["ids"]) < len(d_all["ids"])
    assert d_one["x"].shape[0] == len(d_one["ids"])      # features filtered with the ids
    assert all(s == 1 for s in d_one["scenario"])
    with pytest.raises(ValueError, match="matched no rows"):
        load_aligned(fpath, csv_path, where="scenario=99")


def test_probe_reports_both_binary_tasks(tmp_path):
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=40, dim=16, separable=True)
    out = tmp_path / "probe_two"
    assert probe_main(["--features", str(fpath), "--csv", str(csv_path), "--out", str(out),
                       "--epochs", "20", "--device", "cpu"]) == 0
    r = json.loads((out / "metrics.json").read_text())
    assert "text_fake" in r and "binary" in r
    assert r["text_fake"]["accuracy"] > 0.9             # its own question, learnable
    assert "per_scenario_text_fake" in r


def test_baselines():
    b = baselines([0, 0, 0, 1], [0, 0, 1, 1], 2, seed=0)
    assert b["majority_class"] == 0 and b["majority_accuracy"] == 0.5
    assert baselines([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 5, seed=0)["random_expected"] == 0.2


def test_confusion_matrix_and_format():
    m = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert m == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert sum(sum(r) for r in m) == 4
    assert "genuine" in format_confusion(m, ["genuine", "ooc", "fake"])


def test_probe_learns_separable_features(tmp_path):
    """Clusters this clean must score far above the 0.2 random baseline; if
    they do not, the training loop is broken rather than the features."""
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=60, dim=32, separable=True)
    out = tmp_path / "probe"
    assert probe_main(["--features", str(fpath), "--csv", str(csv_path), "--out", str(out),
                       "--epochs", "40", "--device", "cpu"]) == 0

    r = json.loads((out / "metrics.json").read_text())
    assert r["multiclass"]["accuracy"] > 0.9
    assert r["multiclass"]["accuracy"] > r["multiclass"]["baselines"]["majority_accuracy"]
    assert r["binary"]["accuracy"] > 0.9
    assert r["binary"]["auc"] > 0.9

    cm = r["multiclass"]["confusion_matrix"]
    assert sum(sum(row) for row in cm) == r["multiclass"]["n"]
    assert (out / "predictions.csv").exists() and (out / "report.txt").exists()


def test_probe_reports_chance_on_noise(tmp_path):
    """With no signal in the features the probe must land near chance, not
    invent accuracy - a guard against label leakage through the split."""
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=40, dim=16, separable=False)
    out = tmp_path / "probe_noise"
    assert probe_main(["--features", str(fpath), "--csv", str(csv_path), "--out", str(out),
                       "--epochs", "20", "--device", "cpu"]) == 0
    r = json.loads((out / "metrics.json").read_text())
    assert r["multiclass"]["accuracy"] < 0.5          # chance is 0.2


def test_class_weights_match_inverse_frequency():
    """text_fake is 2 groups against 3 (40/60) and label_binary 1 against 4
    (20/80). Weighting the loss is how that is handled - not by deleting
    rows, which would break the row set the other two streams join against."""
    from fnd.probe_textfor import class_weights

    y_txt = torch.tensor([1.0] * 40 + [0.0] * 60)      # text_fake proportions
    assert class_weights(y_txt, 1).item() == pytest.approx(1.5)

    y_bin = torch.tensor([1.0] * 80 + [0.0] * 20)      # label_binary proportions
    assert class_weights(y_bin, 1).item() == pytest.approx(0.25)

    w = class_weights(torch.tensor([0, 0, 1, 2, 2, 2]), 3)
    assert w.shape == (3,) and w[1] > w[2]             # rarer class weighs more

    assert class_weights(torch.tensor([1.0, 1.0]), 1) is None   # one class only


def test_weighted_and_unweighted_both_run(tmp_path):
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=30, dim=16, separable=True)
    for flag in ([], ["--no-class-weight"]):
        out = tmp_path / f"probe{len(flag)}"
        assert probe_main(["--features", str(fpath), "--csv", str(csv_path),
                           "--out", str(out), "--epochs", "15", "--device", "cpu"] + flag) == 0
        assert json.loads((out / "metrics.json").read_text())["text_fake"]["accuracy"] > 0.9


def test_per_subcategory_breakdown(tmp_path):
    """text_fake mixes AI-generated text, human rumours and word edits, and
    the method only detects the first. The breakdown by source sub-category
    is what separates 'the stream scores X' from 'the stream detects Y'."""
    fpath, csv_path = _make_dataset(tmp_path, n_per_class=40, dim=16, separable=True)
    out = tmp_path / "probe_sub"
    assert probe_main(["--features", str(fpath), "--csv", str(csv_path), "--out", str(out),
                       "--epochs", "15", "--device", "cpu"]) == 0

    r = json.loads((out / "metrics.json").read_text())
    per_sub = r["per_subcategory_text_fake"]
    assert len(per_sub) > 1
    assert sum(d["n"] for d in per_sub.values()) == r["text_fake"]["n"]
    assert all(0.0 <= d["accuracy"] <= 1.0 for d in per_sub.values())
