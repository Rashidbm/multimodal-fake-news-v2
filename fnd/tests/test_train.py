"""End-to-end smoke test of fnd.train on tiny synthetic data, random weights.

Proves the loop runs: dataset -> model -> loss -> optimizer -> val selection
-> best.pt -> test metrics + predictions, including per-scenario accuracy and
the real/fake oversampling.  No downloads: dummy tokenizers, random weights.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL.Image")

from fnd.data.records import GROUPS, LABEL_INDEX, SCENARIO_NUMBER, binary_label


class DummyTokenizer:
    """Same call signature as a HF tokenizer; ids come from a hash of the words."""

    def __call__(self, text, truncation=True, max_length=16, padding="max_length", return_tensors="pt"):
        ids = [1] + [2 + (hash(w) % 900) for w in text.split()][: max_length - 2] + [3]
        mask = [1] * len(ids)
        ids += [0] * (max_length - len(ids))
        mask += [0] * (max_length - len(mask))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


def make_csv(tmp_path, per_group=4):
    rows = []
    k = 0
    for g in GROUPS:
        for i in range(per_group):
            p = tmp_path / f"{g}_{i}.png"
            PIL.new("RGB", (40, 30), (k * 7 % 255, 90, 160)).save(p)
            split = "train" if i < per_group - 2 else ("val" if i == per_group - 2 else "test")
            rows.append({"sample_id": f"s{k}", "scenario": SCENARIO_NUMBER[g], "group": g,
                         "label_index": LABEL_INDEX[g], "label_binary": binary_label(g),
                         "text": f"caption number {k} about {g}", "image_path": str(p), "split": split})
            k += 1
    csv_path = tmp_path / "tiny.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return csv_path


def test_train_loop_end_to_end(tmp_path, monkeypatch):
    import fnd.train as train
    monkeypatch.setattr(train, "load_tokenizers", lambda cfg: (DummyTokenizer(), DummyTokenizer()))
    csv_path = make_csv(tmp_path)
    out = tmp_path / "run"
    rc = train.main(["--csv", str(csv_path), "--out", str(out), "--epochs", "2", "--batch-size", "4",
                     "--num-workers", "0", "--device", "cpu", "--random-init", "--patience", "5"])
    assert rc == 0
    assert (out / "best.pt").is_file()
    hist = json.load(open(out / "history.json"))
    assert len(hist) == 2 and "f1" in hist[0]["val"]
    metrics = json.load(open(out / "test_metrics.json"))
    assert set(metrics["per_scenario"]) == {"1", "2", "3", "4", "5"}
    preds = list(csv.DictReader(open(out / "test_predictions.csv")))
    assert len(preds) == 5 and {"prob", "pred", "clip_similarity", "attn_text"} <= set(preds[0])


def test_scenario_task_runs(tmp_path, monkeypatch):
    import fnd.train as train
    monkeypatch.setattr(train, "load_tokenizers", lambda cfg: (DummyTokenizer(), DummyTokenizer()))
    csv_path = make_csv(tmp_path)
    out = tmp_path / "run5"
    rc = train.main(["--csv", str(csv_path), "--out", str(out), "--epochs", "1", "--batch-size", "4",
                     "--num-workers", "0", "--device", "cpu", "--random-init", "--task", "scenario"])
    assert rc == 0
    assert "f1_macro" in json.load(open(out / "test_metrics.json"))
