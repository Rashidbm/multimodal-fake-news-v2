"""CPU-only checks of the actual dataset -> epoch -> model input boundary.

No pretrained encoders, downloads, optimizer, or weight updates are used.
"""
import csv

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fnd.data.torch_dataset import FakeNewsDataset, collate
from fnd.train import run_epoch


class RecordingTokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, text, *, max_length, **kwargs):
        self.texts.append(text)
        ids = list(text.encode("utf-8"))[:max_length]
        mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids += [0] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


class InputSpy(torch.nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.calls = []

    def forward(self, resnet_pixels, bert_input_ids, bert_attention_mask,
                clip_pixels, clip_input_ids, clip_attention_mask):
        features = (resnet_pixels, bert_input_ids, bert_attention_mask,
                    clip_pixels, clip_input_ids, clip_attention_mask)
        assert all(x.device.type == "cpu" for x in features)
        self.calls.append(tuple(x.clone() for x in features))
        score = bert_input_ids.float().mean(1) / 100 + clip_pixels.mean((1, 2, 3))
        logits = score[:, None]
        if self.task == "scenario":
            logits = logits + torch.arange(5)[None, :]
        return {"logits": logits, "clip_similarity": score,
                "attention": torch.full((len(score), 3), 1 / 3)}


@pytest.mark.parametrize("task", ["binary", "scenario"])
def test_labels_metadata_and_path_names_do_not_enter_model(tmp_path, task):
    caption = "A person speaks at a podium."
    tokenizer = RecordingTokenizer()
    model = InputSpy(task)
    results = []
    for name, label, scenario in [("real", 0, 4), ("fake", 1, 1)]:
        image_path = tmp_path / name / f"known_{name}.png"
        image_path.parent.mkdir()
        Image.new("RGB", (32, 24), (80, 120, 160)).save(image_path)
        row = {"sample_id": name, "split": "val", "text": caption,
               "image_path": str(image_path), "label_binary": label,
               "label_index": scenario - 1, "scenario": scenario,
               "group": name, "source": name, "fake_cls": name,
               "gt_answers": name, "text_fake": label, "image_fake": label}
        csv_path = tmp_path / f"{name}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        dataset = FakeNewsDataset(csv_path, "val", tokenizer, tokenizer, train=False)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
        results.append(run_epoch(model, loader, torch.device("cpu"), task))

    assert tokenizer.texts == [caption] * 4  # No label or path appended to either tokenizer.
    assert len(model.calls) == 2
    assert all(torch.equal(a, b) for a, b in zip(*model.calls))
    assert results[0]["_rows"][0]["prob"] == results[1]["_rows"][0]["prob"]
    assert results[0]["_rows"][0]["pred"] == results[1]["_rows"][0]["pred"]
    assert results[0]["loss"] != results[1]["loss"]  # Labels still supervise the loss.
