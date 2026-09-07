"""PyTorch Dataset over the built CSV.

For each row it opens the image file and returns everything the model needs:

    resnet_pixels      image resized 224, ImageNet normalisation   (2.1)
    bert_input_ids     caption tokens for BERT-base-uncased          (2.2)
    clip_pixels        image resized 224, CLIP normalisation         (2.3)
    clip_input_ids     caption tokens for CLIP's own tokenizer       (2.3)
    label_binary       0 real / 1 fake        label_index 0..4        scenario 1..5

Only the ``split`` requested is loaded.  Nothing here looks at the labels
except to pass them through; the split assignment was fixed by the builder.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)


def read_split(csv_path: str | Path, split: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise ValueError(f"no rows with split={split!r} in {csv_path}")
    return rows


class FakeNewsDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, bert_tokenizer, clip_tokenizer,
                 max_len_bert: int = 64, max_len_clip: int = 77, train: bool = False,
                 image_root: str | Path | None = None):
        self.rows = read_split(csv_path, split)
        self.bert_tok, self.clip_tok = bert_tokenizer, clip_tokenizer
        self.max_len_bert, self.max_len_clip = max_len_bert, max_len_clip
        self.image_root = Path(image_root) if image_root else None
        aug = [transforms.RandomHorizontalFlip()] if train else []
        self.resnet_tf = transforms.Compose([transforms.Resize((224, 224)), *aug, transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.clip_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                           transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: dict) -> Path:
        p = Path(row["image_path"])
        if self.image_root and not p.is_absolute():
            p = self.image_root / p
        return p

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        with Image.open(self._image_path(row)) as im:
            im = im.convert("RGB")
            resnet_pixels = self.resnet_tf(im)
            clip_pixels = self.clip_tf(im)
        b = self.bert_tok(row["text"], truncation=True, max_length=self.max_len_bert,
                          padding="max_length", return_tensors="pt")
        c = self.clip_tok(row["text"], truncation=True, max_length=self.max_len_clip,
                          padding="max_length", return_tensors="pt")
        return {
            "resnet_pixels": resnet_pixels,
            "clip_pixels": clip_pixels,
            "bert_input_ids": b["input_ids"][0],
            "bert_attention_mask": b["attention_mask"][0],
            "clip_input_ids": c["input_ids"][0],
            "clip_attention_mask": c["attention_mask"][0],
            "label_binary": torch.tensor(int(row["label_binary"]), dtype=torch.float32),
            "label_index": torch.tensor(int(row["label_index"]), dtype=torch.long),
            "scenario": torch.tensor(int(row["scenario"]), dtype=torch.long),
            "sample_id": row["sample_id"],
        }


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "sample_id"}
    out["sample_id"] = [b["sample_id"] for b in batch]
    return out
