"""Train FND-CLIP on the built CSV and report on the test split.

    python -m fnd.train --csv data/processed/balanced_5group.csv --out outputs/fnd_clip_v1

What happens:
  1. train split -> optimise; val split -> pick the best epoch (binary F1)
  2. the best checkpoint is evaluated once on the test split
  3. outputs/<run>/  best.pt, history.json, test_metrics.json, test_predictions.csv

test_metrics.json contains accuracy / precision / recall / F1 / AUC overall
AND accuracy per scenario (1..5), which tells us which kind of fake the
model misses.  test_predictions.csv has one row per test pair with the
probability, the prediction, the CLIP similarity and the three attention
weights, for error analysis.

--task binary     one sigmoid output, real vs fake (the spec)      [default]
--task scenario   five outputs, one per scenario (softmax)

Real/fake balance: the CSV has five equal scenarios, so real (genuine) is 1:4
against fake.  The PDF allows oversampling MMFakeBench, so by default the
train loader draws genuine pairs 4x as often (WeightedRandomSampler), making
each epoch half real / half fake.  --no-oversample turns this off.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data.torch_dataset import FakeNewsDataset, collate
from .metrics import binary_metrics, multiclass_metrics, per_scenario_accuracy
from .models.fnd_clip import FNDCLIP, FNDCLIPConfig


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_epoch(model, loader, device, task, optimizer=None, log_every=50) -> dict:
    """One pass over ``loader``.  With an optimizer -> training, else evaluation.
    Returns metrics and, for evaluation, the per-sample predictions."""
    training = optimizer is not None
    model.train(training)
    total_loss, n_seen, t0 = 0.0, 0, time.time()
    ids, scen, y_bin, y_idx, probs, preds, sims, alphas = [], [], [], [], [], [], [], []

    for step, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        with torch.set_grad_enabled(training):
            out = model(batch["resnet_pixels"], batch["bert_input_ids"], batch["bert_attention_mask"],
                        batch["clip_pixels"], batch["clip_input_ids"], batch["clip_attention_mask"])
            logits = out["logits"]
            if task == "binary":
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), batch["label_binary"])
            else:
                loss = F.cross_entropy(logits, batch["label_index"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = logits.shape[0]
        total_loss += loss.item() * bs
        n_seen += bs
        if training and step % log_every == 0:
            print(f"    step {step}/{len(loader)}  loss {total_loss / n_seen:.4f}  "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)

        if task == "binary":
            p = torch.sigmoid(logits.squeeze(-1))
            probs += p.tolist()
            preds += (p >= 0.5).long().tolist()
        else:
            p = torch.softmax(logits, dim=-1)
            probs += p.tolist()
            preds += p.argmax(dim=-1).tolist()
        ids += batch["sample_id"]
        scen += batch["scenario"].tolist()
        y_bin += batch["label_binary"].long().tolist()
        y_idx += batch["label_index"].tolist()
        sims += out["clip_similarity"].tolist()
        alphas += out["attention"].tolist()

    result = {"loss": total_loss / max(n_seen, 1), "seconds": time.time() - t0}
    if task == "binary":
        result.update(binary_metrics(y_bin, probs))
        result["per_scenario"] = per_scenario_accuracy(scen, y_bin, preds)
    else:
        result.update(multiclass_metrics(y_idx, preds, 5))
        result["per_scenario"] = per_scenario_accuracy(scen, y_idx, preds)
    result["_rows"] = [
        {"sample_id": i, "scenario": s, "label_binary": yb, "label_index": yi,
         "prob": (pr if task == "binary" else max(pr)), "pred": pd, "clip_similarity": si,
         "attn_text": a[0], "attn_image": a[1], "attn_clip": a[2]}
        for i, s, yb, yi, pr, pd, si, a in zip(ids, scen, y_bin, y_idx, probs, preds, sims, alphas)
    ]
    return result


def summary(m: dict, task: str) -> str:
    if task == "binary":
        head = (f"loss {m['loss']:.4f}  acc {m['accuracy']:.4f}  f1 {m['f1']:.4f}  "
                f"auc {m['auc']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}")
    else:
        head = f"loss {m['loss']:.4f}  acc {m['accuracy']:.4f}  f1_macro {m['f1_macro']:.4f}"
    per = "  ".join(f"s{s}={v['accuracy']:.3f}" for s, v in m["per_scenario"].items())
    return f"{head}\n    per scenario: {per}"


def load_tokenizers(cfg: FNDCLIPConfig):
    """Separate function so the smoke test can swap in dummy tokenizers."""
    from transformers import AutoTokenizer, CLIPTokenizer
    return AutoTokenizer.from_pretrained(cfg.bert_name), CLIPTokenizer.from_pretrained(cfg.clip_name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train FND-CLIP.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/fnd_clip_v1")
    ap.add_argument("--task", choices=["binary", "scenario"], default="binary")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=2, help="stop after this many epochs without val improvement")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-oversample", action="store_true",
                    help="do not oversample genuine pairs to balance real/fake in training")
    ap.add_argument("--freeze-bert", action="store_true")
    ap.add_argument("--freeze-resnet", action="store_true")
    ap.add_argument("--image-root", default=None, help="prefix for relative image paths in the CSV")
    ap.add_argument("--limit", type=int, default=None, help="debug: use only this many rows per split")
    ap.add_argument("--random-init", action="store_true", help="smoke test only: no pretrained weights")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   task: {args.task}   out: {out_dir}")

    cfg = FNDCLIPConfig(num_outputs=1 if args.task == "binary" else 5, pretrained=not args.random_init,
                        fine_tune_bert=not args.freeze_bert, fine_tune_resnet=not args.freeze_resnet)
    model = FNDCLIP(cfg).to(device)

    bert_tok, clip_tok = load_tokenizers(cfg)

    def loader(split, train):
        ds = FakeNewsDataset(args.csv, split, bert_tok, clip_tok, train=train, image_root=args.image_root)
        if args.limit:
            ds.rows = ds.rows[: args.limit]
        sampler = None
        if train and not args.no_oversample:
            n_real = sum(1 for r in ds.rows if r["label_binary"] == "0")
            n_fake = len(ds.rows) - n_real
            if n_real and n_fake:
                w_real, w_fake = 0.5 / n_real, 0.5 / n_fake      # real and fake each get half the draws
                weights = [w_real if r["label_binary"] == "0" else w_fake for r in ds.rows]
                sampler = WeightedRandomSampler(weights, num_samples=len(ds.rows), replacement=True)
                print(f"oversampling: train has {n_real} real / {n_fake} fake rows; "
                      f"each epoch draws {len(ds.rows)} with real:fake = 1:1")
        return DataLoader(ds, batch_size=args.batch_size, shuffle=(train and sampler is None), sampler=sampler,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=(device.type == "cuda"))

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)
    print(f"rows: train {len(train_loader.dataset)}  val {len(val_loader.dataset)}  test {len(test_loader.dataset)}")

    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr_backbone, args.lr_head, args.weight_decay))
    key = "f1" if args.task == "binary" else "f1_macro"
    best, best_epoch, bad_epochs, history = -1.0, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}")
        tr = run_epoch(model, train_loader, device, args.task, optimizer)
        print("  train: " + summary(tr, args.task))
        va = run_epoch(model, val_loader, device, args.task)
        print("  val:   " + summary(va, args.task))
        history.append({"epoch": epoch,
                        "train": {k: v for k, v in tr.items() if k != "_rows"},
                        "val": {k: v for k, v in va.items() if k != "_rows"}})
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        if va[key] > best:
            best, best_epoch, bad_epochs = va[key], epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch,
                        "val": history[-1]["val"], "args": vars(args)}, out_dir / "best.pt")
            print(f"  saved best.pt (val {key} {best:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  no val improvement for {args.patience} epochs, stopping")
                break

    print(f"\nbest epoch {best_epoch} (val {key} {best:.4f}); evaluating on test")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    te = run_epoch(model, test_loader, device, args.task)
    print("  test:  " + summary(te, args.task))
    rows = te.pop("_rows")
    te["best_epoch"] = best_epoch
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(te, f, indent=2)
    with open(out_dir / "test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir / 'test_metrics.json'} and {out_dir / 'test_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
