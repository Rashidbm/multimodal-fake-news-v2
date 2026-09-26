"""Train controlled frozen/adapted CLIP matchers on caption-paired train/dev data."""
import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizerFast

from .models.clip_matcher import CLIPMatcher


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path, split=None):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    return [r for r in rows if split is None or r['split'] == split]


def clip_transform():
    return transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


class PairedDataset(Dataset):
    def __init__(self, rows, tokenizer):
        groups = defaultdict(dict)
        for row in rows:
            key, scenario = row['caption_id'], int(row['scenario'])
            if scenario in groups[key]:
                raise ValueError('Duplicate caption/scenario')
            groups[key][scenario] = row
        if not groups or any(set(g) != {1, 4} for g in groups.values()):
            raise ValueError('Each caption must have one genuine and one OOC image')
        self.pairs = [(groups[k][4], groups[k][1]) for k in sorted(groups)]
        if any(r['text'] != f['text'] for r, f in self.pairs):
            raise ValueError('Different captions inside a pair')
        self.tokens = tokenizer([r['text'] for r, _ in self.pairs], padding='max_length',
                                truncation=True, max_length=77, return_tensors='pt')
        self.transform = clip_transform()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pixels = []
        for row in self.pairs[i]:
            with Image.open(row['image_path']) as image:
                pixels.append(self.transform(image.convert('RGB')))
        return dict(real=pixels[0], fake=pixels[1], input_ids=self.tokens['input_ids'][i],
                    attention_mask=self.tokens['attention_mask'][i], index=i)


def pair_loss(logits):
    if len(logits) % 2:
        raise ValueError('Need real half followed by mismatched half')
    real, fake = logits.chunk(2)
    bce = (F.binary_cross_entropy_with_logits(real, torch.zeros_like(real))
           + F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake)))/2
    return bce + 0.2*F.relu(0.5-(fake-real)).mean()


def score_pairs(real, fake, threshold=0.5):
    real, fake = np.asarray(real), np.asarray(fake)
    rc, fc = real < threshold, fake >= threshold
    return dict(genuine_recall=float(rc.mean()), ooc_recall=float(fc.mean()),
                balanced_accuracy=float((rc.mean()+fc.mean())/2), both_correct=float((rc & fc).mean()),
                within_caption_ranking=float(((fake > real)+0.5*(fake == real)).mean()),
                auc=float(roc_auc_score([0]*len(real)+[1]*len(fake), np.r_[real, fake])),
                caption_pairs=len(real), threshold=float(threshold))


def select_threshold(real, fake):
    # Development data only. Tie break toward balanced sensitivity/specificity, then 0.5.
    fpr, tpr, thresholds = roc_curve([0]*len(real)+[1]*len(fake), np.r_[real, fake], drop_intermediate=False)
    valid = np.flatnonzero(np.isfinite(thresholds))
    best = max(valid, key=lambda i: (round(float(tpr[i]-fpr[i]), 12),
                                    min(float(tpr[i]), 1-float(fpr[i])), -abs(float(thresholds[i])-0.5)))
    return float(thresholds[best])


def batch_inputs(batch, device):
    return dict(pixel_values=torch.cat([batch['real'], batch['fake']]).to(device),
                input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))


def cache_features(model, loader, device):
    n = len(loader.dataset)
    images, texts = None, None
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            image, text = model.encode(**batch_inputs(batch, device))
            image = torch.stack(image.cpu().chunk(2), dim=1)
            text = text[:len(batch['index'])].cpu()
            if images is None:
                images = torch.empty(n, 2, image.shape[-1])
                texts = torch.empty(n, text.shape[-1])
            images[batch['index']], texts[batch['index']] = image, text
            if step % 50 == 0 or step == len(loader):
                print(f'cache {step}/{len(loader)}', flush=True)
    return images.to(device).clone(), texts.to(device).clone()


def predictions(model, loader, device, cache=None):
    model.eval()
    real, fake = np.empty(len(loader.dataset)), np.empty(len(loader.dataset))
    with torch.inference_mode():
        if cache is not None:
            image, text = cache
            for indices in torch.arange(len(real), device=device).split(128):
                r = model.classify(image[indices, 0], text[indices]).sigmoid().cpu().numpy()
                f = model.classify(image[indices, 1], text[indices]).sigmoid().cpu().numpy()
                key = indices.cpu().numpy()
                real[key], fake[key] = r, f
        else:
            for batch in loader:
                r, f = model(**batch_inputs(batch, device)).sigmoid().cpu().numpy().reshape(2, -1)
                real[batch['index']], fake[batch['index']] = r, f
    return real, fake


def save_predictions(path, dataset, real, fake):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'caption_id', 'scenario', 'label_binary', 'prob'])
        writer.writeheader()
        for pair, pr, pf in zip(dataset.pairs, real, fake):
            for row, probability in zip(pair, (pr, pf)):
                writer.writerow({**{k: row[k] for k in writer.fieldnames if k != 'prob'}, 'prob': float(probability)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tune-layers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--backbone-lr', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
    datasets = {split: PairedDataset(read_rows(args.csv, split), tokenizer) for split in ('train', 'val')}
    train_keys = {r['caption_id'] for r, _ in datasets['train'].pairs}
    if train_keys & {r['caption_id'] for r, _ in datasets['val'].pairs}:
        raise ValueError('Caption leakage between train and validation')
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, persistent_workers=args.workers > 0)
               for split, ds in datasets.items()}
    device = torch.device(args.device)
    model = CLIPMatcher(tune_layers=args.tune_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(args.head_lr, args.backbone_lr, args.weight_decay))
    out.mkdir(parents=True)
    manifest = dict(args=vars(args), csv_sha256=digest(args.csv), torch_version=torch.__version__,
                    trainable_parameters=[name for name, p in model.named_parameters() if p.requires_grad],
                    counts={k: len(ds) for k, ds in datasets.items()},
                    preprocessing='CLIP bicubic shorter-edge resize 224, center crop 224, official normalization; 77 tokens',
                    model_inputs=['pixel_values', 'input_ids', 'attention_mask'],
                    target='0: genuine matched pair, 1: out-of-context pair; not a general factual truth label',
                    confirmation_used=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    cache = {split: cache_features(model, loader, device) for split, loader in loaders.items()} if not args.tune_layers else {}
    if cache:
        torch.save({k: [v.cpu() for v in values] for k, values in cache.items()}, out/'frozen_features.pt')
    history, best, stale = [], (-1.0, -1.0), 0
    for epoch in range(1, args.epochs+1):
        start = time.time()
        model.train()
        losses, seen = 0., 0
        generator = torch.Generator().manual_seed(args.seed+epoch)
        order = torch.randperm(len(datasets['train']), generator=generator)
        if cache:
            batches = order.split(args.batch_size)
        else:
            batches = DataLoader(datasets['train'], batch_size=args.batch_size, sampler=order.tolist(),
                                 num_workers=args.workers)
        for step, batch in enumerate(batches, 1):
            if cache:
                indices = batch.to(device)
                image, text = cache['train']
                logits = model.classify(torch.cat([image[indices, 0], image[indices, 1]]),
                                        torch.cat([text[indices], text[indices]]))
            else:
                logits = model(**batch_inputs(batch, device))
            loss = pair_loss(logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses += loss.item()*len(logits); seen += len(logits)
            if step % 50 == 0:
                print(f'epoch {epoch} step {step}/{len(batches)} loss {losses/seen:.4f} seconds {time.time()-start:.1f}', flush=True)
        real, fake = predictions(model, loaders['val'], device, cache.get('val'))
        threshold = select_threshold(real, fake)
        fixed, selected = score_pairs(real, fake), score_pairs(real, fake, threshold)
        result = dict(epoch=epoch, train_loss=losses/seen, seconds=time.time()-start, fixed=fixed, selected=selected)
        history.append(result)
        save_predictions(out/f'val_epoch_{epoch}.csv', datasets['val'], real, fake)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        score = (selected['balanced_accuracy'], selected['auc'])
        print(json.dumps(result), flush=True)
        if score > best:
            best, stale = score, 0
            checkpoint = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                tune_layers=args.tune_layers, model_name=model.model_name, epoch=epoch,
                threshold=threshold, validation=result, csv_sha256=manifest['csv_sha256'])
            torch.save(checkpoint, out/'best.pt')
            (out/'best.json').write_text(json.dumps({k: v for k, v in checkpoint.items() if k != 'model'}, indent=2))
        else:
            stale += 1
            if stale >= args.patience:
                break
    print('COMPLETE '+str(out/'best.json'), flush=True)


if __name__ == '__main__':
    main()
