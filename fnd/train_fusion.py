"""Train the V3 fusion module on cached, ID-joined branch features.

Inputs are npz files with arrays sample_ids, split, scenario (original 1..5), semantic [N,768],
image [N,768] and text [N,4096]: the release's fusion-training-features.npz for train/val, and
an optional test file in the same layout. Only the fusion module trains; the branches are frozen.

    python -m fnd.train_fusion --features fusion-training-features.npz --out outputs/fusion_v3
    python -m fnd.train_fusion --features train_val.npz --test-features test.npz --out outputs/fusion_v3

Recipe (as fnd.train_team_fusion): AdamW 1e-4, batch 64, class-balanced sampler, cross-entropy
plus 0.1 x image auxiliary loss, checkpoint selection on validation macro-F1, patience 10.
Test rows are scored once, by the selected checkpoint, after training has finished.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fnd.models.pipeline_v3 import V3FusionModule

CLASSES = ['genuine', 'out_of_context', 'real_text_fake_image',
           'fake_or_edited_text_real_image', 'fake_or_edited_text_fake_image']
# Original scenario -> class: S4->0 genuine, S1->1 OOC, S3->2, S2->3, S5->4 (index 0 unused).
SCENARIO_TO_CLASS = np.array([-1, 1, 3, 2, 0, 4])
IMAGE_FAKE_CLASSES = (2, 4)


def load(path, splits):
    with np.load(path, allow_pickle=False) as data:
        missing = {'sample_ids', 'split', 'scenario', 'semantic', 'image', 'text'} - set(data.files)
        if missing:
            raise ValueError(f'{path}: missing arrays {missing}')
        keep = np.isin(data['split'].astype(str), splits)
        ids = data['sample_ids'].astype(str)[keep]
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f'{path}: duplicate sample IDs')
        scenario = data['scenario'][keep].astype(np.int64)
        if not np.isin(scenario, [1, 2, 3, 4, 5]).all():
            raise ValueError(f'{path}: scenarios must be the original 1..5')
        arrays = [data[k][keep].astype(np.float32) for k in ['semantic', 'image', 'text']]
        if [a.shape[1] for a in arrays] != [768, 768, 4096] or not all(np.isfinite(a).all() for a in arrays):
            raise ValueError(f'{path}: expected finite semantic/image [N,768] and text [N,4096]')
        return dict(ids=ids, split=data['split'].astype(str)[keep], y=SCENARIO_TO_CLASS[scenario],
                    x=[torch.from_numpy(a) for a in arrays])


def predict(model, x, device, batch_size=256):
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(x[0]), batch_size):
            logits = model(*[a[start:start+batch_size].to(device) for a in x])['main_logits']
            out.append(logits.softmax(-1).cpu())
    return torch.cat(out).numpy()


def report(y, probabilities):
    pred = probabilities.argmax(1)
    binary_true, binary_pred = (y != 0).astype(int), (pred != 0).astype(int)
    return dict(
        n=int(len(y)), accuracy=float((pred == y).mean()),
        macro_f1=float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0)),
        recall_by_class=dict(zip(CLASSES, recall_score(y, pred, labels=list(range(5)), average=None, zero_division=0).tolist())),
        f1_by_class=dict(zip(CLASSES, f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0).tolist())),
        confusion=confusion_matrix(y, pred, labels=list(range(5))).tolist(),
        binary=dict(accuracy=float((binary_pred == binary_true).mean()),
                    macro_f1=float(f1_score(binary_true, binary_pred, average='macro', zero_division=0)),
                    recall_fake=float(recall_score(binary_true, binary_pred, zero_division=0)),
                    recall_genuine=float(recall_score(1-binary_true, 1-binary_pred, zero_division=0))))


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True, help='train/val npz')
    ap.add_argument('--test-features', help='test npz, scored once after training')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True)   # a new folder per run keeps earlier results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train, val = load(args.features, ['train']), load(args.features, ['val'])
    for name, part in [('train', train), ('val', val)]:
        if set(part['y'].tolist()) != set(range(5)):
            raise ValueError(f'All five classes are required in {name}')
    counts = np.bincount(train['y'], minlength=5)
    sampler = WeightedRandomSampler(torch.from_numpy(1/counts[train['y']]), len(train['y']), replacement=True)
    loader = DataLoader(TensorDataset(*train['x'], torch.from_numpy(train['y'])), batch_size=64,
                        sampler=sampler, drop_last=True)
    model = V3FusionModule().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best, stale, history, start = -1., 0, [], time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for semantic, image, text, y in loader:
            semantic, image, text, y = (t.to(args.device) for t in (semantic, image, text, y))
            result = model(semantic, image, text)
            image_target = torch.isin(y, torch.tensor(IMAGE_FAKE_CLASSES, device=y.device)).long()
            loss = nn.functional.cross_entropy(result['main_logits'], y) + \
                .1*nn.functional.cross_entropy(result['aux_logits'], image_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics = report(val['y'], predict(model, val['x'], args.device))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_macro_f1=metrics['macro_f1'],
                            val_accuracy=metrics['accuracy'], minutes=(time.perf_counter()-start)/60))
        print(f"epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  val macro-F1 {metrics['macro_f1']:.4f}  "
              f"acc {metrics['accuracy']:.4f}", flush=True)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            torch.save(dict(model=model.state_dict(), classes=CLASSES, epoch=epoch, validation=metrics, seed=args.seed,
                            architecture='fnd.models.pipeline_v3.V3FusionModule (tokenized attention, 607d9f5)',
                            features_sha256=digest(args.features)), out/'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(out/'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    summary = dict(selected_epoch=checkpoint['epoch'], validation=checkpoint['validation'], test_used=False)
    if args.test_features:
        test = load(args.test_features, ['test'])
        probabilities = predict(model, test['x'], args.device)
        summary.update(test=report(test['y'], probabilities), test_used=True,
                       test_features_sha256=digest(args.test_features))
        with open(out/'test_predictions.csv', 'w', encoding='utf-8') as f:
            f.write('sample_id,true_class,predicted_class,' + ','.join(f'p_{c}' for c in CLASSES) + '\n')
            for sample, y, p in zip(test['ids'], test['y'], probabilities):
                f.write(f'{sample},{CLASSES[y]},{CLASSES[p.argmax()]},' + ','.join(f'{v:.6f}' for v in p) + '\n')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
