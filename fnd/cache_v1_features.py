"""Cache existing V1 representations without training or changing its checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.torch_dataset import FakeNewsDataset, collate
from .models.fnd_clip import FNDCLIP, FNDCLIPConfig
from .train import load_tokenizers, to_device
from .train_clip_matcher import digest, read_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = FNDCLIPConfig(**checkpoint['config'])
    model = FNDCLIP(config).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval().requires_grad_(False)
    tokenizers = load_tokenizers(config)
    rows = [r for r in read_rows(args.csv) if r['split'] in args.splits]
    ds = FakeNewsDataset(args.csv, rows[0]['split'], *tokenizers, train=False, clip_preprocess=config.clip_preprocess)
    ds.rows = rows
    loader = DataLoader(ds, batch_size=32, num_workers=4, collate_fn=collate)
    ids, features, logits = [], [], []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            batch = to_device(batch, args.device)
            result = model(batch['resnet_pixels'], batch['bert_input_ids'], batch['bert_attention_mask'],
                           batch['clip_pixels'], batch['clip_input_ids'], batch['clip_attention_mask'])
            ids.extend(batch['sample_id'])
            features.append(result['semantic'].cpu())
            logits.append(result['logits'].squeeze(-1).cpu())
            if step % 50 == 0 or step == len(loader):
                print(f'V1 features {step}/{len(loader)}', flush=True)
    payload = dict(sample_ids=ids, hidden=torch.cat(features), logit=torch.cat(logits),
                   csv_sha256=digest(args.csv), checkpoint_sha256=digest(args.checkpoint),
                   checkpoint=str(Path(args.checkpoint).resolve()), splits=args.splits, epoch=checkpoint['epoch'])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out.with_suffix('.json').write_text(json.dumps({k: v for k, v in payload.items() if k not in ['hidden', 'logit', 'sample_ids']}, indent=2))
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
