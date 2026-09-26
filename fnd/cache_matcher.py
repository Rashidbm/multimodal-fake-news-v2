"""Cache matcher outputs/features for explicit splits; evaluation is separate."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizerFast

from .models.clip_matcher import CLIPMatcher
from .train_clip_matcher import clip_transform, digest, read_rows


class RowDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.transform = rows, clip_transform()
        self.tokens = tokenizer([r['text'] for r in rows], padding='max_length', truncation=True,
                                max_length=77, return_tensors='pt')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]['image_path']) as image:
            pixels = self.transform(image.convert('RGB'))
        return dict(pixel_values=pixels, input_ids=self.tokens['input_ids'][i],
                    attention_mask=self.tokens['attention_mask'][i])


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
    checkpoint_sha = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CLIPMatcher(checkpoint['model_name'], tune_layers=0)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizerFast.from_pretrained(checkpoint['model_name'])
    rows = [r for r in read_rows(args.csv) if r['split'] in args.splits]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique sample IDs')
    ds = RowDataset(rows, tokenizer)
    loader = DataLoader(ds, batch_size=32, num_workers=4)
    collected = {k: [] for k in ['image', 'text', 'hidden', 'logit']}
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            image, text = model.encode(**{k: v.to(args.device) for k, v in batch.items()})
            product = image*text
            feature = torch.cat([image, text, product, (image-text).abs(), product.sum(-1, keepdim=True)], -1)
            hidden = model.head[:-1](feature)
            logit = model.head[-1](hidden).squeeze(-1)
            for key, value in dict(image=image, text=text, hidden=hidden, logit=logit).items():
                collected[key].append(value.cpu())
            if step % 50 == 0 or step == len(loader):
                print(f'matcher features {step}/{len(loader)}', flush=True)
    payload = dict(sample_ids=[r['sample_id'] for r in rows], csv_sha256=digest(args.csv),
                   checkpoint_sha256=checkpoint_sha, checkpoint=str(Path(args.checkpoint).resolve()),
                   splits=args.splits, threshold=checkpoint['threshold'], epoch=checkpoint['epoch'],
                   **{k: torch.cat(v) for k, v in collected.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out.with_suffix('.json').write_text(json.dumps({k: v for k, v in payload.items() if k not in collected and k != 'sample_ids'}, indent=2))
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
