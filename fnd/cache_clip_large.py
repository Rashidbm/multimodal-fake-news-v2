"""Cache frozen CLIP-L/14 image/text embeddings with official preprocessing."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPTokenizerFast

from .data.torch_dataset import clip_image_transform
from .train_clip_matcher import digest, read_rows


class ClipRows(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.transform = rows, clip_image_transform('official')
        self.tokens = tokenizer([r['text'] for r in rows], max_length=77, padding='max_length', truncation=True, return_tensors='pt')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]['image_path']) as image:
            pixels = self.transform(image.convert('RGB'))
        return dict(pixel_values=pixels, input_ids=self.tokens['input_ids'][index], attention_mask=self.tokens['attention_mask'][index])


def embedding(value):
    value = value if torch.is_tensor(value) else value.pooler_output
    return torch.nn.functional.normalize(value.float(), dim=-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    source = json.loads(Path(args.source).read_text())
    model = CLIPModel.from_pretrained(source['name'], revision=source['revision']).to(args.device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizerFast.from_pretrained(source['name'], revision=source['revision'])
    rows = [r for r in read_rows(args.csv) if r['split'] in args.splits]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique rows')
    loader = DataLoader(ClipRows(rows, tokenizer), batch_size=32, num_workers=4)
    images, texts = [], []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            images.append(embedding(model.get_image_features(pixel_values=batch['pixel_values'].to(args.device))).cpu())
            texts.append(embedding(model.get_text_features(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))).cpu())
            if step % 25 == 0 or step == len(loader):
                print(f'CLIP-L/14 features {step}/{len(loader)}', flush=True)
    payload = dict(sample_ids=[r['sample_id'] for r in rows], csv_sha256=digest(args.csv), model_name=source['name'],
                   model_revision=source['revision'], splits=args.splits, preprocessing='official OpenAI bicubic resize+center crop',
                   image=torch.cat(images), text=torch.cat(texts))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out.with_suffix('.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in ['image', 'text', 'sample_ids']}, indent=2))
    print('COMPLETE '+str(out), flush=True)


if __name__ == '__main__':
    main()
