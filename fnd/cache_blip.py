"""Cache a pretrained BLIP matching representation using image/caption inputs only."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BlipForImageTextRetrieval, BlipProcessor

from .cache_matcher import RowDataset
from .train_clip_matcher import digest, read_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--checkpoint', help='Optional task-adapted BLIP state; otherwise official pretrained weights')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    name = 'Salesforce/blip-itm-base-coco'
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForImageTextRetrieval.from_pretrained(name).to(args.device).eval().requires_grad_(False)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
    rows = [r for r in read_rows(args.csv) if r['split'] in args.splits]
    ds = RowDataset(rows, processor.tokenizer)
    ds.transform = transforms.Compose([transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                                      transforms.ToTensor(), transforms.Normalize(processor.image_processor.image_mean,
                                                                               processor.image_processor.image_std)])
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
    features, logits = [], []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            output = model(**{k: v.to(args.device) for k, v in batch.items()}, use_itm_head=True)
            features.append(output.question_embeds[:, 0].cpu())
            logits.append((output.itm_score[:, 0]-output.itm_score[:, 1]).cpu())
            if step % 25 == 0 or step == len(loader):
                print(f'BLIP {step}/{len(loader)}', flush=True)
    payload = dict(sample_ids=[r['sample_id'] for r in rows], csv_sha256=digest(args.csv),
                   splits=args.splits, model_name=name, model_revision=model.config._commit_hash,
                   checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
                   hidden=torch.cat(features), logit=torch.cat(logits),
                   preprocessing='384x384 bicubic RGB, official normalization, BERT tokenizer max_length=77',
                   logit_definition='pretrained ITM mismatch logit minus match logit; not factual falsity')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out.with_suffix('.json').write_text(json.dumps({k: v for k, v in payload.items() if k not in ['hidden', 'logit', 'sample_ids']}, indent=2))
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
