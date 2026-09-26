"""Cache image-only representations; final tests require explicit locked evaluation."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel

from fnd.cache_clip_large import embedding
from fnd.data.torch_dataset import clip_image_transform
from .data import ForensicDataset
from .model import ForensicModel
from .preprocess import read_rows, sha256
from .remote_genimage import OUT
from .train import atomic_save, worker_seed

CLIP_NAME='openai/clip-vit-large-patch14'
CLIP_REVISION='32bd64288804d66eefd0ccbe215aa642df71cc41'


class ClipImages(Dataset):
    def __init__(self,rows,corruption=None):
        self.rows=rows;self.corruption=corruption;self.transform=clip_image_transform('official')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image_path']) as original:image=original.convert('RGB')
        if self.corruption=='jpeg75':
            buffer=BytesIO();image.save(buffer,format='JPEG',quality=75);buffer.seek(0)
            with Image.open(buffer) as jpeg:image=jpeg.convert('RGB')
        if self.corruption=='blur1':image=image.filter(ImageFilter.GaussianBlur(1))
        return self.transform(image),torch.tensor(int(self.rows[i]['label']),dtype=torch.float32),i


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['clip','rgb','dct'],required=True)
    ap.add_argument('--csv',required=True);ap.add_argument('--out',required=True);ap.add_argument('--checkpoint')
    ap.add_argument('--splits',nargs='+',default=['train','val']);ap.add_argument('--test-lock')
    ap.add_argument('--dct-dir',default=str(OUT/'dct'));ap.add_argument('--dynamic-dct',action='store_true')
    ap.add_argument('--corruption',choices=['jpeg75','blur1']);ap.add_argument('--device',default='mps')
    ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if 'test' in args.splits:
        if not args.test_lock:raise ValueError('Final test requires a prewritten selection lock')
        lock=json.loads(Path(args.test_lock).read_text())
        if not lock.get('selection_complete'):raise ValueError('Selection not complete')
        if sha256(args.csv) not in lock['allowed_csv_sha256']:raise ValueError('Test data not covered by lock')
        if args.kind=='clip':
            if lock['allowed_clip_revision']!=CLIP_REVISION:raise ValueError('CLIP version is not locked')
        elif sha256(args.checkpoint) not in lock['allowed_checkpoint_sha256']:raise ValueError('Checkpoint is not covered by the final-test lock')
    torch.set_num_threads(4);rows=[r for r in read_rows(args.csv) if r['split'] in args.splits]
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Need unique image rows')
    if args.kind=='clip':
        model=CLIPModel.from_pretrained(CLIP_NAME,revision=CLIP_REVISION,local_files_only=True).to(args.device).eval().requires_grad_(False)
        dataset=ClipImages(rows,args.corruption)
    else:
        if not args.checkpoint:raise ValueError('CNN cache requires a checkpoint')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if checkpoint['kind']!=args.kind:raise ValueError('Wrong checkpoint architecture')
        model=ForensicModel(args.kind,pretrained=False)
        model.load_state_dict(checkpoint['model'],strict=True);model=model.to(args.device).eval().requires_grad_(False)
        dataset=ForensicDataset(rows,args.kind,dct_dir=args.dct_dir,corruption=args.corruption,dynamic_dct=args.dynamic_dct)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,worker_init_fn=worker_seed)
    features=[];logits=[];order=[]
    with torch.inference_mode():
        for step,(pixels,labels,indices) in enumerate(loader,1):
            if args.kind=='clip':representation=embedding(model.get_image_features(pixel_values=pixels.to(args.device)))
            else:
                scores,representation=model(pixels.to(args.device),True);logits.append(scores.cpu())
            features.append(representation.cpu());order.append(indices)
            if step%25==0 or step==len(loader):print(args.kind,args.corruption or 'clean',step,'/',len(loader),flush=True)
    if torch.cat(order).tolist()!=list(range(len(rows))):raise ValueError('Cache row order changed')
    payload=dict(sample_ids=[r['sample_id'] for r in rows],splits=[r['split'] for r in rows],labels=[int(r['label']) for r in rows],
                 features=torch.cat(features),csv_sha256=sha256(args.csv),kind=args.kind,corruption=args.corruption,
                 checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
                 source_sha256=sha256(__file__),input='Image pixels only',test_used='test' in args.splits)
    if args.kind=='clip':payload.update(model_name=CLIP_NAME,model_revision=CLIP_REVISION,preprocessing='official OpenAI bicubic resize+center crop')
    if logits:payload['logits']=torch.cat(logits)
    out.parent.mkdir(parents=True,exist_ok=True);atomic_save(payload,out)
    out.with_suffix('.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in ('features','logits','sample_ids','labels','splits')},indent=2))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
