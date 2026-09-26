"""Single-image inference for the specified GenImage AI-generation detector."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .model import ForensicModel
from .preprocess import dct_map


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--image',required=True)
    ap.add_argument('--dct-stats');ap.add_argument('--threshold',type=float,default=.5);ap.add_argument('--device',default='mps')
    ap.add_argument('--features-out');args=ap.parse_args();torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    kind=saved['kind'];model=ForensicModel(kind,pretrained=False)
    model.load_state_dict(saved['model'],strict=True);model=model.to(args.device).eval().requires_grad_(False)
    with Image.open(args.image) as original:image=original.convert('RGB')
    if kind=='rgb':
        transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
                   transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        pixels=transform(image)[None]
    else:
        if not args.dct_stats:raise ValueError('DCT inference requires saved training statistics')
        stats=json.loads(Path(args.dct_stats).read_text())
        pixels=torch.from_numpy(((dct_map(image)-stats['mean'])/(stats['std']+1e-8)).astype(np.float32))[None]
    with torch.inference_mode():logit,features=model(pixels.to(args.device),True)
    probability=float(logit.sigmoid().item())
    if args.features_out:torch.save(features.cpu(),args.features_out)
    print(json.dumps(dict(probability_ai_generated=probability,prediction=int(probability>=args.threshold),threshold=args.threshold,
                         target='0 authentic, 1 AI-generated; this is not ground truth for all Photoshop edits',
                         feature_dimension=features.shape[-1],input_shape=list(pixels.shape),input='image pixels only'),indent=2))


if __name__=='__main__':main()
