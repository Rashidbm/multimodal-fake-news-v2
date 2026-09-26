"""Deployable image-only inference for an exported image-branch bundle."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPModel

from fnd.cache_clip_large import embedding
from fnd.data.torch_dataset import clip_image_transform
from .cache import CLIP_NAME,CLIP_REVISION
from .model import ForensicModel
from .preprocess import dct_map,sha256


class ImageBranch(nn.Module):
    def __init__(self,bundle,device='cpu'):
        super().__init__();self.path=Path(bundle);self.config=json.loads(self.path.read_text());self.device=device
        directory=self.path.parent;self.encoders=nn.ModuleDict();self.kinds=self.config['encoders']
        self.rgb_transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.clip_transform=clip_image_transform('official')
        for name,source in self.kinds.items():
            if source['kind']=='clip':
                self.encoders[name]=CLIPModel.from_pretrained(CLIP_NAME,revision=CLIP_REVISION,local_files_only=True)
            else:
                path=directory/source['checkpoint']
                if sha256(path)!=source['checkpoint_sha256']:raise ValueError('Encoder checksum mismatch')
                saved=torch.load(path,map_location='cpu',weights_only=True);model=ForensicModel(source['kind'],pretrained=False)
                model.load_state_dict(saved['model'],strict=True);self.encoders[name]=model
        state_path=directory/self.config['head']
        if sha256(state_path)!=self.config['head_sha256']:raise ValueError('Head checksum mismatch')
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        self.register_buffer('weight',state['weight']);self.register_buffer('bias',state['bias'])
        if 'projection' in state:
            self.register_buffer('projection',state['projection']);self.register_buffer('center',state['center'])
        else:self.projection=None
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_images(self,images):
        images=[image.convert('RGB') for image in images];features=[]
        for name,source in self.kinds.items():
            kind=source['kind'];encoder=self.encoders[name]
            if kind=='clip':
                pixels=torch.stack([self.clip_transform(image) for image in images]).to(self.device)
                features.append(embedding(encoder.get_image_features(pixel_values=pixels)))
            elif kind=='rgb':
                pixels=torch.stack([self.rgb_transform(image) for image in images]).to(self.device)
                features.append(encoder(pixels,True)[1])
            else:
                stats=source['dct_stats']
                pixels=torch.from_numpy(np.stack([(dct_map(image)-stats['mean'])/(stats['std']+1e-8) for image in images])).float().to(self.device)
                features.append(encoder(pixels,True)[1])
        return torch.cat(features,dim=1)

    @torch.inference_mode()
    def forward(self,images):
        raw=self.encode_images(images);logits=raw@self.weight+self.bias
        features=(raw-self.center)@self.projection.T if self.projection is not None else raw
        return dict(logits=logits,probability=logits.sigmoid(),features=features)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--image',required=True)
    ap.add_argument('--device',default='mps');ap.add_argument('--features-out')
    args=ap.parse_args();torch.set_num_threads(4);model=ImageBranch(args.bundle,args.device)
    with Image.open(args.image) as image:output=model([image])
    probability=float(output['probability'].item());threshold=model.config['threshold']
    if args.features_out:torch.save(output['features'].cpu(),args.features_out)
    print(json.dumps(dict(probability_fake_image=probability,prediction=int(probability>=threshold),threshold=threshold,
                         target=model.config['target'],feature_dimension=output['features'].shape[-1],input='image pixels only'),indent=2))


if __name__=='__main__':main()
