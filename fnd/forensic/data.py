"""Image-only inputs; folder/generator/scenario labels remain external targets."""
from io import BytesIO
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .model import vendor_module
from .preprocess import dct_map


class ForensicDataset(Dataset):
    def __init__(self,rows,kind,training=False,dct_dir=None,mask=True,robust=False,corruption=None,dynamic_dct=False,blur_probability=0.):
        self.rows=rows;self.kind=kind;self.training=training;self.robust=robust;self.corruption=corruption
        self.dct_dir=Path(dct_dir) if dct_dir else None
        self.stats=json.loads((self.dct_dir/'dct_stats.json').read_text()) if self.dct_dir and kind=='dct' else None
        self.mask=bool(mask and training and kind=='rgb');self.masker=None
        self.dynamic_dct=dynamic_dct
        self.blur_probability=blur_probability
        self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        if kind=='dct' and not self.stats:raise ValueError('DCT requires saved training statistics')

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        if self.kind=='dct' and not self.corruption and not self.robust and not self.dynamic_dct:
            path=self.dct_dir/'normalized'/(row['sample_id']+'.pt')
            pixels=torch.load(path,weights_only=True)
            if pixels.shape!=(1,224,224) or pixels.dtype!=torch.float32:raise ValueError(f'Invalid tensor {path}')
        else:
            with Image.open(row['image_path']) as original:image=original.convert('RGB')
            if self.corruption=='jpeg75' or (self.training and self.robust and random.random()<.5):
                quality=75 if self.corruption else random.randint(60,95)
                buffer=BytesIO();image.save(buffer,format='JPEG',quality=quality);buffer.seek(0)
                with Image.open(buffer) as jpeg:image=jpeg.convert('RGB')
            if self.corruption=='blur1':image=image.filter(ImageFilter.GaussianBlur(1))
            elif self.training and self.blur_probability>0 and random.random()<self.blur_probability:image=image.filter(ImageFilter.GaussianBlur(random.uniform(.2,1.)))
            image=image.resize((224,224),Image.Resampling.BILINEAR)
            if self.mask and random.random()<.5:
                if self.masker is None:self.masker=vendor_module('mask.py','forensic_vendor_mask').FrequencyMaskGenerator(ratio=.15,band='all',transform_type='fourier',channel='all')
                image=self.masker.transform(image)
            if self.kind=='rgb':pixels=self.tensor(image)
            else:pixels=torch.from_numpy(((dct_map(image)-self.stats['mean'])/(self.stats['std']+1e-8)).astype(np.float32))
        return pixels,torch.tensor(int(row['label']),dtype=torch.float32),i
