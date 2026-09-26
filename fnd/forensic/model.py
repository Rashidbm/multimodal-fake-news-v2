"""The two image-specific guide models; explicit freezing includes BN buffers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'external/FakeImageDetection'
RGB_WEIGHTS=VENDOR/'checkpoints/mask_15/rn50ft_spectralmask.pth'


def vendor_module(filename,name):
    spec=importlib.util.spec_from_file_location(name,VENDOR/filename)
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(VENDOR))
    try:spec.loader.exec_module(module)
    finally:sys.path.remove(str(VENDOR))
    return module


class ForensicModel(nn.Module):
    def __init__(self,kind,pretrained=True,replace_rgb_head=True):
        super().__init__();self.kind=kind;self.phase=1
        if kind=='dct':
            self.encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder.conv1=nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)
            nn.init.kaiming_normal_(self.encoder.conv1.weight,mode='fan_out',nonlinearity='relu')
            self.encoder.fc=nn.Linear(2048,1)
        elif kind=='rgb':
            self.encoder=vendor_module('networks/resnet.py','forensic_vendor_resnet').resnet50(pretrained=False)
            if pretrained:
                # The author's optimizer metadata contains NumPy scalar AP values.
                # Allow only these inspected numeric types, keeping weights_only enabled.
                allowed=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,
                         np.dtypes.Float64DType,np.dtypes.Float32DType]
                with torch.serialization.safe_globals(allowed):
                    checkpoint=torch.load(RGB_WEIGHTS,map_location='cpu',weights_only=True)
                state=checkpoint.get('model_state_dict',checkpoint.get('model',checkpoint.get('state_dict',checkpoint)))
                state={k.removeprefix('module.'):v for k,v in state.items()}
                if state['fc.weight'].shape[0]!=1000:self.encoder.change_output(state['fc.weight'].shape[0])
                self.encoder.load_state_dict(state,strict=True)
            if replace_rgb_head or not pretrained:self.encoder.change_output(1)
        else:raise ValueError(kind)
        self.set_phase(1)

    def set_phase(self,phase):
        self.phase=phase
        self.frozen_names=(['conv1','bn1','layer1','layer2'] if self.kind=='rgb' and phase==1 else ['layer1','layer2'] if self.kind=='dct' and phase==1 else [])
        for p in self.encoder.parameters():p.requires_grad_(True)
        for name in self.frozen_names:getattr(self.encoder,name).requires_grad_(False)
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if mode:
            for name in getattr(self,'frozen_names',[]):getattr(self.encoder,name).eval()
        return self

    def forward(self,pixels,return_features=False):
        e=self.encoder
        x=e.maxpool(e.relu(e.bn1(e.conv1(pixels))))
        x=e.layer1(x);x=e.layer2(x);x=e.layer3(x);x=e.layer4(x)
        features=e.avgpool(x).flatten(1);logits=e.fc(features).squeeze(-1)
        return (logits,features) if return_features else logits
