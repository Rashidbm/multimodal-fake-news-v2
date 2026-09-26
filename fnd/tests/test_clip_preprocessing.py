import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fnd.data.torch_dataset import clip_image_transform
from fnd.models.fnd_clip import FNDCLIPConfig


def test_official_preprocessing_matches_published_transform_and_preserves_old_default():
    # A wide image exposes accidental stretching that square smoke images conceal.
    im = Image.fromarray(np.random.default_rng(7).integers(0, 256, size=(240, 640, 3), dtype=np.uint8))
    published = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
    official = clip_image_transform('official')(im)
    assert torch.equal(official, published(im))
    assert not torch.allclose(official, clip_image_transform('legacy')(im))
    assert torch.equal(clip_image_transform()(im), clip_image_transform('legacy')(im))
    assert FNDCLIPConfig(**{}).clip_preprocess == 'legacy'
