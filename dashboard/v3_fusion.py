"""Plug the trained V3 fusion checkpoint (fnd.train_fusion's best.pt) into the dashboard.

    python -m dashboard.server --fusion dashboard.v3_fusion:load_fusion

The dashboard calls the factory with the device only, so the checkpoint path comes from
FUSION_CHECKPOINT (default outputs/fusion_v3/best.pt, where fnd.train_fusion writes it).
"""
import os
from pathlib import Path

import torch

from fnd.models.pipeline_v3 import V3FusionModule

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / 'outputs' / 'fusion_v3' / 'best.pt'


class V3Fusion:
    """Callable the dashboard expects: classes + (semantic, image, text) -> probabilities [1, 5]."""

    def __init__(self, model, classes):
        self.model, self.classes = model.eval(), list(classes)

    def __call__(self, semantic, image, text):
        with torch.inference_mode():
            return self.model(semantic, image, text)['main_logits'].softmax(-1)


def load_fusion(device, checkpoint=None):
    path = Path(checkpoint or os.environ.get('FUSION_CHECKPOINT', DEFAULT_CHECKPOINT))
    if not path.is_file():
        raise FileNotFoundError(f'fusion checkpoint {path} not found: set FUSION_CHECKPOINT to the '
                                'best.pt written by fnd.train_fusion')
    saved = torch.load(path, map_location=device, weights_only=True)
    model = V3FusionModule().to(device)
    model.load_state_dict(saved['model'])   # strict: fails loudly if the architecture changed
    print(f'fusion checkpoint {path} (epoch {saved.get("epoch")}, '
          f'val macro-F1 {saved.get("validation", {}).get("macro_f1", float("nan")):.4f})', flush=True)
    return V3Fusion(model, saved['classes'])
