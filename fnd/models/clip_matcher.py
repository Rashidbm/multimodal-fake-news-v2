"""CLIP image–caption matcher; no labels or provenance enter its forward method."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel


class CLIPMatcher(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', tune_layers=0, clip=None):
        super().__init__()
        self.model_name, self.tune_layers = model_name, tune_layers
        self.clip = clip if clip is not None else CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        if tune_layers:
            for encoder in (self.clip.vision_model, self.clip.text_model):
                if not 0 < tune_layers <= len(encoder.encoder.layers):
                    raise ValueError('Invalid number of trainable CLIP layers')
                for layer in encoder.encoder.layers[-tune_layers:]:
                    layer.requires_grad_(True)
            for module in (self.clip.visual_projection, self.clip.text_projection,
                           self.clip.vision_model.post_layernorm, self.clip.text_model.final_layer_norm):
                module.requires_grad_(True)
        dim = self.clip.config.projection_dim
        self.head = nn.Sequential(nn.LayerNorm(4*dim+1), nn.Linear(4*dim+1, 256),
                                  nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64),
                                  nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def train(self, mode=True):
        super().train(mode)
        if not self.tune_layers:
            self.clip.eval()
        return self

    @staticmethod
    def embedding(output):
        value = output if torch.is_tensor(output) else output.pooler_output
        return F.normalize(value, dim=-1)

    def encode(self, pixel_values, input_ids, attention_mask):
        image = self.embedding(self.clip.get_image_features(pixel_values=pixel_values))
        text = self.embedding(self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
        if len(image) == 2*len(text):
            text = torch.cat([text, text])
        elif len(image) != len(text):
            raise ValueError('Expected one text per image or one text per two paired images')
        return image, text

    def classify(self, image, text):
        product = image*text
        feature = torch.cat([image, text, product, (image-text).abs(), product.sum(-1, keepdim=True)], -1)
        return self.head(feature).squeeze(-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        return self.classify(*self.encode(pixel_values, input_ids, attention_mask))

    def parameter_groups(self, head_lr, backbone_lr, weight_decay):
        groups = [dict(params=list(self.head.parameters()), lr=head_lr, weight_decay=weight_decay)]
        backbone = [p for p in self.clip.parameters() if p.requires_grad]
        if backbone:
            groups.append(dict(params=backbone, lr=backbone_lr, weight_decay=weight_decay))
        return groups
