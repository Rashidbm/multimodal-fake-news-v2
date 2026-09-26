"""Small components for testing image-caption alignment in V1."""
import torch
from torch import nn
from torch.nn import functional as F


class StandardizedSimilarityGate(nn.Module):
    """Training batch standardization with saved running moments for inference."""
    def __init__(self, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('num_batches', torch.tensor(0, dtype=torch.long))

    def forward(self, similarity):
        if self.training:
            mean = similarity.detach().mean()
            variance = similarity.detach().var(unbiased=False)
            with torch.no_grad():
                if self.num_batches.item() == 0:
                    self.running_mean.copy_(mean)
                    self.running_var.copy_(variance)
                else:
                    self.running_mean.lerp_(mean, self.momentum)
                    self.running_var.lerp_(variance, self.momentum)
                self.num_batches.add_(1)
        else:
            mean, variance = self.running_mean, self.running_var
        return torch.sigmoid((similarity - mean) / torch.sqrt(variance + self.eps))


class PairInteractionHead(nn.Module):
    """Learn matching from cross-modal interactions, with a separate supervision head."""
    def __init__(self, clip_dim=512, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(nn.LayerNorm(2 * clip_dim + 1),
                                      nn.Linear(2 * clip_dim + 1, 256), nn.GELU(), nn.Dropout(0.2),
                                      nn.Linear(256, hidden_dim), nn.GELU())
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, image, text):
        product = image * text
        difference = (image-text).abs()
        similarity = product.sum(-1, keepdim=True)
        hidden = self.features(torch.cat([product, difference, similarity], dim=-1))
        return hidden, self.classifier(hidden).squeeze(-1)


def matching_loss(logits, scenarios):
    """Only real/OOC rows specify whether two authentic modalities belong together."""
    eligible = (scenarios == 1) | (scenarios == 4)
    if not eligible.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[eligible], (scenarios[eligible] == 1).float())
