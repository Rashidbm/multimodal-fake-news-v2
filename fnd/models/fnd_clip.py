"""FND-CLIP, implemented from the project instructions (sections 2.1-2.4).

Reference: "Multimodal Fake News Detection via CLIP-Guided Learning",
Zhou et al., IEEE ICME 2023.

    2.1  image  -> ResNet-50 (ImageNet weights, fine-tuned)  -> 2048
    2.2  text   -> BERT-base-uncased, [CLS] token             ->  768
    2.3  image  -> CLIP image encoder (frozen)                ->  512
         text   -> CLIP text encoder  (frozen)                ->  512
         sim = cosine(clip_img, clip_txt)
         fused = concat(clip_img, clip_txt) = 1024, re-weighted by sim
    2.4  three streams (text, image, fused) -> each projected to D
         attention gives one scalar per stream, softmax, weighted sum -> D
         two-layer MLP -> 1 logit (sigmoid = P(fake))

Every stream is projected to the same size D so the attention can compare
them.  The forward pass also returns the attention weights and the CLIP
similarity, so we can inspect which stream the model relied on per sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import StandardizedSimilarityGate, PairInteractionHead


@dataclass
class FNDCLIPConfig:
    bert_name: str = "bert-base-uncased"
    clip_name: str = "openai/clip-vit-base-patch32"
    proj_dim: int = 256            # D: common size of the three streams
    dropout: float = 0.1
    num_outputs: int = 1           # 1 = binary real/fake (spec); 5 = one logit per scenario
    fine_tune_resnet: bool = True  # spec 2.1: ResNet-50 is fine-tuned
    fine_tune_bert: bool = True    # existing project baseline; the cited preprint freezes BERT
    similarity_weighting: str = "relu"   # 'relu' | 'sigmoid' | 'none' | 'standardized'
    alignment_head: bool = False
    normalize_streams: bool = False
    fine_tune_matching: bool = True
    pretrained: bool = True        # False = random weights, used only by unit tests (no downloads)
    clip_preprocess: str = 'legacy'  # checkpointed; old models retain their original inputs
    augment_unimodal_clip: bool = False  # separate author-inspired fusion experiment


class FNDCLIP(nn.Module):
    def __init__(self, cfg: FNDCLIPConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or FNDCLIPConfig()
        D = cfg.proj_dim

        # ---- 2.1 ResNet-50 image stream -> 2048 ----------------------------
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        resnet.fc = nn.Identity()                 # keep the 2048-d pooled feature
        self.resnet = resnet
        for p in self.resnet.parameters():
            p.requires_grad = cfg.fine_tune_resnet

        # ---- 2.2 BERT text stream -> 768 ([CLS]) ---------------------------
        from transformers import BertConfig, BertModel
        if cfg.pretrained:
            self.bert = BertModel.from_pretrained(cfg.bert_name, add_pooling_layer=False)
        else:
            self.bert = BertModel(BertConfig(), add_pooling_layer=False)
        for p in self.bert.parameters():
            p.requires_grad = cfg.fine_tune_bert

        # ---- 2.3 CLIP encoders (frozen) -> 512 + 512 -----------------------
        from transformers import CLIPConfig, CLIPModel
        self.clip = CLIPModel.from_pretrained(cfg.clip_name) if cfg.pretrained else CLIPModel(CLIPConfig())
        for p in self.clip.parameters():
            p.requires_grad = False
        clip_dim = self.clip.config.projection_dim   # 512 for ViT-B/32

        # ---- projections to the common size D ------------------------------
        def proj(in_dim):
            return nn.Sequential(nn.Linear(in_dim, D), nn.ReLU(), nn.Dropout(cfg.dropout))
        extra_clip = clip_dim if cfg.augment_unimodal_clip else 0
        self.text_proj = proj(self.bert.config.hidden_size + extra_clip)
        self.image_proj = proj(2048 + extra_clip)
        self.fused_proj = proj(2 * clip_dim)                   # 1024 -> D

        # ---- 2.4 modality-wise attention + classifier ----------------------
        self.attention = nn.Sequential(nn.Linear(D, D // 2), nn.Tanh(), nn.Linear(D // 2, 1))
        self.classifier = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(D // 2, cfg.num_outputs)
        )
        if cfg.similarity_weighting == 'standardized':
            self.similarity_gate = StandardizedSimilarityGate()
        if cfg.normalize_streams:
            self.stream_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(3)])
        if cfg.alignment_head:
            self.matching_head = PairInteractionHead(clip_dim)
            self.matching_projection = nn.Linear(128, D)
            self.matching_head.requires_grad_(cfg.fine_tune_matching)

    # 2.3: turn the cosine similarity into a weight for the fused CLIP vector.
    # Poorly aligned pairs (low similarity) should contribute less.
    def _weight_from_sim(self, sim: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.similarity_weighting
        if mode == "relu":
            return sim.clamp(min=0.0)          # negative similarity -> 0 contribution
        if mode == "sigmoid":
            return torch.sigmoid(sim * 10.0)   # smooth 0..1, centred at sim = 0
        if mode == "none":
            return torch.ones_like(sim)
        if mode == "standardized":
            return self.similarity_gate(sim)
        raise ValueError(f"unknown similarity_weighting {mode!r}")

    def _clip_embed(self, out) -> torch.Tensor:
        """transformers < 5 returns a tensor; >= 5 returns an output object whose
        pooler_output is the projected embedding.  Accept both, check the size."""
        if not torch.is_tensor(out):
            out = getattr(out, "pooler_output", None) if getattr(out, "pooler_output", None) is not None else out[0]
        if out.shape[-1] != self.clip.config.projection_dim:
            raise RuntimeError(f"CLIP embedding has size {out.shape[-1]}, expected {self.clip.config.projection_dim}")
        return out

    def forward(self, resnet_pixels, bert_input_ids, bert_attention_mask,
                clip_pixels, clip_input_ids, clip_attention_mask) -> dict:
        # 2.1 image stream
        img = self.resnet(resnet_pixels)                                          # (B, 2048)
        # 2.2 text stream
        txt = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        txt = txt.last_hidden_state[:, 0]                                         # [CLS] (B, 768)
        # 2.3 CLIP alignment (no gradient: frozen)
        with torch.no_grad():
            c_img = self._clip_embed(self.clip.get_image_features(pixel_values=clip_pixels))      # (B, 512)
            c_txt = self._clip_embed(self.clip.get_text_features(input_ids=clip_input_ids,
                                                                 attention_mask=clip_attention_mask))  # (B, 512)
        raw_clip_image, raw_clip_text = c_img, c_txt
        c_img = F.normalize(c_img, dim=-1)
        c_txt = F.normalize(c_txt, dim=-1)
        sim = (c_img * c_txt).sum(dim=-1)                                         # cosine (B,)
        weight = self._weight_from_sim(sim).unsqueeze(-1)                         # (B, 1)
        fused = torch.cat([c_img, c_txt], dim=-1)                                # (B, 1024)
        projected_clip = self.fused_proj(fused if self.cfg.similarity_weighting == 'standardized' else fused * weight)
        if self.cfg.augment_unimodal_clip:
            txt = torch.cat([txt, raw_clip_text], dim=-1)
            img = torch.cat([img, raw_clip_image], dim=-1)
        projected_text, projected_image = self.text_proj(txt), self.image_proj(img)
        if self.cfg.normalize_streams:
            projected_text = self.stream_norms[0](projected_text)
            projected_image = self.stream_norms[1](projected_image)
            projected_clip = self.stream_norms[2](projected_clip)
        if self.cfg.similarity_weighting == 'standardized':
            projected_clip = projected_clip * weight
        extra = {}
        if self.cfg.alignment_head:
            matching_features, matching_logits = self.matching_head(c_img, c_txt)
            # Put the trained pair interaction into the third stream, preserving
            # the 3*D semantic export and its exact relationship to V1 logits.
            projected_clip = projected_clip + self.matching_projection(matching_features)
            extra['match_logits'] = matching_logits

        # project the three streams to D
        streams = torch.stack([projected_text, projected_image, projected_clip], dim=1)  # (B, 3, D)

        # 2.4 modality-wise attention: one score per stream, softmax, weighted sum
        scores = self.attention(streams).squeeze(-1)                              # (B, 3)
        alpha = torch.softmax(scores, dim=-1)                                     # (B, 3)
        weighted_streams = alpha.unsqueeze(-1) * streams
        pooled = weighted_streams.sum(dim=1)                                     # (B, D)
        logits = self.classifier(pooled)                                          # (B, num_outputs)
        # V2 consumes the three trained modality representations before the sum.
        # Default D=256 gives 3*256=768 features, with no new/random projection.
        return {"logits": logits, "attention": alpha, "clip_similarity": sim,
                "semantic": weighted_streams.flatten(1), **extra}

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        if not self.cfg.fine_tune_bert:
            self.bert.eval()
        if not self.cfg.fine_tune_resnet:
            self.resnet.eval()
        if self.cfg.alignment_head and not self.cfg.fine_tune_matching:
            self.matching_head.eval()
        return self

    def parameter_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """Backbones (ResNet, BERT) get a small learning rate, new layers a larger one."""
        backbone = [p for m in (self.resnet, self.bert) for p in m.parameters() if p.requires_grad]
        head = [p for m in (self.text_proj, self.image_proj, self.fused_proj, self.attention, self.classifier)
                for p in m.parameters()]
        if self.cfg.normalize_streams:
            head.extend(self.stream_norms.parameters())
        if self.cfg.alignment_head:
            head.extend(p for p in self.matching_head.parameters() if p.requires_grad)
            head.extend(self.matching_projection.parameters())
        groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
        return groups
