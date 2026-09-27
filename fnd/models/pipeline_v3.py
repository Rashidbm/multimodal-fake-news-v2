"""V3 Fusion Architecture + Classifier (Implementation Guidelines V3 §5-6).

Consumes three feature vectors produced upstream:
  v_semantic  [B, 768]   — semantic branch (docs/SEMANTIC_BRANCH.md)
  v_imgfor    [B, 768]   — image forensic branch (docs/IMAGE_BRANCH.md)
  v_textfor   [B, 4096]  — Qwen3.5-9B layer-30, cached *unprojected*
                           (docs/TEXT_FLUOROSCOPY.md); the 4096→768
                           projection of §4.5 lives here so it trains.

Outputs:
  main_logits [B, 5]    — raw logits for CrossEntropyLoss
  aux_logits  [B, 2]    — binary real/fake image head (training only)
  fused       [B, 1024] — intermediate fused vector (debug / ablation)

Architecture stages (PDF §4.5, §5-6):
  0. TextForensicProjection: v_textfor 4096 → 768 (Linear + GELU).
  1. Independent LayerNorm on each input vector.
  2. Tokenize each vector into T tokens, then three pairwise bidirectional
     cross-attention channels (8 heads). Each pair sums — NOT concatenates —
     both attention directions → [B, 768].
  3. Conv1d → GELU → Conv1d across the 3 relational channels → [B, 1024].
  4. MLP classifier with BatchNorm + GELU → 5 raw logits.
  5. Auxiliary binary head on v_imgfor.detach() (gradient isolation).

Why tokenize (stage 2): attention over a single key is a no-op. With one
token per vector, softmax over one score is always 1, so the output is just
out_proj(W_v · y): the query has no effect and W_q / W_k never receive a
gradient. Expanding each vector into T learned tokens gives every query T
keys to weigh, which is what §5.3's "x queries y" needs to mean anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fnd.models.text_fluoroscopy import TextForensicProjection


# ---------------------------------------------------------------------------
# Sub-module 1 — Vector tokenizer
# ---------------------------------------------------------------------------

class VectorTokenizer(nn.Module):
    """Expand one [B, dim] vector into a [B, T, dim] token sequence.

    A learned Linear(dim → T·dim) produces T different views of the vector,
    plus a learned position embedding so tokens stay distinguishable.
    """

    def __init__(self, dim: int = 768, num_tokens: int = 8):
        super().__init__()
        if num_tokens < 2:
            raise ValueError(
                f"num_tokens must be >= 2, got {num_tokens}: with one token "
                "the attention softmax is always 1 and attention is a no-op"
            )
        self.dim, self.num_tokens = dim, num_tokens
        self.proj = nn.Linear(dim, num_tokens * dim)
        self.pos = nn.Parameter(torch.zeros(num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, dim] → tokens: [B, T, dim]"""
        tokens = self.proj(x).view(x.size(0), self.num_tokens, self.dim)
        return self.norm(tokens + self.pos)


# ---------------------------------------------------------------------------
# Sub-module 2 — Pairwise bidirectional cross-attention
# ---------------------------------------------------------------------------

class PairwiseCrossAttention(nn.Module):
    """Bidirectional cross-attention between two token sequences.

    Both attention directions are summed element-wise (V3 §5.3).

    Direction 1: x queries y  (Q=x, K=y, V=y)
    Direction 2: y queries x  (Q=y, K=x, V=x)
    Each direction's T output tokens are mean-pooled → [B, dim].
    Output:      LayerNorm(dir1 + dir2)  → [B, dim]
    """

    def __init__(self, dim: int = 768, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.attn_x_to_y = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_y_to_x = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_tok: torch.Tensor, y_tok: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tok: [B, T, dim]
            y_tok: [B, T, dim]
        Returns:
            relational vector: [B, dim]
        """
        # Direction 1: x queries y.
        dir1, _ = self.attn_x_to_y(query=x_tok, key=y_tok, value=y_tok,
                                   need_weights=False)
        # Direction 2: y queries x.
        dir2, _ = self.attn_y_to_x(query=y_tok, key=x_tok, value=x_tok,
                                   need_weights=False)

        # Pool tokens, sum both directions (not concatenate) → [B, dim].
        return self.norm(dir1.mean(dim=1) + dir2.mean(dim=1))


# ---------------------------------------------------------------------------
# Sub-module 3 — Three-way fusion (pairwise attention + 1D-Conv)
# ---------------------------------------------------------------------------

class ThreeWayFusion(nn.Module):
    """Three-signal pairwise cross-attention followed by 1D-Conv integration.

    Stages:
      - Project v_textfor 4096 → 768 (§4.5).
      - LayerNorm each input independently (§5.1).
      - Tokenize each signal once; the tokens are shared by both pairs it
        belongs to.
      - Run 3 PairwiseCrossAttention channels (§5.2-5.3):
            Pair 1: semantic ↔ imgfor
            Pair 2: semantic ↔ textfor
            Pair 3: imgfor   ↔ textfor
      - Stack relational vectors → [B, 3, 768], permute → [B, 768, 3] (§5.4).
      - Conv1d(768→768, k=3, p=1) → GELU → Conv1d(768→1024, k=1).
      - AdaptiveAvgPool1d(1) + squeeze → [B, 1024] fused vector.
    """

    def __init__(self, feat_dim: int = 768, fused_dim: int = 1024,
                 num_heads: int = 8, attn_dropout: float = 0.1,
                 text_in_dim: int = 4096, num_tokens: int = 8):
        super().__init__()

        # Stage 0 — text projection owned (and trained) by fusion.
        self.text_proj = (
            TextForensicProjection(text_in_dim, feat_dim)
            if text_in_dim != feat_dim else nn.Identity()
        )

        # Stage 1 — independent LayerNorm per signal.
        self.ln_semantic = nn.LayerNorm(feat_dim)
        self.ln_imgfor   = nn.LayerNorm(feat_dim)
        self.ln_textfor  = nn.LayerNorm(feat_dim)

        # Stage 2 — tokenizers + three pairwise bidirectional channels.
        self.tok_semantic = VectorTokenizer(feat_dim, num_tokens)
        self.tok_imgfor   = VectorTokenizer(feat_dim, num_tokens)
        self.tok_textfor  = VectorTokenizer(feat_dim, num_tokens)

        self.pair_sem_img  = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.pair_sem_text = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.pair_img_text = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)

        # Stage 3 — 1D-Conv across the 3 relational channels.
        # After permute the tensor is [B, feat_dim, 3]:
        #   - axis 1 (feat_dim=768) is the "channels" Conv1d sees
        #   - axis 2 (3)            is the "length" Conv1d slides over
        self.conv1 = nn.Conv1d(
            in_channels=feat_dim, out_channels=feat_dim,
            kernel_size=3, padding=1,   # kernel covers all 3 positions
        )
        self.conv2 = nn.Conv1d(
            in_channels=feat_dim, out_channels=fused_dim,
            kernel_size=1,              # pointwise bottleneck to 1024
        )
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        v_semantic: torch.Tensor,
        v_imgfor: torch.Tensor,
        v_textfor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_semantic: [B, 768]
            v_imgfor:   [B, 768]
            v_textfor:  [B, text_in_dim]
        Returns:
            fused: [B, 1024]
        """
        # --- Text projection, then LayerNorm ---
        s = self.ln_semantic(v_semantic)
        i = self.ln_imgfor(v_imgfor)
        t = self.ln_textfor(self.text_proj(v_textfor))

        # --- Tokenize ---
        s_tok = self.tok_semantic(s)    # [B, T, 768]
        i_tok = self.tok_imgfor(i)
        t_tok = self.tok_textfor(t)

        # --- Pairwise bidirectional cross-attention ---
        r1 = self.pair_sem_img(s_tok, i_tok)    # Pair 1: semantic ↔ image forensic      [B, 768]
        r2 = self.pair_sem_text(s_tok, t_tok)   # Pair 2: semantic ↔ text forensic       [B, 768]
        r3 = self.pair_img_text(i_tok, t_tok)   # Pair 3: image forensic ↔ text forensic [B, 768]

        # --- Stack → [B, 3, 768], permute → [B, 768, 3] ---
        x = torch.stack([r1, r2, r3], dim=1).permute(0, 2, 1)

        # --- 1D-Conv fusion: Conv → GELU → Conv (§5.4) ---
        x = self.conv2(self.act(self.conv1(x)))   # [B, 1024, 3]

        # --- Global average pool across the 3 positions → [B, 1024] ---
        return self.pool(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Sub-module 4 — MLP Classifier (V3 §6)
# ---------------------------------------------------------------------------

class V3Classifier(nn.Module):
    """Five-class MLP head for the fused [B, 1024] vector.

    Layer 1: Linear(1024 → 512) + BatchNorm1d(512) + GELU + Dropout(0.5)
    Layer 2: Linear(512  → 256) + GELU
    Layer 3: Linear(256  →   5) — raw logits, no softmax

    BatchNorm1d cannot train on a batch of 1: use drop_last=True in the
    training DataLoader.
    """

    def __init__(self, in_dim: int = 1024, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused: [B, 1024] → logits: [B, num_classes]"""
        return self.net(fused)


# ---------------------------------------------------------------------------
# Top-level module — V3FusionModule
# ---------------------------------------------------------------------------

class V3FusionModule(nn.Module):
    """Complete V3 fusion + classification head.

    Expects three pre-computed feature vectors and returns main logits and
    aux logits (use aux only for the training loss, §7.1).

    Usage::

        model = V3FusionModule()
        out   = model(v_semantic, v_imgfor, v_textfor)
        loss  = criterion(out["main_logits"], labels)

    Gradient isolation (PDF §5.5):
        The aux classifier receives v_imgfor.detach(), so the aux loss only
        trains the aux head itself and never reaches the fusion layers or
        any upstream encoder.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        fused_dim: int = 1024,
        num_classes: int = 5,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        text_in_dim: int = 4096,
        num_tokens: int = 8,
    ):
        super().__init__()

        self.fusion = ThreeWayFusion(
            feat_dim=feat_dim,
            fused_dim=fused_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            text_in_dim=text_in_dim,
            num_tokens=num_tokens,
        )
        self.classifier = V3Classifier(
            in_dim=fused_dim,
            num_classes=num_classes,
        )

        # Auxiliary binary head: real image (0) vs fake image (1).
        self.aux_classifier = nn.Linear(feat_dim, 2)

    def forward(
        self,
        v_semantic: torch.Tensor,
        v_imgfor: torch.Tensor,
        v_textfor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            v_semantic: [B, 768]
            v_imgfor:   [B, 768]
            v_textfor:  [B, text_in_dim] — raw cached Qwen vector

        Returns dict with keys:
            "main_logits"  [B, 5]    — 5-class logits
            "aux_logits"   [B, 2]    — binary real/fake image logits
            "fused"        [B, 1024] — fused representation
        """
        fused = self.fusion(v_semantic, v_imgfor, v_textfor)
        main_logits = self.classifier(fused)
        aux_logits = self.aux_classifier(v_imgfor.detach())

        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "fused": fused,
        }
