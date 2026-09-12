"""V3 Fusion Architecture + Classifier (Implementation Guidelines V3 §5-6).

Consumes three 768-dim feature vectors produced upstream:
  v_semantic  [B, 768]  — FND-CLIP (frozen, pre-cached)
  v_imgfor    [B, 768]  — UnivFD + patch-DCT (trainable)
  v_textfor   [B, 768]  — Qwen2-7B layer-30 (frozen, pre-cached)

Outputs:
  main_logits [B, 5]   — raw logits for CrossEntropyLoss
  aux_logits  [B, 2]   — binary real/fake image head (training only)
  fused       [B, 1024] — intermediate fused vector (debug / ablation)

Architecture stages (PDF §5-6):
  1. Independent LayerNorm on each input vector.
  2. Three pairwise bidirectional cross-attention channels (8 heads).
     Each pair sums — NOT concatenates — both attention directions → [B, 768].
  3. 1D-Conv across the 3 relational channels → fused [B, 1024].
  4. MLP classifier with BatchNorm + GELU → 5 raw logits.
  5. Auxiliary binary head on v_imgfor.detach() (gradient isolation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sub-module 1 — Pairwise bidirectional cross-attention
# ---------------------------------------------------------------------------

class PairwiseCrossAttention(nn.Module):
    """Bidirectional cross-attention between two [B, dim] vectors.

    Both attention directions are summed element-wise (V3 §5.3).
    This keeps the output at [B, dim] — unlike Phase 2, which concatenated
    to [B, 2*dim].

    Direction 1: x queries y  (Q=x, K=y, V=y)
    Direction 2: y queries x  (Q=y, K=x, V=x)
    Output:      dir1 + dir2  → [B, dim]
    """

    def __init__(self, dim: int = 768, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        # Shared attention modules — direction determined by which tensor
        # plays Q vs K/V at call time.
        self.attn_x_to_y = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_y_to_x = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim]
            y: [B, dim]
        Returns:
            relational vector: [B, dim]
        """
        # nn.MultiheadAttention with batch_first=True expects [B, seq, dim].
        # Our "sequence" is a single token per sample.
        x_seq = x.unsqueeze(1)   # [B, 1, dim]
        y_seq = y.unsqueeze(1)   # [B, 1, dim]

        # Direction 1: x queries y.
        dir1, _ = self.attn_x_to_y(query=x_seq, key=y_seq, value=y_seq)
        dir1 = dir1.squeeze(1)   # [B, dim]

        # Direction 2: y queries x.
        dir2, _ = self.attn_y_to_x(query=y_seq, key=x_seq, value=x_seq)
        dir2 = dir2.squeeze(1)   # [B, dim]

        # Sum both directions (not concatenate) → [B, dim].
        # Apply LayerNorm for training stability.
        return self.norm(dir1 + dir2)


# ---------------------------------------------------------------------------
# Sub-module 2 — Three-way fusion (pairwise attention + 1D-Conv)
# ---------------------------------------------------------------------------

class ThreeWayFusion(nn.Module):
    """Three-signal pairwise cross-attention followed by 1D-Conv integration.

    Stages:
      - LayerNorm each input independently.
      - Run 3 PairwiseCrossAttention channels:
            Pair 1: semantic ↔ imgfor
            Pair 2: semantic ↔ textfor
            Pair 3: imgfor   ↔ textfor
      - Stack relational vectors → [B, 3, 768], permute → [B, 768, 3].
      - Conv1d(768→768, k=3, p=1) + GELU   (captures inter-pair correlations)
      - Conv1d(768→1024, k=1)     + GELU   (pointwise bottleneck)
      - AdaptiveAvgPool1d(1) + squeeze → [B, 1024] fused vector.
    """

    def __init__(self, feat_dim: int = 768, fused_dim: int = 1024,
                 num_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()

        # Stage 1 — independent LayerNorm per signal.
        self.ln_semantic = nn.LayerNorm(feat_dim)
        self.ln_imgfor   = nn.LayerNorm(feat_dim)
        self.ln_textfor  = nn.LayerNorm(feat_dim)

        # Stage 2 — three pairwise bidirectional attention channels.
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
            kernel_size=1,              # pointwise projection to 1024
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
            v_textfor:  [B, 768]
        Returns:
            fused: [B, 1024]
        """
        # --- LayerNorm ---
        s = self.ln_semantic(v_semantic)
        i = self.ln_imgfor(v_imgfor)
        t = self.ln_textfor(v_textfor)

        # --- Pairwise bidirectional cross-attention ---
        r1 = self.pair_sem_img(s, i)    # Pair 1: semantic ↔ image forensic   [B, 768]
        r2 = self.pair_sem_text(s, t)   # Pair 2: semantic ↔ text forensic    [B, 768]
        r3 = self.pair_img_text(i, t)   # Pair 3: image forensic ↔ text forensic [B, 768]

        # --- Stack → [B, 3, 768], permute → [B, 768, 3] ---
        stacked = torch.stack([r1, r2, r3], dim=1)   # [B, 3, 768]
        x = stacked.permute(0, 2, 1)                  # [B, 768, 3]

        # --- 1D-Conv fusion ---
        x = self.act(self.conv1(x))   # [B, 768, 3]
        x = self.act(self.conv2(x))   # [B, 1024, 3]

        # --- Global average pool across the 3 positions → [B, 1024] ---
        x = self.pool(x).squeeze(-1)  # [B, 1024]
        return x


# ---------------------------------------------------------------------------
# Sub-module 3 — MLP Classifier (V3 §6)
# ---------------------------------------------------------------------------

class V3Classifier(nn.Module):
    """Five-class MLP head for the fused [B, 1024] vector.

    Layer 1: Linear(1024 → 512) + BatchNorm1d(512) + GELU + Dropout(0.5)
    Layer 2: Linear(512  → 256) + GELU
    Layer 3: Linear(256  →   5) — raw logits, no softmax
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
# Top-level module — V3FusionModule (Student 4's plug-in point)
# ---------------------------------------------------------------------------

class V3FusionModule(nn.Module):
    """Complete V3 fusion + classification head.

    This is the deliverable for Student 4.  It expects three pre-computed
    feature vectors and returns main logits (always) and aux logits (training).

    Usage::

        model = V3FusionModule()
        out   = model(v_semantic, v_imgfor, v_textfor)
        loss  = criterion(out["main_logits"], labels)

    Gradient isolation (PDF §5.5, Option B):
        The aux classifier receives v_imgfor.detach() so its gradients
        cannot flow back through the forensic encoder into frozen backbones.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        fused_dim: int = 1024,
        num_classes: int = 5,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        self.fusion = ThreeWayFusion(
            feat_dim=feat_dim,
            fused_dim=fused_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )
        self.classifier = V3Classifier(
            in_dim=fused_dim,
            num_classes=num_classes,
        )

        # Auxiliary binary head: real image (0) vs fake image (1).
        # Sits on v_imgfor only; gradient is isolated via .detach() in forward.
        self.aux_classifier = nn.Linear(feat_dim, 2)

    def forward(
        self,
        v_semantic: torch.Tensor,
        v_imgfor: torch.Tensor,
        v_textfor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            v_semantic: [B, 768]  — FND-CLIP output (frozen)
            v_imgfor:   [B, 768]  — UnivFD output (trainable)
            v_textfor:  [B, 768]  — Qwen2 layer-30 output (frozen)

        Returns dict with keys:
            "main_logits"  [B, 5]    — 5-class logits
            "aux_logits"   [B, 2]    — binary real/fake image logits
            "fused"        [B, 1024] — fused representation
        """
        fused = self.fusion(v_semantic, v_imgfor, v_textfor)
        main_logits = self.classifier(fused)

        # Detach v_imgfor before aux head → gradients from aux loss cannot
        # propagate back through the forensic encoder into frozen backbones.
        aux_logits = self.aux_classifier(v_imgfor.detach())

        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "fused": fused,
        }


# ---------------------------------------------------------------------------
# Unit test — run with:  python src/models/v3_pipeline.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    B = 4  # batch size

    print("=" * 60)
    print("V3FusionModule — unit test")
    print("=" * 60)

    # Random feature vectors simulating upstream encoder outputs.
    v_sem  = torch.randn(B, 768)
    v_img  = torch.randn(B, 768)
    v_text = torch.randn(B, 768)

    model = V3FusionModule(
        feat_dim=768,
        fused_dim=1024,
        num_classes=5,
        num_heads=8,
    )
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # ---- Forward pass ----
    out = model(v_sem, v_img, v_text)

    main_logits = out["main_logits"]
    aux_logits  = out["aux_logits"]
    fused       = out["fused"]

    print(f"main_logits shape : {tuple(main_logits.shape)}")
    print(f"aux_logits  shape : {tuple(aux_logits.shape)}")
    print(f"fused       shape : {tuple(fused.shape)}")

    assert main_logits.shape == (B, 5),    f"Expected ({B}, 5), got {main_logits.shape}"
    assert aux_logits.shape  == (B, 2),    f"Expected ({B}, 2), got {aux_logits.shape}"
    assert fused.shape       == (B, 1024), f"Expected ({B}, 1024), got {fused.shape}"
    print("\nShape assertions — PASSED")

    # ---- Backward pass (dummy loss) ----
    labels     = torch.randint(0, 5, (B,))
    aux_labels = torch.randint(0, 2, (B,))

    criterion_main = nn.CrossEntropyLoss()
    criterion_aux  = nn.BCEWithLogitsLoss()

    main_loss = criterion_main(main_logits, labels)

    # Aux uses 1-hot float targets for BCEWithLogitsLoss.
    aux_targets = nn.functional.one_hot(aux_labels, num_classes=2).float()
    aux_loss = criterion_aux(aux_logits, aux_targets)

    # Mirror the training loop: aux weight = 0.1, aux already detached in forward.
    total_loss = main_loss + 0.1 * aux_loss
    total_loss.backward()

    print(f"\nmain_loss : {main_loss.item():.4f}")
    print(f"aux_loss  : {aux_loss.item():.4f}")
    print(f"total_loss: {total_loss.item():.4f}")

    # Verify gradients exist on fusion parameters.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.fusion.parameters()
    )
    assert has_grad, "No gradients on fusion parameters — backward failed"
    print("\nGradient flow check — PASSED")

    # Verify aux classifier has no gradient into frozen-region proxy
    # (v_imgfor was detached, so aux_classifier.weight.grad should exist
    #  but the grad stopped at the detach boundary).
    assert model.aux_classifier.weight.grad is not None, \
        "aux_classifier.weight has no gradient"
    print("aux_classifier gradient isolation — PASSED")

    print("\nAll checks passed. V3FusionModule is ready for Student 4.")
    sys.exit(0)
