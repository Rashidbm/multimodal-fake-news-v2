"""V3 Fusion: combines the three branch vectors and predicts one of 5 classes.

Follows Implementation Guidelines V3, sections 4.5, 5 and 6.

Inputs (one row per post, B = batch size):
  v_semantic  [B, 768]   from the semantic branch
  v_imgfor    [B, 768]   from the image branch
  v_textfor   [B, 4096]  from the text branch (Qwen3.5-9B, layer 30)

Outputs (a dict):
  main_logits [B, 5]     one score per class (softmax turns them into %)
  aux_logits  [B, 2]     image real/fake; only used for the training loss
  fused       [B, 1024]  the combined vector, for debugging

The 7 steps (same as slide 14):
  1. Text projection: shrink v_textfor from 4096 to 768, so all three match.
  2. LayerNorm: put each vector on the same scale.
  3. Tokenize: turn each vector into 8 tokens.
  4. Cross-attention: compare the vectors in 3 pairs, both directions,
     and add the two directions together.
  5. 1D-conv: combine the 3 pair results into one 1024 vector.
  6. MLP classifier: 1024 -> 512 -> 256 -> 5 class scores.
  7. Aux head: guesses if the image is fake, from a detached copy of
     v_imgfor, so its loss never changes the rest of the model.

Why step 3 matters: attention picks which keys to focus on. With only one
token per vector there is only one key, so softmax always gives it 100% and
the attention does nothing. With 8 tokens there are 8 keys to choose from.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fnd.models.text_fluoroscopy import TextForensicProjection


class VectorTokenizer(nn.Module):
    """Step 3: turn one vector [B, 768] into 8 tokens [B, 8, 768].

    A learned layer makes 8 different views of the vector, and a position
    embedding keeps the 8 tokens different from each other.
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
        """x: [B, 768] -> tokens: [B, 8, 768]"""
        tokens = self.proj(x).view(x.size(0), self.num_tokens, self.dim)
        return self.norm(tokens + self.pos)


class PairwiseCrossAttention(nn.Module):
    """Step 4, for one pair of vectors x and y.

    Direction 1: x asks y  (query = x, key and value = y)
    Direction 2: y asks x  (query = y, key and value = x)
    Each direction's 8 tokens are averaged into one vector, then the two
    directions are added (not joined side by side) -> [B, 768].
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
        """x_tok, y_tok: [B, 8, 768] -> pair result: [B, 768]"""
        dir1, _ = self.attn_x_to_y(query=x_tok, key=y_tok, value=y_tok,
                                   need_weights=False)
        dir2, _ = self.attn_y_to_x(query=y_tok, key=x_tok, value=x_tok,
                                   need_weights=False)
        return self.norm(dir1.mean(dim=1) + dir2.mean(dim=1))


class ThreeWayFusion(nn.Module):
    """Steps 1-5: from the three branch vectors to one fused vector [B, 1024].

    The 3 pairs:
      semantic <-> image   does the picture fit the story?
      semantic <-> text    does the writing fit the story?
      image    <-> text    do the image and text fingerprints agree?
    """

    def __init__(self, feat_dim: int = 768, fused_dim: int = 1024,
                 num_heads: int = 8, attn_dropout: float = 0.1,
                 text_in_dim: int = 4096, num_tokens: int = 8):
        super().__init__()

        # Step 1: text projection 4096 -> 768 (trains together with fusion).
        self.text_proj = (
            TextForensicProjection(text_in_dim, feat_dim)
            if text_in_dim != feat_dim else nn.Identity()
        )

        # Step 2: one LayerNorm per branch.
        self.ln_semantic = nn.LayerNorm(feat_dim)
        self.ln_imgfor   = nn.LayerNorm(feat_dim)
        self.ln_textfor  = nn.LayerNorm(feat_dim)

        # Step 3: one tokenizer per branch.
        self.tok_semantic = VectorTokenizer(feat_dim, num_tokens)
        self.tok_imgfor   = VectorTokenizer(feat_dim, num_tokens)
        self.tok_textfor  = VectorTokenizer(feat_dim, num_tokens)

        # Step 4: one cross-attention per pair.
        self.pair_sem_img  = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.pair_sem_text = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.pair_img_text = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)

        # Step 5: 1D-conv that slides across the 3 pair results.
        self.conv1 = nn.Conv1d(
            in_channels=feat_dim, out_channels=feat_dim,
            kernel_size=3, padding=1,   # looks at all 3 pairs together
        )
        self.conv2 = nn.Conv1d(
            in_channels=feat_dim, out_channels=fused_dim,
            kernel_size=1,              # 768 -> 1024
        )
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        v_semantic: torch.Tensor,
        v_imgfor: torch.Tensor,
        v_textfor: torch.Tensor,
    ) -> torch.Tensor:
        """Three branch vectors -> fused vector [B, 1024]."""
        # Steps 1-2: same size, same scale.
        s = self.ln_semantic(v_semantic)
        i = self.ln_imgfor(v_imgfor)
        t = self.ln_textfor(self.text_proj(v_textfor))

        # Step 3: 8 tokens each.
        s_tok = self.tok_semantic(s)
        i_tok = self.tok_imgfor(i)
        t_tok = self.tok_textfor(t)

        # Step 4: compare the 3 pairs.
        r1 = self.pair_sem_img(s_tok, i_tok)
        r2 = self.pair_sem_text(s_tok, t_tok)
        r3 = self.pair_img_text(i_tok, t_tok)

        # Step 5: line the 3 results up [B, 768, 3], conv, then average -> [B, 1024].
        x = torch.stack([r1, r2, r3], dim=1).permute(0, 2, 1)
        x = self.conv2(self.act(self.conv1(x)))
        return self.pool(x).squeeze(-1)


class V3Classifier(nn.Module):
    """Step 6: fused vector [B, 1024] -> 5 class scores.

    Train with drop_last=True: BatchNorm cannot train on a batch of 1.
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
        return self.net(fused)


class V3FusionModule(nn.Module):
    """The full fusion model (steps 1-7).

    Usage::

        model = V3FusionModule()
        out   = model(v_semantic, v_imgfor, v_textfor)
        loss  = criterion(out["main_logits"], labels)
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

        # Step 7: aux head, image real (0) vs fake (1).
        self.aux_classifier = nn.Linear(feat_dim, 2)

    def forward(
        self,
        v_semantic: torch.Tensor,
        v_imgfor: torch.Tensor,
        v_textfor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.fusion(v_semantic, v_imgfor, v_textfor)
        main_logits = self.classifier(fused)
        # detach(): the aux loss trains only the aux head, nothing else.
        aux_logits = self.aux_classifier(v_imgfor.detach())

        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "fused": fused,
        }
