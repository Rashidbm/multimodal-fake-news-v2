"""Text Fluoroscopy: the textual-forensic stream (guidelines section 4).

Reference: Yang et al., "Text Fluoroscopy: Detecting LLM-Generated Text
through Intrinsic Features", EMNLP 2024.

    4.1  text -> tokenizer                      -> input_ids, attention_mask
    4.2  frozen forward, output_hidden_states=True
    4.3  one layer's hidden state               -> (B, S, H)
    4.4  masked mean pooling over the sequence  -> (B, H)
    4.5  Linear(H, 768) + GELU                  -> v_textfor (B, 768)

Nothing here trains, and there is no generation: one forward pass under
no_grad.  4.5 is exported rather than applied, so the fusion stage can own
a projection that trains; extract_textfor.py caches the (B, H) vectors.

The layer choice and the two departures from the guidelines are argued in
docs/TEXT_FLUOROSCOPY.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TextFluoroscopyConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    layer: int = 30               # see "which layer" in docs/TEXT_FLUOROSCOPY.md
    max_len: int = 512            # tokens per caption
    proj_dim: int = 768           # must match v_semantic / v_imgfor
    truncate_layers: bool = False  # off by default; see the note in __init__
    dtype: str = "auto"           # auto | bfloat16 | float16 | float32
    pretrained: bool = True       # False = tiny random Qwen2, used by the unit tests


# ---------------------------------------------------------------------------
# 4.3  which hidden state
# ---------------------------------------------------------------------------

def resolve_layer(num_hidden_states: int, requested: int) -> int:
    """Validate a layer index: hidden_states has num_layers + 1 entries,
    index 0 being the embedding output."""
    num_layers = num_hidden_states - 1
    idx = requested if requested >= 0 else num_hidden_states + requested
    if not 0 <= idx < num_hidden_states:
        raise IndexError(
            f"layer={requested} does not exist: this model has {num_layers} "
            f"transformer layers, so hidden_states has {num_hidden_states} entries "
            f"(0 = embeddings, 1..{num_layers} = layers). Valid: 0..{num_layers} "
            f"or -1..-{num_hidden_states}."
        )
    return idx


# ---------------------------------------------------------------------------
# 4.4  masked mean pooling
# ---------------------------------------------------------------------------

def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, S, H) + (B, S) -> (B, H), ignoring padding.

    The mask is not optional: averaging padding in shrinks each vector by an
    amount that depends on caption length, making length a feature.  See
    test_pooling_is_length_invariant.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"expected (B, S, H), got {tuple(hidden_states.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}"
        )
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)   # (B, S, 1)
    summed = (hidden_states * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                      # (B, 1)
    return summed / counts


# ---------------------------------------------------------------------------
# 4.5  projection to the shared 768-d fusion space
# ---------------------------------------------------------------------------

class TextForensicProjection(nn.Module):
    """Linear(in_dim -> out_dim) + GELU.

    Stage 4's cross-attention needs v_semantic, v_imgfor and v_textfor to
    share one embedding size; CLIP and UnivFD emit 768, Qwen3.5-9B emits 4096.
    No transformers import here, so the fusion module can depend on it
    without pulling in an LLM.
    """

    def __init__(self, in_dim: int = 4096, out_dim: int = 768):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"dims must be positive, got {in_dim} -> {out_dim}")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.fc.weight)   # suits a linear feeding a GELU
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"expected (B, {self.in_dim}), got {tuple(x.shape)} - pool over the "
                "sequence dimension before projecting"
            )
        if x.size(-1) != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {x.size(-1)}")
        return self.act(self.fc(x))

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------

def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"unknown dtype {name!r}")
    if device.type != "cuda":
        return torch.float32              # CPU: fp16 is unsupported for many ops
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16             # RTX 4090, A100
    return torch.float16                  # T4, P100


class TextFluoroscopy(nn.Module):
    """Frozen LLM + masked mean pooling -> (B, hidden_size).

    The projection is not applied here; ``projection()`` returns a correctly
    sized TextForensicProjection for the fusion stage to train.
    """

    def __init__(self, cfg: TextFluoroscopyConfig | None = None, device: torch.device | None = None):
        super().__init__()
        self.cfg = cfg = cfg or TextFluoroscopyConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(cfg.dtype, self.device)

        from transformers import AutoTokenizer, AutoModel

        if cfg.pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            # AutoModel, not AutoModelForCausalLM: no vocabulary logits are
            # needed, so the LM head is never loaded.
            self.model = AutoModel.from_pretrained(cfg.model_name, torch_dtype=self.dtype)
        else:
            # Tiny randomly initialised Qwen2 with the same API, for the tests.
            from transformers import Qwen2Config, Qwen2Model
            self.tokenizer = None
            self.model = Qwen2Model(Qwen2Config(
                vocab_size=256, hidden_size=64, intermediate_size=128,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128,
            ))

        if self.tokenizer is not None:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False       # frozen: feature extraction only

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        self.layer = resolve_layer(self.num_layers + 1, cfg.layer)

        # Dropping the layers above the one we read saves compute and VRAM;
        # the setting must match between cached extraction and live inference.
        #
        # Trap: transformers builds hidden_states as
        # [embeddings, out_1, ..., out_{N-1}, norm(out_N)] - every entry is the
        # raw block output EXCEPT the last, which has the final RMSNorm applied.
        # Cutting the stack would make the layer we read the new last one and
        # silently norm it.  Identity keeps both paths equal; see
        # test_layer_truncation_keeps_the_same_vector.
        # Off by default. At layer 30 of 32 it saves about 6% of the forward
        # pass, which does not buy much against the risk: the equality of the
        # two paths is verified (test_layer_truncation_keeps_the_same_vector)
        # only for a standard decoder stack, and Qwen3.5 is a hybrid of Gated
        # DeltaNet and Gated Attention blocks with sparse MoE. Enable it only
        # after checking that a truncated run reproduces an untruncated one on
        # the model actually in use.
        self._truncated = False
        if cfg.truncate_layers and 0 < self.layer < self.num_layers:
            if not (hasattr(self.model, "layers") and hasattr(self.model, "norm")):
                raise RuntimeError(
                    f"truncate_layers=True but {cfg.model_name} does not expose "
                    "model.layers / model.norm; run with truncation off"
                )
            self.model.layers = self.model.layers[: self.layer]
            self.model.norm = nn.Identity()
            self._truncated = True

    @property
    def read_index(self) -> int:
        """Index into outputs.hidden_states after any truncation."""
        return -1 if self._truncated else self.layer

    def projection(self) -> TextForensicProjection:
        return TextForensicProjection(self.hidden_size, self.cfg.proj_dim)

    def tokenize(self, texts: list[str]) -> dict:
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer: this instance was built with pretrained=False")
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.cfg.max_len, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """One forward pass -> pooled (B, hidden_size)."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        hidden = out.hidden_states[self.read_index]          # (B, S, H)
        return masked_mean_pool(hidden, attention_mask)      # (B, H)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenize(texts)
        return self.encode(enc["input_ids"], enc["attention_mask"])

    def describe(self) -> str:
        kept = f"{self.layer}/{self.num_layers} layers kept" if self._truncated else "all layers kept"
        return (f"{self.cfg.model_name} | hidden={self.hidden_size} layers={self.num_layers} "
                f"| reading hidden_states[{self.layer}] ({kept}) | dtype={self.dtype} "
                f"| device={self.device}")
