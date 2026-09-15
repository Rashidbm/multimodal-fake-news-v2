"""Text Fluoroscopy: the textual-forensic stream (guidelines section 4).

Reference: Yang et al., "Text Fluoroscopy: Detecting LLM-Generated Text
through Intrinsic Features", EMNLP 2024.

    4.1  text -> tokenizer                      -> input_ids, attention_mask
    4.2  frozen forward, output_hidden_states=True
    4.3  one layer's hidden state               -> (B, S, H)
    4.4  masked mean pooling over the sequence  -> (B, H)
    4.5  Linear(H, 768) + GELU                  -> v_textfor (B, 768)

Nothing here trains, and there is no generation: one forward pass under
no_grad.  4.5 is exported rather than applied, so the fusion stage can own a
projection that trains; extract_textfor.py caches the (B, H) vectors.

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
    max_len: int = 96             # longest caption measured on this dataset: 45 tokens
    proj_dim: int = 768           # must match v_semantic / v_imgfor
    dtype: str = "auto"           # auto | bfloat16 | float16 | float32
    pretrained: bool = True       # False = tiny random Qwen2, used by the unit tests


# ---------------------------------------------------------------------------
# 4.3  which hidden state
# ---------------------------------------------------------------------------

def resolve_layer(num_hidden_states: int, requested: int) -> int:
    """Validate a layer index: hidden_states has num_layers + 1 entries,
    index 0 being the embedding output.

    The guidelines say "layer 30". That exists on a 32-layer model (Qwen3.5-9B,
    and the paper's gte-Qwen1.5-7B) and does not on a 28-layer one (Qwen2-7B).
    Checked here rather than assumed, so a wrong index fails in the first
    second instead of an hour into a run.
    """
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
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# 4.5  projection to the shared 768-d fusion space
# ---------------------------------------------------------------------------

class TextForensicProjection(nn.Module):
    """Linear(in_dim -> out_dim) + GELU.

    Stage 4's cross-attention needs v_semantic, v_imgfor and v_textfor to
    share one embedding size; CLIP and UnivFD emit 768, Qwen3.5-9B emits 4096.
    No transformers import here, so the fusion module can depend on it without
    pulling in an LLM.
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


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------

def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name in ("float32", "float16", "bfloat16"):
        return getattr(torch, name)
    if name != "auto":
        raise ValueError(f"unknown dtype {name!r}")
    if device.type != "cuda":
        return torch.float32                  # CPU: fp16 is unsupported for many ops
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def config_int(cfg, *names):
    """Find an integer field on a config, including nested sub-configs.

    Not every architecture puts hidden_size and num_hidden_layers at the top
    level. Qwen3.5 is heterogeneous - a mix of Gated DeltaNet and Gated
    Attention blocks - and nests them, so `config.hidden_size` raises
    AttributeError. Returns None rather than guessing.
    """
    for n in names:
        v = getattr(cfg, n, None)
        if isinstance(v, int):
            return v
    subs = []
    getter = getattr(cfg, "get_text_config", None)
    if callable(getter):
        try:
            subs.append(getter())
        except Exception:
            pass
    for attr in ("text_config", "llm_config", "language_model", "decoder"):
        sub = getattr(cfg, attr, None)
        if sub is not None:
            subs.append(sub)
    for sub in subs:
        for n in names:
            v = getattr(sub, n, None)
            if isinstance(v, int):
                return v
    return None


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

        from transformers import AutoModel, AutoTokenizer

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

        # Measure the dimensions from a real forward pass rather than trusting
        # the config to expose them. One short pass, and it is correct for any
        # architecture including nested and heterogeneous ones.
        self.num_layers, self.hidden_size = self.measure_shape()
        self.layer = resolve_layer(self.num_layers + 1, cfg.layer)

        # Cross-check against the config where it does expose them, so a
        # mismatch is visible rather than silent.
        cfg_h = config_int(self.model.config, "hidden_size", "d_model", "n_embd")
        cfg_l = config_int(self.model.config, "num_hidden_layers", "n_layer", "num_layers")
        self.config_agrees = (cfg_h in (None, self.hidden_size)
                              and cfg_l in (None, self.num_layers))

    @torch.no_grad()
    def measure_shape(self) -> tuple[int, int]:
        """(num_layers, hidden_size), read off an actual forward pass."""
        if self.tokenizer is not None:
            enc = self.tokenizer("probe", return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
        else:
            enc = {"input_ids": torch.tensor([[1, 2]], device=self.device)}
        hs = self.model(**enc, output_hidden_states=True).hidden_states
        if not hs:
            raise RuntimeError(f"{self.cfg.model_name} returned no hidden_states")
        return len(hs) - 1, int(hs[-1].shape[-1])

    def projection(self) -> TextForensicProjection:
        return TextForensicProjection(self.hidden_size, self.cfg.proj_dim)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """One forward pass -> pooled (B, hidden_size)."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        return masked_mean_pool(out.hidden_states[self.layer], attention_mask)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> tuple[torch.Tensor, list[int]]:
        """Tokenize and encode -> (pooled (B, H), untruncated token lengths).

        The lengths are returned so the caller can record which captions were
        truncated, which the guidelines ask for explicitly.
        """
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer: this instance was built with pretrained=False")
        lengths = [len(x) for x in self.tokenizer(texts, padding=False,
                                                  truncation=False)["input_ids"]]
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.cfg.max_len, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.encode(enc["input_ids"], enc["attention_mask"]), lengths

    def describe(self) -> str:
        note = "" if self.config_agrees else "  [config reports different dims - measured wins]"
        return (f"{self.cfg.model_name} | hidden={self.hidden_size} layers={self.num_layers} "
                f"| reading hidden_states[{self.layer}] | dtype={self.dtype} "
                f"| device={self.device}{note}")
