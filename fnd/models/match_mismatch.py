"""The match/mismatch stream: does this caption belong to this picture?

Two interchangeable frozen backbones answer the same question, so the project
can pick one on evidence rather than on reputation:

    clip    CLIP image encoder + CLIP text encoder, two separate towers that
            never see each other.  The pair is described by its two embeddings
            and by how they interact (cosine, product, absolute difference).
    qwenvl  Qwen2-VL / Qwen2.5-VL, one decoder that reads the picture and the
            caption in a single sequence, so the caption tokens can attend to
            the image tokens.  The pair is described by one hidden state read
            out of that joint pass.

Both are frozen and neither is asked for a verdict: each returns a vector, and
a small head trained separately (``fnd.probe_match``) turns vectors into
scores.  That is what makes the comparison fair - same rows, same splits, same
head, same protocol, only the encoder differs.  ``fnd.compare_match`` runs it.

The architectural bet being tested: CLIP scores agreement *after* each side has
been summarised independently, which is exactly the operation an out-of-context
pair is built to defeat (both halves are individually plausible - only the
pairing is wrong).  A VLM reads both together, so in principle it can notice
that the caption's named entity is not the thing in the photograph.  Whether
that principle survives contact with this dataset is an empirical question, and
this module exists to answer it.

Nothing here trains and nothing generates text: one forward pass under
no_grad.  See docs/MATCH_MISMATCH.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fnd.models.text_fluoroscopy import (
    TextForensicProjection,
    masked_mean_pool,
    resolve_dtype,
    resolve_layer,
)

# Feature compositions available to the CLIP backbone.  'interaction' is the
# default and the only one used for the headline comparison; the other two
# exist so the ablation can show what each part contributes.
CLIP_FEATURE_MODES = ("interaction", "concat", "sim")

# Pooling strategies for the VLM backbone.
POOLINGS = ("last", "masked_mean")

DEFAULT_MODELS = {
    "clip": "openai/clip-vit-large-patch14",
    "qwenvl": "Qwen/Qwen2.5-VL-3B-Instruct",
}

# One fixed prompt for every sample.  Held constant on purpose: a prompt that
# varied with the row would put information into the features that the encoder
# did not have to read out of the picture.
DEFAULT_PROMPT = "Caption: {text}\nDoes this caption describe the image?"


@dataclass
class MatchConfig:
    backbone: str = "clip"              # 'clip' | 'qwenvl'
    model_name: str = ""                # "" = DEFAULT_MODELS[backbone]
    layer: int = -1                     # qwenvl: index into hidden_states, -1 = last
    max_len: int = 77                   # clip's text encoder is fixed at 77
    pooling: str = "last"               # qwenvl only; see POOLINGS
    features: str = "interaction"       # clip only; see CLIP_FEATURE_MODES
    prompt: str = DEFAULT_PROMPT        # qwenvl only, must contain {text}
    max_pixels: int = 401408            # qwenvl: 512*28*28, caps the image token count
    proj_dim: int = 768                 # the shared fusion width, exported not applied
    dtype: str = "auto"                 # auto | bfloat16 | float16 | float32
    pretrained: bool = True             # False = tiny random weights, for the tests

    def __post_init__(self):
        if self.backbone not in DEFAULT_MODELS:
            raise ValueError(f"backbone must be one of {tuple(DEFAULT_MODELS)}, got {self.backbone!r}")
        if self.features not in CLIP_FEATURE_MODES:
            raise ValueError(f"features must be one of {CLIP_FEATURE_MODES}, got {self.features!r}")
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if "{text}" not in self.prompt:
            raise ValueError("prompt must contain {text}, otherwise the caption is never shown")
        if not self.model_name:
            self.model_name = DEFAULT_MODELS[self.backbone]


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------

def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, S, H) + (B, S) -> (B, H), taking the last non-padding position.

    The right read-out for a causal decoder.  Attention only ever looks left,
    so in a sequence that runs ``<image tokens> <caption tokens>`` the image
    positions have not seen the caption; averaging them in dilutes the one
    position that has seen both.  Mean pooling is still offered
    (``pooling='masked_mean'``) because it is what the textual-forensic stream
    uses, and the ablation should be able to vary one thing at a time.

    The index is found from the mask rather than from ``[:, -1]``, so the
    result is the same whether the processor pads on the left or the right.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"expected (B, S, H), got {tuple(hidden_states.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}"
        )
    mask = attention_mask.to(torch.int64)
    if not (mask.sum(dim=1) > 0).all():
        raise ValueError("a row of attention_mask is all zeros: that sample has no tokens")
    # last index whose mask is 1, for either padding side
    idx = mask.shape[1] - 1 - mask.flip(1).argmax(dim=1)
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), idx]


def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, how: str) -> torch.Tensor:
    if how == "last":
        return last_token_pool(hidden_states, attention_mask)
    if how == "masked_mean":
        return masked_mean_pool(hidden_states, attention_mask)
    raise ValueError(f"unknown pooling {how!r}, expected one of {POOLINGS}")


# ---------------------------------------------------------------------------
# CLIP feature composition
# ---------------------------------------------------------------------------

def clip_pair_features(image_emb: torch.Tensor, text_emb: torch.Tensor,
                       mode: str = "interaction") -> tuple[torch.Tensor, torch.Tensor]:
    """Two (B, D) CLIP embeddings -> (features, cosine similarity).

    Both embeddings are L2-normalised first, so the product and difference
    terms describe direction only and cannot be dominated by one vector simply
    having a larger norm.

    ``interaction`` = ``[img, txt, img * txt, |img - txt|, cos]`` -> 4D + 1.
    The two interaction terms are the point of the whole stream: ``img * txt``
    is per-dimension agreement (its sum *is* the cosine, so the head can learn
    to weight the dimensions the cosine treats equally), and ``|img - txt|``
    is per-dimension disagreement.  Handing over only ``[img, txt]`` would
    leave a two-layer head to discover multiplication on its own; handing over
    only the cosine collapses the entire pair into one number.  Both of those
    are available as ``concat`` and ``sim`` so the ablation can prove it.
    """
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"embeddings differ in shape: {tuple(image_emb.shape)} vs {tuple(text_emb.shape)}")
    if image_emb.dim() != 2:
        raise ValueError(f"expected (B, D) embeddings, got {tuple(image_emb.shape)}")

    img = torch.nn.functional.normalize(image_emb.float(), dim=-1)
    txt = torch.nn.functional.normalize(text_emb.float(), dim=-1)
    sim = (img * txt).sum(dim=-1)                                   # (B,)

    if mode == "sim":
        return sim.unsqueeze(-1), sim
    if mode == "concat":
        return torch.cat([img, txt], dim=-1), sim
    if mode == "interaction":
        return torch.cat([img, txt, img * txt, (img - txt).abs(), sim.unsqueeze(-1)], dim=-1), sim
    raise ValueError(f"unknown feature mode {mode!r}, expected one of {CLIP_FEATURE_MODES}")


def clip_feature_dim(embed_dim: int, mode: str) -> int:
    return {"sim": 1, "concat": 2 * embed_dim, "interaction": 4 * embed_dim + 1}[mode]


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------

class MatchEncoder(nn.Module):
    """Frozen backbone -> one vector per (caption, image) pair.

    Subclasses implement ``encode_pairs``.  Both report ``feature_dim``, so the
    extractor and the probe never have to know which backbone produced a file.
    """

    feature_dim: int

    def __init__(self, cfg: MatchConfig, device: torch.device | None = None):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(cfg.dtype, self.device)

    def freeze(self, module: nn.Module) -> None:
        module.to(self.device).eval()
        for p in module.parameters():
            p.requires_grad = False

    def projection(self) -> TextForensicProjection:
        """Linear(feature_dim -> 768) + GELU for the fusion stage to own and
        train.  Exported, never applied here: the cached vectors stay raw, the
        same decision the textual-forensic stream made."""
        return TextForensicProjection(self.feature_dim, self.cfg.proj_dim)

    def encode_pairs(self, texts: list[str], images: list) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class CLIPMatchEncoder(MatchEncoder):
    """Two frozen CLIP towers, combined by ``clip_pair_features``.

    Returns ``similarity`` alongside the features because the raw cosine is a
    zero-shot detector in its own right, and the probe reports it as the
    "no training at all" line beside the trained head.
    """

    def __init__(self, cfg: MatchConfig | None = None, device: torch.device | None = None,
                 model=None, processor=None):
        cfg = cfg or MatchConfig(backbone="clip")
        if cfg.backbone != "clip":
            raise ValueError(f"CLIPMatchEncoder needs backbone='clip', got {cfg.backbone!r}")
        super().__init__(cfg, device)

        if model is not None:
            self.model, self.processor = model, processor
        elif cfg.pretrained:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(cfg.model_name)
            # The processor, not a hand-written resize: CLIP variants differ in
            # input resolution and crop, and getting that wrong degrades the
            # embeddings quietly rather than loudly.
            self.processor = CLIPProcessor.from_pretrained(cfg.model_name)
        else:
            from transformers import CLIPConfig, CLIPModel
            self.model = CLIPModel(CLIPConfig.from_dict({
                "text_config": {"hidden_size": 32, "intermediate_size": 37, "num_hidden_layers": 2,
                                "num_attention_heads": 2, "vocab_size": 99, "max_position_embeddings": 77},
                "vision_config": {"hidden_size": 32, "intermediate_size": 37, "num_hidden_layers": 2,
                                 "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
                "projection_dim": 16,
            }))
            self.processor = None

        self.freeze(self.model)
        self.embed_dim = int(self.model.config.projection_dim)
        self.feature_dim = clip_feature_dim(self.embed_dim, cfg.features)

    @torch.no_grad()
    def encode_tensors(self, pixel_values, input_ids, attention_mask) -> dict:
        """The tensor-level entry point, so the tests can drive the encoder
        without a processor and without downloading weights."""
        img = self.model.get_image_features(pixel_values=pixel_values.to(self.device))
        txt = self.model.get_text_features(input_ids=input_ids.to(self.device),
                                           attention_mask=attention_mask.to(self.device))
        img, txt = self._embedding(img), self._embedding(txt)
        feats, sim = clip_pair_features(img, txt, self.cfg.features)
        return {"features": feats, "similarity": sim}

    def _embedding(self, out) -> torch.Tensor:
        """transformers < 5 returns a tensor here; >= 5 returns an output
        object.  Accept both and check the width, the same way fnd_clip does."""
        if not torch.is_tensor(out):
            pooled = getattr(out, "pooler_output", None)
            out = pooled if pooled is not None else out[0]
        if out.shape[-1] != self.model.config.projection_dim:
            raise RuntimeError(
                f"CLIP embedding has size {out.shape[-1]}, expected {self.model.config.projection_dim}")
        return out

    @torch.no_grad()
    def encode_pairs(self, texts: list[str], images: list) -> dict:
        if self.processor is None:
            raise RuntimeError("no processor: this instance was built with pretrained=False; "
                               "call encode_tensors instead")
        enc = self.processor(text=texts, images=images, return_tensors="pt",
                             padding="max_length", truncation=True, max_length=self.cfg.max_len)
        return self.encode_tensors(enc["pixel_values"], enc["input_ids"], enc["attention_mask"])

    def describe(self) -> str:
        return (f"{self.cfg.model_name} | two frozen towers, embed={self.embed_dim} "
                f"| features={self.cfg.features} -> {self.feature_dim} "
                f"| dtype={self.dtype} | device={self.device}")


class QwenVLMatchEncoder(MatchEncoder):
    """One frozen vision-language decoder reading picture and caption together.

    ``model`` and ``processor`` can be injected, which is how the tests exercise
    the prompt building and the pooling without downloading a VLM.
    """

    def __init__(self, cfg: MatchConfig | None = None, device: torch.device | None = None,
                 model=None, processor=None):
        cfg = cfg or MatchConfig(backbone="qwenvl", model_name=DEFAULT_MODELS["qwenvl"])
        if cfg.backbone != "qwenvl":
            raise ValueError(f"QwenVLMatchEncoder needs backbone='qwenvl', got {cfg.backbone!r}")
        super().__init__(cfg, device)

        if model is not None:
            self.model, self.processor = model, processor
        else:
            if not cfg.pretrained:
                raise ValueError("pretrained=False is not supported for qwenvl: there is no small "
                                 "stand-in VLM. Inject model= and processor= instead (the tests do).")
            from transformers import AutoModelForVision2Seq, AutoProcessor
            # max_pixels caps how many patches a large photograph becomes.
            # Uncapped, one 4K image can expand into thousands of tokens and
            # the batch no longer fits.
            self.processor = AutoProcessor.from_pretrained(cfg.model_name, max_pixels=cfg.max_pixels)
            self.model = AutoModelForVision2Seq.from_pretrained(cfg.model_name, torch_dtype=self.dtype)

        self.freeze(self.model)
        if getattr(self.processor, "tokenizer", None) is not None:
            tok = self.processor.tokenizer
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

        self.num_layers, self.hidden_size = self.measure_shape()
        self.layer = resolve_layer(self.num_layers + 1, cfg.layer)
        self.feature_dim = self.hidden_size

    @torch.no_grad()
    def measure_shape(self) -> tuple[int, int]:
        """(num_layers, hidden_size), read off the model's own config, falling
        back to a text-only forward pass.  Sending a probe image through a VLM
        is expensive enough that the config is worth trusting here when it
        answers; a disagreement would show up immediately as a shape error on
        the first real batch."""
        from fnd.models.text_fluoroscopy import config_int

        cfg_h = config_int(self.model.config, "hidden_size", "d_model", "n_embd")
        cfg_l = config_int(self.model.config, "num_hidden_layers", "n_layer", "num_layers")
        if isinstance(cfg_h, int) and isinstance(cfg_l, int):
            return cfg_l, cfg_h

        enc = self.processor.tokenizer("probe", return_tensors="pt")
        hs = self.model(**{k: v.to(self.device) for k, v in enc.items()},
                        output_hidden_states=True).hidden_states
        if not hs:
            raise RuntimeError(f"{self.cfg.model_name} returned no hidden_states")
        return len(hs) - 1, int(hs[-1].shape[-1])

    def build_prompts(self, texts: list[str]) -> list[str]:
        """One chat-templated prompt per caption, image placeholder included.

        The chat template matters: a VLM instruction-tuned on a specific
        wrapper produces different activations without it, and the whole stream
        is read out of those activations.
        """
        messages = [
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": self.cfg.prompt.format(text=t)}]}]
            for t in texts
        ]
        return [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages]

    @torch.no_grad()
    def encode_pairs(self, texts: list[str], images: list) -> dict:
        if len(texts) != len(images):
            raise ValueError(f"{len(texts)} captions but {len(images)} images")
        enc = self.processor(text=self.build_prompts(texts), images=images,
                             return_tensors="pt", padding=True)
        enc = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[self.layer]
        pooled = pool(hidden, enc["attention_mask"], self.cfg.pooling)
        # No cosine exists here: the two modalities were never embedded apart.
        return {"features": pooled, "similarity": None}

    def describe(self) -> str:
        return (f"{self.cfg.model_name} | one joint decoder, hidden={self.hidden_size} "
                f"layers={self.num_layers} | reading hidden_states[{self.layer}] "
                f"pooling={self.cfg.pooling} | dtype={self.dtype} | device={self.device}")


def build_encoder(cfg: MatchConfig, device: torch.device | None = None, **kwargs) -> MatchEncoder:
    """The one place that maps a backbone name to a class."""
    if cfg.backbone == "clip":
        return CLIPMatchEncoder(cfg, device, **kwargs)
    if cfg.backbone == "qwenvl":
        return QwenVLMatchEncoder(cfg, device, **kwargs)
    raise ValueError(f"unknown backbone {cfg.backbone!r}")
