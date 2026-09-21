"""Match/mismatch tests: no CLIP download, no VLM, no GPU.

The things worth pinning down in a comparison experiment are the ones that
would make the comparison a lie rather than merely wrong:

  - the pooling reads the position that actually saw both modalities,
  - the CLIP feature composition means what the docstring says it means,
  - the extractor refuses a missing image instead of quietly skipping the row,
  - the probe joins features to labels by id, not by row order,
  - the comparison refuses two feature files that cover different rows.

The VLM is exercised through an injected stub model, so the prompt building
and the read-out are tested without downloading several gigabytes.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from fnd.data.records import GROUPS  # noqa: E402
from fnd.models.match_mismatch import (  # noqa: E402
    CLIPMatchEncoder,
    MatchConfig,
    QwenVLMatchEncoder,
    build_encoder,
    clip_feature_dim,
    clip_pair_features,
    last_token_pool,
    pool,
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        MatchConfig(backbone="resnet")
    with pytest.raises(ValueError):
        MatchConfig(backbone="clip", features="magic")
    with pytest.raises(ValueError):
        MatchConfig(backbone="qwenvl", pooling="first")
    with pytest.raises(ValueError, match="must contain"):
        MatchConfig(backbone="qwenvl", prompt="does this match?")


def test_config_fills_in_the_default_model():
    assert MatchConfig(backbone="clip").model_name.startswith("openai/clip")
    assert "VL" in MatchConfig(backbone="qwenvl").model_name


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------

def test_last_token_pool_takes_the_last_real_token():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])       # right padding
    out = last_token_pool(hidden, mask)
    assert torch.equal(out[0], hidden[0, 2])
    assert torch.equal(out[1], hidden[1, 1])


def test_last_token_pool_handles_left_padding():
    """Processors differ on which side they pad; [:, -1] would be right for one
    and silently wrong for the other."""
    hidden = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    out = last_token_pool(hidden, torch.tensor([[0, 0, 1, 1]]))
    assert torch.equal(out[0], hidden[0, 3])


def test_last_token_pool_is_padding_invariant():
    torch.manual_seed(0)
    hidden = torch.randn(1, 2, 5)
    padded = torch.cat([hidden, torch.randn(1, 3, 5)], dim=1)
    short = last_token_pool(hidden, torch.ones(1, 2, dtype=torch.long))
    long = last_token_pool(padded, torch.tensor([[1, 1, 0, 0, 0]]))
    assert torch.allclose(short, long)


def test_pool_rejects_empty_rows_and_unknown_modes():
    hidden = torch.randn(1, 3, 4)
    with pytest.raises(ValueError, match="all zeros"):
        last_token_pool(hidden, torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="unknown pooling"):
        pool(hidden, torch.ones(1, 3, dtype=torch.long), "median")


def test_pool_dispatches_to_both_strategies():
    hidden = torch.randn(2, 3, 6)
    mask = torch.ones(2, 3, dtype=torch.long)
    assert pool(hidden, mask, "last").shape == (2, 6)
    mean = pool(hidden, mask, "masked_mean")
    assert torch.allclose(mean, hidden.mean(dim=1), atol=1e-6)


# ---------------------------------------------------------------------------
# CLIP feature composition
# ---------------------------------------------------------------------------

def test_interaction_features_have_the_documented_width():
    img, txt = torch.randn(4, 16), torch.randn(4, 16)
    feats, sim = clip_pair_features(img, txt, "interaction")
    assert feats.shape == (4, clip_feature_dim(16, "interaction")) == (4, 65)
    assert sim.shape == (4,)
    assert clip_pair_features(img, txt, "concat")[0].shape == (4, 32)
    assert clip_pair_features(img, txt, "sim")[0].shape == (4, 1)


def test_identical_embeddings_are_a_perfect_match():
    v = torch.randn(3, 8)
    feats, sim = clip_pair_features(v, v.clone(), "interaction")
    assert torch.allclose(sim, torch.ones(3), atol=1e-5)
    # the |img - txt| block must be exactly zero for an identical pair
    assert torch.allclose(feats[:, 24:32], torch.zeros(3, 8), atol=1e-6)


def test_opposite_embeddings_are_a_perfect_mismatch():
    v = torch.randn(3, 8)
    _, sim = clip_pair_features(v, -v, "interaction")
    assert torch.allclose(sim, -torch.ones(3), atol=1e-5)


def test_similarity_ignores_vector_length():
    """Normalisation first, so a brighter image cannot look like a better match
    simply by having a larger embedding norm."""
    img, txt = torch.randn(2, 8), torch.randn(2, 8)
    _, a = clip_pair_features(img, txt, "interaction")
    _, b = clip_pair_features(img * 17.0, txt * 0.03, "interaction")
    assert torch.allclose(a, b, atol=1e-5)


def test_product_block_sums_to_the_cosine():
    """The head is handed per-dimension agreement whose sum IS the cosine, so
    it can re-weight dimensions the cosine treats equally."""
    img, txt = torch.randn(5, 12), torch.randn(5, 12)
    feats, sim = clip_pair_features(img, txt, "interaction")
    assert torch.allclose(feats[:, 24:36].sum(dim=1), sim, atol=1e-5)


def test_pair_features_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        clip_pair_features(torch.randn(2, 8), torch.randn(2, 4))
    with pytest.raises(ValueError):
        clip_pair_features(torch.randn(2, 3, 8), torch.randn(2, 3, 8))


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_clip():
    pytest.importorskip("transformers")
    return CLIPMatchEncoder(MatchConfig(backbone="clip", pretrained=False))


def test_clip_encoder_is_frozen_and_declares_its_width(tiny_clip):
    assert all(not p.requires_grad for p in tiny_clip.model.parameters())
    assert tiny_clip.feature_dim == clip_feature_dim(tiny_clip.embed_dim, "interaction")


def test_clip_encoder_shapes_and_projection(tiny_clip):
    size = tiny_clip.model.config.vision_config.image_size
    out = tiny_clip.encode_tensors(
        pixel_values=torch.randn(2, 3, size, size),
        input_ids=torch.randint(0, 90, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
    )
    assert out["features"].shape == (2, tiny_clip.feature_dim)
    assert out["similarity"].shape == (2,)
    proj = tiny_clip.projection()
    assert proj(out["features"]).shape == (2, 768)


def test_clip_encoder_without_a_processor_says_so(tiny_clip):
    with pytest.raises(RuntimeError, match="no processor"):
        tiny_clip.encode_pairs(["a caption"], [object()])


def test_build_encoder_rejects_a_crossed_backbone():
    with pytest.raises(ValueError):
        CLIPMatchEncoder(MatchConfig(backbone="qwenvl"))
    with pytest.raises(ValueError):
        QwenVLMatchEncoder(MatchConfig(backbone="clip"))


# ---- the VLM, through a stub ---------------------------------------------

class _StubVLM(nn.Module):
    """A decoder-shaped stand-in: returns hidden states whose last real token
    encodes the row, so the read-out can be checked exactly."""

    class _Cfg:
        hidden_size = 6
        num_hidden_layers = 3

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.dummy = nn.Linear(2, 2)
        self.seen: list[str] = []

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kw):
        b, s = attention_mask.shape
        layers = []
        for layer in range(self.config.num_hidden_layers + 1):
            h = torch.zeros(b, s, self.config.hidden_size)
            for i in range(b):
                h[i] = float(layer)
                h[i, attention_mask[i].sum() - 1] = float(100 + i)   # the last real token
            layers.append(h)
        return type("Out", (), {"hidden_states": tuple(layers)})()


class _StubProcessor:
    """Records the prompts it is given, pads on the right, ignores the pixels."""

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"

    def __init__(self):
        self.tokenizer = self._Tok()
        self.prompts: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = messages[0]["content"][1]["text"]
        return f"<|im_start|>user<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>"

    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        self.prompts.extend(text)
        lengths = [3 + i for i in range(len(text))]
        s = max(lengths)
        mask = torch.zeros(len(text), s, dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        return {"input_ids": torch.ones(len(text), s, dtype=torch.long),
                "attention_mask": mask,
                "pixel_values": torch.randn(len(text), 3, 8, 8)}


@pytest.fixture
def stub_vlm():
    proc = _StubProcessor()
    enc = QwenVLMatchEncoder(MatchConfig(backbone="qwenvl", layer=-1, dtype="float32"),
                             device=torch.device("cpu"), model=_StubVLM(), processor=proc)
    return enc, proc


def test_vlm_reads_the_requested_layer_and_the_right_position(stub_vlm):
    enc, _ = stub_vlm
    out = enc.encode_pairs(["first caption", "second caption"], [object(), object()])
    assert out["features"].shape == (2, 6)
    assert out["similarity"] is None, "the VLM never embeds the two sides apart"
    # last layer, last real token, per row
    assert torch.allclose(out["features"][0], torch.full((6,), 100.0))
    assert torch.allclose(out["features"][1], torch.full((6,), 101.0))


def test_vlm_layer_index_is_validated(stub_vlm):
    enc, proc = stub_vlm
    with pytest.raises(IndexError):
        QwenVLMatchEncoder(MatchConfig(backbone="qwenvl", layer=30),
                           device=torch.device("cpu"), model=_StubVLM(), processor=proc)
    middle = QwenVLMatchEncoder(MatchConfig(backbone="qwenvl", layer=1, dtype="float32"),
                                device=torch.device("cpu"), model=_StubVLM(), processor=proc)
    assert middle.layer == 1


def test_vlm_prompt_carries_the_caption_and_an_image_placeholder(stub_vlm):
    enc, proc = stub_vlm
    enc.encode_pairs(["a dog on a beach"], [object()])
    assert "a dog on a beach" in proc.prompts[0]
    assert "image" in proc.prompts[0], "the image placeholder must survive the chat template"


def test_vlm_prompt_is_identical_apart_from_the_caption(stub_vlm):
    """A prompt that varied with the row would smuggle information into the
    features that the encoder never had to read out of the picture."""
    enc, proc = stub_vlm
    enc.encode_pairs(["alpha", "beta"], [object(), object()])
    a, b = (p.replace("alpha", "X").replace("beta", "X") for p in proc.prompts[:2])
    assert a == b


def test_vlm_refuses_unequal_captions_and_images(stub_vlm):
    enc, _ = stub_vlm
    with pytest.raises(ValueError, match="captions but"):
        enc.encode_pairs(["one", "two"], [object()])


def test_vlm_needs_real_weights_without_injection():
    with pytest.raises(ValueError, match="pretrained=False"):
        build_encoder(MatchConfig(backbone="qwenvl", pretrained=False))


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------

def _tiny_clip_model():
    from transformers import CLIPConfig, CLIPModel

    return CLIPModel(CLIPConfig.from_dict({
        "text_config": {"hidden_size": 32, "intermediate_size": 37, "num_hidden_layers": 2,
                        "num_attention_heads": 2, "vocab_size": 99, "max_position_embeddings": 77},
        "vision_config": {"hidden_size": 32, "intermediate_size": 37, "num_hidden_layers": 2,
                         "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
        "projection_dim": 16,
    }))


class _TinyCLIPProcessor:
    """Stands in for CLIPProcessor: resizes to the tiny model's input size and
    turns each caption into deterministic ids, so no tokenizer is downloaded
    and two different captions still produce two different vectors."""

    def __call__(self, text=None, images=None, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        import numpy as np

        pixels = torch.stack([
            torch.tensor(np.asarray(im.resize((32, 32)), dtype="float32") / 255.0).permute(2, 0, 1)
            for im in images
        ])
        ids = torch.tensor([[(ord(c) % 98) + 1 for c in t[:8].ljust(8, "_")] for t in text])
        return {"pixel_values": pixels, "input_ids": ids,
                "attention_mask": torch.ones_like(ids)}


def _make_pair_csv(tmp_path, n=6, with_images=True, missing_image=False):
    from PIL import Image

    img_dir = tmp_path / "img"
    img_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        group = GROUPS[i % len(GROUPS)]
        name = f"{i}.png"
        if with_images and not (missing_image and i == n - 1):
            Image.new("RGB", (40, 40), color=(i * 20 % 255, 80, 200)).save(img_dir / name)
        rows.append({
            "sample_id": f"s{i:03d}",
            "text": f"caption number {i}",
            "image_path": str(img_dir / name),
            "group": group,
            "label_index": GROUPS.index(group),
            "scenario": GROUPS.index(group) + 1,
            "label_binary": 0 if group == "genuine" else 1,
            "subcategory": f"src_{i % 2}",
            "split": "train" if i % 3 == 0 else ("val" if i % 3 == 1 else "test"),
        })
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_read_pair_rows_needs_both_halves(tmp_path):
    from fnd.extract_match import read_pair_rows

    path = tmp_path / "text_only.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "text", "split"])
        w.writeheader()
        w.writerow({"sample_id": "a", "text": "hi", "split": "train"})
    with pytest.raises(KeyError, match="image_path"):
        read_pair_rows(path)


def test_read_pair_rows_rejects_duplicate_ids(tmp_path):
    from fnd.extract_match import read_pair_rows

    path = tmp_path / "dupes.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "text", "image_path", "split"])
        w.writeheader()
        w.writerow({"sample_id": "a", "text": "x", "image_path": "a.png", "split": "train"})
        w.writerow({"sample_id": "a", "text": "y", "image_path": "b.png", "split": "train"})
    with pytest.raises(ValueError, match="duplicate"):
        read_pair_rows(path)


def test_a_missing_image_names_the_row_and_stops(tmp_path):
    """Skipping it would put the two backbones on different row sets, and the
    comparison would then be between two different test sets."""
    pytest.importorskip("PIL")
    from fnd.extract_match import load_images, read_pair_rows

    csv_path = _make_pair_csv(tmp_path, n=4, missing_image=True)
    rows, id_col = read_pair_rows(csv_path)
    with pytest.raises(RuntimeError, match="s003"):
        load_images(rows, id_col, None)


def test_image_root_is_prefixed_only_to_relative_paths(tmp_path):
    from fnd.extract_match import resolve_image_path

    assert resolve_image_path({"image_path": "a/b.png"}, "/data") == tmp_path.__class__("/data/a/b.png")
    assert resolve_image_path({"image_path": "/abs/b.png"}, "/data") == tmp_path.__class__("/abs/b.png")
    assert resolve_image_path({"image_path": "a/b.png"}, None) == tmp_path.__class__("a/b.png")


def test_extract_end_to_end_with_a_tiny_clip(tmp_path):
    """The whole CLI on 6 rows and a randomly initialised CLIP: proves the
    files, the shapes and the metadata, and downloads nothing."""
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    import numpy as np

    from fnd import extract_match

    csv_path = _make_pair_csv(tmp_path, n=6)
    out = tmp_path / "v_match_clip.pt"

    # Point the CLI at an untrained tiny CLIP with a stand-in processor, so
    # the run exercises the real CSV -> images -> encode -> save path.
    real_build = extract_match.build_encoder
    extract_match.build_encoder = lambda cfg, device=None, **kw: CLIPMatchEncoder(
        MatchConfig(backbone="clip", features=cfg.features, pretrained=False),
        device, model=_tiny_clip_model(), processor=_TinyCLIPProcessor())
    try:
        assert extract_match.main(["--csv", str(csv_path), "--out", str(out),
                                   "--device", "cpu", "--batch-size", "2"]) == 0
    finally:
        extract_match.build_encoder = real_build

    payload = torch.load(out, weights_only=False)
    assert payload["features"].shape[0] == 6
    assert len(payload["ids"]) == 6 and payload["ids"][0] == "s000"
    assert payload["similarity"].shape == (6,)
    assert torch.isfinite(payload["features"]).all()

    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["backbone"] == "clip" and meta["has_similarity"] is True
    assert meta["feature_dim"] == payload["features"].shape[1]
    assert meta["projected"] is False

    with np.load(out.with_suffix(".npz")) as z:
        assert z["pooled_features"].shape == tuple(payload["features"].shape)
        assert list(z["sample_id"]) == payload["ids"]
