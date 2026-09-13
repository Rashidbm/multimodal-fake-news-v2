"""Text Fluoroscopy tests with a tiny random Qwen2 (no downloads, no GPU).

Checks the five steps of guidelines section 4 are wired as described:
  4.2 the backbone is frozen, 4.3 the layer index is validated against the
  model that actually exists, 4.4 padding cannot influence the pooled
  vector, 4.5 the projection lands in the 768-d fusion space.
"""
import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fnd.extract_textfor import main as extract_main, read_rows
from fnd.models.text_fluoroscopy import (
    TextFluoroscopy,
    TextFluoroscopyConfig,
    TextForensicProjection,
    masked_mean_pool,
    resolve_layer,
)


# --- 4.3 layer selection ---------------------------------------------------

def test_resolve_layer_accepts_valid_indices():
    n = 33                                  # Qwen3.5-9B: 32 layers + embeddings
    assert resolve_layer(n, -1) == 32
    assert resolve_layer(n, -2) == 31
    assert resolve_layer(n, 30) == 30
    assert resolve_layer(n, 0) == 0


def test_layer_30_depends_on_the_model():
    """The guidelines' layer 30 is valid on a 32-layer model (Qwen3.5-9B)
    and impossible on a 28-layer one (Qwen2-7B). The index is validated
    against whatever model is loaded, not assumed."""
    assert resolve_layer(33, 30) == 30                      # Qwen3.5-9B: fine
    with pytest.raises(IndexError) as e:
        resolve_layer(29, 30)                               # Qwen2-7B: not fine
    assert "28 transformer layers" in str(e.value)


# --- 4.4 masked mean pooling ----------------------------------------------

def test_pooling_ignores_padding():
    hidden = torch.zeros(1, 5, 4)
    hidden[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    hidden[0, 1] = torch.tensor([3.0, 3.0, 3.0, 3.0])
    hidden[0, 2:] = torch.tensor([999.0, -999.0, 999.0, -999.0])   # padding junk
    mask = torch.tensor([[1, 1, 0, 0, 0]])
    assert torch.allclose(masked_mean_pool(hidden, mask), torch.full((1, 4), 2.0))


def test_pooling_is_length_invariant():
    """Same caption at two padding lengths must give one vector, or caption
    length has leaked in as a feature."""
    real = torch.randn(1, 3, 8)
    short = torch.cat([real, torch.randn(1, 1, 8)], dim=1)
    long = torch.cat([real, torch.randn(1, 20, 8)], dim=1)
    a = masked_mean_pool(short, torch.tensor([[1, 1, 1, 0]]))
    b = masked_mean_pool(long, torch.tensor([[1, 1, 1] + [0] * 20]))
    assert torch.allclose(a, b, atol=1e-6)


def test_pooling_rejects_mismatched_mask():
    with pytest.raises(ValueError):
        masked_mean_pool(torch.randn(2, 5, 4), torch.ones(2, 6, dtype=torch.long))


# --- 4.5 projection --------------------------------------------------------

def test_projection_shape_and_dims():
    proj = TextForensicProjection(4096, 768)          # Qwen3.5-9B -> fusion space
    out = proj(torch.randn(6, 4096))
    assert out.shape == (6, 768) and torch.isfinite(out).all()
    assert TextForensicProjection(896, 768)(torch.randn(2, 896)).shape == (2, 768)


def test_projection_rejects_unpooled_input():
    proj = TextForensicProjection(4096, 768)
    with pytest.raises(ValueError):
        proj(torch.randn(2, 10, 4096))                # forgot to pool
    with pytest.raises(ValueError):
        proj(torch.randn(2, 512))                     # wrong hidden size


# --- the extractor end to end ---------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return TextFluoroscopy(TextFluoroscopyConfig(pretrained=False, layer=-2, proj_dim=768))


def test_backbone_is_frozen(tiny):
    assert all(not p.requires_grad for p in tiny.model.parameters())
    assert not tiny.model.training


def test_encode_shapes_and_projection(tiny):
    ids = torch.randint(1, 200, (3, 12), device=tiny.device)
    mask = torch.ones(3, 12, dtype=torch.long, device=tiny.device)
    mask[2, 6:] = 0                                    # third row is padded
    pooled = tiny.encode(ids, mask)
    assert pooled.shape == (3, tiny.hidden_size)
    assert torch.isfinite(pooled).all()
    assert tiny.projection()(pooled).shape == (3, 768)


def test_padding_does_not_change_a_row(tiny):
    """A caption encoded alone and in a padded batch must match."""
    ids = torch.randint(1, 200, (1, 7), device=tiny.device)
    alone = tiny.encode(ids, torch.ones(1, 7, dtype=torch.long, device=tiny.device))

    padded_ids = torch.cat([ids, torch.randint(1, 200, (1, 9), device=tiny.device)], dim=1)
    padded_mask = torch.tensor([[1] * 7 + [0] * 9], device=tiny.device)
    padded = tiny.encode(padded_ids, padded_mask)

    assert torch.allclose(alone, padded, atol=1e-4)


def test_layer_truncation_keeps_the_same_vector():
    """Truncation is an optimisation, not a change of output."""
    torch.manual_seed(0)
    full = TextFluoroscopy(TextFluoroscopyConfig(pretrained=False, layer=2, truncate_layers=False))
    torch.manual_seed(0)
    cut = TextFluoroscopy(TextFluoroscopyConfig(pretrained=False, layer=2, truncate_layers=True))
    cut.model.load_state_dict(
        {k: v for k, v in full.model.state_dict().items() if k in cut.model.state_dict()},
        strict=False,
    )
    ids = torch.randint(1, 200, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    assert cut._truncated and not full._truncated
    assert torch.allclose(full.encode(ids, mask), cut.encode(ids, mask), atol=1e-5)


# --- CSV reading + the saved artefact --------------------------------------

def _write_csv(path, n=6):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "text", "split"])
        w.writeheader()
        for i in range(n):
            w.writerow({"sample_id": f"mmfb_{i:04d}", "text": f"caption number {i}",
                        "split": "train" if i % 2 else "test"})
    return path


def test_read_rows_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "text", "split"])
        w.writeheader()
        w.writerow({"sample_id": "a", "text": "x", "split": "train"})
        w.writerow({"sample_id": "a", "text": "y", "split": "train"})
    with pytest.raises(ValueError):
        read_rows(p)


def test_read_rows_filters_split(tmp_path):
    p = _write_csv(tmp_path / "rows.csv")
    assert len(read_rows(p)) == 6
    assert all(r["split"] == "test" for r in read_rows(p, "test"))


def test_extract_end_to_end_smoke(tmp_path):
    """Real CLI against the real Qwen2-0.5B; skipped unless cached."""
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2-0.5B-Instruct not cached locally")

    csv_path = _write_csv(tmp_path / "rows.csv", n=4)
    out = tmp_path / "v_textfor.pt"
    rc = extract_main(["--csv", str(csv_path), "--out", str(out),
                       "--model", "Qwen/Qwen2-0.5B-Instruct",
                       "--batch-size", "2", "--device", "cpu"])
    assert rc == 0

    payload = torch.load(out, weights_only=False)
    assert payload["features"].shape == (4, 896)
    assert payload["sample_ids"] == [f"mmfb_{i:04d}" for i in range(4)]
    assert payload["meta"]["pooling"] == "masked_mean"
    assert payload["meta"]["projected"] is False


def test_limit_spreads_across_the_file():
    """build.py writes the CSV grouped by scenario, so the first N rows would
    be one class. --limit must sample across the file instead, or the subset
    handed to the fusion stage would contain a single scenario."""
    from fnd.extract_textfor import evenly_spaced

    rows = [{"sample_id": f"s{i}", "group": g}
            for g in ("ooc", "ftri", "rtfi", "genuine", "ftfi")
            for i in range(100)]

    picked = evenly_spaced(rows, 10)
    assert len(picked) == 10
    assert len({r["group"] for r in picked}) == 5      # every scenario present
    assert evenly_spaced(rows, 999) is rows            # limit >= len is a no-op
    assert len(evenly_spaced(rows, 1)) == 1
