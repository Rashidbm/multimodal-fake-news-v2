"""V3 fusion tests (guidelines §5-6), random tensors only.

The key regression: attention must actually attend. With one token per
vector the softmax over a single key is always 1, the query is ignored, and
W_q / W_k never get a gradient. These tests fail on that version.
"""
import pytest

torch = pytest.importorskip("torch")

from fnd.models.pipeline_v3 import (
    PairwiseCrossAttention,
    V3FusionModule,
    VectorTokenizer,
)

B = 4


def inputs(batch=B, text_dim=4096):
    return torch.randn(batch, 768), torch.randn(batch, 768), torch.randn(batch, text_dim)


def test_output_shapes_with_raw_4096_text_vector():
    torch.manual_seed(0)
    out = V3FusionModule().eval()(*inputs())
    assert out["main_logits"].shape == (B, 5)
    assert out["aux_logits"].shape == (B, 2)
    assert out["fused"].shape == (B, 1024)


def test_text_already_projected_is_accepted():
    torch.manual_seed(0)
    out = V3FusionModule(text_in_dim=768).eval()(*inputs(text_dim=768))
    assert out["main_logits"].shape == (B, 5)


def test_attention_weights_are_not_trivial():
    torch.manual_seed(0)
    tok = VectorTokenizer(768, 8)
    pair = PairwiseCrossAttention(768, 8, dropout=0.0).eval()
    x, y = tok(torch.randn(B, 768)), tok(torch.randn(B, 768))
    _, w = pair.attn_x_to_y(x, y, y, average_attn_weights=False)
    assert w.shape == (B, 8, 8, 8)                       # B, heads, T queries, T keys
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)))
    assert w.max() < 1.0                                 # no single key takes everything


def test_attention_output_depends_on_query():
    torch.manual_seed(0)
    tok = VectorTokenizer(768, 8)
    pair = PairwiseCrossAttention(768, 8, dropout=0.0).eval()
    y = tok(torch.randn(B, 768))
    a, _ = pair.attn_x_to_y(tok(torch.randn(B, 768)), y, y)
    b, _ = pair.attn_x_to_y(tok(torch.randn(B, 768)), y, y)
    assert not torch.allclose(a, b, atol=1e-4)


def test_query_and_key_weights_receive_gradient():
    torch.manual_seed(0)
    model = V3FusionModule().train()
    model(*inputs(batch=8))["main_logits"].sum().backward()
    for pair in (model.fusion.pair_sem_img, model.fusion.pair_sem_text, model.fusion.pair_img_text):
        for attn in (pair.attn_x_to_y, pair.attn_y_to_x):
            g = attn.in_proj_weight.grad
            q, k, v = g[:768], g[768:1536], g[1536:]
            assert q.abs().sum() > 0 and k.abs().sum() > 0 and v.abs().sum() > 0
    assert model.fusion.text_proj.fc.weight.grad.abs().sum() > 0


def test_single_token_is_rejected():
    with pytest.raises(ValueError):
        VectorTokenizer(768, 1)


def test_aux_loss_does_not_reach_fusion_or_inputs():
    torch.manual_seed(0)
    model = V3FusionModule().train()
    s, i, t = inputs(batch=8)
    i.requires_grad_(True)
    model(s, i, t)["aux_logits"].sum().backward()
    assert i.grad is None
    assert all(p.grad is None for p in model.fusion.parameters())
    assert model.aux_classifier.weight.grad is not None


def test_eval_is_deterministic():
    torch.manual_seed(0)
    model = V3FusionModule().eval()
    x = inputs()
    assert torch.equal(model(*x)["main_logits"], model(*x)["main_logits"])
