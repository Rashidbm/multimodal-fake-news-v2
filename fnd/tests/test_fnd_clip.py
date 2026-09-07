"""Shape and behaviour tests for FND-CLIP with random weights (no downloads).

Checks the four parts of the spec are wired as described:
  2.1 image stream is 2048-d, 2.2 text stream is 768-d, 2.3 CLIP streams are
  512-d with a cosine similarity in [-1, 1] that scales the fused vector,
  2.4 three attention weights that sum to 1 and a single sigmoid logit.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from fnd.metrics import auc, binary_metrics, multiclass_metrics, per_scenario_accuracy
from fnd.models.fnd_clip import FNDCLIP, FNDCLIPConfig


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return FNDCLIP(FNDCLIPConfig(pretrained=False, proj_dim=64)).eval()


def fake_batch(B=2):
    return dict(
        resnet_pixels=torch.randn(B, 3, 224, 224),
        bert_input_ids=torch.randint(1, 1000, (B, 16)),
        bert_attention_mask=torch.ones(B, 16, dtype=torch.long),
        clip_pixels=torch.randn(B, 3, 224, 224),
        clip_input_ids=torch.randint(1, 1000, (B, 16)),
        clip_attention_mask=torch.ones(B, 16, dtype=torch.long),
    )


def test_forward_shapes_and_attention(model):
    out = model(**fake_batch(2))
    assert out["logits"].shape == (2, 1)                      # 2.4 one sigmoid logit
    assert out["attention"].shape == (2, 3)                   # 2.4 three stream weights
    assert torch.allclose(out["attention"].sum(-1), torch.ones(2), atol=1e-5)
    assert out["clip_similarity"].shape == (2,)               # 2.3 cosine similarity
    assert (out["clip_similarity"].abs() <= 1.0 + 1e-5).all()


def test_stream_dimensions(model):
    assert model.image_proj[0].in_features == 2048            # 2.1 ResNet-50
    assert model.text_proj[0].in_features == 768              # 2.2 BERT [CLS]
    assert model.fused_proj[0].in_features == 1024            # 2.3 concat(512, 512)


def test_clip_is_frozen_and_backbones_train(model):
    assert all(not p.requires_grad for p in model.clip.parameters())
    assert any(p.requires_grad for p in model.resnet.parameters())
    assert any(p.requires_grad for p in model.bert.parameters())
    groups = model.parameter_groups(lr_backbone=1e-5, lr_head=1e-3, weight_decay=0.0)
    assert len(groups) == 2 and groups[0]["lr"] == 1e-3 and groups[1]["lr"] == 1e-5


def test_similarity_weighting():
    m = FNDCLIP.__new__(FNDCLIP)
    m.cfg = FNDCLIPConfig(similarity_weighting="relu")
    s = torch.tensor([-0.5, 0.0, 0.3])
    assert torch.equal(m._weight_from_sim(s), torch.tensor([0.0, 0.0, 0.3]))
    m.cfg = FNDCLIPConfig(similarity_weighting="none")
    assert torch.equal(m._weight_from_sim(s), torch.ones(3))


def test_five_output_variant():
    torch.manual_seed(0)
    m = FNDCLIP(FNDCLIPConfig(pretrained=False, proj_dim=32, num_outputs=5)).eval()
    assert m(**fake_batch(1))["logits"].shape == (1, 5)


def test_metrics():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.2, 0.6]
    m = binary_metrics(y, p)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert m["accuracy"] == 0.5 and abs(m["auc"] - 0.75) < 1e-9
    assert auc([1, 0], [0.5, 0.5]) == 0.5                      # tie -> chance
    mc = multiclass_metrics([0, 1, 2], [0, 1, 1], 3)
    assert abs(mc["accuracy"] - 2 / 3) < 1e-9
    ps = per_scenario_accuracy([1, 1, 4], [1, 1, 0], [1, 0, 0])
    assert ps[1]["accuracy"] == 0.5 and ps[4]["accuracy"] == 1.0
