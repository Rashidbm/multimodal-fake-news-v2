"""The dashboard's fusion plug loads a checkpoint in fnd.train_fusion's format."""
import pytest

torch = pytest.importorskip("torch")

from dashboard.v3_fusion import load_fusion
from fnd.models.pipeline_v3 import V3FusionModule
from fnd.train_fusion import CLASSES


def save_checkpoint(path):
    torch.manual_seed(0)
    model = V3FusionModule()
    torch.save(dict(model=model.state_dict(), classes=CLASSES, epoch=8, seed=42,
                    validation=dict(macro_f1=0.8428, confusion=[[1, 0], [0, 1]]),
                    architecture='fnd.models.pipeline_v3.V3FusionModule (tokenized attention, 607d9f5)',
                    features_sha256='0' * 64), path)
    return model.eval()


def test_loads_train_fusion_checkpoint_and_returns_probabilities(tmp_path):
    model = save_checkpoint(tmp_path / 'best.pt')
    fusion = load_fusion('cpu', tmp_path / 'best.pt')
    assert fusion.classes == CLASSES

    x = torch.randn(1, 768), torch.randn(1, 768), torch.randn(1, 4096)
    probs = fusion(*x)
    assert probs.shape == (1, 5)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    with torch.inference_mode():
        assert torch.allclose(probs, model(*x)['main_logits'].softmax(-1))


def test_checkpoint_path_from_environment(tmp_path, monkeypatch):
    save_checkpoint(tmp_path / 'best.pt')
    monkeypatch.setenv('FUSION_CHECKPOINT', str(tmp_path / 'best.pt'))
    assert load_fusion('cpu').classes == CLASSES


def test_missing_checkpoint_says_what_to_set(tmp_path):
    with pytest.raises(FileNotFoundError, match='FUSION_CHECKPOINT'):
        load_fusion('cpu', tmp_path / 'nope.pt')
