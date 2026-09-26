import inspect

import numpy as np
import pytest
import torch
from transformers import CLIPConfig, CLIPModel

from fnd.models.clip_matcher import CLIPMatcher
from fnd.train_clip_matcher import pair_loss, score_pairs, select_threshold


def small_model(tune):
    config = CLIPConfig(projection_dim=8,
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
                         eos_token_id=2, bos_token_id=1, pad_token_id=0),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, image_size=32, patch_size=16))
    return CLIPMatcher(tune_layers=tune, clip=CLIPModel(config))


def inputs():
    return dict(pixel_values=torch.randn(4, 3, 32, 32),
                input_ids=torch.tensor([[1, 4, 5, 2], [1, 8, 9, 2]]),
                attention_mask=torch.ones(2, 4, dtype=torch.long))


@pytest.mark.parametrize('tune', [0, 1])
def test_only_authorized_parameters_receive_gradient_and_optimizer(tune):
    model = small_model(tune).train()
    pair_loss(model(**inputs())).backward()
    optimized = {id(p) for group in model.parameter_groups(1e-3, 1e-5, 0) for p in group['params']}
    assert all((id(p) in optimized) == p.requires_grad for p in model.parameters())
    for name, parameter in model.clip.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
        else:
            assert tune == 1
            assert ('layers.1.' in name or 'projection' in name or 'post_layernorm' in name
                    or 'final_layer_norm' in name)
    assert model.head[-1].weight.grad.abs().sum() > 0
    if tune:
        assert model.clip.visual_projection.weight.grad.abs().sum() > 0
        assert model.clip.text_projection.weight.grad.abs().sum() > 0


def test_cached_and_live_features_agree_and_paired_text_is_repeated_correctly():
    model = small_model(0).eval()
    batch = inputs()
    with torch.no_grad():
        image, text = model.encode(**batch)
        torch.testing.assert_close(text[:2], text[2:])
        torch.testing.assert_close(model(**batch), model.classify(image, text))
        single = {key: value[:1] for key, value in batch.items()}
        torch.testing.assert_close(model(**single)[0], model(**batch)[0], atol=1e-6, rtol=1e-5)


def test_forward_cannot_accept_ground_truth_or_source_metadata():
    assert list(inspect.signature(CLIPMatcher.forward).parameters) == [
        'self', 'pixel_values', 'input_ids', 'attention_mask']
    with pytest.raises(TypeError):
        small_model(0)(**inputs(), label_binary=torch.zeros(4))


def test_loss_rewards_both_correct_predictions_and_pair_order():
    assert pair_loss(torch.tensor([-2., -1., 2., 1.])) < pair_loss(torch.zeros(4))
    assert pair_loss(torch.tensor([2., 1., -2., -1.])) > pair_loss(torch.zeros(4))
    with pytest.raises(ValueError):
        pair_loss(torch.ones(3))


def test_threshold_selection_and_tied_ranking():
    real, fake = [0.1, 0.2, 0.4], [0.3, 0.5, 0.8]
    threshold = select_threshold(real, fake)
    selected = score_pairs(real, fake, threshold)
    exhaustive = max(score_pairs(real, fake, t)['balanced_accuracy'] for t in np.r_[real, fake, 1.01])
    assert selected['balanced_accuracy'] == exhaustive
    assert score_pairs([0.5], [0.5])['within_caption_ranking'] == 0.5
    assert score_pairs([0.1], [0.9])['both_correct'] == 1
