import torch
from torch import nn
from types import SimpleNamespace

from fnd.models.alignment import StandardizedSimilarityGate, matching_loss
from fnd.alignment_selection import constrained_score
from fnd.models.fnd_clip import FNDCLIP, FNDCLIPConfig
from fnd.models.alignment import PairInteractionHead


def test_gate_train_equation_and_saved_eval_statistics():
    gate = StandardizedSimilarityGate()
    values = torch.tensor([0.1, 0.2, 0.4])
    actual = gate(values)
    expected = torch.sigmoid((values-values.mean()) / torch.sqrt(values.var(unbiased=False)+gate.eps))
    torch.testing.assert_close(actual, expected)
    restored = StandardizedSimilarityGate()
    restored.load_state_dict(gate.state_dict())
    restored.eval()
    before = {k: v.clone() for k,v in restored.state_dict().items()}
    single = restored(torch.tensor([0.2]))
    mixed = restored(torch.tensor([0.2, 0.99, -0.99]))
    torch.testing.assert_close(single[0], mixed[0])
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_constant_similarity_is_finite():
    gate = StandardizedSimilarityGate()
    torch.testing.assert_close(gate(torch.ones(3)), torch.full((3,), 0.5))
    gate.eval()
    assert torch.isfinite(gate(torch.ones(1))).all()


def test_matching_supervision_does_not_relabel_other_manipulations():
    logits = torch.zeros(5, requires_grad=True)
    loss = matching_loss(logits, torch.arange(1,6))
    loss.backward()
    assert logits.grad[0] < 0  # OOC -> mismatch
    assert logits.grad[3] > 0  # genuine -> matching
    assert torch.equal(logits.grad[[1,2,4]], torch.zeros(3))
    absent = torch.ones(3, requires_grad=True)
    zero = matching_loss(absent, torch.tensor([2,3,5]))
    zero.backward()
    assert torch.equal(absent.grad, torch.zeros(3))


def test_constrained_threshold_improves_real_without_sacrificing_ooc():
    rows=[]
    for scenario,values in [(1,[.3,.6,.9]),(4,[.1,.4,.7]),(2,[.8,.9,.95]),(3,[.85,.9,.95]),(5,[.8,.9,.95])]:
        rows.extend({'scenario':scenario,'label_binary':int(scenario!=4),'prob':p} for p in values)
    result=constrained_score(rows,{'1':2/3,'2':1,'3':1,'5':1})
    assert result['threshold']==.6
    assert result['real_recall']==2/3
    assert result['per_scenario'][1]['accuracy']==2/3


def tiny_model(alignment=False):
    class TextEncoder(nn.Module):
        def forward(self,input_ids,attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(1))
    class Clip(nn.Module):
        config=SimpleNamespace(projection_dim=512)
        def get_image_features(self,pixel_values):
            return pixel_values
        def get_text_features(self,input_ids,attention_mask):
            return input_ids.float()
    model=FNDCLIP.__new__(FNDCLIP)
    nn.Module.__init__(model)
    model.cfg=FNDCLIPConfig(proj_dim=16,similarity_weighting='standardized',alignment_head=alignment,normalize_streams=alignment)
    model.resnet=nn.Identity();model.bert=TextEncoder();model.clip=Clip()
    model.text_proj=nn.Linear(768,16);model.image_proj=nn.Linear(2048,16);model.fused_proj=nn.Linear(1024,16)
    model.attention=nn.Linear(16,1);model.classifier=nn.Linear(16,1)
    model.similarity_gate=StandardizedSimilarityGate()
    if alignment:
        model.stream_norms=nn.ModuleList([nn.LayerNorm(16) for _ in range(3)])
        model.matching_head=PairInteractionHead()
        model.matching_projection=nn.Linear(128,16)
    return model


def tiny_inputs():
    return [torch.randn(2,2048),torch.randn(2,768),torch.ones(2,768),
            torch.randn(2,512),torch.randn(2,512),torch.ones(2,512)]


def test_gate_scales_the_projected_bias_and_features():
    model=tiny_model().eval()
    observed=[]
    handle=model.fused_proj.register_forward_hook(lambda module,args,out:observed.append(out))
    out=model(*tiny_inputs())
    handle.remove()
    weight=model.similarity_gate(out['clip_similarity'])
    expected=observed[0]*weight[:,None]*out['attention'][:,2,None]
    torch.testing.assert_close(out['semantic'].reshape(2,3,16)[:,2],expected)


def test_matching_branch_is_trainable_and_preserves_semantic_export():
    model=tiny_model(alignment=True).train()
    out=model(*tiny_inputs())
    loss=out['logits'].square().mean()+matching_loss(out['match_logits'],torch.tensor([1,4]))
    loss.backward()
    assert model.matching_head.classifier.weight.grad.abs().sum()>0
    assert model.matching_projection.weight.grad.abs().sum()>0
    optimized={id(p) for group in model.parameter_groups(1e-5,1e-3,0) for p in group['params']}
    assert all(id(p) in optimized for p in model.parameters() if p.requires_grad)
    pooled=out['semantic'].reshape(2,3,16).sum(1)
    torch.testing.assert_close(model.classifier(pooled),out['logits'])


def test_frozen_matcher_stays_in_eval_mode():
    model=tiny_model(alignment=True)
    model.cfg.fine_tune_matching=False
    model.matching_head.requires_grad_(False)
    model.train()
    assert not model.matching_head.training
    assert not any(p.requires_grad for p in model.matching_head.parameters())
    assert model.matching_projection.training
