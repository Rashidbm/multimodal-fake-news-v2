"""Behavioral checks for image labels, DCT math and actual frozen-module state."""
import json

import numpy as np
from PIL import Image
import pytest
from scipy.fftpack import dct
import torch

from fnd.forensic.data import ForensicDataset
from fnd.forensic.model import ForensicModel
from fnd.forensic.preprocess import dct_map
from fnd.forensic.metrics import binary_metrics
from fnd.forensic.build_genimage import HashTree


def test_dct_matches_literal_patch_reference():
    rng=np.random.default_rng(123)
    image=Image.fromarray(rng.integers(0,256,(241,271,3),dtype=np.uint8))
    y=np.asarray(image.resize((224,224),Image.Resampling.BILINEAR).convert('YCbCr'),dtype=np.float32)[:,:,0]
    references=[]
    for size in (8,16):
        result=np.empty((224,224),dtype=np.float32)
        for top in range(0,224,size):
            for left in range(0,224,size):
                patch=y[top:top+size,left:left+size]
                coefficients=dct(dct(patch,type=2,axis=0,norm='ortho'),type=2,axis=1,norm='ortho')
                result[top:top+size,left:left+size]=np.log(np.abs(coefficients)+1e-8)
        references.append(result)
    np.testing.assert_allclose(dct_map(image)[0],.5*(references[0]+references[1]),rtol=0,atol=1e-6)


def test_image_only_input_is_independent_of_caption_label_and_metadata(tmp_path):
    p=tmp_path/'image.png';Image.new('RGB',(300,250),(40,100,180)).save(p)
    rows=[dict(image_path=str(p),sample_id='a',label=0,caption='real',scenario=4,generator='nature'),
          dict(image_path=str(p),sample_id='b',label=1,caption='fake',scenario=3,generator='ai')]
    data=ForensicDataset(rows,'rgb',training=False)
    assert torch.equal(data[0][0],data[1][0])
    assert data[0][1].item()==0 and data[1][1].item()==1


def test_missing_dct_is_an_error_not_a_zero_tensor(tmp_path):
    (tmp_path/'dct_stats.json').write_text(json.dumps(dict(mean=0.,std=1.)))
    data=ForensicDataset([dict(sample_id='missing',label=0)],'dct',dct_dir=tmp_path)
    with pytest.raises(FileNotFoundError):data[0]


def test_frozen_rgb_weights_and_bn_buffers_do_not_update():
    torch.set_num_threads(2);model=ForensicModel('rgb',pretrained=False).train()
    names=['conv1','bn1','layer1','layer2']
    before={name:{k:v.clone() for k,v in getattr(model.encoder,name).state_dict().items()} for name in names}
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
    values=model(torch.randn(2,3,64,64));values.sum().backward();opt.step()
    for name in names:
        for k,v in getattr(model.encoder,name).state_dict().items():assert torch.equal(v,before[name][k]),(name,k)
    assert model.encoder.fc.weight.grad is not None


def test_dct_phase_transition_changes_freezing():
    model=ForensicModel('dct',pretrained=False).train()
    assert model.encoder.conv1.weight.requires_grad
    assert not model.encoder.layer1[0].conv1.weight.requires_grad
    assert not model.encoder.layer1[0].bn1.training
    model.set_phase(2)
    assert all(p.requires_grad for p in model.parameters())
    assert model.encoder.layer1[0].bn1.training
    assert model.encoder.conv1.in_channels==1


def test_metric_orientation_and_near_duplicate_split_policy():
    metrics=binary_metrics([0,0,1,1],[.1,.9,.8,.7])
    assert metrics['real_recall']==.5 and metrics['fake_recall']==1.
    tree=HashTree();tree.add(0,'test')
    assert tree.cross_split_near(3,'train')
    assert not tree.cross_split_near(3,'test')
    assert not tree.cross_split_near(255,'train')


def test_compact_projection_preserves_classifier_and_does_not_mutate_weights():
    from fnd.forensic.export import compact_projection
    rng=np.random.default_rng(71)
    train=rng.normal(size=(80,12)).astype(np.float32)
    weight=rng.normal(size=12)*3;original=weight.copy();bias=.7
    center,projection=compact_projection(train,weight,dimension=5)
    np.testing.assert_array_equal(weight,original)
    # Verify on fresh synthetic examples, not the projection's training rows.
    unseen=rng.normal(size=(23,12)).astype(np.float32)
    mapped=(unseen-center)@projection.T
    output=mapped[:,0]*np.linalg.norm(weight)+bias+center@weight
    np.testing.assert_allclose(output,unseen@weight+bias,rtol=1e-5,atol=1e-5)


def test_global_operating_point_maximizes_worst_recall():
    from fnd.forensic.development import operating_threshold
    conditions={'a':(np.array([0,0,1,1]),np.array([.1,.6,.5,.9])),
                'b':(np.array([0,0,1,1]),np.array([.2,.3,.4,.7]))}
    threshold=operating_threshold(conditions)
    worst=lambda t:min(min(binary_metrics(y,p,t)[k] for k in ('real_recall','fake_recall')) for y,p in conditions.values())
    assert worst(threshold)==max(worst(t) for t in np.linspace(0,1,1001))


def test_final_test_extraction_requires_a_locked_dataset(tmp_path,monkeypatch):
    from fnd.forensic.cache import main
    import sys
    csv=tmp_path/'images.csv';csv.write_text('sample_id,split,label,image_path\n')
    arguments=['cache','--kind','clip','--csv',str(csv),'--out',str(tmp_path/'features.pt'),'--splits','test']
    monkeypatch.setattr(sys,'argv',arguments)
    with pytest.raises(ValueError,match='prewritten selection lock'):main()
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(dict(selection_complete=True,allowed_csv_sha256=['wrong hash'])))
    monkeypatch.setattr(sys,'argv',arguments+['--test-lock',str(lock)])
    with pytest.raises(ValueError,match='not covered by lock'):main()


def test_image_feature_join_rejects_wrong_labels(tmp_path):
    from fnd.forensic.combine_features import aligned_concatenation
    first=dict(sample_ids=['one','two'],splits=['train','val'],labels=[0,1],csv_sha256='example',features=torch.randn(2,3))
    second={**first,'labels':[0,0]}
    a=tmp_path/'a.pt';b=tmp_path/'b.pt';torch.save(first,a);torch.save(second,b)
    with pytest.raises(ValueError,match='target or split mismatch'):aligned_concatenation([a,b],tmp_path/'combined.pt')
