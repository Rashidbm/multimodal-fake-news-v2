"""Check candidate image/caption inference and 768-feature export on development."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from scipy.special import softmax

from fnd.fit_semantic_fusion import feature_matrix, load_cache
from fnd.predict_semantic_fusion import SemanticFusionPredictor
from fnd.train_clip_matcher import digest, read_rows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semantic_resolution'


def main():
    torch.set_num_threads(2)
    bundle_path=OUT/'candidate/bundle.json'
    lock=json.loads((OUT/'candidate/selection_lock.json').read_text())
    assert digest(bundle_path)==lock['candidate_bundle_sha256']
    csv_path=ROOT/'data/processed/semantic_resolution_train_dev.csv'
    rows=read_rows(csv_path)
    chosen=[next(i for i,r in enumerate(rows) if r['split']=='val' and r['evaluation_group']=='v1_five_scenarios'
                 and int(r['scenario'])==s) for s in range(1,6)]
    caption=next(r['caption_id'] for r in rows if r['split']=='val' and r['evaluation_group']=='paired_news')
    chosen += [i for i,r in enumerate(rows) if r['split']=='val' and r['evaluation_group']=='paired_news' and r['caption_id']==caption]
    assert len(chosen)==7
    caches=[load_cache(OUT/f'expanded_{k}.pt',csv_path,rows) for k in ['blip','v1','clip']]
    x=feature_matrix(caches[0],caches[1],'blip_v1_clip_large',caches[2])[chosen]
    bundle=json.loads(bundle_path.read_text())
    model=joblib.load(bundle['classifier']['path'])['model']
    expected=1-model.predict_proba(x)[:,0]
    predictor=SemanticFusionPredictor(bundle_path,device='cpu')
    images=[]
    for i in chosen:
        with Image.open(rows[i]['image_path']) as image:
            images.append(image.convert('RGB'))
    result=predictor.predict(images,[rows[i]['text'] for i in chosen],return_semantic=True)
    actual=np.array([p['fake_probability'] for p in result['predictions']])
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
    assert result['semantic'].shape==(7,768)
    projection=predictor.semantic_projection
    logits=result['semantic'].double().numpy()@projection['coefficients'].T+projection['intercept']
    reconstructed=1-softmax(logits,axis=1)[:,0]
    np.testing.assert_allclose(actual,reconstructed,atol=1e-6,rtol=1e-6)
    assert (actual>=predictor.threshold).tolist()==(expected>=predictor.threshold).tolist()
    report=dict(passed=True,bundle_sha256=digest(bundle_path),rows=7,split='validation only',
        sample_ids=[rows[i]['sample_id'] for i in chosen],device='cpu',
        maximum_live_cached_probability_error=float(abs(actual-expected).max()),
        maximum_projection_probability_error=float(abs(actual-reconstructed).max()),semantic_shape=[7,768],
        inputs=['Image pixels','Caption tokens'],label_metadata_entered_predictor=False)
    (OUT/'inference_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
