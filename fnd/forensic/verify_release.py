"""Compare exported pixel inference with selected cached models on training images."""
import gc
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy.special import expit
import torch

from .development import OUT, DATASETS, CASES
from .inference import ImageBranch
from .preprocess import read_rows, sha256


def main():
    torch.set_num_threads(4);lock=json.loads((OUT/'selection_lock.json').read_text())
    bundles=json.loads((OUT/'release_bundles.json').read_text());reports={}
    rows={}
    for domain,path in DATASETS.items():
        for row in read_rows(path):
            if row['split']=='train':rows[row['sample_id']]={**row,'domain':domain}
    for purpose,path in bundles.items():
        path=Path(path);directory=path.parent
        source=json.loads((directory/'projection_source.json').read_text())
        if sha256(source['path'])!=source['sha256']:raise ValueError('Projection source changed')
        cache=torch.load(source['path'],map_location='cpu',weights_only=True)
        if 'test' in cache['splits']:raise ValueError('Export verification must use development images')
        selected=[];counts={}
        for i,key in enumerate(cache['sample_ids']):
            if cache['splits'][i]!='train':continue
            row=rows[key];group=(row['domain'],int(row['label']))
            if counts.get(group,0)<4:selected.append(i);counts[group]=counts.get(group,0)+1
        raw_reference=cache['features'][selected].numpy();name=lock['selected'][purpose];candidate=lock['candidates'][name]
        if candidate['kind']=='linear':
            model=joblib.load(candidate['classifier']);reference_probability=model.predict_proba(raw_reference)[:,1]
        else:
            state=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=True)['model']
            weight=state['encoder.fc.weight'].numpy()[0];bias=float(state['encoder.fc.bias'].item())
            reference_probability=expit(raw_reference@weight+bias)
        branch=ImageBranch(path,'mps');raws=[];outputs=[];projected=[]
        for start in range(0,len(selected),8):
            images=[]
            for i in selected[start:start+8]:
                with Image.open(rows[cache['sample_ids'][i]]['image_path']) as image:images.append(image.convert('RGB'))
            raws.append(branch.encode_images(images).cpu().numpy())
            output=branch(images);outputs.append(output['probability'].cpu().numpy());projected.append(output['features'].cpu().numpy())
        raw=np.concatenate(raws);probability=np.concatenate(outputs);features=np.concatenate(projected)
        np.testing.assert_allclose(raw,raw_reference,rtol=1e-3,atol=5e-4)
        np.testing.assert_allclose(probability,reference_probability,rtol=1e-3,atol=1e-4)
        threshold=branch.config['threshold']
        np.testing.assert_array_equal(probability>=threshold,reference_probability>=threshold)
        if features.shape!=(len(selected),768) or not np.isfinite(features).all():raise ValueError('Invalid fusion interface')
        reports[purpose]=dict(model=name,bundle_sha256=sha256(path),training_images_checked=len(selected),
                             groups={f'{domain}/{label}':n for (domain,label),n in counts.items()},
                             max_feature_abs_error=float(np.abs(raw-raw_reference).max()),
                             max_probability_abs_error=float(np.abs(probability-reference_probability).max()),
                             predictions_identical=True,feature_dimension=768,final_test_used=False)
        # Post-lock verification of the exported head on all reserved test features.
        # This checks the implementation; it never fits weights or changes a threshold.
        from .evaluate_locked import aligned_test
        test_errors={}
        for domain,condition in CASES:
            reference_rows=[r for r in read_rows(DATASETS[domain]) if r['split']=='test']
            x=np.concatenate([aligned_test(e,domain,condition,reference_rows,OUT/'selection_lock.json') for e in candidate['encoders']],axis=1)
            if candidate['kind']=='linear':expected=joblib.load(candidate['classifier']).predict_proba(x)[:,1]
            else:
                state=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=True)['model']
                expected=expit(x.astype(np.float64)@state['encoder.fc.weight'].numpy()[0].astype(np.float64)+float(state['encoder.fc.bias'].item()))
            actual=expit(x@branch.weight.cpu().numpy()+float(branch.bias.cpu()))
            np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
            np.testing.assert_array_equal(actual>=threshold,expected>=threshold)
            test_errors[domain+'/'+condition]=dict(n=len(x),max_probability_abs_error=float(np.abs(actual-expected).max()),predictions_identical=True)
        reports[purpose].update(test_head_equivalence=test_errors,final_test_used=True,test_used_for_fitting_or_selection=False)
        print('VERIFIED',purpose,reports[purpose],flush=True)
        del branch,cache;gc.collect();torch.mps.empty_cache()
    (OUT/'release_verification.json').write_text(json.dumps(reports,indent=2))
    (OUT/'RELEASE_VERIFIED').write_text('Pixel inference matches on checked training images; exported-head decisions also match on all locked test features. No fitting or selection uses test labels.\n')


if __name__=='__main__':main()
