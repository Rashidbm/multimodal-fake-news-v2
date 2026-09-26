"""Export an inspectable linear image head and a train-fitted 768-dimensional interface."""
import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.decomposition import PCA
import torch

from .cache import CLIP_NAME,CLIP_REVISION
from .preprocess import sha256


def linear_parameters(pipeline):
    scale=pipeline.steps[0][1];classifier=pipeline.steps[-1][1]
    if classifier.coef_.shape[0]!=1:raise ValueError('Expected one binary logit')
    if classifier.classes_.tolist()!=[0,1]:raise ValueError('Wrong positive-class orientation')
    weight=classifier.coef_[0]/scale.scale_
    bias=float(classifier.intercept_[0]-weight@scale.mean_)
    return weight,bias


def compact_projection(x,weight,dimension=768):
    if x.shape[1]==dimension:return None,None
    if x.shape[1]<dimension:raise ValueError('Cannot invent extra learned feature dimensions')
    center=x.mean(axis=0,dtype=np.float64).astype(np.float32)
    # Never normalize the caller's classifier weights in place.
    direction=np.array(weight,dtype=np.float64,copy=True)
    norm=np.linalg.norm(direction)
    if not np.isfinite(norm) or norm==0:raise ValueError('A nonzero classifier direction is required')
    direction/=norm
    residual=x-center;residual-=np.outer(residual@direction,direction).astype(np.float32)
    pca=PCA(n_components=dimension-1,svd_solver='randomized',random_state=2026091414,iterated_power=3)
    pca.fit(residual)
    basis=np.vstack([direction,pca.components_]).astype(np.float32)
    return center,basis


def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--classifier');group.add_argument('--cnn-head');ap.add_argument('--cache',required=True)
    ap.add_argument('--selection',required=True);ap.add_argument('--encoder-spec',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--target',choices=['broad','ai'],default='broad')
    args=ap.parse_args();torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'bundle.json').exists():raise FileExistsError(out/'bundle.json')
    selection=json.loads(Path(args.selection).read_text());model=None
    if args.classifier:
        model=joblib.load(args.classifier)
        if selection['classifier_sha256']!=sha256(args.classifier):raise ValueError('Selected classifier changed')
        weight,bias=linear_parameters(model)
    else:
        if selection['checkpoint_sha256']!=sha256(args.cnn_head):raise ValueError('Selected CNN changed')
        saved=torch.load(args.cnn_head,map_location='cpu',weights_only=True)
        weight=saved['model']['encoder.fc.weight'].numpy()[0].astype(np.float64)
        bias=float(saved['model']['encoder.fc.bias'].item())
    cache=torch.load(args.cache,map_location='cpu',weights_only=True)
    if 'test' in cache['splits']:raise ValueError('Projection fitting must not access final-test features')
    x=cache['features'].numpy();train=np.array(cache['splits'])=='train'
    if not train.any():raise ValueError('Projection needs training images')
    raw_logits=x.astype(np.float64)@weight+bias
    # Check the algebra in float64; sklearn rounds intermediate scaling for float32 inputs.
    if model is not None:
        np.testing.assert_allclose(raw_logits,model.decision_function(x.astype(np.float64)),rtol=1e-10,atol=1e-9)
        float32_scaling_error=float(np.max(np.abs(raw_logits-model.decision_function(x))))
    else:
        comparison=cache['logits'].numpy() if 'logits' in cache else x@weight.astype(np.float32)+np.float32(bias)
        float32_scaling_error=float(np.max(np.abs(raw_logits-comparison)))
    if float32_scaling_error>1e-3:raise ValueError('Unexpected scaling conversion error')
    center,projection=compact_projection(x[train],weight)
    state=dict(weight=torch.tensor(weight,dtype=torch.float32),bias=torch.tensor(bias,dtype=torch.float32))
    error=0.
    if projection is not None:
        projected_weight=np.zeros(768,dtype=np.float32);projected_weight[0]=np.linalg.norm(weight)
        projected_bias=bias+float(center.astype(np.float64)@weight)
        mapped=(x-center)@projection.T
        error=float(np.max(np.abs(mapped@projected_weight+projected_bias-raw_logits)))
        if error>1e-3:raise ValueError(f'Projection changes detector logits by {error}')
        state.update(projection=torch.from_numpy(projection),center=torch.from_numpy(center),
                     projected_head_weight=torch.from_numpy(projected_weight),projected_head_bias=torch.tensor(projected_bias,dtype=torch.float32))
    torch.save(state,out/'head.pt')
    encoders=json.loads(Path(args.encoder_spec).read_text())
    for name,source in encoders.items():
        if source['kind']=='clip':
            if source.get('model_revision',CLIP_REVISION)!=CLIP_REVISION:raise ValueError('Wrong CLIP revision')
            source.update(model_name=CLIP_NAME,model_revision=CLIP_REVISION)
        else:
            original=Path(source['checkpoint']);dest=out/(name+'.pt');shutil.copy2(original,dest)
            source.update(checkpoint=dest.name,checkpoint_sha256=sha256(dest))
    bundle=dict(format_version=1,encoders=encoders,head='head.pt',head_sha256=sha256(out/'head.pt'),
                threshold=selection['threshold'],target='0 authentic image; 1 AI-generated image' if args.target=='ai' else '0 authentic image; 1 edited/manipulated or AI-generated image',
                feature_dimension=768,raw_feature_dimension=x.shape[1],projection='Identity' if projection is None else 'Classifier direction plus 767 training-only residual PCA components',
                projection_fit_rows=int(train.sum()),max_projection_logit_error=error,max_float32_scaling_logit_error=float32_scaling_error,
                source_classifier_sha256=sha256(args.classifier or args.cnn_head),selection_sha256=sha256(args.selection),
                source_cache_sha256=sha256(args.cache),input='Image pixels only; no text, labels, filenames or metadata enter encoders',
                training_target=selection.get('training_target','See source experiment selection'),
                integration_note='No fusion with the project text or semantic branch has been trained or evaluated by this export')
    (out/'bundle.json').write_text(json.dumps(bundle,indent=2));print(json.dumps(bundle,indent=2))


if __name__=='__main__':main()
