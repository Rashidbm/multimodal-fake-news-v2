"""Package each pre-test selected purpose; selection is never changed by test scores."""
import json
from pathlib import Path
import subprocess
import sys

import torch

from .development import OUT, DATASETS
from .preprocess import sha256


def run(name,module,args):
    with (OUT/f'{name}.log').open('a',buffering=1) as log:
        subprocess.run([sys.executable,'-u','-m',module,*args],stdout=log,stderr=subprocess.STDOUT,check=True)


def projection_cache(name,candidate,purpose):
    if candidate['kind']=='cnn':
        if purpose=='ai':
            path=OUT/name/'genimage_train_projection.pt'
            if not path.exists():
                run('export_'+name+'_features','fnd.forensic.cache',['--kind','dct' if name=='dct_guide' else 'rgb',
                    '--checkpoint',candidate['checkpoint'],'--csv',str(DATASETS['genimage']),'--out',str(path),'--splits','train','--batch-size','64'])
            return path
        return OUT/name/('mixed_development.pt' if name=='rgb_broad' else 'news_development.pt')
    if len(candidate['encoders'])==1:
        return OUT/('clip_genimage/development.pt' if purpose=='ai' else 'news/clip_development.pt')
    if name=='broad_features':return OUT/'broad_feature_head/projection_training.pt'
    directory=Path(candidate['classifier']).parent.parent
    if name.endswith('_news'):return directory/'clean.pt'
    # Include both training domains seen by the frozen fusion classifier.
    destination=directory/'projection_training.pt'
    if not destination.exists():
        tensors=[];ids=[];labels=[]
        for filename in ('clean.pt','dgm4.pt'):
            saved=torch.load(directory/filename,map_location='cpu',weights_only=True)
            if 'test' in saved['splits']:raise ValueError('Projection cannot use final-test features')
            keep=[i for i,split in enumerate(saved['splits']) if split=='train']
            tensors.append(saved['features'][keep]);ids.extend(saved['sample_ids'][i] for i in keep);labels.extend(saved['labels'][i] for i in keep)
        if len(set(ids))!=len(ids):raise ValueError('Duplicated physical training images')
        torch.save(dict(features=torch.cat(tensors),sample_ids=ids,labels=labels,splits=['train']*len(ids),test_used=False),destination)
    return destination


def main():
    torch.set_num_threads(4);lock_path=OUT/'selection_lock.json';lock=json.loads(lock_path.read_text())
    if not lock['selection_complete']:raise ValueError('Selection is incomplete')
    outputs={}
    for purpose,name in lock['selected'].items():
        candidate=lock['candidates'][name];out=OUT/('selected' if purpose=='broad' else 'selected_'+purpose)
        out.mkdir(exist_ok=True);outputs[purpose]=str(out/'bundle.json')
        if (out/'bundle.json').exists():continue
        selection=dict(model=name,purpose=purpose,threshold=lock['results'][name][purpose]['threshold'],
                       selection_lock_sha256=sha256(lock_path),test_used_for_selection=False)
        selection['training_target']='0 nature, 1 AI-generated' if name in ('rgb_guide','dct_guide','rgb_robust','clip_genimage') else '0 dataset-labeled authentic image, 1 manipulated or AI-generated image'
        if candidate['kind']=='cnn':
            expected=lock['checkpoint_sha256'][name]
            if sha256(candidate['checkpoint'])!=expected:raise ValueError('Locked CNN changed')
            selection['checkpoint_sha256']=expected;head=['--cnn-head',candidate['checkpoint']]
        else:
            expected=lock['classifier_sha256'][name]
            if sha256(candidate['classifier'])!=expected:raise ValueError('Locked classifier changed')
            selection['classifier_sha256']=expected;head=['--classifier',candidate['classifier']]
        encoders={}
        for encoder in candidate['encoders']:
            if encoder=='clip':encoders[encoder]=dict(kind='clip',model_revision=lock['allowed_clip_revision'])
            else:
                checkpoint=OUT/encoder/'best.pt'
                if sha256(checkpoint)!=lock['checkpoint_sha256'][encoder]:raise ValueError('Locked encoder changed')
                encoders[encoder]=dict(kind='dct' if encoder=='dct_guide' else 'rgb',checkpoint=str(checkpoint))
                if encoder=='dct_guide':encoders[encoder]['dct_stats']=json.loads((DATASETS['genimage'].parent/'dct/dct_stats.json').read_text())
        selection_path=out/'selection.json';selection_path.write_text(json.dumps(selection,indent=2))
        encoder_path=out/'encoders.json';encoder_path.write_text(json.dumps(encoders,indent=2))
        cache=projection_cache(name,candidate,purpose)
        run('export_'+purpose,'fnd.forensic.export',[*head,'--cache',str(cache),'--selection',str(selection_path),
            '--encoder-spec',str(encoder_path),'--out',str(out),'--target','ai' if purpose=='ai' else 'broad'])
        (out/'projection_source.json').write_text(json.dumps(dict(path=str(cache),sha256=sha256(cache)),indent=2))
        print('EXPORTED',purpose,name,flush=True)
    (OUT/'release_bundles.json').write_text(json.dumps(outputs,indent=2))


if __name__=='__main__':main()
