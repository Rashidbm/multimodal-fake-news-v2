"""Extract fresh features and evaluate the already frozen candidate once."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/semantic_resolution'


def main():
    env = dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
        TQDM_DISABLE='1',TRANSFORMERS_VERBOSITY='error',TOKENIZERS_PARALLELISM='false')
    common=['--csv','data/processed/semantic_resolution_confirmation.csv','--splits','test']
    jobs=[
        ('confirmation_blip',['-m','fnd.cache_blip',*common,'--out',str(OUT/'confirmation_blip.pt')]),
        ('confirmation_v1',['-m','fnd.cache_v1_features',*common,'--checkpoint',
            'outputs/v1_alignment_round/standardized_gate/epoch_5.pt','--out',str(OUT/'confirmation_v1.pt')]),
        ('confirmation_clip',['-m','fnd.cache_clip_large',*common,'--source',
            'outputs/recall_improvement/clip_large_source.json','--out',str(OUT/'confirmation_clip.pt')]),
        ('confirmation_evaluate',['-m','fnd.evaluate_semantic_resolution'])]
    for name,args in jobs:
        state=dict(stage=name,started_unix=time.time())
        (OUT/'CONFIRMATION_STATUS.json').write_text(json.dumps(state,indent=2))
        print('Starting '+name,flush=True)
        with (OUT/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(stage=name+'_failed',returncode=result.returncode)
            (OUT/'CONFIRMATION_STATUS.json').write_text(json.dumps(state,indent=2))
            raise RuntimeError(name+' failed')
    (OUT/'CONFIRMATION_STATUS.json').write_text(json.dumps(dict(stage='complete',completed_unix=time.time()),indent=2))


if __name__=='__main__':
    main()
