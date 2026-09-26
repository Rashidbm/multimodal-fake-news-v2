"""Train guide RGB/DCT baselines with validation-only selection and resumable state."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .data import ForensicDataset
from .metrics import report
from .model import ForensicModel, RGB_WEIGHTS, VENDOR
from .preprocess import read_rows,sha256
from .remote_genimage import OUT


def worker_seed(worker):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed);torch.set_num_threads(1)


def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix('.partial');torch.save(value,temporary);temporary.replace(path)


@torch.inference_mode()
def predict(model,loader,device,features=False):
    model.eval();p=np.empty(len(loader.dataset),dtype=np.float64);loss=0.;n=0;hidden=[];indices=[]
    for pixels,labels,idx in loader:
        pixels=pixels.to(device);labels=labels.to(device)
        if features:
            logits,representation=model(pixels,True);hidden.append(representation.cpu());indices.append(idx)
        else:logits=model(pixels)
        loss+=F.binary_cross_entropy_with_logits(logits,labels,reduction='sum').item();n+=len(labels)
        p[idx.numpy()]=logits.sigmoid().cpu().numpy()
    result=dict(probabilities=p,loss=loss/n)
    if features:
        joined=torch.cat(hidden);order=torch.cat(indices);result['features']=joined[order.argsort()]
    return result


def loader(rows,kind,args,training=False,corruption=None):
    data=ForensicDataset(rows,kind,training,args.dct_dir,mask=not args.no_mask,robust=args.robust and training,corruption=corruption,
                         blur_probability=getattr(args,'blur_probability',0.))
    return DataLoader(data,batch_size=args.batch_size,shuffle=training,num_workers=args.workers,
                      worker_init_fn=worker_seed,pin_memory=args.device.startswith('cuda'),
                      persistent_workers=False,generator=torch.Generator().manual_seed(args.seed))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['rgb','dct'],required=True)
    ap.add_argument('--csv',default=str(OUT/'manifest.csv'));ap.add_argument('--dct-dir',default=str(OUT/'dct'))
    ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--device',default='mps');ap.add_argument('--seed',type=int,default=2026091408)
    ap.add_argument('--no-mask',action='store_true');ap.add_argument('--robust',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--initial-checkpoint');ap.add_argument('--balance-domains',action='store_true')
    ap.add_argument('--blur-probability',type=float,default=0.);ap.add_argument('--unfreeze-epoch',type=int,default=0)
    ap.add_argument('--selection',choices=['AP','worst_domain_AUC'],default='AP')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    rows=read_rows(args.csv);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    if not train or not val:raise ValueError('Missing train/validation split')
    if any(int(r['label']) not in (0,1) for r in rows):raise ValueError('Image targets must be binary')
    if any({int(r['label']) for r in subset}!={0,1} for subset in (train,val)):raise ValueError('Both classes are required in each development split')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate image identities')
    train_ids={r['sample_id'] for r in train}
    if train_ids&{r['sample_id'] for r in val}:raise ValueError('Train/validation overlap')
    manifest=dict(args=vars(args),csv_sha256=sha256(args.csv),training_rows=len(train),validation_rows=len(val),
                  counts=dict(Counter(str(r['label']) for r in train)),test_used=False,input='image pixels only',
                  label_definition='0 authentic, 1 AI generated',source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  batchnorm_policy='Running statistics in frozen modules are frozen with their parameters',
                  hardware_adaptation='Single-process MPS; no CUDA-only distributed launcher; pin_memory false on MPS')
    if args.kind=='rgb':
        manifest['initial_weights_sha256']=sha256(RGB_WEIGHTS)
        manifest['upstream_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()
        manifest['checkpoint_name_note']='Author Drive uses legacy rn50ft_spectralmask.pth; legacy implementation uses Fourier FFT.'
        manifest['masking']='Upstream FrequencyMaskGenerator unchanged; band all, ratio .15, probability .5; train only'
    else:
        manifest['initial_weights']='torchvision ResNet50 IMAGENET1K_V1, new Kaiming one-channel conv1'
        manifest['dct_stats_sha256']=sha256(Path(args.dct_dir)/'dct_stats.json')
    if args.initial_checkpoint:
        manifest['adaptation_initial_checkpoint_sha256']=sha256(args.initial_checkpoint)
    if args.balance_domains:
        manifest['label_definition']='0 authentic image, 1 locally edited or AI-generated image'
        manifest['extension']='Broad manipulation adaptation; distinct from the GenImage-only guide experiments'
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and not args.resume:raise FileExistsError('Existing experiment requires --resume or new output directory')
    if manifest_path.exists() and args.resume:
        previous=json.loads(manifest_path.read_text())
        fixed=['kind','csv','dct_dir','batch_size','seed','no_mask','robust','lr','initial_checkpoint','balance_domains','blur_probability','unfreeze_epoch','selection']
        if any(previous['args'].get(key)!=vars(args).get(key) for key in fixed):raise ValueError('Resume hyperparameters changed; use a new experiment directory')
    if not manifest_path.exists():manifest_path.write_text(json.dumps(manifest,indent=2))
    model=ForensicModel(args.kind).to(args.device)
    if args.initial_checkpoint:
        initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
        if initial['kind']!=args.kind:raise ValueError('Initial checkpoint architecture mismatch')
        model.load_state_dict(initial['model'],strict=True)
    sample_weights=torch.ones(len(train),dtype=torch.float32)
    if args.balance_domains:
        domains={r['domain'] for r in train};counts=Counter((r['domain'],int(r['label'])) for r in train)
        if any(counts[d,l]==0 for d in domains for l in (0,1)):raise ValueError('Each domain must contain both labels')
        sample_weights=torch.tensor([len(train)/(len(domains)*2*counts[r['domain'],int(r['label'])]) for r in train],dtype=torch.float32)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3)
    start=1;best=-1.;bad=0;history=[]
    if args.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if saved['csv_sha256']!=manifest['csv_sha256']:raise ValueError('Resume data changed')
        model.set_phase(saved['phase']);model.load_state_dict(saved['model'])
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=saved['lr'],weight_decay=1e-4)
        optimizer.load_state_dict(saved['optimizer']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3)
        scheduler.load_state_dict(saved['scheduler']);start=saved['epoch']+1;best=saved['best_score'];bad=saved['bad_epochs'];history=saved['history']
        torch.set_rng_state(saved['rng']);random.setstate(saved['python_rng'])
        np.random.set_state((saved['numpy_rng'][0],np.array(saved['numpy_rng'][1],dtype=np.uint32),*saved['numpy_rng'][2:]))
    train_loader=loader(train,args.kind,args,True);val_loader=loader(val,args.kind,args)
    print('START',args.kind,'train',len(train),'val',len(val),'device',args.device,'trainable',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    for epoch in range(start,args.epochs+1):
        if bad>=5:
            print('Early stopping was already reached before restart; finalizing the checkpoint',flush=True)
            break
        # Epoch-specific shuffling and worker seeds make restarts independent of earlier loader iterations.
        train_loader.generator.manual_seed(args.seed+epoch)
        if (args.kind=='dct' and epoch==6) or (args.unfreeze_epoch and epoch==args.unfreeze_epoch):
            model.set_phase(2)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3)
            print('PHASE 2: all layers trainable; optimizer reset; clipping=1',flush=True)
        model.train();begin=time.monotonic();loss_sum=0.;n=0
        for step,(pixels,labels,indices) in enumerate(train_loader,1):
            pixels=pixels.to(args.device);labels=labels.to(args.device)
            optimizer.zero_grad(set_to_none=True);logits=model(pixels)
            element_loss=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
            loss=(element_loss*sample_weights[indices].to(args.device)).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward()
            if model.phase==2:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach())*len(labels);n+=len(labels)
            if step%50==0:
                status=dict(epoch=epoch,step=step,steps=len(train_loader),train_loss=loss_sum/n,seconds=time.monotonic()-begin)
                print(json.dumps(status),flush=True);(out/'status.json').write_text(json.dumps(status))
        evaluated=predict(model,val_loader,args.device);metrics=report(val,evaluated['probabilities'])
        ap_value=metrics['overall']['AP'] if args.selection=='AP' else min(m['AUC'] for m in metrics['per_generator'].values())
        metrics['selection_metric_name']=args.selection;metrics['selection_metric_value']=ap_value;scheduler.step(ap_value)
        improved=ap_value>best
        if improved:best=ap_value;bad=0
        else:bad+=1
        record=dict(epoch=epoch,phase=model.phase,train_loss=loss_sum/n,val_loss=evaluated['loss'],metrics=metrics,
                    seconds=time.monotonic()-begin,lr=optimizer.param_groups[0]['lr'],best_score=best,bad_epochs=bad)
        history.append(record);(out/'history.json').write_text(json.dumps(history,indent=2))
        rng_np=np.random.get_state();rng_np=(rng_np[0],rng_np[1].tolist(),*rng_np[2:])
        saved=dict(kind=args.kind,epoch=epoch,phase=model.phase,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),lr=optimizer.param_groups[0]['lr'],csv_sha256=manifest['csv_sha256'],
                   best_score=best,bad_epochs=bad,history=history,rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=rng_np)
        atomic_save(saved,out/'last.pt')
        if improved:
            atomic_save(dict(kind=args.kind,epoch=epoch,phase=model.phase,model=model.state_dict(),
                             csv_sha256=manifest['csv_sha256'],metrics=metrics),out/'best.pt')
            with (out/'val_predictions.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['sample_id','generator','label','probability']);writer.writeheader()
                writer.writerows(dict(sample_id=r['sample_id'],generator=r['generator'],label=r['label'],probability=float(p)) for r,p in zip(val,evaluated['probabilities']))
        print('EPOCH',epoch,args.selection,round(ap_value,5),'real',round(metrics['overall']['real_recall'],4),'fake',round(metrics['overall']['fake_recall'],4),'seconds',round(record['seconds']),flush=True)
        if bad>=5:print('EARLY STOP',flush=True);break
    (out/'complete.json').write_text(json.dumps(dict(best_selection_metric=best,selection_metric=args.selection,epochs_completed=len(history),checkpoint_sha256=sha256(out/'best.pt'),test_used=False),indent=2))
    print('COMPLETE',args.kind,best,flush=True)


if __name__=='__main__':main()
