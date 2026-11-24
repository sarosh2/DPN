"""Restart-safe GPU/CPU final training with exact model + data fingerprints."""
from __future__ import annotations
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, roc_auc_score)
from src.config_registry import atomic_json,json_hash
from src.data_io import hash_file,load_splits
from src.model_factory import (instantiate,model_counts,seed_all)
from src.results import upsert,METRICS


SOURCE_FILES=('src/utils.py','src/dpn.py','src/model_factory.py','src/training.py',
              'src/data_io.py','src/config_registry.py','src/models/dpn.py',
              'src/models/dpn_flexible.py','src/models/resnet.py')


def fingerprint(repo,dataset_config,model_config,manifest,variant):
    digest=hashlib.sha256()
    for obj in (dataset_config,model_config,manifest,{'variant':variant}):
        digest.update(json.dumps(obj,sort_keys=True,allow_nan=False).encode())
    for source in SOURCE_FILES:
        digest.update(source.encode());digest.update(hash_file(Path(repo)/source).encode())
    return digest.hexdigest()


def atomic_save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    torch.save(data,temporary)
    os.replace(temporary,path)


def logits_loss(logits,target,out):
    if out==1:
        return F.binary_cross_entropy_with_logits(logits.reshape(-1),target.reshape(-1).float())
    return F.cross_entropy(logits,target.long().reshape(-1))


def probabilities(logits,out):
    if out==1:
        p=torch.sigmoid(logits.reshape(-1))
        return torch.stack((1-p,p),dim=-1)
    return torch.softmax(logits,dim=1)


def eval_metrics_from_arrays(y,proba):
    nclass=proba.shape[1];y=np.asarray(y).ravel().astype(np.int64)
    cls=np.asarray(proba).argmax(axis=1)
    result={
        'accuracy':float(accuracy_score(y,cls)),
        'macro_f1':float(f1_score(y,cls,average='macro',zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,cls)),
        'log_loss':float(log_loss(y,proba,labels=list(range(nclass)))),
        'brier':float(np.mean(np.sum((proba-np.eye(nclass)[y])**2,axis=1))),
    }
    conf=proba.max(axis=1); correct=(cls==y).astype(float)
    bin_edges=np.linspace(0.,1.,16);ece=0.
    for i in range(15):
        mask=((conf>=bin_edges[i])&(conf<=bin_edges[i+1])) if i==14 else ((conf>=bin_edges[i])&(conf<bin_edges[i+1]))
        if mask.any():ece+=(mask.mean())*abs(float(correct[mask].mean())-float(conf[mask].mean()))
    result['ece15']=float(ece)
    try:
        if nclass==2:result['auroc']=float(roc_auc_score(y,proba[:,1]))
        else:result['auroc']=float(roc_auc_score(y,proba,multi_class='ovr',average='macro',labels=list(range(nclass))))
    except ValueError:result['auroc']=None
    return result


def evaluate(model,x,y,out,device,batch=8192):
    model.eval();loss_total=0.;n=0;all_p=[]
    with torch.no_grad():
        for pos in range(0,len(y),batch):
            xx=x[pos:pos+batch].to(device)
            yy=y[pos:pos+batch].to(device)
            z=model(xx)
            loss=logits_loss(z,yy,out)
            loss_total+=float(loss)*len(yy);n+=len(yy)
            all_p.append(probabilities(z,out).detach().cpu().numpy())
    if n==0:raise ValueError('Empty evaluation split')
    prob=np.concatenate(all_p)
    return {'loss':loss_total/n,**eval_metrics_from_arrays(y.cpu().numpy(),prob)}


def epoch_train(model,opt,x,y,out,device,batch,seed,epoch):
    model.train();loss_total=0.; n=0;correct=0
    # Shuffle using an explicit epoch seed. Checkpoints also retain RNG state.
    g=torch.Generator(device='cpu').manual_seed(int(seed)*1000003+int(epoch))
    order=torch.randperm(len(y),generator=g)
    if device.type=='cuda':order=order.to(device)
    for ix in range(0,len(order),batch):
        batch_indices=order[ix:ix+batch]
        xx=x[batch_indices].to(device);yy=y[batch_indices].to(device)
        opt.zero_grad(set_to_none=True)
        z=model(xx)
        loss=logits_loss(z,yy,out)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
        loss.backward();opt.step()
        loss_total+=float(loss.detach())*len(yy);n+=len(yy)
        preds=(z.reshape(-1)>0).long() if out==1 else z.argmax(dim=1)
        correct+=int((preds==yy).sum())
    return {'loss':loss_total/n,'accuracy':correct/n}


def _restore_rng(checkpoint,device):
    torch.set_rng_state(checkpoint['rng_cpu'])
    random.setstate(checkpoint['rng_py'])
    np.random.set_state(checkpoint['rng_numpy'])
    if device.type=='cuda' and checkpoint.get('rng_cuda') is not None:
        torch.cuda.set_rng_state(checkpoint['rng_cuda'],device=device)


def _save_state(path,model,opt,epoch,best_score,best_loss,best_epoch,stale,wall,finger,device,best_path):
    state={'schema':'dpn_epoch_v2','fingerprint':finger,'epoch':epoch,
           'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
           'optimizer':opt.state_dict(),'best_score':best_score,'best_epoch':best_epoch,
           'stale':stale,'elapsed_wall_seconds':wall,'best_loss':best_loss,
           'best_checkpoint_sha256':hash_file(best_path),
           'rng_cpu':torch.get_rng_state(),'rng_py':random.getstate(),
           'rng_numpy':np.random.get_state(),
           'rng_cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
    atomic_save(path,state)


def restore_lth_reparameterization(model, state, device):
    """Attach saved LTH masks before loading weight_orig from checkpoint."""
    from torch.nn.utils import prune
    for name,layer in [*(('layers.'+str(i),m) for i,m in enumerate(model.layers)),
                       ('head',model.head)]:
        mask_key=name+'.weight_mask'
        if mask_key in state and not hasattr(layer,'weight_orig'):
            prune.custom_from_mask(layer,'weight',state[mask_key].to(device))


def _load_state(path,model,opt,expected,device):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('schema')!='dpn_epoch_v2' or obj.get('fingerprint')!=expected:
        raise RuntimeError(f'INCOMPATIBLE CHECKPOINT: {path}. Archive it before changing protocol.')
    if hasattr(model,'head') and hasattr(model,'layers'):
        restore_lth_reparameterization(model,obj['model'],device)
    model.load_state_dict(obj['model'],strict=True)
    opt.load_state_dict(obj['optimizer'])
    for state in opt.state.values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.to(device)
    _restore_rng(obj,device)
    return obj


def _lth_layers(model):
    return [*model.layers,model.head]


def _prepare_lth(model,seed_config,root,x,y,device,out,finger,deadline):
    """The two cumulative global magnitude mask rounds with true rewinding."""
    from torch.nn.utils import prune
    seed=int(seed_config['seed']); pre=int(seed_config['regularization']['pretrain_epochs_per_round'])
    targets=[float(v) for v in seed_config['regularization']['cumulative_mask_targets']]
    cp=root/'lth_mask_search';cp.mkdir(parents=True,exist_ok=True)
    initial=[(layer.weight.detach().clone(),layer.bias.detach().clone()) for layer in _lth_layers(model)]
    search_time=0.
    for stage,target in enumerate(targets,1):
        completed=cp/f'stage_{stage}_complete.pt'
        if completed.exists():
            obj=torch.load(completed,map_location='cpu',weights_only=False)
            if obj['fingerprint']!=finger:raise RuntimeError('LTH mask stage checkpoint mismatch')
            for layer,mask in zip(_lth_layers(model),obj['masks']):
                if hasattr(layer,'weight_orig'):prune.remove(layer,'weight')
                prune.custom_from_mask(layer,'weight',mask.to(device))
            model.load_state_dict(obj['model'],strict=True)
            search_time=obj['elapsed_seconds'];continue
        opt=torch.optim.AdamW(model.parameters(),lr=float(seed_config['training']['learning_rate']),
                              weight_decay=float(seed_config['training']['weight_decay']))
        epoch_file=cp/f'stage_{stage}_latest.pt';begin=1
        if epoch_file.exists():
            item=torch.load(epoch_file,map_location='cpu',weights_only=False)
            if item['fingerprint']!=finger:raise RuntimeError('LTH pretrain fingerprint mismatch')
            # If the current model has no pruning reparameterization, restore masks first.
            for layer,mask in zip(_lth_layers(model),item['masks']):
                if not hasattr(layer,'weight_orig') and mask is not None:
                    prune.custom_from_mask(layer,'weight',mask.to(device))
            model.load_state_dict(item['model'],strict=True)
            opt.load_state_dict(item['optimizer'])
            for state in opt.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v):state[k]=v.to(device)
            _restore_rng(item,device)
            begin=item['epoch']+1
        started=time.perf_counter()
        for epoch in range(begin,pre+1):
            if deadline and time.time()>deadline-25:
                raise TimeoutError('Approaching allocation deadline, LTH checkpoint saved')
            epoch_train(model,opt,x['train'],y['train'],out,device,
                        int(seed_config['training']['batch_size']),seed,100*stage+epoch)
            state={'fingerprint':finger,'epoch':epoch,'model':model.state_dict(),
                   'optimizer':opt.state_dict(),
                   'masks':[getattr(layer,'weight_mask',None) for layer in _lth_layers(model)],
                   'rng_cpu':torch.get_rng_state(),'rng_py':random.getstate(),
                   'rng_numpy':np.random.get_state(),
                   'rng_cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
            atomic_save(epoch_file,state)
        search_time+=time.perf_counter()-started
        layers=_lth_layers(model)
        # Prune only still active linear weights; masks remain on the model's device.
        active=[]
        for layer in layers:
            mask=getattr(layer,'weight_mask',torch.ones_like(layer.weight)).detach()
            w=getattr(layer,'weight_orig',layer.weight).detach()
            active.append((w.abs()*mask)[mask.bool()])
        active_values=torch.cat(active)
        allocated=sum(layer.weight.numel() for layer in layers)
        current_zeros=sum(int((getattr(layer,'weight_mask',torch.ones_like(layer.weight))==0).sum()) for layer in layers)
        required=max(0,round(target*allocated)-current_zeros)
        if required>0:
            if required>active_values.numel():raise RuntimeError('LTH target exceeds available weights')
            threshold=torch.kthvalue(active_values,required).values
            for layer in layers:
                oldmask=getattr(layer,'weight_mask',torch.ones_like(layer.weight)).detach()
                weight=getattr(layer,'weight_orig',layer.weight).detach()
                newmask=oldmask*(weight.abs()>threshold).to(oldmask.dtype)
                if hasattr(layer,'weight_orig'):
                    prune.remove(layer,'weight')
                prune.custom_from_mask(layer,'weight',newmask.to(device))
        # True lottery-ticket rewind; keep masks, restore both weights and biases.
        with torch.no_grad():
            for layer,(w,b) in zip(layers,initial):
                layer.weight_orig.copy_(w.to(device));layer.bias.copy_(b.to(device))
        atomic_save(completed,{'fingerprint':finger,'model':model.state_dict(),
             'masks':[layer.weight_mask.detach().cpu() for layer in layers],
             'elapsed_seconds':search_time})
    return search_time


def _checkpoint_dir(repo,dataset,seed,model,variant):
    suffix=('__'+variant) if variant else ''
    return Path(repo)/'checkpoints'/dataset/f'seed_{seed}'/(model+suffix)


def run_training(repo,ds_cfg,seed_config,device='cpu',deadline=None,variant=None):
    """Train one arm/variant; safely skip complete or resume unfinished runs."""
    repo=Path(repo);ds=ds_cfg['dataset'];seed=int(seed_config['seed']);name=seed_config['model']
    device=torch.device(device)
    x_np,y_np,manifest=load_splits(repo,ds_cfg,seed,include_test=False)
    fp=fingerprint(repo,ds_cfg,seed_config,manifest,variant)
    cp=_checkpoint_dir(repo,ds,seed,name,variant);cp.mkdir(parents=True,exist_ok=True)
    for part, title, message in (
        (cp.parent, f'{ds}, seed {seed}', 'Each model folder contains independent per-epoch recovery files.'),
        (cp, f'{ds}, seed {seed}: {name}',
         'Latest and best checkpoints use frozen dataset, source and model fingerprints. '
         'The best checkpoint is selected on validation only. '
         'Resume with `python run.py --datasets '+ds+' --seeds '+str(seed)+' --models '+name+'`.')):
        doc=part/'README.md'
        if not doc.exists():doc.write_text(f'# {title}\n\n{message}\n')
    done=cp/'completed.json';latest=cp/'latest.pt';best=cp/'best.pt'
    if done.exists():
        obj=json.loads(done.read_text())
        if obj['fingerprint']!=fp:raise RuntimeError(f'Incompatible completed model: {done}')
        if not best.is_file():raise RuntimeError('Completed model best.pt missing')
        if name!='staged_dpn':upsert(repo,'training',obj['summary']['result'])
        return obj['summary']
    ex=ds_cfg['expected'];out=int(ex['output_logits'])
    x={k:torch.tensor(v,device=device,dtype=torch.float32) for k,v in x_np.items()}
    y={k:torch.tensor(v,device=device,dtype=torch.long) for k,v in y_np.items()}
    model=instantiate(seed_config,ds_cfg,seed,
                      repo/'configs'/ds/str(seed),variant).to(device)
    train_spec=seed_config['training']
    if latest.exists() and name=='lth_mlp':
        saved_state=torch.load(latest,map_location='cpu',weights_only=False)
        if saved_state.get('fingerprint')!=fp:
            raise RuntimeError(f'INCOMPATIBLE CHECKPOINT: {latest}')
        restore_lth_reparameterization(model,saved_state['model'],device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(train_spec['learning_rate']),
                          weight_decay=float(train_spec['weight_decay']))
    start=1;best_acc=-float('inf');best_loss=float('inf');best_epoch=0;stale=0;elapsed=0.;search_time=0.
    if latest.exists():
        obj=_load_state(latest,model,opt,fp,device)
        start=int(obj['epoch'])+1;best_acc=float(obj['best_score']);best_epoch=int(obj['best_epoch'])
        best_loss=float(obj.get('best_loss',float('inf')))
        stale=int(obj['stale']);elapsed=float(obj['elapsed_wall_seconds'])
        expected_best=obj.get('best_checkpoint_sha256')
        if not expected_best:
            raise RuntimeError('Checkpoint lacks transactional best checkpoint hash')
        if not best.is_file() or hash_file(best)!=expected_best:
            previous=cp/'best_previous.pt'
            if not previous.is_file() or hash_file(previous)!=expected_best:
                raise RuntimeError('Neither best checkpoint nor committed backup matches latest.pt')
            staged=best.with_name(best.name+'.recover')
            shutil.copyfile(previous,staged)
            os.replace(staged,best)
    elif name=='lth_mlp':
        search_time=_prepare_lth(model,seed_config,cp,x,y,device,out,fp,deadline)
        opt=torch.optim.AdamW(model.parameters(),lr=float(train_spec['learning_rate']),
                              weight_decay=float(train_spec['weight_decay']))
    clock=time.perf_counter();max_epochs=int(train_spec['max_epochs']);patience=int(train_spec['patience'])
    last={};history_path=cp/'epochs.jsonl'
    for epoch in range(start,max_epochs+1):
        if deadline and time.time()>deadline-30:
            print(f'[checkpoint-safe-stop] {ds}/{seed}/{name} epoch={epoch}',flush=True)
            return None
        t0=time.perf_counter()
        tr=epoch_train(model,opt,x['train'],y['train'],out,device,
                       int(train_spec['batch_size']),seed,epoch)
        val=evaluate(model,x['val'],y['val'],out,device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elif device.type=='mps':torch.mps.synchronize()
        epoch_seconds=time.perf_counter()-t0
        improved=(val['accuracy']>best_acc or
                  (val['accuracy']==best_acc and val['log_loss']<best_loss))
        if improved:
            best_acc=val['accuracy'];best_loss=val['log_loss'];best_epoch=epoch;stale=0
            if best.is_file():
                temporary=cp/'best_previous.pt.tmp'
                shutil.copyfile(best,temporary)
                os.replace(temporary,cp/'best_previous.pt')
            atomic_save(best,{'schema':'dpn_best_v2','fingerprint':fp,'epoch':epoch,
                'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'validation':val})
        else:stale+=1
        elapsed+=epoch_seconds
        state={'epoch':epoch,'elapsed_seconds':epoch_seconds,'train':tr,'validation':val,
               'best_accuracy':best_acc,'best_epoch':best_epoch,'stale':stale}
        with history_path.open('a') as f:f.write(json.dumps(state,sort_keys=True)+'\n')
        _save_state(latest,model,opt,epoch,best_acc,best_loss,best_epoch,stale,elapsed,fp,device,best)
        print(f'[epoch] {ds}/{seed}/{name}{"/"+variant if variant else ""} '
              f'{epoch}/{max_epochs} val_acc={val["accuracy"]:.5f} '
              f'val_f1={val["macro_f1"]:.5f} time={epoch_seconds:.2f}s',flush=True)
        last=state
        if stale>patience:break
    if not best.is_file():raise RuntimeError('No validation winner checkpoint')
    selected=torch.load(best,map_location='cpu',weights_only=False)
    if selected['fingerprint']!=fp:raise RuntimeError('Best checkpoint fingerprint mismatch')
    if name=='lth_mlp':restore_lth_reparameterization(model,selected['model'],device)
    model.load_state_dict(selected['model'],strict=True)
    train_final=evaluate(model,x['train'],y['train'],out,device)
    val_final=evaluate(model,x['val'],y['val'],out,device)
    active,allocated=model_counts(model)
    architecture=seed_config['architecture']
    if variant is not None:architecture=architecture['candidates'][variant]
    target=int(architecture.get('target_allocated',allocated))
    widths=architecture.get('hidden_widths',[])
    row={'dataset':ds,'seed':seed,'model':name,'variant':variant or '',
      'split':'validation','status':'complete','epochs_completed':last.get('epoch',start-1),
      'best_epoch':best_epoch,'early_stopped':last.get('epoch',0)<max_epochs,
      'training_wall_seconds':elapsed,'initialization_seed':seed,
      'optimizer':'AdamW','learning_rate':train_spec['learning_rate'],
      'weight_decay':train_spec['weight_decay'],'batch_size':train_spec['batch_size'],
      'max_epochs':max_epochs,'patience':patience,
      'normalization':seed_config['regularization']['normalization'],
      'dropout':seed_config['regularization'].get('dropout',0.),
      'lth_search_seconds':search_time,'parameters_allocated':allocated,
      'parameters_active':active,'parameter_budget_target':target,
      'parameter_budget_gap_percent':100*(allocated-target)/target,
      'hidden_blocks':len(widths),'output_blocks':len(architecture.get('output_widths',[])),
      'mlp_hidden_depth':len(ds_cfg['reference_mlp']['hidden_widths']),
      'checkpoint_fingerprint':fp,'dataset_config_sha256':json_hash(ds_cfg),
      'model_config_sha256':json_hash(seed_config),
      'ticket_sha256':seed_config['ticket_sha256'],'train_loss':train_final['loss'],
      'validation_loss':val_final['loss'],'test_accessed':False,
      'data_source':ds_cfg['source']['path'],'data_split_provenance':'official_train_validation'}
    for m in METRICS:
        row['train_'+m]=train_final[m] if train_final[m] is not None else ''
        row['val_'+m]=val_final[m] if val_final[m] is not None else ''
    summary={'fingerprint':fp,'result':row,'validation':val_final}
    atomic_json(done,{'fingerprint':fp,'summary':summary})
    if name!='staged_dpn':upsert(repo,'training',row)
    return summary


def run_staged(repo,ds_cfg,seed_config,device='cpu',deadline=None):
    options=seed_config['architecture']['candidates']
    successes={}
    for variant in ('staged_compressed','staged_input_skip'):
        if options[variant]['status']=='infeasible':continue
        result=run_training(repo,ds_cfg,seed_config,device,deadline,variant)
        if result is None:return None
        successes[variant]=result
    if not successes:raise RuntimeError('Both staged variants are infeasible; cannot fabricate staged model')
    chosen=min(successes,key=lambda k:(-successes[k]['validation']['accuracy'],
                successes[k]['validation']['loss'],k))
    result=successes[chosen]
    upsert(repo,'training',result['result'])
    atomic_json(_checkpoint_dir(repo,ds_cfg['dataset'],seed_config['seed'],'staged_dpn',None)/'selected.json',
                {'variant':chosen,'validation':result['validation'],
                 'selection':'highest_accuracy_then_lowest_loss','test_accessed':False})
    return result


def isolated_latency(repo,ds_cfg,seed_config,device='cpu',variant=None):
    repo=Path(repo);ds=ds_cfg['dataset'];seed=int(seed_config['seed']);name=seed_config['model']
    if name=='staged_dpn' and variant is None:
        selected=_checkpoint_dir(repo,ds,seed,name,None)/'selected.json'
        if not selected.exists():raise FileNotFoundError('Staged winner not selected')
        variant=json.loads(selected.read_text())['variant']
    cp=_checkpoint_dir(repo,ds,seed,name,variant)
    best=cp/'best.pt';done=cp/'completed.json'
    if not best.is_file() or not done.is_file():raise FileNotFoundError('Cannot benchmark an unfinished training run')
    model=instantiate(seed_config,ds_cfg,seed,repo/'configs'/ds/str(seed),variant).to(device)
    state=torch.load(best,map_location='cpu',weights_only=False)
    if state['fingerprint']!=json.loads(done.read_text())['fingerprint']:
        raise RuntimeError('Latency ticket fingerprint mismatch')
    if name=='lth_mlp':restore_lth_reparameterization(model,state['model'],device)
    model.load_state_dict(state['model'],strict=True);model.eval()
    x=torch.zeros((1,int(ds_cfg['expected']['input_features'])),device=device)
    device=torch.device(device)
    with torch.no_grad():
        for _ in range(10):model(x)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elif device.type=='mps':torch.mps.synchronize()
        timings=[]
        for _ in range(int(ds_cfg['evaluation'].get('latency_repeats',50))):
            t0=time.perf_counter();model(x)
            if device.type=='cuda':torch.cuda.synchronize(device)
            elif device.type=='mps':torch.mps.synchronize()
            timings.append(1000*(time.perf_counter()-t0))
    median=float(np.median(timings))
    rows=[r for r in json.loads(done.read_text())['summary']['result'].items()]
    row=dict(rows);row['inference_latency_batch1_ms']=median
    upsert(repo,'training',row)
    atomic_json(cp/'latency.json',{'ms_median':median,'repeats':len(timings),
             'measurement':'isolated_batch1_no_training_processes','device':str(device)})
    return median
