"""Original outer structural LTH pruning with seed-local restart files."""
from __future__ import annotations
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import time
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from src import utils
from src.dpn import DPN
from src.model_factory import seed_all
from src.data_io import load_splits,hash_file
from src.config_registry import atomic_json,json_hash,per_seed_dir
from src.results import upsert


def canonical_regularized_forward(model,x,dropout):
    """Non-affine pre-linear LN on growing input; original pruning topology."""
    hidden=model.total_nodes-model.output_nodes
    offset=0
    for i,w in enumerate(model.weights):
        z=F.linear(F.layer_norm(x,(x.shape[-1],),eps=1e-5),
                   w*getattr(model,f'mask_{i}'),model.biases[offset:offset+w.shape[0]])
        if offset<hidden:
            z=F.relu(z)
            if dropout:z=F.dropout(z,p=dropout,training=model.training)
        x=torch.cat((x,z),dim=-1);offset+=w.shape[0]
    if offset!=model.total_nodes:raise RuntimeError('Pruning DPN lost logits')
    return x[:,-model.output_nodes:]


def adapted_pruner_train(model,train_loader,val_loader,epochs,optimizer,criterion,
                         flatten=True,device=None,*,dropout):
    """Step79 train bridge, not a replacement for the original pruning search."""
    device=device or next(model.parameters()).device
    old_forward=model.__dict__.get('forward',None)
    model.forward=types.MethodType(lambda self,x:canonical_regularized_forward(self,x,dropout),model)
    tr=[];va=[]
    try:
        for epoch in range(int(epochs)):
            model.train();correct=total=0;loss_total=0.;t0=time.monotonic()
            for xx,yy in train_loader:
                xx=xx.to(device);yy=yy.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(xx)
                if model.output_nodes==1:
                    logits=logits.reshape(-1);target=yy.float().reshape(-1)
                    preds=(logits>0).long()
                else:
                    target=yy.long().reshape(-1);preds=logits.argmax(dim=1)
                loss=criterion(logits,target)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pruning loss')
                loss.backward();optimizer.step()
                correct+=int((preds==yy.reshape(-1)).sum());total+=len(yy)
                loss_total+=float(loss.detach())*len(yy)
            tr.append([time.monotonic()-t0,loss_total/total,correct/total])
            model.eval();correct=total=0;loss_total=0.
            with torch.no_grad():
                for xx,yy in val_loader:
                    xx=xx.to(device);yy=yy.to(device)
                    logits=model(xx)
                    if model.output_nodes==1:
                        logits=logits.reshape(-1);target=yy.float().reshape(-1)
                        preds=(logits>0).long()
                    else:
                        target=yy.long().reshape(-1);preds=logits.argmax(dim=1)
                    loss=criterion(logits,target)
                    correct+=int((preds==yy.reshape(-1)).sum());total+=len(yy)
                    loss_total+=float(loss)*len(yy)
            va.append([loss_total/total,correct/total])
            print(f'[prune-epoch] {epoch+1}/{epochs} train_acc={tr[-1][2]:.4f} '
                  f'val_acc={va[-1][1]:.4f}',flush=True)
        return np.asarray(tr),np.asarray(va)
    finally:
        if old_forward is None:model.__dict__.pop('forward',None)
        else:model.forward=old_forward


@contextmanager
def pruning_bridge(dropout,rounds):
    """The original outer/inner pruners are not replaced or rewritten.

    Only the train() callback is bridged for binary logits and pre-linear LN.
    For rounds < 10 a local nonpersistent decrement guard prevents a loop.
    """
    old_train=utils.train;old_outer=utils.apply_lth_pruning
    def replacement(*args,**kwargs):
        return adapted_pruner_train(*args,**kwargs,dropout=dropout)
    utils.train=replacement
    if rounds<10:
        text=inspect.getsource(old_outer)
        before='step = rounds // 10'
        if text.count(before)!=1:raise RuntimeError('Cannot apply one-round decrement guard')
        env=dict(utils.__dict__)
        exec(compile(text.replace(before,'step = max(1, rounds // 10)'),'<round-guard>','exec'),env)
        utils.apply_lth_pruning=env['apply_lth_pruning']
    try:yield
    finally:
        utils.train=old_train;utils.apply_lth_pruning=old_outer


def _measure(model,features,device,dropout,repeats):
    model.eval();x=torch.zeros(1,features,device=device)
    with torch.no_grad():
        for _ in range(4):canonical_regularized_forward(model,x,dropout)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elif device.type=='mps':torch.mps.synchronize()
        times=[]
        for _ in range(repeats):
            t0=time.perf_counter();canonical_regularized_forward(model,x,dropout)
            if device.type=='cuda':torch.cuda.synchronize(device)
            elif device.type=='mps':torch.mps.synchronize()
            times.append(1000*(time.perf_counter()-t0))
    return float(np.median(times))


def discover(repo,cfg,seed,device='cuda',deadline=None):
    """Save one real accepted pruning ticket and all original inner snapshots."""
    repo=Path(repo);ds=cfg['dataset'];seed=int(seed);folder=per_seed_dir(repo,ds,seed)
    folder.mkdir(parents=True,exist_ok=True)
    ticket_path=folder/'accepted_pruned_ticket.pt';meta_path=folder/'pruning_metadata.json'
    if ticket_path.exists() or meta_path.exists():
        if not ticket_path.exists() or not meta_path.exists():
            raise RuntimeError('Incomplete final pruning artifact; inspect or restore it, do not overwrite')
        obj=json.loads(meta_path.read_text())
        if obj['ticket_sha256']!=hash_file(ticket_path) or obj['dataset_config_sha256']!=json_hash(cfg):
            raise RuntimeError('Wrong-source accepted pruning ticket')
        return obj
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but is not available')
    if device.type=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple Metal requested but is not available')
    X,y,data_manifest=load_splits(repo,cfg,seed,include_test=False)
    xx={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in X.items()}
    yy={k:torch.tensor(v,dtype=torch.long,device=device) for k,v in y.items()}
    p=cfg['pruning'];hidden=int(p['initial_hidden_blocks']);out=int(cfg['expected']['output_logits'])
    features=int(cfg['expected']['input_features']);widths=[1]*(hidden+out)
    seed_all(seed)
    model=DPN(features,sum(widths),out,minimal=False,hidden_dims=widths).to(device)
    initial_params=sum(v.numel() for v in model.parameters());initial_blocks=len(model.weights)
    dropout=float(p['dropout']);lr=float(cfg['training']['learning_rate'])
    wd=float(cfg['training']['weight_decay'])
    # Original outer pruning rebuilds optimizers via type(opt)(..., **defaults).
    # PyTorch 2.10 AdamW exposes a decoupled_weight_decay default that its
    # own constructor does not accept. Use equivalent Adam(decoupled=True)
    # when supported, as in the successful legacy Roihu experiments.
    import inspect
    if 'decoupled_weight_decay' in inspect.signature(torch.optim.Adam).parameters:
        opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd,decoupled_weight_decay=True)
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    train=DataLoader(TensorDataset(xx['train'],yy['train']),batch_size=int(cfg['training']['batch_size']),shuffle=False,num_workers=0)
    val=DataLoader(TensorDataset(xx['val'],yy['val']),batch_size=max(64,int(cfg['training']['batch_size'])),shuffle=False,num_workers=0)
    criterion=nn.BCEWithLogitsLoss() if out==1 else nn.CrossEntropyLoss()
    rounds=int(p['outer_initial_rounds']);epochs=int(p['candidate_training_epochs'])
    start_lat=_measure(model,features,device,dropout,5)
    print(f'[discover] {ds}/{seed} initial={initial_params} {initial_blocks} blocks '
          f'outer={rounds} epochs={epochs}',flush=True)
    t0=time.monotonic()
    with pruning_bridge(dropout,rounds):
        returned=utils.apply_lth_pruning(model,train,val,epochs,opt,criterion,
                         float(p['target_prune_fraction']),rounds,
                         checkpoint_path=folder/'pruning_anchor.pt')
    final,_,accepted,frac,accuracy=returned
    if not isinstance(final,DPN):raise RuntimeError('Original pruner did not return a DPN')
    if final.input_features!=features or final.output_nodes!=out:
        raise ValueError('Pruned DPN dimension mismatch')
    end_lat=_measure(final,features,device,dropout,5)
    elapsed=time.monotonic()-t0
    state={k:v.detach().cpu().clone() for k,v in final.state_dict().items()}
    layout=[int(w.shape[0]) for w in final.weights]
    # Obtain verified separate hidden and output block boundaries.
    hidden_total=final.total_nodes-out;split=0;cur=0
    while cur<hidden_total:
        cur+=layout[split];split+=1
    if cur!=hidden_total:raise ValueError('Pruner ended on a hidden/output boundary split')
    allocated=sum(v.numel() for v in final.parameters())
    active=int(final.count_parameters())
    if not (0<active<=allocated<=initial_params):
        raise RuntimeError('Pruned DPN physical parameter count is invalid')
    max_rounds=sum(range(rounds,0,-max(1,rounds//10)))
    actual=0
    for file in folder.glob('pruning_rounds_*.pt'):
        try:
            checkpoint=torch.load(file,map_location='cpu',weights_only=False)
            actual+=int(checkpoint.get('next_round',0))
            del checkpoint
        except Exception as exc:
            raise RuntimeError(f'Unreadable accepted pruning checkpoint: {file}') from exc
    # The accepted canonical state is derived solely from the original outer.
    from src.training import atomic_save
    atomic_save(ticket_path,{'schema':'dpn_accepted_pruning_v2','dataset':ds,'seed':seed,
       'features':features,'output_logits':out,'widths':layout,'state':state,
       'parameters_allocated':allocated,'parameters_active':active,
       'source_dataset_config_sha256':json_hash(cfg)})
    digest=hash_file(ticket_path)
    metadata={'dataset':ds,'seed':seed,'dataset_config_sha256':json_hash(cfg),
       'ticket_sha256':digest,'hidden_widths':layout[:split],
       'output_widths':layout[split:],'parameters_allocated':allocated,
       'parameters_active':active,'pruning_completed':True,
       'accepted_inner_round':int(accepted),'returned_prune_fraction':float(frac),
       'selection_validation_accuracy':float(accuracy),
       'data_manifest':data_manifest,
       'max_rounds':max_rounds,'actual_rounds':actual,
       'starting_block_count':initial_blocks,'final_block_count':len(layout),
       'starting_parameters':initial_params,'elapsed_seconds':elapsed,
       'pruner':'original_src.utils.apply_lth_pruning'}
    atomic_json(meta_path,metadata)
    row={'dataset':ds,'seed':seed,'max_rounds':max_rounds,'actual_rounds':actual,
       'percent_pruned_parameters':100*(initial_params-allocated)/initial_params,
       'starting_block_count':initial_blocks,'final_block_count':len(layout),
       'block_reduction_percent':100*(initial_blocks-len(layout))/initial_blocks,
       'final_block_layout':json.dumps(layout),'validation_epochs':(actual*epochs)+epochs,
       'total_time_seconds':elapsed,'starting_parameter_size':initial_params,
       'final_parameter_size':allocated,'active_final_parameters':active,
       'starting_inference_latency_ms':start_lat,'ending_inference_latency_ms':end_lat,
       'inference_latency_reduction_percent':100*(start_lat-end_lat)/start_lat,
       'reference_parameter_target':p['reference_target_diagnostic_only'],
       'reference_parameter_deviation_percent':100*(allocated/p['reference_target_diagnostic_only']-1),
       'ticket_sha256':digest,'original_pruning_entrypoint':'src.utils.apply_lth_pruning',
       'source_config_sha256':json_hash(cfg),
       'latency_measurement_context':'discovery_worker_may_share_GPU;not_isolated'}
    upsert(repo,'pruning',row)
    print(f'[discover-done] {ds}/{seed} allocated={allocated} active={active} '
          f'blocks={len(layout)} rounds={actual}/{max_rounds}',flush=True)
    return metadata
