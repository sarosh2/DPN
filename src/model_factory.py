"""Seven reference-consistent models with explicit, frozen seed configs."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.dpn import DPN, FlexibleDPN, FlexibleDPNConfig
from src.models.resnet import ResNet

MODEL_NAMES=('pruned_dpn','matched_mlp','matched_resnet','lth_mlp',
             'block_matched_dpn','parameter_matched_max_dpn','staged_dpn')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


class PaperMLP(nn.Module):
    """Original paper-shaped numeric MLP, single logit for binary HIGGS Small."""
    def __init__(self,features,widths,output,dropout):
        super().__init__()
        self.layers=nn.ModuleList()
        prev=int(features)
        for width in widths:
            self.layers.append(nn.Linear(prev,int(width)));prev=int(width)
        self.head=nn.Linear(prev,int(output))
        self.dropout=float(dropout)

    def forward(self,x):
        for layer in self.layers:
            x=F.relu(layer(x))
            if self.dropout:x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.head(x)
        return x.squeeze(-1) if x.shape[-1]==1 else x


def model_counts(model):
    allocated=sum(p.numel() for p in model.parameters())
    if isinstance(model,FlexibleDPN):
        return model.accounting()['parameters_active'],allocated
    masked=0
    for m in model.modules():
        if isinstance(m,nn.Linear) and hasattr(m,'weight_mask'):
            masked+=int((m.weight_mask==0).sum().item())
    return allocated-masked,allocated


def flex_cfg(spec, dataset_cfg, variant=None):
    ex=dataset_cfg['expected'];reg=spec['regularization']
    src=spec['architecture']
    if variant is not None:
        src=src['candidates'][variant]
        if src.get('status')=='infeasible':
            raise ValueError(f'Staged candidate {variant} is infeasible')
    norm=reg['normalization']
    # Jannis uses hidden BatchNorm after linear, before ReLU, without affine.
    actual_norm='none' if norm=='batchnorm' else norm
    return FlexibleDPNConfig(int(ex['input_features']),
        tuple(src['hidden_widths']),tuple(src['output_widths']),
        stage_boundaries=tuple(src.get('stage_boundaries',())),
        transition_widths=tuple(src.get('transition_widths',())),
        raw_input_bypass=bool(src.get('raw_input_bypass',False)),
        normalization=actual_norm,layernorm_affine=False,
        dropout=float(reg.get('dropout',0.)))


def load_ticket(config_dir,model_spec,device='cpu'):
    path=config_dir/str(model_spec['architecture']['ticket_file'])
    if not path.is_file():raise FileNotFoundError('Missing saved accepted pruned ticket: '+str(path))
    from src.data_io import hash_file
    if hash_file(path)!=model_spec['architecture']['ticket_sha256']:
        raise RuntimeError('Accepted DPN ticket hash changed')
    t=torch.load(path,map_location='cpu',weights_only=True)
    widths=t['widths'];features=int(t['features']);output=int(t['output_logits'])
    model=DPN(features,sum(widths),output,minimal=False,hidden_dims=widths)
    model.load_state_dict(t['state'],strict=True)
    return model.to(device)


def instantiate(model_spec, dataset_cfg, seed, config_dir, variant=None):
    name=model_spec['model'];ex=dataset_cfg['expected'];reg=model_spec['regularization']
    features=int(ex['input_features']);out=int(ex['output_logits'])
    # Each arm gets its own deterministic initialization (MLP and LTH share it).
    seed_all(int(seed))
    if name in ('matched_mlp','lth_mlp'):
        model=PaperMLP(features,model_spec['architecture']['hidden_widths'],out,reg['dropout'])
    elif name=='matched_resnet':
        spec=model_spec['architecture'];model=ResNet(features,max(2,out),d=spec['d'],
            d_hidden_factor=spec['d_hidden_factor'],n_layers=spec['n_layers'],
            activation=spec['activation'],normalization='batchnorm',
            hidden_dropout=reg.get('hidden_dropout',0.),
            residual_dropout=reg.get('residual_dropout',0.))
        if out==1:model.head=nn.Linear(spec['d'],1)
    elif name=='pruned_dpn':
        canonical=load_ticket(config_dir,model_spec)
        model=FlexibleDPN.from_canonical(canonical,normalization=reg['normalization'],
            layernorm_affine=False,dropout=reg['dropout'])
    elif name in ('block_matched_dpn','parameter_matched_max_dpn','staged_dpn'):
        config=flex_cfg(model_spec,dataset_cfg,variant)
        model=FlexibleDPN(config,init_seed=int(seed))
        if reg['normalization']=='batchnorm':
            model.pre_activation_norms=nn.ModuleList([
                nn.BatchNorm1d(w,eps=1e-5,momentum=.1,affine=False,
                               track_running_stats=True)
                for w in config.hidden_widths]+
                [nn.Identity() for _ in config.output_widths])
    else:raise ValueError(f'Unsupported model: {name}')
    architecture=(model_spec['architecture'] if variant is None else
                  model_spec['architecture']['candidates'][variant])
    if name!='pruned_dpn' and 'actual_allocated' in architecture:
        got=model_counts(model)[1]
        if got != int(architecture['actual_allocated']):
            raise ValueError(f'Model {name} physical allocation changed: {got} vs {architecture["actual_allocated"]}')
    return model
