"""Dependency-driven, immutable per-seed model configs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from src import utils
from src.data_io import hash_file
from src.model_factory import MODEL_NAMES


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    tmp.write_text(data)
    tmp.replace(path)


def json_hash(obj):
    # Acquisition metadata (including verified source pins) is excluded for
    # backward compatibility. The loader enforces the pinned data bytes.
    # Exclude this optional, additive section so previously imported model
    # configs and accepted pruning tickets keep their original fingerprints.
    # The exact downloaded data bytes are separately hashed by split_manifest.
    if isinstance(obj, dict) and obj.get('schema') == 'dpn_dataset_v2' and 'data_acquisition' in obj:
        obj = {key: value for key, value in obj.items() if key != 'data_acquisition'}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def per_seed_dir(repo,dataset,seed):
    return Path(repo)/'configs'/dataset/str(int(seed))


def model_path(repo,dataset,seed,model):
    if model not in MODEL_NAMES:raise ValueError('Unknown model name')
    return per_seed_dir(repo,dataset,seed)/(model+'.config')


def read_model(repo,dataset,seed,model,dataset_cfg):
    path=model_path(repo,dataset,seed,model)
    if not path.is_file():return None
    obj=json.loads(path.read_text())
    if obj.get('model')!=model or obj.get('dataset')!=dataset or obj.get('seed')!=int(seed):
        raise ValueError(f'Wrong-model or wrong-seed config {path}')
    cfg_hash=json_hash(dataset_cfg)
    if obj.get('dataset_config_sha256')!=cfg_hash:
        raise RuntimeError(f'Dataset config changed. Archive old experiment first: {path}')
    return obj


def _commit(repo,dataset,seed,model,cfg,dataset_cfg):
    path=model_path(repo,dataset,seed,model)
    config={'schema':'dpn_seed_model_v2','dataset':dataset,'seed':int(seed),'model':model,
            'dataset_config_sha256':json_hash(dataset_cfg),**cfg}
    if path.exists():
        old=read_model(repo,dataset,seed,model,dataset_cfg)
        if old!=config:
            raise RuntimeError(f'Existing model config differs. Never rewrite silently: {path}')
        return old
    atomic_json(path,config)
    return config


def ticket_info(repo,dataset,seed,ds_cfg):
    folder=per_seed_dir(repo,dataset,seed)
    path=folder/'accepted_pruned_ticket.pt'
    meta=folder/'pruning_metadata.json'
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(f'Accepted pruned ticket absent for {dataset}/{seed}. Run discovery first.')
    obj=json.loads(meta.read_text())
    if obj['dataset']!=dataset or obj['seed']!=int(seed):raise ValueError('Wrong ticket seed')
    if obj.get('dataset_config_sha256')!=json_hash(ds_cfg):
        raise RuntimeError('Accepted pruning ticket belongs to a different dataset configuration')
    if obj['ticket_sha256']!=hash_file(path):raise RuntimeError('Accepted pruning ticket bytes changed')
    return obj


def ensure_model(repo,dataset,seed,name,dataset_cfg):
    """Resolve requested model config, with strict, ordered dependencies.

    `pruned_dpn` requires a real accepted outer-pruning ticket; this function
    does not invent a topology or substitute a reference-paper parameter count.
    """
    if name not in MODEL_NAMES:raise ValueError(f'Unknown model {name}')
    previous=read_model(repo,dataset,seed,name,dataset_cfg)
    if previous is not None:
        # Critical: reject a config whose source ticket was replaced.
        ticket=ticket_info(repo,dataset,seed,dataset_cfg)
        if previous.get('ticket_sha256')!=ticket['ticket_sha256']:
            raise RuntimeError('Model config references an outdated pruned ticket')
        return previous
    ticket=ticket_info(repo,dataset,seed,dataset_cfg)
    ex=dataset_cfg['expected'];features=int(ex['input_features']);out=int(ex['output_logits'])
    target=int(ticket['parameters_allocated']);reg=dataset_cfg['regularization'][name]
    architecture=None
    if name=='pruned_dpn':
        architecture={'hidden_widths':ticket['hidden_widths'],
                      'output_widths':ticket['output_widths'],
                      'ticket_file':'accepted_pruned_ticket.pt','ticket_sha256':ticket['ticket_sha256'],
                      'actual_allocated':target,'parameters_active':ticket['parameters_active'],
                      'source':'original_apply_lth_pruning_accepted_ticket'}
    elif name in ('matched_mlp','lth_mlp'):
        # Paired seed initialization is supplied by model_factory.seed_all(seed).
        if name=='lth_mlp':
            mlp=ensure_model(repo,dataset,seed,'matched_mlp',dataset_cfg)
            architecture={**mlp['architecture'],'paired_with':'matched_mlp',
                          'lth_cumulative_targets':reg['cumulative_mask_targets'],
                          'lth_pretrain_epochs':reg['pretrain_epochs_per_round']}
        else:
            architecture=utils.match_mlp_to_ticket(features,out,
                dataset_cfg['reference_mlp']['hidden_widths'],target,
                dataset_cfg['reference_mlp'].get('matching_tolerance',.01))
    elif name=='matched_resnet':
        architecture=utils.match_resnet_to_ticket(features,out,target)
    elif name=='parameter_matched_max_dpn':
        architecture=utils.match_maximal_singleton_dpn(features,out,target)
    elif name=='block_matched_dpn':
        mlp=ensure_model(repo,dataset,seed,'matched_mlp',dataset_cfg)
        aw=mlp['architecture']
        architecture=utils.match_shallow_block_dpn(features,out,aw['hidden_widths'],
            aw['actual_allocated'],reg.get('parameter_tolerance',.005))
        architecture['reference_model_config']='matched_mlp.config'
    elif name=='staged_dpn':
        options=utils.staged_dpn_candidates(features,out,target)
        architecture={'candidates':options,'selection':'highest_validation_accuracy_then_lowest_loss',
                      'actual_allocated':target,'target_allocated':target}
    else:raise AssertionError(name)
    return _commit(repo,dataset,seed,name,
        {'architecture':architecture,'regularization':reg,
         'training':dict(dataset_cfg['training']),
         'ticket_sha256':ticket['ticket_sha256'],
         'initialization_seed':int(seed),
         'provenance':{'reference_budget':'accepted_pruned_ticket_physical_allocation',
                       'pruning_accepted':ticket['pruning_completed'],
                       'test_data_accessed':False}},dataset_cfg)
