#!/usr/bin/env python3
"""ONE-TIME migration of verified historical Step83/85/86/87/89 artifacts.

After migration, this file may be deleted. Runtime src/ modules never import it.
Reads no test arrays, copies no old training checkpoints, and refuses data guesses.
Run this where original results/final/ directories actually exist (Roihu).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from src import utils
from src.dpn import DPN
from src.data_io import read_config,hash_file
from src.config_registry import per_seed_dir,atomic_json,json_hash,ensure_model,model_path
from src.runner import ensure_seed_readme
from src.model_factory import MODEL_NAMES,instantiate,model_counts
from src.results import upsert,initialize,METRICS,make_plots
from src.training import atomic_save
from src.statistics import compute

SOURCES={'covtype':'step86_covtype_full_12h','jannis':'step83_jannis_full_4h',
         'helena':'step85_helena_full_12h','higgs_small':'step87_higgs_small_full_12h'}
SEEDS=(42,77,53,31,18)
ARMS=('pruned_dpn','matched_mlp','matched_resnet','lth_mlp',
      'block_matched_dpn','parameter_matched_max_dpn','staged_dpn')


def read_committed(path,expect_ds,expect_seed,expect_arm,variant=None):
    """Only accept checksum-committed, final original results."""
    digest=path.with_name('result.sha256')
    if not (path.is_file() and digest.is_file()):return None
    if digest.read_text().strip()!=hash_file(path):
        raise RuntimeError('Result checksum mismatch: '+str(path))
    obj=json.loads(path.read_text())
    if (obj.get('dataset')!=expect_ds or int(obj.get('seed',-1))!=expect_seed
        or obj.get('arm')!=expect_arm or not obj.get('completed',False)
        or obj.get('metrics_split')!='validation' or obj.get('test_data_accessed') is not False):
        raise RuntimeError('Invalid/mismatched validation result: '+str(path))
    if variant is not None and obj.get('variant')!=variant:
        raise RuntimeError('Wrong staged candidate: '+str(path))
    return obj


def source_arm(base,ds,seed,arm):
    return base/'tasks'/ds/f'seed_{seed}'/'arms'/arm


def isolated_latency_map(root):
    result={}
    p=root/'reports'/'isolated_latency'/'latency_per_seed_isolated.csv'
    if not p.is_file():return result
    with p.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('median_latency_ms'):
                result[(row['dataset'],int(row['seed']),row['model'])]=float(row['median_latency_ms'])
    return result


def layout_from_contract(arm,contract,features,out,result):
    """Read the old frozen architecture instead of regenerating an approximation."""
    if not contract:return None
    spec=contract.get('spec') or {}
    if arm in ('matched_mlp','lth_mlp'):
        w=spec.get('widths')
        if not w:return None
        got=utils.allocated_parameters(features,out,w)
        if got!=int(result['allocated']):
            raise RuntimeError(f'MLP contract allocation differs: {got} vs {result["allocated"]}')
        return {'hidden_widths':[int(x) for x in w],
                'actual_allocated':got,'target_allocated':int(result['budget']),
                'relative_error':abs(got-int(result['budget']))/int(result['budget'])}
    if arm=='matched_resnet':
        if not {'d','n_layers','d_hidden_factor','activation'}<=spec.keys():return None
        return {'d':int(spec['d']),'n_layers':int(spec['n_layers']),
                'd_hidden_factor':float(spec['d_hidden_factor']),
                'activation':spec['activation'],'normalization':'batchnorm',
                'actual_allocated':int(result['allocated']),
                'target_allocated':int(result['budget'])}
    if arm in ('parameter_matched_max_dpn','staged_dpn'):
        cfg=spec.get('config')
        if not cfg:return None
        return {'hidden_widths':[int(x) for x in cfg['hidden_widths']],
                'output_widths':[int(x) for x in cfg['output_widths']],
                'stage_boundaries':[int(x) for x in cfg.get('stage_boundaries',[])],
                'transition_widths':[int(x) for x in cfg.get('transition_widths',[])],
                'raw_input_bypass':bool(cfg.get('raw_input_bypass',False)),
                'actual_allocated':int(result['allocated']),
                'target_allocated':int(result['budget'])}
    if arm=='block_matched_dpn':
        cfg=spec.get('config')
        if not cfg:return None
        return {'hidden_widths':[int(x) for x in cfg['hidden_widths']],
                'output_widths':[int(x) for x in cfg['output_widths']],
                'actual_allocated':int(result['allocated']),
                'target_allocated':int(result['budget'])}
    return None


def source_result(base,block_root,ds,seed,model):
    if model=='staged_dpn':
        chosen=source_arm(base,ds,seed,'staged_dpn')/'selected.json'
        if not chosen.is_file():return None,None,None
        selection=json.loads(chosen.read_text())
        v=selection.get('variant')
        if v not in ('staged_compressed','staged_input_skip'):
            raise RuntimeError('Unrecognized selected staged DPN')
        arm_dir=source_arm(base,ds,seed,'staged_dpn__'+v)
        result=read_committed(arm_dir/'result.json',ds,seed,model,v)
        return result,arm_dir,v
    arm_dir=(source_arm(block_root,ds,seed,model) if model=='block_matched_dpn'
             else source_arm(base,ds,seed,model))
    result=read_committed(arm_dir/'result.json',ds,seed,model)
    return result,arm_dir,None


def convert_ticket(repo,cfg,ds,seed,task):
    source=task/'discovery'/'ticket.pt'
    metadata=task/'discovery'/'discovery.json'
    if not source.is_file() or not metadata.is_file():
        return False
    old=json.loads(metadata.read_text())
    if old.get('ticket_sha256')!=hash_file(source):
        raise RuntimeError(f'Corrupt legacy discovery ticket: {source}')
    if old.get('dataset')!=ds or int(old.get('seed',-1))!=seed:
        raise RuntimeError('Legacy discovery identity mismatch')
    if not old.get('qualification',{}).get('near_reference_scale',False):
        raise RuntimeError(f'Unqualified legacy ticket: {ds}/{seed}')
    raw=torch.load(source,map_location='cpu',weights_only=True)
    if raw['dataset']!=ds or int(raw['seed'])!=seed:
        raise RuntimeError('Legacy checkpoint identity mismatch')
    dims=[int(x) for x in raw['widths']]
    out=int(cfg['expected']['output_logits']);features=int(cfg['expected']['input_features'])
    canonical=DPN(features,sum(dims),out,minimal=False,hidden_dims=dims)
    canonical.load_state_dict(raw['state'],strict=True)
    allocated=sum(t.numel() for t in canonical.parameters())
    active=int(canonical.count_parameters())
    if (allocated!=int(old['ticket']['parameters_allocated'])
        or active!=int(old['ticket']['parameters_active'])):
        raise RuntimeError('Historical parameter accounting mismatch')
    hidden=sum(dims)-out;split=0;pos=0
    while pos<hidden:
        pos+=dims[split];split+=1
    if pos!=hidden:raise ValueError('Output split is not a block boundary')
    target=per_seed_dir(repo,ds,seed);target.mkdir(parents=True,exist_ok=True)
    ensure_seed_readme(repo,ds,seed)
    dest=target/'accepted_pruned_ticket.pt';met=target/'pruning_metadata.json'
    if dest.exists() or met.exists():
        if not dest.exists() or not met.exists():raise RuntimeError('Half-imported pruning ticket')
        saved=json.loads(met.read_text())
        if (saved.get('dataset_config_sha256')!=json_hash(cfg)
            or saved.get('ticket_sha256')!=hash_file(dest)
            or saved.get('original_sha256') not in (None,hash_file(source))):
            raise RuntimeError('Existing accepted ticket has different provenance')
    else:
        atomic_save(dest,{'schema':'dpn_accepted_pruning_v2','dataset':ds,'seed':seed,
            'features':features,'output_logits':out,'widths':dims,'state':raw['state'],
            'parameters_allocated':allocated,'parameters_active':active,
            'source_dataset_config_sha256':json_hash(cfg)})
        pr=old.get('pruning_result',{})
        actual=0
        for ck in sorted((task/'discovery').glob('pruning_rounds_*.pt')):
            point=torch.load(ck,map_location='cpu',weights_only=False)
            actual+=int(point.get('next_round',0))
        atomic_json(met,{
            'dataset':ds,'seed':seed,'dataset_config_sha256':json_hash(cfg),
            'ticket_sha256':hash_file(dest),'hidden_widths':dims[:split],
            'output_widths':dims[split:],'parameters_allocated':allocated,
            'parameters_active':active,'pruning_completed':True,
            'legacy_source':str(source.resolve()),'original_sha256':hash_file(source),
            'actual_rounds':actual,'max_rounds':275,
            'accepted_inner_round':pr.get('round_accepted_by_original'),
            'selection_validation_accuracy':pr.get('selection_validation_accuracy'),
            'pruning_wall_seconds':pr.get('wall_seconds')})
    saved=json.loads(met.read_text());pr=old.get('pruning_result',{})
    initial=utils.singleton_capacity(features,cfg['pruning']['initial_hidden_blocks'],out)
    initial_blocks=cfg['pruning']['initial_hidden_blocks']+out
    actual=saved.get('actual_rounds','')
    upsert(repo,'pruning',{'dataset':ds,'seed':seed,'max_rounds':saved.get('max_rounds',275),
       'actual_rounds':actual,'percent_pruned_parameters':100*(initial-allocated)/initial,
       'starting_block_count':initial_blocks,'final_block_count':len(dims),
       'block_reduction_percent':100*(1-len(dims)/initial_blocks),
       'final_block_layout':json.dumps(dims),
       'validation_epochs':int(actual)*cfg['pruning']['candidate_training_epochs'] if actual!='' else '',
       'total_time_seconds':pr.get('wall_seconds',''),
       'starting_parameter_size':initial,'final_parameter_size':allocated,
       'reference_parameter_target':cfg['pruning']['reference_target_diagnostic_only'],
       'reference_parameter_deviation_percent':100*(allocated/cfg['pruning']['reference_target_diagnostic_only']-1),
       'active_final_parameters':active,'ticket_sha256':hash_file(dest),
       'original_pruning_entrypoint':'src.utils.apply_lth_pruning',
       'source_config_sha256':json_hash(cfg),
       'latency_measurement_context':'not available in historical discovery artifacts'})
    return True


def store_legacy_architecture(repo,cfg,ds,seed,model,source_result_obj,arm_dir,variant,base):
    """Store exact old layer geometry where a committed contract proves it."""
    path=model_path(repo,ds,seed,model)
    config=json.loads(path.read_text())
    target=config['architecture'];actual=source_result_obj
    contract_path=arm_dir/'contract.json'
    contract=json.loads(contract_path.read_text()) if contract_path.is_file() else None
    features=cfg['expected']['input_features'];out=cfg['expected']['output_logits']
    if contract and (contract.get('arm')!=model or contract.get('dataset')!=ds
                     or int(contract.get('seed',-1))!=seed):
        raise RuntimeError('Frozen training contract does not match requested model')
    found=layout_from_contract(model,contract,features,out,actual)
    if model=='staged_dpn':
        # Collect BOTH historical candidate architectures. Validation decides the
        # winner again if the user chooses to retrain this seed.
        candidates={}
        for v in ('staged_compressed','staged_input_skip'):
            vdir=source_arm(base,ds,seed,'staged_dpn__'+v)
            vres=read_committed(vdir/'result.json',ds,seed,model,v)
            if vres:
                vf=vdir/'contract.json'
                if not vf.exists():raise RuntimeError('Staged candidate contract missing')
                vspec=layout_from_contract(model,json.loads(vf.read_text()),features,out,vres)
                if vspec is None:raise RuntimeError('Staged candidate geometry missing')
                vspec['status']='candidate';candidates[v]=vspec
            else:
                # Only allow the standard engine's original infeasible marker.
                if (vdir/'INFEASIBLE.json').is_file():
                    candidates[v]={'status':'infeasible','reason':'Original validation candidate infeasible'}
                else:raise RuntimeError(f'Missing staged variant provenance: {ds}/{seed}/{v}')
        target={'candidates':candidates,'selection':'highest_validation_accuracy_then_lowest_loss',
                'actual_allocated':int(actual['budget']),'target_allocated':int(actual['budget'])}
    elif found:
        if model=='lth_mlp':
            paired=json.loads(model_path(repo,ds,seed,'matched_mlp').read_text())['architecture']
            if found['hidden_widths']!=paired['hidden_widths']:
                raise RuntimeError('Historical LTH and MLP architectures are not paired')
            found={**paired,'paired_with':'matched_mlp',
                   'lth_cumulative_targets':cfg['regularization']['lth_mlp']['cumulative_mask_targets'],
                   'lth_pretrain_epochs':cfg['regularization']['lth_mlp']['pretrain_epochs_per_round']}
        if model=='block_matched_dpn':
            mlp=json.loads(model_path(repo,ds,seed,'matched_mlp').read_text())
            w=found['hidden_widths']
            if len(w)!=len(mlp['architecture']['hidden_widths']):
                raise RuntimeError('Historical Block DPN does not match the original MLP hidden depth')
            allowed=cfg['regularization']['block_matched_dpn']['parameter_tolerance']
            if abs(found['actual_allocated']-mlp['architecture']['actual_allocated'])/mlp['architecture']['actual_allocated']>allowed:
                raise RuntimeError('Historical Block DPN parameter matching failed')
            found['target_allocated']=int(mlp['architecture']['actual_allocated'])
            found['relative_error']=abs(found['actual_allocated']-found['target_allocated'])/found['target_allocated']
            found['reference_model_config']='matched_mlp.config'
        target=found
    config['architecture']=target
    if contract:
        config['legacy_original_spec']=contract.get('spec',{})
        config['legacy_original_hyperparameters']=contract.get('hyperparameters',{})
        config['legacy_original_init_fingerprint']=(contract.get('spec',{}) or {}).get('paired_init_sha256','')
        # Save per-arm training settings stated by the historical source. Do
        # not infer any setting absent from that frozen contract.
        hp=contract.get('hyperparameters',{})
        for source_key,config_key in (('learning_rate','learning_rate'),
                                       ('weight_decay','weight_decay'),
                                       ('batch_size','batch_size'),
                                       ('patience','patience')):
            if source_key in hp and hp[source_key] is not None:
                config['training'][config_key]=hp[source_key]
    config['legacy_original_contract_sha256']=hash_file(contract_path) if contract else ''
    config['legacy_validation_result_sha256']=hash_file(arm_dir/'result.json')
    config['legacy_source']='verified_original_regularized_benchmark'
    # Validate the exact geometry with the new model. No silent shape replacement.
    model_obj=instantiate(config,cfg,seed,per_seed_dir(repo,ds,seed),variant)
    got=model_counts(model_obj)[1]
    if got!=int(actual['allocated']):
        raise RuntimeError(f'Cannot reproduce old physical tensor allocation for {ds}/{seed}/{model}: '
                           f'constructed={got}, historical={actual["allocated"]}')
    atomic_json(path,config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-results',required=True,type=Path)
    p.add_argument('--new-root',default='.',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--datasets',nargs='+',default=list(SOURCES))
    p.add_argument('--step89-folder',default='step89_shallow_block_regularized_4gpu')
    p.add_argument('--skip-validation',action='store_true',help='Only tickets and configs; no per-seed result CSV')
    args=p.parse_args();base=args.previous_results.resolve();repo=args.new_root.resolve()
    if not base.is_dir():raise SystemExit(f'Historical results folder does not exist: {base}')
    initialize(repo)
    count_tickets=count_models=count_results=0
    for ds in args.datasets:
        if ds not in SOURCES:raise ValueError('No verified legacy mapping for '+ds)
        cfg=read_config(repo,ds);oldbase=base/SOURCES[ds]
        block_root=base/args.step89_folder
        latencies=isolated_latency_map(oldbase)
        for seed in args.seeds:
            task=oldbase/'tasks'/ds/f'seed_{seed}'
            if not convert_ticket(repo,cfg,ds,seed,task):
                print(f'[missing verified discovery] {ds}/{seed}: {task}',flush=True)
                continue
            count_tickets+=1
            # Dependency order is important. Corrected Block DPN uses exact MLP.
            for model in ('pruned_dpn','matched_mlp','lth_mlp','matched_resnet',
                          'parameter_matched_max_dpn','staged_dpn','block_matched_dpn'):
                result,arm_dir,variant=source_result(oldbase,block_root,ds,seed,model)
                if result is None:
                    raise RuntimeError(f'Missing committed final result for {ds}/{seed}/{model} at {arm_dir}')
                config=ensure_model(repo,ds,seed,model,cfg)
                # Do not overwrite previously frozen configs when re-running import.
                if not config.get('legacy_original_contract_sha256'):
                    store_legacy_architecture(repo,cfg,ds,seed,model,result,arm_dir,variant,oldbase)
                else:
                    expected=hash_file(arm_dir/'contract.json') if (arm_dir/'contract.json').is_file() else ''
                    if expected!=config['legacy_original_contract_sha256']:
                        raise RuntimeError('Legacy contract changed after import')
                count_models+=1
                if args.skip_validation:continue
                measurements=result['metrics']
                spec=json.loads(model_path(repo,ds,seed,model).read_text())
                reg=spec['regularization']
                row={'dataset':ds,'seed':seed,'model':model,'variant':variant or '',
                   'status':'complete','split':'validation','epochs_completed':result.get('epochs',''),
                   'best_epoch':result.get('best_epoch',''),
                   'early_stopped':(result.get('epochs',300)<cfg['training']['max_epochs']),
                   'training_wall_seconds':result.get('train_wall_seconds',''),
                   'initialization_seed':seed,'optimizer':'AdamW',
                   'learning_rate':spec['training']['learning_rate'],
                   'weight_decay':spec['training']['weight_decay'],
                   'batch_size':spec['training']['batch_size'],
                   'max_epochs':spec['training']['max_epochs'],
                   'patience':spec['training']['patience'],
                   'normalization':reg['normalization'],'dropout':reg.get('dropout',0.),
                   'lth_search_seconds':result.get('lth_mask_search_seconds',''),
                   'parameters_allocated':result['allocated'],'parameters_active':result['active'],
                   'parameter_budget_target':result.get('budget',''),
                   'parameter_budget_gap_percent':100*result.get('relative_gap',0.),
                   'validation_loss':measurements.get('log_loss',''),
                   'checkpoint_fingerprint':'LEGACY: new-code checkpoint not reusable',
                   'dataset_config_sha256':json_hash(cfg),'model_config_sha256':json_hash(spec),
                   'ticket_sha256':json.loads((per_seed_dir(repo,ds,seed)/'pruning_metadata.json').read_text())['ticket_sha256'],
                   'data_source':cfg['source']['path'],
                   'data_split_provenance':'official_validation_from_verified_legacy_run',
                   'test_accessed':False}
                for field in METRICS:
                    v=measurements.get(field)
                    row['val_'+field]=v if v is not None else ''
                a=spec['architecture']
                if model=='staged_dpn':a=a['candidates'][variant]
                row['hidden_blocks']=len(a.get('hidden_widths',[]))
                row['output_blocks']=len(a.get('output_widths',[]))
                mlp_file=model_path(repo,ds,seed,'matched_mlp')
                row['mlp_hidden_depth']=(len(json.loads(mlp_file.read_text())['architecture']['hidden_widths'])
                    if mlp_file.is_file() else len(cfg['reference_mlp']['hidden_widths']))
                if model=='block_matched_dpn':
                    rec=arm_dir/'step89_isolated_latency.json'
                    if rec.is_file():
                        measured=json.loads(rec.read_text())
                        if measured['best_sha256']!=hash_file(arm_dir/'best.pt') or measured['result_sha256']!=hash_file(arm_dir/'result.json'):
                            raise RuntimeError('Step89 uncontended latency provenance failed')
                        row['inference_latency_batch1_ms']=measured['median_ms']
                elif (ds,seed,model) in latencies:
                    row['inference_latency_batch1_ms']=latencies[(ds,seed,model)]
                upsert(repo,'training',row)
                count_results+=1
            print(f'[imported] {ds}/{seed}: all seven model configs and results',flush=True)
    # Plots recompute from the imported CSV; held-out test remains sealed.
    make_plots(repo,args.datasets)
    if not args.skip_validation:compute(repo,'val','accuracy')
    print(f'[migration] tickets={count_tickets}, configs={count_models}, validation rows={count_results}')
    print('[migration] Finished. This one-time script can now be deleted.')
    print('[migration] Historical *training* checkpoints are NOT compatible with the new engine.')

if __name__=='__main__':main()
