"""One-way freeze gate and explicit, reproducible held-out test evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from filelock import FileLock
from src.data_io import read_config,load_splits,split_manifest,hash_file
from src.config_registry import read_model,atomic_json,json_hash
from src.training import _checkpoint_dir,evaluate,restore_lth_reparameterization,fingerprint
from src.model_factory import instantiate
from src.results import read_rows,upsert,make_plots,METRICS


def evaluate_frozen(repo,datasets,seeds,models,device='cpu',confirm_frozen=False):
    if not confirm_frozen:
        raise RuntimeError('Test remains sealed. Set --confirm-frozen after freezing the model comparison grid.')
    repo=Path(repo).resolve();device=torch.device(device)
    lock=repo/'checkpoints'/'.runner.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock),timeout=0):
        return _evaluate_locked(repo,datasets,seeds,models,device)


def _evaluate_locked(repo,datasets,seeds,models,device):
    if not datasets or not seeds or not models:
        raise ValueError('Test comparison grid must not be empty')
    spec={'datasets':sorted(set(datasets)),'seeds':sorted(set(int(x) for x in seeds)),
          'models':sorted(set(models)),'schema':'dpn_heldout_test_frozen_v2'}
    manifest=repo/'results'/'TEST_EVALUATION_FROZEN.json'
    configs=[]
    rows={(r['dataset'],int(r['seed']),r['model']):r for r in read_rows(repo,'training')}
    for ds in spec['datasets']:
        ds_cfg=read_config(repo,ds)
        train_manifest=None
        for seed in spec['seeds']:
            for model in spec['models']:
                seed_config=read_model(repo,ds,seed,model,ds_cfg)
                if seed_config is None:
                    raise RuntimeError(f'Test not allowed: model config missing for {ds}/{seed}/{model}')
                base=_checkpoint_dir(repo,ds,seed,model,None)
                if model=='staged_dpn':
                    selected=base/'selected.json'
                    if not selected.exists():raise RuntimeError('Staged validation selection not frozen')
                    variant=json.loads(selected.read_text())['variant']
                else:variant=None
                cp=_checkpoint_dir(repo,ds,seed,model,variant)
                if not (cp/'completed.json').exists() or not (cp/'best.pt').exists():
                    raise RuntimeError(f'Test not allowed: incomplete {ds}/{seed}/{model}')
                obj=json.loads((cp/'completed.json').read_text())
                if train_manifest is None:
                    train_manifest=split_manifest(repo,ds_cfg,include_test=False)
                expected=fingerprint(repo,ds_cfg,seed_config,train_manifest,variant)
                row=rows.get((ds,seed,model))
                if obj.get('fingerprint')!=expected:
                    raise RuntimeError(f'Test checkpoint fingerprint mismatch: {ds}/{seed}/{model}')
                if not row or row.get('status')!='complete' or row.get('checkpoint_fingerprint')!=expected:
                    raise RuntimeError(f'Test requires matching completed CSV row: {ds}/{seed}/{model}')
                if (row.get('variant') or None)!=variant:
                    raise RuntimeError(f'Test CSV variant mismatch: {ds}/{seed}/{model}')
                saved=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
                if saved.get('schema')!='dpn_best_v2' or saved.get('fingerprint')!=expected:
                    raise RuntimeError(f'Test best checkpoint fingerprint mismatch: {ds}/{seed}/{model}')
                if saved.get('epoch')!=obj['summary']['result']['best_epoch']:
                    raise RuntimeError(f'Test best checkpoint epoch mismatch: {ds}/{seed}/{model}')
                del saved
                configs.append({'dataset':ds,'seed':seed,'model':model,'variant':variant,
                                'fingerprint':expected,
                                'best_sha256':hash_file(cp/'best.pt'),
                                'config_hash':json_hash(seed_config)})
    spec['frozen_checkpoints']=configs
    if manifest.exists():
        if json.loads(manifest.read_text())!=spec:
            raise RuntimeError('Test set already frozen under a DIFFERENT comparison grid; stop.')
    else:
        if any(r.get('test_accuracy') for r in rows.values()):
            raise RuntimeError('Existing test scores without a matching freeze manifest; stop.')
        atomic_json(manifest,spec)
    for ds in spec['datasets']:
        cfg=read_config(repo,ds)
        for seed in spec['seeds']:
            # The official test array is first opened ONLY after manifest frozen.
            X,Y,_=load_splits(repo,cfg,seed,include_test=True)
            xx=torch.as_tensor(X['test'],dtype=torch.float32)
            yy=torch.as_tensor(Y['test'],dtype=torch.long)
            for name in spec['models']:
                key=(ds,seed,name);row=rows.get(key)
                if row is None:raise RuntimeError('Finished training has no canonical CSV row')
                if row.get('test_accuracy'):
                    print(f'[test-skip] {ds}/{seed}/{name} already evaluated');continue
                seed_config=read_model(repo,ds,seed,name,cfg)
                if name=='staged_dpn':
                    p=_checkpoint_dir(repo,ds,seed,name,None)/'selected.json'
                    variant=json.loads(p.read_text())['variant']
                else:variant=None
                cp=_checkpoint_dir(repo,ds,seed,name,variant)
                model=instantiate(seed_config,cfg,seed,repo/'configs'/ds/str(seed),variant).to(device)
                saved=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
                if saved['fingerprint']!=json.loads((cp/'completed.json').read_text())['fingerprint']:
                    raise RuntimeError('Test checkpoint fingerprint mismatch')
                if name=='lth_mlp':restore_lth_reparameterization(model,saved['model'],device)
                model.load_state_dict(saved['model'],strict=True)
                result=evaluate(model,xx,yy,int(cfg['expected']['output_logits']),device)
                row['test_loss']=result['loss']
                for metric in METRICS:row['test_'+metric]=result[metric] if result[metric] is not None else ''
                row['test_accessed']=True;row['test_evaluation_frozen']=True
                upsert(repo,'training',row)
                print(f'[test] {ds}/{seed}/{name}: {100*result["accuracy"]:.2f}%',flush=True)
    make_plots(repo,spec['datasets'])
    print('[frozen-test-PASS] Test scores written; never use them for model selection')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--confirm-frozen',action='store_true')
    a=p.parse_args()
    evaluate_frozen(a.repo,a.datasets,a.seeds,a.models,a.device,a.confirm_frozen)

if __name__=='__main__':main()
