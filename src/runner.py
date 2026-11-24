"""Resource-aware DAG scheduler. One dataset per GPU where possible.

Discovery -> deterministic architecture configs -> eight train arms (seven model
families) -> isolated latency. The filesystem, not a volatile queue, records
finished work, so Slurm timeouts do not lose the prior allocation's progress.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter,defaultdict
from filelock import FileLock,Timeout
from src.data_io import read_config,split_manifest
from src.config_registry import ensure_model,per_seed_dir,ticket_info,atomic_json,model_path
from src.model_factory import MODEL_NAMES
from src.results import initialize,make_plots


def available_gpus():
    """CUDA devices visible to this process, plus a portable MPS fallback.

    Device identifiers are always stable CUDA-visible ordinals, not physical
    nvidia-smi indexes (which may differ under Slurm/container allocations).
    """
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def detected_accelerator(preference='auto'):
    """Select CUDA, Apple Metal (MPS), or CPU without a hardware profile."""
    import torch
    if preference not in ('auto','cuda','mps','cpu'):
        raise ValueError(f'Unknown device choice: {preference}')
    if preference=='cpu':return 'cpu',[]
    if torch.cuda.is_available():
        if preference in ('auto','cuda'):return 'cuda',available_gpus()
    if preference=='cuda':raise RuntimeError('CUDA requested but no CUDA GPU is available')
    has_mps=bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())
    if has_mps and preference in ('auto','mps'):return 'mps',['mps']
    if preference=='mps':raise RuntimeError('MPS requested but Apple Metal is not available')
    return 'cpu',[]


def gpu_telemetry(physical_id):
    """nvidia-smi sample; returns (util_fraction, used_fraction), or None."""
    if physical_id=='mps' or shutil.which('nvidia-smi') is None:return None
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    ordinal=int(physical_id)
    ids=[x.strip() for x in visible.split(',') if x.strip()]
    nvidia_id=ids[ordinal] if ids and ordinal<len(ids) else str(ordinal)
    cmd=['nvidia-smi','--id='+nvidia_id,
         '--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']
    try:
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=4,check=True)
        line=result.stdout.strip().splitlines()[0]
        u,used,total=(float(v.strip()) for v in line.split(',')[:3])
        return u/100,used/max(1,total)
    except (OSError,IndexError,ValueError,subprocess.SubprocessError):return None


def parse_selection(repo,datasets,seeds,models):
    folder=Path(repo)/'configs'
    if datasets==['all']:
        datasets=sorted(p.parent.name for p in folder.glob('*/dataset.json'))
    if not datasets:raise ValueError('No dataset configs found')
    if models==['all']:models=list(MODEL_NAMES)
    for model in models:
        if model not in MODEL_NAMES:raise ValueError(f'Unknown model {model}')
    cfgs={ds:read_config(repo,ds) for ds in datasets}
    if seeds==['all']:
        allseeds=sorted({int(seed) for cfg in cfgs.values() for seed in cfg['seeds']})
    else:allseeds=[int(s) for s in seeds]
    if not allseeds:raise ValueError('No seeds selected')
    for cfg in cfgs.values():
        for seed in allseeds:
            if seed<0:raise ValueError('Seed must be nonnegative')
    return cfgs,allseeds,models


def ensure_seed_readme(repo,ds,seed):
    folder=per_seed_dir(repo,ds,seed);folder.mkdir(parents=True,exist_ok=True)
    file=folder/'README.md'
    if not file.exists():file.write_text(
        f'# {ds}, seed {seed}\n\n'
        'Generated `*.config` files contain frozen, seed-specific architecture, '
        'initialization and regularization details. `pruning_metadata.json` and '
        '`accepted_pruned_ticket.pt` record the verified original outer-pruning '
        'winner. Delete neither during a checkpoint resume.\n\n'
        f'Example: `python run.py --datasets {ds} --seeds {seed} '
        '--models matched_mlp block_matched_dpn`.\n')


def assignment(cfgs,gpus):
    if not gpus:return {ds:None for ds in cfgs}
    return {ds:gpus[i%len(gpus)] for i,ds in enumerate(cfgs)}


def _active_gpu_tasks(tasks,device):
    return sum(1 for item in tasks if item['device']==device)


def _run_stage(repo,cfgs,seedlist,targets,stage,devices,deadline,max_workers_per_gpu,
               gpu_util_target,gpu_memory_limit,stop_on_errors=True):
    """Start pending jobs progressively; poll with no in-memory resume state."""
    pending=list(targets);active=[];failed=[];finished=[]
    group={ds:devices[ds] for ds in cfgs}
    # Preferred dataset affinity is not a hard GPU pin. If one dataset is
    # slow, idle visible accelerators can help process its remaining seeds.
    accelerator_pool=list(dict.fromkeys(d for d in group.values() if d is not None))
    logs=Path(repo)/'logs';logs.mkdir(parents=True,exist_ok=True)
    while pending or active:
        if deadline and time.time()>=deadline-35:
            print('[scheduler] Near wall limit; draining workers and preserving checkpoints',flush=True)
            # Children cooperate through DPN_DEADLINE. The Slurm signal may
            # terminate the batch regardless; last completed epoch is safe.
            for record in active:
                record['process'].terminate()
            for record in active:
                try:record['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:record['process'].kill()
                record['log_handle'].close()
            return {'finished':finished,'pending':pending,'running':[r['job'] for r in active]}
        made=False
        for job in list(pending):
            ds,seed,model=job
            preferred=group[ds]
            if preferred is None:
                device=None
                if _active_gpu_tasks(active,None)>=max(1,min(os.cpu_count() or 1,4)):
                    continue
            else:
                # First keep a seed on its dataset's preferred GPU. If that
                # GPU is busy, fill another GPU with free capacity. There is
                # no hardware-specific mapping or per-cluster profile.
                order=sorted(accelerator_pool,
                    key=lambda gpu:(_active_gpu_tasks(active,gpu),gpu!=preferred))
                device=None
                for candidate in order:
                    busy=_active_gpu_tasks(active,candidate)
                    limit=(min(2,max_workers_per_gpu) if candidate=='mps'
                           else max_workers_per_gpu)
                    if busy>=limit:continue
                    if candidate!='mps':
                        sample=gpu_telemetry(candidate)
                        if sample is not None:
                            util,used=sample
                            if used>=gpu_memory_limit or (util>=gpu_util_target and busy>=2):
                                continue
                    device=candidate
                    break
                if device is None:continue
            name=f'{ds}-{seed}-{model}-{stage}'
            logfile=logs/(name+'.log')
            env=dict(os.environ);env.update({'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1',
                   'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            if device not in (None,'mps'):
                # The worker sees exactly one CUDA device, indexed as cuda:0.
                # Map through the parent's existing visibility mask.
                parent_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
                if parent_visible:
                    visible=[x.strip() for x in parent_visible.split(',') if x.strip()]
                    env['CUDA_VISIBLE_DEVICES']=visible[int(device)]
                else:env['CUDA_VISIBLE_DEVICES']=str(device)
            else:env['CUDA_VISIBLE_DEVICES']=''
            cmd=[sys.executable,'-u','-m','src.worker','--repo',str(repo),
                 '--dataset',ds,'--seed',str(seed),'--stage',stage,'--device',
                 'cpu' if device is None else 'mps' if device=='mps' else 'cuda']
            if model is not None:cmd+=['--model',model]
            if deadline:cmd+=['--deadline',str(deadline)]
            handle=logfile.open('a')
            try:process=subprocess.Popen(cmd,env=env,cwd=repo,stdout=handle,
                                         stderr=subprocess.STDOUT,start_new_session=True)
            except Exception:
                handle.close();raise
            active.append({'process':process,'log_handle':handle,'log':logfile,
                           'device':device,'name':name,'job':job})
            pending.remove(job);made=True
            print(f'[launch] {name} pid={process.pid} gpu={device}',flush=True)
        for record in list(active):
            rc=record['process'].poll()
            if rc is None:continue
            record['log_handle'].close();active.remove(record)
            if rc==0:
                finished.append(record['job'])
                print(f'[finished] {record["name"]}',flush=True)
            elif rc==75:
                pending.append(record['job'])
                print(f'[checkpointed] {record["name"]}: deadline approaching',flush=True)
            else:
                failed.append((record['name'],rc,str(record['log'])))
                print(f'[FAILED] {record["name"]} exit={rc} see {record["log"]}',flush=True)
                if stop_on_errors:
                    # Other running tasks may finish to their own checkpoints.
                    pending=[]
        if not made:time.sleep(1.)
    # Do not claim success if Slurm shutdown left a running process.
    if failed:
        raise RuntimeError('Failed stages: '+repr(failed))
    return {'finished':finished,'pending':pending,'running':[x['job'] for x in active]}


def _task_completed(repo,ds,seed,model,stage):
    if stage=='discover':
        return (per_seed_dir(repo,ds,seed)/'pruning_metadata.json').is_file()
    base=Path(repo)/'checkpoints'/ds/f'seed_{seed}'/model
    if stage=='train':
        return (base/'completed.json').is_file() if model!='staged_dpn' else (base/'selected.json').is_file()
    if stage=='latency':
        if model=='staged_dpn':
            select=base/'selected.json'
            if not select.exists():return False
            variant=json.loads(select.read_text())['variant']
            base=Path(str(base)+'__'+variant)
        return (base/'latency.json').is_file()
    return False


def run(repo='.',datasets=None,seeds=None,models=None,*,device='auto',
        max_workers_per_gpu=16,gpu_util_target=.92,gpu_memory_limit=.9,
        max_seconds=None,dry_run=False,allow_discovery=True):
    repo=Path(repo).resolve();cfgs,seedlist,modelnames=parse_selection(repo,
        datasets or ['all'],seeds or ['all'],models or ['all'])
    if max_workers_per_gpu<1 or max_workers_per_gpu>32:raise ValueError('Workers/GPU must be in 1..32')
    backend,gpus=detected_accelerator(device)
    assignments=assignment(cfgs,gpus)
    deadline=time.time()+max_seconds if max_seconds else None
    print('[resources]',json.dumps({'visible_gpus':gpus,
         'backend':backend,'dataset_gpu_assignment':assignments,'max_workers_per_gpu':max_workers_per_gpu,
         'util_target':gpu_util_target,'vram_limit':gpu_memory_limit}),flush=True)
    for ds,cfg in cfgs.items():
        # Validate the exact prepared split before spending GPU allocation.
        if not dry_run:split_manifest(repo,cfg,include_test=False)
        print('[dataset]',ds,'seeds',seedlist,'models',modelnames,flush=True)
    if dry_run:return {'datasets':list(cfgs),'seeds':seedlist,'models':modelnames,'gpus':gpus}
    if (repo/'results'/'TEST_EVALUATION_FROZEN.json').exists():
        raise RuntimeError('Held-out test was frozen; further training is blocked. '
                           'Use a separate new experiment for a new study.')
    initialize(repo)
    # Inter-allocation lock is held in the parent process only, not subprocesses.
    lock=repo/'checkpoints'/'.runner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock),timeout=0):
        if (repo/'results'/'TEST_EVALUATION_FROZEN.json').exists():
            raise RuntimeError('Held-out test was frozen; further training is blocked.')
        for ds in cfgs:
            for seed in seedlist:ensure_seed_readme(repo,ds,seed)
        discover=[(ds,seed,None) for ds in cfgs for seed in seedlist
                  if not _task_completed(repo,ds,seed,None,'discover')]
        if discover:
            if not allow_discovery:
                missing=', '.join(f'{ds}/{seed}' for ds,seed,_ in discover)
                raise RuntimeError('Discovery disabled. Import verified pruned tickets first: '+missing)
            if backend!='cuda':
                print('[warning] Original structural pruning on CPU/Apple Metal may be very slow',flush=True)
            result=_run_stage(repo,cfgs,seedlist,discover,'discover',assignments,deadline,
                              min(5,max_workers_per_gpu),gpu_util_target,gpu_memory_limit)
            if result['pending'] or result['running']:
                print('[PARTIAL] pruning stage has unfinished tickets');return result
        for ds,cfg in cfgs.items():
            for seed in seedlist:
                # Model configs can now be resolved without touching TEST.
                for name in modelnames:ensure_model(repo,ds,seed,name,cfg)
        training=[(ds,seed,name) for ds in cfgs for seed in seedlist
                  for name in modelnames if not _task_completed(repo,ds,seed,name,'train')]
        if training:
            result=_run_stage(repo,cfgs,seedlist,training,'train',assignments,deadline,
                        max_workers_per_gpu,gpu_util_target,gpu_memory_limit)
            if result['pending'] or result['running']:
                print('[PARTIAL] final model stage incomplete; safe to rerun');return result
        # Avoid competing with training tasks during isolated latency profiling.
        latency=[(ds,seed,name) for ds in cfgs for seed in seedlist
                 for name in modelnames if not _task_completed(repo,ds,seed,name,'latency')]
        if latency:
            result=_run_stage(repo,cfgs,seedlist,latency,'latency',assignments,deadline,
                  1,1.,gpu_memory_limit)
            if result['pending'] or result['running']:
                print('[PARTIAL] isolated latency stage unfinished');return result
        images=make_plots(repo,list(cfgs))
        print(f'[PASS] Requested dataset/seed/model grid complete. Test remains sealed. '
              f'{len(images)} graphs refreshed.',flush=True)
        return {'complete':True,'images':images}
