"""One independent GPU process / task; GPU visibility is set by the parent."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from src.data_io import read_config
from src.pruning import discover
from src.config_registry import read_model
from src.training import run_training,run_staged,isolated_latency


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--model')
    p.add_argument('--stage',choices=('discover','train','latency'),required=True)
    p.add_argument('--device',choices=('cuda','mps','cpu'),required=True)
    p.add_argument('--deadline',type=float)
    args=p.parse_args();cfg=read_config(args.repo,args.dataset)
    import torch
    torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA_VISIBLE_DEVICES did not expose assigned GPU')
    if device.type=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple Metal device was not available')
    print(f'[worker] ds={args.dataset} seed={args.seed} model={args.model} '
          f'stage={args.stage} device={device}',flush=True)
    if args.stage=='discover':
        discover(args.repo,cfg,args.seed,device,args.deadline);return
    spec=read_model(args.repo,args.dataset,args.seed,args.model,cfg)
    if spec is None:raise RuntimeError('Missing model config; scheduler must resolve dependencies')
    if args.stage=='train':
        result=(run_staged(args.repo,cfg,spec,device,args.deadline)
                 if args.model=='staged_dpn' else
                 run_training(args.repo,cfg,spec,device,args.deadline))
        if result is None:sys.exit(75)
    elif args.stage=='latency':
        ms=isolated_latency(args.repo,cfg,spec,device)
        print(f'[latency] {args.dataset}/{args.seed}/{args.model}: {ms:.4f} ms',flush=True)

if __name__=='__main__':main()
