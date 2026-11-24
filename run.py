#!/usr/bin/env python3
"""CLI version of the notebook. Example: python run.py --datasets jannis --seeds 42 --models matched_mlp"""
from __future__ import annotations
import argparse
from pathlib import Path
from src.model_factory import MODEL_NAMES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets',nargs='+',default=['all'],help='Dataset directory names or all')
    ap.add_argument('--seeds',nargs='+',default=['all'],help='Seed integers or all')
    ap.add_argument('--models',nargs='+',default=['all'],help='Model names or all')
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--device',choices=('auto','cuda','mps','cpu'),default='auto')
    ap.add_argument('--max-workers-per-gpu',type=int,default=16)
    ap.add_argument('--gpu-util-target',type=float,default=.92)
    ap.add_argument('--gpu-memory-limit',type=float,default=.9)
    ap.add_argument('--max-seconds',type=float)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--no-pruning',action='store_true',
        help='Fail if any accepted pruning ticket is absent; never start discovery')
    ap.add_argument('--plots-only',action='store_true')
    args=ap.parse_args()
    if args.plots_only:
        from src.results import make_plots
        print('Created:',make_plots(args.repo,args.datasets if args.datasets!=['all'] else None))
        return
    from src.runner import run
    run(args.repo,args.datasets,args.seeds,args.models,device=args.device,
        max_workers_per_gpu=args.max_workers_per_gpu,
        gpu_util_target=args.gpu_util_target,
        gpu_memory_limit=args.gpu_memory_limit,
        max_seconds=args.max_seconds,dry_run=args.dry_run,
        allow_discovery=not args.no_pruning)

if __name__=='__main__':main()
