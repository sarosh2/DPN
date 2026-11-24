"""Atomic CSV result storage, reproducible plots, and readable folder indexes."""
from __future__ import annotations
from contextlib import contextmanager
from filelock import FileLock
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

PRUNE_COLUMNS=('dataset','seed','max_rounds','actual_rounds','percent_pruned_parameters',
 'starting_block_count','final_block_count','block_reduction_percent',
 'final_block_layout','validation_epochs','total_time_seconds',
 'starting_parameter_size','final_parameter_size',
 'starting_inference_latency_ms','ending_inference_latency_ms',
 'inference_latency_reduction_percent','reference_parameter_target',
 'reference_parameter_deviation_percent','active_final_parameters',
 'ticket_sha256','original_pruning_entrypoint','source_config_sha256',
 'latency_measurement_context')

METRICS=('accuracy','macro_f1','balanced_accuracy','log_loss','brier','ece15','auroc')
TRAIN_COLUMNS=('dataset','seed','model','variant','split','status','epochs_completed',
 'best_epoch','early_stopped','training_wall_seconds','training_gpu_seconds',
 'initialization_seed','optimizer','learning_rate','weight_decay','batch_size',
 'max_epochs','patience','normalization','dropout','lth_search_seconds',
 'parameters_allocated','parameters_active','parameter_budget_target',
 'parameter_budget_gap_percent','hidden_blocks','output_blocks','mlp_hidden_depth',
 'inference_latency_batch1_ms','checkpoint_fingerprint',
 'dataset_config_sha256','model_config_sha256','ticket_sha256',
 'train_loss','validation_loss',
 *tuple('train_'+m for m in METRICS),
 *tuple('val_'+m for m in METRICS),
 *tuple('test_'+m for m in METRICS),'test_loss','test_evaluation_frozen',
 'test_accessed','data_source','data_split_provenance')


def path_for(repo,which):
    assert which in ('training','pruning')
    return Path(repo)/'results'/'csvs'/(which+'.csv')


def initialize(repo):
    for name,cols in [('training',TRAIN_COLUMNS),('pruning',PRUNE_COLUMNS)]:
        p=path_for(repo,name);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            with p.open('w',newline='') as f:csv.writer(f).writerow(cols)
    (Path(repo)/'results'/'images').mkdir(parents=True,exist_ok=True)


@contextmanager
def locked(path):
    """Cross-platform CSV writer lock (Linux, macOS, Windows)."""
    lock_path=Path(str(path)+'.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock_path),timeout=30):
        yield


def read_rows(repo,which):
    initialize(repo)
    with path_for(repo,which).open(newline='') as f:return list(csv.DictReader(f))


def upsert(repo,which,row):
    p=path_for(repo,which);initialize(repo)
    columns=TRAIN_COLUMNS if which=='training' else PRUNE_COLUMNS
    unique=('dataset','seed','model') if which=='training' else ('dataset','seed')
    if any(key not in row for key in unique):raise ValueError('Missing result key')
    with locked(p):
        with p.open(newline='') as f:old=list(csv.DictReader(f))
        old=[r for r in old if tuple(str(r[k]) for k in unique)!=tuple(str(row[k]) for k in unique)]
        old.append({key:row.get(key,'') for key in columns})
        old.sort(key=lambda r:tuple(str(r[k]) for k in unique))
        tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
            writer.writeheader();writer.writerows(old)
        os.replace(tmp,p)


def make_plots(repo,datasets=None):
    """Generate boxplots from **actual five-seed TEST entries only**.

    When test is sealed, an explicitly marked pending figure is generated,
    never a synthetic test distribution or a validation-as-test substitution.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    repo=Path(repo);initialize(repo)
    rows=read_rows(repo,'training')
    historical_path=repo/'results'/'csvs'/'historical_validation_summary.csv'
    historical_names=set()
    if historical_path.is_file():
        with historical_path.open(newline='') as fh:
            historical_names={r['dataset'] for r in csv.DictReader(fh)}
    known=sorted(set(r['dataset'] for r in rows)|set(datasets or [])|historical_names)
    all_images=[]
    for ds in known:
        models=defaultdict(list)
        for r in rows:
            if r['dataset']!=ds or r['status']!='complete':continue
            val=r.get('test_accuracy','')
            if val:
                try:models[r['model']].append(100*float(val))
                except ValueError:pass
        fig,ax=plt.subplots(figsize=(max(7,len(models)*1.3),4.7))
        if models:
            names=list(sorted(models))
            values=[models[n] for n in names]
            bp=ax.boxplot(values,tick_labels=names,patch_artist=True,showmeans=True)
            for patch in bp['boxes']:patch.set_facecolor('#c6d8ed')
            for i,v in enumerate(values,1):
                jitter=np.random.default_rng(i).uniform(-.06,.06,len(v))
                ax.scatter(np.full(len(v),i)+jitter,v,alpha=.7,s=22,color='#134276')
                if len(v)>1:
                    ax.errorbar(i,float(np.mean(v)),yerr=float(np.std(v,ddof=1)),
                                fmt='D',color='#bd5e29',capsize=5)
            ax.tick_params(axis='x',labelrotation=22)
            ax.set_ylabel('Held-out test accuracy (%)')
        else:
            ax.text(.5,.5,'TEST RESULTS PENDING\nNo held-out test values recorded',
                    ha='center',va='center',transform=ax.transAxes,fontsize=13)
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_ylabel('Held-out test accuracy (%)')
        ax.set_title(f'{ds}: test accuracy across seeds')
        ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        out=repo/'results'/'images'/f'{ds}_test_accuracy.png'
        fig.savefig(out,dpi=175);plt.close(fig)
        all_images.append(out.name)
    # Additional validation figures never claim to be held-out test data.
    # Import uses true per-seed values. Before import, the aggregate display
    # uses only the rounded five-seed summary supplied by the paper author.
    summary_file=repo/'results'/'csvs'/'historical_validation_summary.csv'
    historical=[]
    if summary_file.is_file():
        with summary_file.open(newline='') as fh:
            historical=list(csv.DictReader(fh))
    validation_images=[]
    validation_datasets=sorted(set(known)|{r['dataset'] for r in historical})
    for ds in validation_datasets:
        observed=defaultdict(list)
        for row in rows:
            if row.get('dataset')!=ds or row.get('status')!='complete':continue
            v=row.get('val_accuracy','')
            if v not in ('',None):observed[row['model']].append(100*float(v))
        fig,ax=plt.subplots(figsize=(max(9,1.16*max(len(observed),7)),4.9))
        if observed:
            names=sorted(observed);ys=[observed[n] for n in names]
            ax.boxplot(ys,showmeans=True,patch_artist=False)
            for i,vals in enumerate(ys,1):
                ax.scatter(np.full(len(vals),i),vals,s=15,alpha=.65)
                if len(vals)>1:ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),
                                             fmt='D',capsize=3)
            ax.set_xticks(range(1,len(names)+1),names,rotation=24,ha='right')
            ax.set_title(f'{ds}: per-seed validation accuracy (observed values)')
            provenance='Per-seed values from verified training.csv'
        else:
            dataset_rows=sorted([r for r in historical if r['dataset']==ds],key=lambda r:r['model'])
            if dataset_rows:
                names=[r['model'] for r in dataset_rows]
                means=[float(r['accuracy_mean_percent']) for r in dataset_rows]
                stdevs=[float(r['accuracy_sd_percent']) for r in dataset_rows]
                xx=np.arange(len(names))
                ax.errorbar(xx,means,yerr=stdevs,fmt='o',capsize=4,markersize=5)
                ax.set_xticks(xx,names,rotation=24,ha='right')
                ax.set_title(f'{ds}: historical validation mean and SD (five seeds)')
                provenance='Rounded aggregate report ONLY; not a per-seed boxplot'
            else:
                ax.text(.5,.5,'VALIDATION DATA PENDING',ha='center',va='center',transform=ax.transAxes)
                ax.set_xticks([])
                provenance='No per-seed or historical summary available'
        ax.set_ylabel('Validation accuracy (%)');ax.grid(axis='y',alpha=.18)
        ax.text(.01,-.26,provenance,transform=ax.transAxes,fontsize=8,alpha=.8)
        fig.tight_layout()
        filename=f'{ds}_validation_accuracy.png'
        fig.savefig(repo/'results'/'images'/filename,dpi=160,bbox_inches='tight')
        plt.close(fig);validation_images.append(filename)
    lines=['# Validation and test accuracy plots','',
        'Figures use per-seed test accuracy values from `../csvs/training.csv`. '
        'The box spans the interquartile range, the line is the median, and '
        'the diamond marks the mean. Orange bars mark mean ± sample SD. '
        'A dataset without test scores shows an explicit **pending** figure.', '']
    for filename in all_images:
        lines.extend([f'## {filename.removesuffix("_test_accuracy.png")}',
                      f'![Test accuracy for {filename}](./{filename})',''])
    lines.extend(['# Validation accuracy figures','',
         'Before per-seed migration, validation figures show the paper-author supplied rounded mean and SD, not synthetic seed values. After import, they use actual per-seed observations.',''])
    for filename in validation_images:
        lines.extend([f'## {filename.removesuffix("_validation_accuracy.png")}: validation',
                      f'![Validation accuracy for {filename}](./{filename})',''])
    (repo/'results'/'images'/'README.md').write_text('\n'.join(lines)+'\n')
    return all_images
