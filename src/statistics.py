"""Two-sided Welch comparisons from actual per-seed metrics; Holm adjustment."""
from __future__ import annotations
import argparse
import csv
from itertools import product
from pathlib import Path
import math
import os
import numpy as np
from scipy import stats
from src.results import read_rows

MLPS=('matched_mlp','matched_resnet','lth_mlp')
DPNS=('pruned_dpn','block_matched_dpn','parameter_matched_max_dpn','staged_dpn')
FIELDS=('dataset','metric','split','mlp','dpn','n_mlp','n_dpn',
        'mean_difference_pp','welch_df','ci95_lower_pp','ci95_upper_pp',
        'welch_t','p_value_two_sided','holm_p_value','reject_holm_005',
        'paired_n','paired_difference_pp','paired_ci95_lower_pp',
        'paired_ci95_upper_pp','paired_t','paired_p_value_two_sided')


def compute(repo,split='val',metric='accuracy',min_seeds=2):
    if split not in ('val','test'):raise ValueError('Only val or test split is valid')
    from src.results import METRICS
    if metric not in METRICS:raise ValueError('Unknown metric')
    key=f'{split}_{metric}'
    records=read_rows(repo,'training')
    values={}
    for row in records:
        if row['status']!='complete' or not row.get(key):continue
        ds=row['dataset'];mod=row['model'];seed=int(row['seed'])
        values.setdefault((ds,mod),{})[seed]=float(row[key])
    candidates=[]
    for ds in sorted(set(k[0] for k in values)):
        for mlp,dpn in product(MLPS,DPNS):
            a=values.get((ds,mlp),{});b=values.get((ds,dpn),{})
            if len(a)<min_seeds or len(b)<min_seeds:continue
            av=np.array(list(a.values()));bv=np.array(list(b.values()))
            # Welch-Satterthwaite independent-group variance estimate.
            va=np.var(av,ddof=1)/len(av);vb=np.var(bv,ddof=1)/len(bv)
            se=math.sqrt(va+vb)
            df=((va+vb)**2)/(va**2/(len(av)-1)+vb**2/(len(bv)-1)) if se>0 else float('inf')
            diff=float(np.mean(av)-np.mean(bv))
            t=diff/se if se else 0.
            p=float(2*stats.t.sf(abs(t),df)) if se else (1. if diff==0 else 0.)
            crit=stats.t.ppf(.975,df) if math.isfinite(df) else stats.norm.ppf(.975)
            common=sorted(set(a)&set(b));paired=np.array([a[s]-b[s] for s in common])
            pn=len(paired);pdiff=pt=pp=pl=pu=None
            if pn>=2:
                pdiff=float(np.mean(paired));sd=float(np.std(paired,ddof=1))
                sp=sd/math.sqrt(pn)
                pt=pdiff/sp if sp else 0.
                pp=float(2*stats.t.sf(abs(pt),pn-1)) if sp else (1. if pdiff==0 else 0.)
                pc=float(stats.t.ppf(.975,pn-1))*sp
                pl=pdiff-pc;pu=pdiff+pc
            candidates.append({'dataset':ds,'metric':metric,'split':split,'mlp':mlp,
                'dpn':dpn,'n_mlp':len(a),'n_dpn':len(b),
                'mean_difference_pp':100*diff,'welch_df':df,
                'ci95_lower_pp':100*(diff-crit*se),
                'ci95_upper_pp':100*(diff+crit*se),'welch_t':t,
                'p_value_two_sided':p,'paired_n':pn,
                'paired_difference_pp':100*pdiff if pdiff is not None else '',
                'paired_ci95_lower_pp':100*pl if pl is not None else '',
                'paired_ci95_upper_pp':100*pu if pu is not None else '',
                'paired_t':pt if pt is not None else '',
                'paired_p_value_two_sided':pp if pp is not None else ''})
    ordered=sorted(range(len(candidates)),key=lambda i:candidates[i]['p_value_two_sided'])
    last=0.
    for rank,i in enumerate(ordered):
        raw=candidates[i]['p_value_two_sided']
        last=max(last,min(1.,(len(ordered)-rank)*raw))
        candidates[i]['holm_p_value']=last
        candidates[i]['reject_holm_005']=last<.05
    target=Path(repo)/'results'/'csvs'/'welch_tests.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+f'.tmp-{os.getpid()}')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(candidates)
    temp.replace(target)
    print(f'[Welch] {len(candidates)} tests, split={split}, metric={metric}, saved {target}')
    print('Differences are MLP minus DPN; positive values favor MLP.')
    return candidates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--metric',default='accuracy')
    args=p.parse_args()
    rows=compute(args.repo,args.split,args.metric)
    for r in rows:
        print(f"{r['dataset']:12} {r['mlp']:16} vs {r['dpn']:30} "
              f"diff={r['mean_difference_pp']:+.3f}pp "
              f"95%CI=[{r['ci95_lower_pp']:+.3f},{r['ci95_upper_pp']:+.3f}] "
              f"p={r['p_value_two_sided']:.4g} holm={r['holm_p_value']:.4g}")

if __name__=='__main__':main()
