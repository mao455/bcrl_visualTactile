"""Publish small, reproducible audit receipts and charts; never raw observations."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from tools.runtime import CODE, read, write


def publish(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=CODE/'research_docs/evidence/dataset_trust';out.mkdir(exist_ok=True)
    def clean(x):
        if isinstance(x,dict):return {clean(k):clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,str):
            for prefix in (str(CODE),'/home/CNS2025331827/docs/bc+rl_RSS'):
                x=x.replace(prefix,'$REPO')
            x=x.replace('/home/CNS2025331827','$HOME')
        return x
    rows=[read(p) for p in sorted((root/'episodes').glob('*.json'))]
    assert len(rows)==275 and read(root/'static_summary.json')['passed']
    def table(name,values):
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    table('episodes.csv',[dict(id=r['id'],family=r['family'],split=r['split'],origin=r['origin'],
        alpha=r['alpha'],steps=r['steps'],duration_s=r['duration_s'],sha256=r['sha256'],bytes=r['bytes'],
        state_valid=r['state_valid'],rgbd_valid=r['sensors']['rgbd_valid'],wrist_valid=r['sensors']['wrist_valid'],
        tactile_valid=r['sensors']['tactile_numeric_valid'],camera_clock_error=r['sensors']['camera_clock_max_error_s'],
        contact_clock_error=r['sensors']['tactile_clock_max_error_s'],q_continuity_error=r['q_continuity_max'],
        target_q_rmse_rad=r['target_vs_q_rmse'],slow_formula_error=r['slow_formula_max'],
        force_nonzero_ticks=r['disturbances']['force_w_N_nonzero_ticks'],force_peak_N=r['disturbances']['force_w_N_peak_norm'],
        decoded_rgb_sha256=r['sensors']['decoded_rgb_sha256'],decoded_depth_sha256=r['sensors']['decoded_depth_sha256']) for r in rows])
    families=read(root/'families_actual.json');rr=[]
    for f in families:
        d={k:f[k] for k in ('identity','split','episodes','table_z','fingerprint')};d['alphas']=json.dumps(f['alphas'])
        for k in ('object_xyz','hole_xyz','relative_xyz','quaternion','arm','hand','velocity','object_velocity'):
            d.update({f'{k}_{i}':x for i,x in enumerate(f[k])})
        rr.append(d)
    table('families.csv',rr)
    names=('static_summary','source_audit','normalization','coverage','attempts','failures','pairs','configurations','replay_plan')
    for name in names:
        value=clean(read(root/(name+'.json')))
        if name=='configurations':
            for variants in value.values():
                if len(variants)==1 and variants[0]['count']==275:
                    variants[0].pop('ids',None);variants[0]['ids_scope']='all275'
        write(out/(name+'.json'),value)
    for name in ('motion','physics_dr','runtime_commits','training_source','early_reset_ledger','sensor_and_label_extra','speed_details','initial_environment_rng','failed_capture_checks','historical_source_recovery','physical_details','fresh_pair_plan','fresh_pair_details','final_audit','videos'):
        if (root/(name+'.json')).exists():
            (out/(name+'.json')).write_text(json.dumps(clean(read(root/(name+'.json'))),separators=(',',':'))+'\n')
    physical=[]
    for mode in ('expert','replay'):
        p=root/(mode+'_results.json')
        if p.exists():physical+=read(p)
    if physical:
        write(out/'physical_results.json',clean(physical))
        groups=defaultdict(list)
        for r in physical:groups[(r['mode'],r['source_success'])].append(r)
        write(out/'physical_summary.json',{f'{mode}/source_success={ok}':dict(starts=len(v),
            legal=sum(r['result']['legal'] for r in v),success=sum(r['result']['success'] for r in v),
            complete=sum(r['result']['software_complete'] for r in v),
            reason_counts=dict(Counter(k for r in v for k,yes in r['result'].get('reasons',{}).items() if yes)),
            source_exhausted_censored=sum(r['result'].get('action_exhausted',False) for r in v)) for (mode,ok),v in groups.items()})
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for origin,color in [('old200','#3876b7'),('new75','#d66b21')]:
        r=[x for x in rows if x['origin']==origin]
        ax[0].hist([x['duration_s'] for x in r],bins=np.arange(2,13.6,.5),alpha=.6,label=origin,color=color)
        ax[1].scatter([x['alpha'] for x in r],[x['duration_s'] for x in r],s=12,alpha=.6,color=color,label=origin)
    pairs=read(root/'pairs.json');ax[2].scatter([x['alpha'] for x in pairs],[x['median_reference_ratio'] for x in pairs],s=12)
    ax[2].axhline(1,color='gray',ls='--');ax[2].axhline(1.4,color='gray',ls=':')
    for a in ax:a.grid(alpha=.2)
    ax[0].set(xlabel='Real simulated duration (s)',ylabel='Episodes',title='275 admitted successes');ax[0].legend()
    ax[1].set(xlabel='Stable execution alpha',ylabel='Duration (s)',title='Alpha is not inverse duration')
    ax[2].set(xlabel='Alpha',ylabel='Slow / median native duration',title='151 observable-initial pairs; 59 unpaired')
    fig.savefig(out/'speed.png',dpi=150);plt.close(fig)
    c=read(root/'coverage.json');fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for cohort,color in [('old','#3876b7'),('new','#d66b21')]:
        for selected,marker in [(False,'x'),(True,'o')]:
            v=[r for r in c['candidates'] if r['cohort']==cohort and r['selected']==selected]
            for a,key in zip(ax[:2],('object_xyz','relative_xyz')):
                a.scatter([r[key][0]*1000 for r in v],[r[key][1]*1000 for r in v],s=20,marker=marker,alpha=.65,label=cohort+(' selected' if selected else ' not selected'))
    ax[0].set(xlabel='Object x (mm)',ylabel='Object y (mm)',title='All 148 initial candidates')
    ax[1].set(xlabel='Hole - object x (mm)',ylabel='Hole - object y (mm)',title='Relative positions and selection')
    rejected=[r for r in c['candidates'] if not r['legal']]
    for a,key in zip(ax[:2],('object_xyz','relative_xyz')):
        a.scatter([r[key][0]*1000 for r in rejected],[r[key][1]*1000 for r in rejected],s=65,facecolors='none',edgecolors='red',label='OBB rejected')
        a.grid(alpha=.2)
    ax[0].legend(fontsize=7)
    counts=Counter(f['episodes'] for f in families);ax[2].bar(list(counts),list(counts.values()))
    ax[2].set(xlabel='Successful trajectories per family',ylabel='Families',title='Repeated initial states are intentional')
    fig.savefig(out/'coverage.png',dpi=150);plt.close(fig)
    write(out/'provenance.json',dict(run='dataset_trust_20261106_01',source_root='$REPO/outputs/visuotactile/run/dataset_trust_20261106_01',
        session=clean(read(root/'session.json')),status=clean(read(root/'status.json')),
        all_275_files_hashed=True,all_64663_frames_pixels_read=True,original_artifacts_modified=False,
        physical_complete=(root/'phases/physical_acceptance/complete.json').exists(),
        fresh_pair_complete=(root/'phases/fresh_pair_check/complete.json').exists()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();publish(a.root.resolve())
