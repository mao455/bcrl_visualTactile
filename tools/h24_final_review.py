"""Read-only final artifact review and compact public export; no new inference."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch

from tools.h24_runtime_recovery import verify
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def pooled_spatial(rows, field='ee_mm'):
    count=sum(row[field]['n'] for row in rows)
    return dict(n=count, mean=sum(row[field]['n']*row[field]['mean'] for row in rows)/count,
        rms=math.sqrt(sum(row[field]['n']*row[field]['rms']**2 for row in rows)/count))


def review(root):
    verify(root)
    status=read(root/'status.json')
    if status.get('training_state')!='completed' or status.get('updates')!=10000:
        raise ValueError('Final training result not complete')
    public=Path('research_docs/evidence/h24_parameterization_20261103')
    local=root/'final_review';local.mkdir(exist_ok=True)
    full=local/'full_report_summary.json'
    if not full.exists():shutil.copy2(public/'summary.json',full)
    summary=read(full)
    if set(summary['groups'])!={'sample_3000','epsilon_3000','sample_5000','sample_8000','sample_10000'}:
        raise ValueError('Final milestone report incomplete')
    checkpoints={}
    for key,value in read(root/'checkpoint_manifest.json').items():
        path=Path(value['path'])
        if sha256(path)!=value['sha256']:raise ValueError('Checkpoint drift: '+key)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        steps=sorted({int(state['step']) for state in saved['optimizer']['state'].values()})
        if steps!=[saved['updates']] or saved['ema_optimization_step']!=saved['updates']:
            raise ValueError('Optimizer/EMA count mismatch')
        if saved['config_sha256']!=value['configuration_sha256']:
            raise ValueError('Checkpoint configuration mismatch')
        checkpoints[key]=dict(sha256=value['sha256'],updates=saved['updates'],optimizer_steps=steps,
            ema_steps=saved['ema_optimization_step'],complete_rng_sampler=all(k in saved for k in
                ('schedule_rng','noise_rng','torch_rng','cuda_rng','schedule_order','schedule_cursor')))
        del saved
    committed={}
    for marker in root.glob('*_committed.json'):
        name=marker.stem.removesuffix('_committed');path=root/name/'result.json'
        if sha256(path)!=read(marker)['sha256']:raise ValueError('Stage drift: '+name)
        committed[name]=sha256(path)
    logs={}
    for group,folder,updates in [('sample','sample_resume/h24_train',10000),('epsilon','h24_epsilon_train',3000)]:
        rows=[json.loads(line) for line in (root/folder/'train.jsonl').read_text().splitlines()]
        if [r['update'] for r in rows]!=list(range(1,updates+1)):
            raise ValueError('Noncontiguous training progress')
        error=max(abs(r['lr']-1e-4*min(1.,r['update']/500)*.5*(1+math.cos(math.pi*max(0,r['update']-500)/18500))) for r in rows)
        if error or not all(math.isfinite(r[k]) for r in rows for k in ('loss','norm','history_grad')):
            raise ValueError('Nonfinite training or LR schedule mismatch')
        logs[group]=dict(updates=len(rows),lr_formula_max_error=error,finite=True)
    live=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:parts=(proc/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if '--root' in parts and root.name in parts[parts.index('--root')+1] and 'tools.h24_final_review' not in parts:
            live.append(int(proc.name))
    if live:raise RuntimeError('Owned experiment process still alive')
    success=[];cohorts={};actual={}
    for name,group in summary['groups'].items():
        loop=read(root/(name+'_loop/result.json'))
        if len(loop['trials'])+len(loop['missing'])!=12:raise ValueError('Incomplete cohort denominator')
        cohorts[name]=dict(completed=len(loop['trials']),successful=sum(bool(t['success']) for t in loop['trials']),
            startup_faults=sum(m['classification']=='pre_scene_native_fault' for m in loop['missing']),
            unstarted=sum(m['classification'].startswith('not_started') for m in loop['missing']),
            splits=group['closed_loop_counts'])
        for trial in loop['trials']:
            if not trial['success']:continue
            path=root/(name+'_loop')/trial['identity']
            with h5py.File(path/'numeric.h5') as file:
                terminal=bool(file['episode/transition/termination_reasons/max_successes'][-1])
            with np.load(path/'trace.npz') as trace:
                subgoals=int(trace['subgoals'][-1]);steps=len(trace['joint']);alignment=bool(trace['alignment'][-1])
            if not terminal or subgoals!=2 or steps!=trial['steps']:
                raise ValueError('Native success evidence mismatch')
            success.append(dict(group=name,identity=trial['identity'],seed=trial['seed'],cohort=trial['cohort'],
                steps=steps,native_terminal_max_successes=terminal,subgoals=subgoals,alignment=alignment,
                checkpoint_sha256=trial['checkpoint_sha256'],numeric_sha256=sha256(path/'numeric.h5'),
                trace_sha256=sha256(path/'trace.npz')))
        actual[name]={row['identity']:row['approach'] for row in group['spatial']['closed_actual_fk']['episodes']
            if row['model']=='Student' and row['approach']}
    val_ids={s['identity'] for s in read(root/'run_plan.json')['formal_cohort'] if s['cohort']=='validation'}
    milestones=['sample_5000','sample_8000','sample_10000']
    paired=sorted(val_ids.intersection(*(set(actual[name]) for name in milestones)))
    for identity in paired:
        initial=[read(root/(name+'_loop')/identity/'initial_observable.json') for name in milestones]
        if not all(state==initial[0] for state in initial):raise ValueError('Observable reset mismatch')
    result=dict(reviewed_epoch=time.time(),status=status,source_count=len(read(root/'session.json')['implementation_sha256']),
        committed_stage_hashes=committed,checkpoints=checkpoints,logs=logs,cohorts=cohorts,
        verified_native_successes=success,live_owned_workers=live,new_optimizer_updates=0,new_kit_starts=0,
        frozen_runtime_verified=True,resources=read(root/'resources/resources.json'),
        common_validation_actual_fk=dict(ids=paired,metrics={name:pooled_spatial([actual[name][i] for i in paired]) for name in milestones},
            scope='Same7 validation IDs; expert precontact common tick prefix; observed resets equal, hidden simulator exactness unproven'),
        full_report_local_sha256=sha256(full),cleanup='No artifact deletion; all checkpoints and failed attempts retained')
    # Public reports retain cohort/trial metrics. Large per-episode phase tables
    # remain in the full local report; publish pooled metrics and 10Hz mean curves.
    for name,group in summary['groups'].items():
        phases=group['spatial'].pop('phases');aggregate=[]
        for model,phase in sorted({(r['model'],r['phase']) for r in phases}):
            rows=[r for r in phases if (r['model'],r['phase'])==(model,phase)]
            aggregate.append(dict(model=model,phase=phase,episodes=len(rows),ticks=sum(r['ticks'] for r in rows),
                **{key:pooled_spatial(rows,key) for key in ('ee_mm','orientation_degree','along_expert_mm')}))
        group['spatial']['phase_aggregates']=aggregate
        curves=group['spatial']['closed_actual_fk'].pop('mean_curves')
        selected=[]
        for model in sorted({r['model'] for r in curves}):
            rows=[r for r in curves if r['model']==model]
            selected += [row for i,row in enumerate(rows) if i%6==0 or i==len(rows)-1]
        destination=public/(name+'_actual_fk_mean10hz.csv')
        if selected:
            with destination.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(selected[0]),lineterminator='\n')
                writer.writeheader();writer.writerows(selected)
        group['spatial']['closed_actual_fk']['mean_curve_csv']=destination.name
    summary['publication']=dict(version=2,phase_reduction='Count-weighted pooled mean/RMS; no averaged quantiles',
        curves='Every sixth aggregate60Hz point plus final; full-resolution plotted PNG and local report retained',
        full_report_sha256=sha256(full),new_inference=False)
    atomic_json(public/'summary.json',summary)
    atomic_json(local/'completed_review.json',result)
    atomic_json(public/'completed_review.json',result)
    print(json.dumps(dict(checkpoints=len(checkpoints),committed_stages=len(committed),cohorts=cohorts,
        common_validation_actual_fk=result['common_validation_actual_fk'],native_successes=success,
        public_summary_bytes=(public/'summary.json').stat().st_size)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    torch.set_num_threads(2);review(parser.parse_args().root.resolve())
