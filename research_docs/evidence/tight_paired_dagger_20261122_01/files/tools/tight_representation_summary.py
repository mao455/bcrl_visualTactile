"""Reproducible small public summaries and figures; raw weights/features stay local."""
import csv,gzip,json,collections,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.tight_representation_audit import PUBLIC,RUN,ROOTS,read,write,sha,load,table,quant


def summary():
    exposure=read(PUBLIC/'exposure.json');episodes=list(csv.DictReader((PUBLIC/'episode_exposure.csv').open()))
    errors=list(csv.DictReader((PUBLIC/'offline_errors.csv').open()));groups=collections.defaultdict(list)
    for r in errors:groups[tuple(r[k] for k in ('model','kind','stage','metric'))].append(r)
    er=[]
    for key,rows in groups.items():
        n=sum(int(r['ticks']) for r in rows);v=dict(zip(('model','kind','stage','metric'),key));v.update(ticks=n,episodes=len(rows))
        for limb,d in [('arm',7),('hand',22)]:
            v[limb+'_rmse_rad']=math.sqrt(sum(float(r[limb+'_squared_sum']) for r in rows)/n/d)
            v[limb+'_rmse_fraction_of_full_reachable_span']=math.sqrt(sum(float(r[limb+'_span_squared_sum']) for r in rows)/n/d)
        er.append(v)
    table(PUBLIC/'offline_summary.csv',er)
    common_ids={r['identity'] for r in errors if r['model']=='accepted72_A' and r['kind']=='correction'}
    matched=collections.defaultdict(list)
    for r in errors:
        if r['identity'] in common_ids:matched[(r['model'],r['stage'],r['metric'])].append(r)
    mm=[]
    for (model,stage,metric),rs in matched.items():
        n=sum(int(r['ticks']) for r in rs)
        mm.append(dict(model=model,stage=stage,metric=metric,episodes=len(rs),ticks=n,arm_rmse_rad=math.sqrt(sum(float(r['arm_squared_sum']) for r in rs)/n/7),hand_rmse_rad=math.sqrt(sum(float(r['hand_squared_sum']) for r in rs)/n/22)))
    table(PUBLIC/'offline_common42_corrections.csv',mm)

    ab=list(csv.DictReader((PUBLIC/'interventions.csv').open()));aa=[]
    for model in sorted({r['model'] for r in ab}):
        for kind in ('force_zero','visual_zero','visual_reverse'):
            r=[x for x in ab if x['model']==model and x['intervention']==kind];n=sum(int(x['ticks']) for x in r);legal=sum(int(x['legal_ticks']) for x in r)
            a=dict(model=model,intervention=kind,episodes=len(r),command_delta_rms_rad=math.sqrt(sum(float(x['command_delta_rms_rad'])**2*int(x['ticks']) for x in r)/n))
            for limb in ('arm','hand'):
                a[limb+'_mse_change_percent']=100*(sum(float(x['changed_'+limb+'_mse'])*int(x['legal_ticks']) for x in r)/sum(float(x['normal_'+limb+'_mse'])*int(x['legal_ticks']) for x in r)-1)
            aa.append(a)
    table(PUBLIC/'intervention_summary.csv',aa)
    stage_rows=[]
    for s in exposure['stages']:
        stage_rows.append(dict(path=s['path'],role=s['role'],component=s['component'],pool=s['train_trajectories'],selected_epoch=s['selected_epoch'],selected_updates=s['selected_updates'],actual_updates=s['actual_log_updates'],selected_visits=s['selected_visits'],minimum_visits=min(s['episode_visits'].values()),maximum_visits=max(s['episode_visits'].values()),learning_rate=s['lr'],parent=s['parent']))
    table(PUBLIC/'stage_summary.csv',stage_rows)
    restricted={}
    for name in exposure['summary']:
        rows=[r for r in episodes if r['model']==name];seen=[r for r in rows if int(r['visits'])>0]
        restricted[name]=dict(current130=exposure['summary'][name],seen_population=dict(trajectories=len(seen),visits=quant([int(r['visits']) for r in seen]),unseen_trajectories=len(rows)-len(seen)))
    write(PUBLIC/'exposure_population_summary.json',restricted)
    plt.rcParams.update({'font.size':9})
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    tick=list(csv.DictReader(gzip.open(PUBLIC/'tick_exposure.csv.gz','rt')))
    for name in exposure['summary']:
        if not name.startswith('candidate'):continue
        rows=[r for r in episodes if r['model']==name];x=np.sort([int(r['visits']) for r in rows]);axs[0].step(x,np.arange(1,len(x)+1)/len(x),label=name.replace('candidate130_',''))
        x=np.sort([int(r['slots']) for r in tick if r['model']==name]);axs[1].step(x,np.arange(1,len(x)+1)/len(x),label=name.replace('candidate130_',''))
    for ax,title in zip(axs,['Selected ancestry: trajectory visits','Selected ancestry: label slots (overlap included)']):
        ax.axvline(50,color='black',ls='--',lw=.8);ax.set(title=title,xlabel='Count',ylabel='Cumulative fraction',xscale='symlog');ax.legend();ax.grid(alpha=.2)
    fig.savefig(PUBLIC/'exposure_cdf.png',dpi=170);plt.close(fig)
    curves=list(csv.DictReader(gzip.open(PUBLIC/'training_curves.csv.gz','rt')))
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(role,component) in zip(axs.flat,[(r,c) for r in 'AB' for c in ('DP','rear')]):
        node=next(s for s in stage_rows if 'tight_recovery_' in s['path'] and s['role']==role and s['component']==component)
        rr=[r for r in curves if r['node']==node['path']];by=collections.defaultdict(list)
        for r in rr:by[int(r['epoch'])].append(float(r['loss']))
        ax.plot(list(by),[np.mean(v) for v in by.values()]);ax.axvline(node['selected_epoch'],color='red',ls='--',label='selected checkpoint');ax.legend();ax.set(title=role+' '+component+' (different objectives/scales)',xlabel='Local epoch',ylabel='Mean logged update loss');ax.grid(alpha=.2)
    fig.savefig(PUBLIC/'candidate_training_curves.png',dpi=170);plt.close(fig)
    probes=list(csv.DictReader((PUBLIC/'probe_metrics.csv').open()));methods=['proprio','visual','touch_proprio','vision_touch_proprio','sensor_history','decoder_history']
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,metric in zip(axs,['object_palm_mm','object_hole_mm']):
        for model in ['accepted72_A','accepted72_B','candidate130_A','candidate130_B']:
            values=[float(next(r[metric] for r in probes if r['model']==model and r['method']==m and r['population']=='independent6_all')) for m in methods];ax.plot(range(len(methods)),values,marker='o',label=model)
        ax.set_xticks(range(len(methods)),['q/qdot','vision','touch+q','vision+touch+q','sensor history','decoder'],rotation=20);ax.set(ylabel='3D vector RMSE (mm)',title='Independent6: '+metric);ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(PUBLIC/'representation_readouts.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    phases=['grasp_without_contact','contact_not_held','held_far','held_rotating','held_near_goal60','after_first_native_subgoal','contact_lost']
    for role,ax in zip('AB',axs):
        s=exposure['summary']['candidate130_'+role+'_DP'];ax.bar(range(len(phases)),[s['phases'][k]['below50']/s['phases'][k]['n']*100 for k in phases]);ax.set_xticks(range(len(phases)),phases,rotation=35,ha='right',fontsize=7);ax.set(title=role+' DP selected ancestry',ylabel='% labels with <50 loss slots',ylim=(0,100))
    fig.savefig(PUBLIC/'phase_underexposure.png',dpi=170);plt.close(fig)
    write(PUBLIC/'summary_scope.json',dict(policy_updates=0,new_simulator_starts=0,new_dagger=0,exposure_denominator='130 training trajectories; 27508 legal ticks. accepted72 unobserved additions explicitly zero, seen population separate.',errors='Rad RMSE and error/full reachable target span. DP uses same recorded histories as matched rear; A own-target error differs semantically from command error. Tiny rad errors can be large relative to narrow reachable interval.',curves='Mean logged stochastic train loss, selected epoch marked; no inference of convergence from monotonicity.'))

def validation_curves():
    rows=[];sources=[]
    for role in 'AB':
        for component in ('DP','rear'):
            folder=ROOTS['candidate130']/role
            source=folder/('training/validation.jsonl' if component=='DP' else f'backend/{role}/training/validation.json')
            data=[json.loads(line) for line in source.open()] if source.suffix=='.jsonl' else read(source)
            sources.append(dict(path=str(source),sha256=sha(source)))
            for r in data:
                for split in ('train','validation'):
                    value=r[split] if component=='DP' else r[split]['all']
                    rows.append(dict(role=role,component=component,epoch=r['epoch'],update=r.get('updates',r.get('update')),population='train2_diagnostic' if split=='train' else 'validation6',arm_rmse_rad=value['arm_rmse_rad'],hand_rmse_rad=value['hand_rmse_rad']))
    table(PUBLIC/'validation_curves.csv',rows)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(role,component) in zip(axs.flat,[(r,c) for r in 'AB' for c in ('DP','rear')]):
        for pop in ('train2_diagnostic','validation6'):
            r=[x for x in rows if x['role']==role and x['component']==component and x['population']==pop]
            ax.plot([x['epoch'] for x in r],[x['hand_rmse_rad'] for x in r],label=pop)
        selected=next(s for s in read(PUBLIC/'exposure.json')['stages'] if 'tight_recovery_' in s['path'] and s['role']==role and s['component']==component)
        ax.axvline(selected['selected_epoch'],ls='--',color='black');ax.set(title=role+' '+component,xlabel='Local epoch',ylabel='Hand RMSE (rad)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(PUBLIC/'validation_curves.png',dpi=170);plt.close(fig)
    write(PUBLIC/'validation_curve_scope.json',dict(sources=sources,scope='Actual historical fixed train2 and validation6, not full-pool train error; DP own semantic target, rear command. Both arm/hand in CSV; selected checkpoint marked. No new training/inference.'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',nargs='?',default='all',choices=['all','validation']);a=p.parse_args()
    if a.mode=='all':summary()
    validation_curves()
