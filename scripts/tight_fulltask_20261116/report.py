"""Small real metrics/plots only; censored or illegal cases never become failures."""
import json,time
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tight_fulltask_20261116.run import ROOT,OLD,TREE,read
from visuotactile.grasp_curriculum import holding_metrics
OUT=TREE/'research_docs/evidence/tight_diagnosis_20261116/fulltask'

def small(x):
    if isinstance(x,dict):return {k:small(v) for k,v in x.items() if k not in ['episodes','sampling_exposure','rows','training_sampling','visit_sampling_coverage','source_sha256','process_io','batch_prefetch','prefetch']}
    if isinstance(x,list):return [small(v) for v in x]
    return x

def behavior(directory):
    plan=read(directory/'plan.json');rows=[]
    for c in plan['cases']:
        p=directory/c['name'];r=read(p/'result.json');row=dict(case=c['name'],role=c['role'],seed=c['spec']['seed'],legal=r['legal'])
        if r['legal']:
            with np.load(p/'trace.npz') as f:a={k:f[k] for k in f.files}
            fingers=(np.linalg.norm(a['object_contact'],axis=-1)>.1).sum(-1).min(-1);idx=np.arange(len(fingers))
            h={w:holding_metrics(a['object'],a['object_contact'],window=w) for w in [12,60,120,180]}
            row.update(controls=r['steps'],success=r['success'],native_lift=bool(a['lifted'].any()),alignment=bool(a['alignment'].any()),subgoal=int(a['subgoals'].max()),
                contact=bool(((fingers>0)&(idx>=48)).any()),two_finger=bool(((fingers>=2)&(idx>=48)).any()),
                hold={str(w):h[w] for w in h},reasons=r.get('reasons'),full_task=not r.get('grasp_stage_censored',False) and not r.get('censored',False),
                video=str(p/'autonomous.mp4') if (p/'autonomous.mp4').exists() else None)
        rows.append(row)
    roles=list(plan['pairs']);common=set.intersection(*[{r['seed'] for r in rows if r['role']==role and r['legal']} for role in roles])
    counts=[]
    for role in roles:
        for scope in ['all_legal','common_legal']:
            selected=[r for r in rows if r['role']==role and r['legal'] and (scope=='all_legal' or r['seed'] in common)]
            full=[r for r in selected if r['full_task']]
            counts.append(dict(role=role,scope=scope,n=len(selected),full_n=len(full),success=sum(r['success'] for r in full),
                **{k:sum(bool(r[k]) for r in selected) for k in ['contact','two_finger','native_lift','alignment','subgoal']},
                **{'hold'+str(w):sum(r['hold'][str(w)]['sustained'] for r in selected) for w in [12,60,120,180]}))
    return dict(rows=rows,counts=counts,common_seeds=sorted(common),protocol=plan.get('paired_native_reset',plan.get('evaluation_scope')),
        paired_initial=read(directory/'paired_initial_check.json') if (directory/'paired_initial_check.json').exists() else None)

def main():
    OUT.mkdir(parents=True,exist_ok=True);result=dict(utc=time.time(),deadline=read(ROOT/'session.json')['deadline'],collection={},training={},evaluations={},timing={})
    entries=read(ROOT/'collection/correction_entries.json')
    result['collection']=dict(attempts=4,successes=len(entries),new_ids=list(entries),families=sorted({v['family'] for v in entries.values()}),
        result=read(ROOT/'collection/result.json'),videos=read(ROOT/'collection/videos.json'))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,role in enumerate('AB'):
        root=ROOT/'update'/role/'training';front={}
        if (root/'validation.jsonl').exists():
            history=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
            front['epochs']=[small(x) for x in history];front['initial']=small(read(root/'initial_metrics.json'))
            updates=[];loss=[]
            with (root/'learning.jsonl').open() as f:
                for line in f:
                    x=json.loads(line);updates.append(x['updates']);loss.append(x['loss'])
            axes[0,i].plot(updates,loss,alpha=.5,lw=.6,label='actual train batch loss');axes[0,i].set_title(role+' DP');axes[0,i].set_xlabel('Optimizer updates');axes[0,i].legend()
            if (root/'finished.json').exists():front['finished']=small(read(root/'finished.json'))
            # Exact selected exposure from best, not last or planned repeats.
            import torch
            if (root/'best.pt').exists():
                ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False,mmap=True)
                front['selected']=dict(epoch=ck['epoch'],updates=ck['updates'],latest_exposure={k:ck.get('sampling_exposure',{}).get(k) for k in entries})
        rearpath=ROOT/'update'/role/'backend'/role/'training';rear={}
        if (rearpath/'validation.json').exists():
            rear['epochs']=small(read(rearpath/'validation.json'))
            x=[];y=[]
            if (rearpath/'loss.jsonl').exists():
                for line in (rearpath/'loss.jsonl').read_text().splitlines():
                    v=json.loads(line);x.append(v['update']);y.append(v['loss'])
            axes[1,i].plot(x,y,lw=.6,alpha=.5,label='actual train update loss');axes[1,i].set_title(role+' Transformer');axes[1,i].set_xlabel('Optimizer updates');axes[1,i].legend()
            for name in ['finished','selected','latest_fit_initial']:
                if (rearpath/(name+'.json')).exists():rear[name]=small(read(rearpath/(name+'.json')))
            if (rearpath/'best.pt').exists():
                ck=torch.load(rearpath/'best.pt',map_location='cpu',weights_only=False,mmap=True)
                rear['selected_exposure']={k:ck.get('episode_counts',{}).get(k) for k in entries}
        result['training'][role]=dict(dp=front,rear=rear)
    fig.tight_layout();fig.savefig(OUT/'training_losses.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for i,role in enumerate('AB'):
        front=result['training'][role]['dp'];rear=result['training'][role]['rear']
        if front:
            records=[dict(epoch=0,**front['initial'])]+front['epochs']
            for key,label in [('train','Original2 training diagnosis'),('validation','Independent validation6'),('correction','New successful corrections')]:
                chosen=[r for r in records if key in r]
                xs=[r['epoch'] for r in chosen];ys=[r[key]['selection'] for r in chosen]
                if key=='correction':ax[0,i].scatter(xs,ys,label=label,marker='x')
                else:ax[0,i].plot(xs,ys,'o-',label=label)
            ax[0,i].set_title(role+' DP target-fit score (not final command)');ax[0,i].set_xlabel('Incremental epoch');ax[0,i].legend(fontsize=8)
        if rear.get('epochs'):
            for key,label in [('train','Training'),('validation','Independent validation')]:
                chosen=[r for r in rear['epochs'] if key in r]
                ax[1,i].plot([r['epoch'] for r in chosen],[r[key]['all']['normalized_equal_mse'] for r in chosen],'o-',label=label)
            ax[1,i].set_title(role+' rear executed-command normalized MSE');ax[1,i].set_xlabel('Incremental epoch');ax[1,i].legend()
    fig.suptitle('Only actually measured checkpoints; sparse new-correction measurements shown as points')
    fig.tight_layout();fig.savefig(OUT/'validation_errors.png',dpi=140);plt.close(fig)
    for group in ['baseline_evaluation','updated_evaluation']:
        if (ROOT/group/'videos.json').exists() and read(ROOT/group/'videos.json')['complete']:
            result['evaluations'][group]=behavior(ROOT/group)
    for p in ROOT.glob('*_worker.json'):
        r=read(p);result['timing'][r['stage']]={k:r.get(k) for k in ['pid','started','ended','seconds','exit_code']}
    result['disk_free_bytes']=__import__('shutil').disk_usage(ROOT).free
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    # A single fixed baseline video gives the entire grasp-then-slip story.
    p=ROOT/'baseline_evaluation/B_fulltask_development_28500101'
    with np.load(p/'trace.npz') as f:a={k:f[k] for k in f.files}
    fingers=(np.linalg.norm(a['object_contact'],axis=-1)>.1).sum(-1).min(-1);t=(np.arange(len(fingers))+1)/60
    fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
    ax[0].plot(t,a['object'][:,2]-np.median(a['object'][24:48,2]));ax[0].axhline(.02,c='gray',ls=':');ax[0].set_ylabel('Object height delta (m)')
    ax[1].step(t,fingers,where='post');ax[1].set_ylabel('Contact fingers / both physics samples')
    ax[2].plot(t[1:],np.sqrt(np.mean(np.diff(a['target'][:,7:],axis=0)**2,axis=1)));ax[2].set_ylabel('Hand target change RMS (rad)');ax[2].set_xlabel('Simulation seconds (60 control ticks/s)')
    fig.suptitle('Recorded B source seed101: real grasp then loss, no expert control');fig.tight_layout();fig.savefig(OUT/'grasp_then_loss.png',dpi=140);plt.close(fig)
    print(json.dumps({g:x['counts'] for g,x in result['evaluations'].items()},indent=2))
if __name__=='__main__':main()
