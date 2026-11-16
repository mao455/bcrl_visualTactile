"""Same-input rear intervention, with independently rebuilt causal memories."""
import gzip,json,time
from pathlib import Path
import h5py,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools import tight_fresh_expert_counterfactual as expert
from visuotactile.compact import sha256
from visuotactile.precision import state_fp32
from scripts.tight_stability_20261116.analyze_existing import OLD,OUT,TREE,read,first
from scripts.tight_stability_20261116.common_expert import load_model
from scripts.tight_stability_20261116.run import ROOT

def diagnostic_expert(model,path):
    """Observe the unchanged, previously validated mapper's returned targets."""
    original=expert.processed_expert_target;targets=[]
    def observe(*args,**kwargs):
        value=original(*args,**kwargs);targets.append(value[1].detach().clone());return value
    with h5py.File(path) as f:n=len(f['episode/transition/target_after_rad'])
    expert.processed_expert_target=observe
    try:result=expert.query(model,path,TREE,ticks=n)
    finally:expert.processed_expert_target=original
    assert len(targets)==1
    return targets[0],result

def expert_model():
    old=read(TREE/'research_docs/evidence/tight_diagnosis_20261116/round19_command_chain.json')
    session=read(OLD/'session.json')
    assert old['source_sha256']==sha256(Path(expert.__file__))
    assert old['expert_sha256']==sha256(session['expert_checkpoint'])
    assert old['validation']['recorded_raw_mean_max_abs_difference']<=1e-4
    assert old['validation']['recorded_mapped_target_max_abs_difference']<=2e-6
    return expert.load_model(session['expert_checkpoint'],141,TREE)

def sensor_slice(inputs,start,end,device):
    def take(v):return v[start*2:end*2].to(device)[None]
    return {k:({name:take(value) for name,value in v.items()} if isinstance(v,dict)
               else (v[start*2:end*2] if k=='tactile_valid' else v[start:end]).to(device)[None])
            for k,v in inputs.items()}

@torch.no_grad()
def replay(m,record):
    state=None;out={k:[] for k in ['u_pre','u_exec','delta']};device=next(m.parameters()).device
    for start in range(0,len(record['base']),32):
        end=min(start+32,len(record['base']));x=sensor_slice(record['inputs'],start,end,device)
        raw,state=m(x,state)
        chain=m.compose_chain(record['base'][None,start:end].to(device),raw,torch.ones(1,end-start,dtype=torch.bool,device=device),record['limits'].to(device),previous=x['robot'][...,58:87])
        for k in out:out[k].append(chain[k][0].cpu())
    return {k:torch.cat(v) for k,v in out.items()}

def rmse(e):
    if not len(e):return dict(ticks=0,arm=None,hand=None)
    return dict(ticks=len(e),arm=float(e[:,:7].square().mean().sqrt()),hand=float(e[:,7:].square().mean().sqrt()))

def main():
    torch.set_num_threads(2);state_fp32();began=time.time();output=OUT/'common_student.json'
    assert not output.exists();assert read(ROOT/'candidates/input_capture_complete.json')['complete']
    teacher=expert_model();rows=[];series=[]
    plan=read(ROOT/'candidates/plan.json')
    for role in 'AB':
        models={};checkpoints={}
        for label,file in [('short','best_through_epoch8.pt'),('long','best.pt')]:
            p=OLD/role/'backend'/role/'training'/file;models[label],saved=load_model(p)
            checkpoints[label]=dict(epoch=saved['epoch'],sha256=sha256(p),base_sha256=saved['configuration']['base_sha256'])
        for case in plan['cases']:
            if case['role']!=role:continue
            d=ROOT/'candidates'/case['name'];result=read(d/'result.json')
            if not result['legal']:continue
            record=torch.load(d/'rear_inputs.pt',map_location='cpu',weights_only=False,mmap=True)
            assert record['complete'] and record['checkpoint']['frontend_sha256']==checkpoints['short']['base_sha256']==checkpoints['long']['base_sha256']
            active='long' if role=='A' else 'short'
            assert record['checkpoint']['backend_sha256']==checkpoints[active]['sha256']
            predicted={k:replay(m,record) for k,m in models.items()}
            error=float((predicted[active]['u_exec']-record['executed']).abs().max())
            assert error<2e-5,('Replay differs from actual execution',case['name'],error)
            mapped,query=diagnostic_expert(teacher,d/'numeric.h5')
            with np.load(d/'trace.npz') as f:
                force=f['object_contact'];obj=f['object'];actual=torch.from_numpy(f['target']);lift=f['lifted'];sub=f['subgoals']
            assert torch.equal(actual,record['executed'])
            fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1);n=len(fingers);idx=np.arange(n)
            onset=first((fingers>0)&(idx>=48));onset=n if onset is None else onset-1
            masks=dict(first48=idx<48,pre_contact=idx<onset,contact12=(idx>=onset)&(idx<onset+12),
                before150=idx<150,grasp_contact=fingers>=2,later=idx>=150)
            from visuotactile.position_control import project_processed_position_targets
            nominal=project_processed_position_targets(record['base'],record['inputs']['robot'][:,58:87],record['limits'],models[active].execution_adapter)
            one=dict(case=case['name'],role=role,controls=n,active=active,checkpoints=checkpoints,replay_max_abs_error_rad=error,metrics={})
            outputs={k:v['u_exec'] for k,v in predicted.items()};outputs['projected_nominal_diagnostic']=nominal
            for name,mask in masks.items():
                mask=torch.as_tensor(mask);one['metrics'][name]={k:rmse((v-mapped)[mask]) for k,v in outputs.items()}
            one['projection_fraction']={k:float(((v['u_exec']-v['u_pre']).abs()>1e-6).float().mean()) for k,v in predicted.items()}
            rows.append(one)
            for i in range(n):
                series.append(dict(case=case['name'],role=role,tick=i+1,simulation_s=(i+1)/60,fingers=int(fingers[i]),
                    height_m=float(obj[i,2]-np.median(obj[24:48,2])),
                    short_hand_error=float((outputs['short'][i,7:]-mapped[i,7:]).square().mean().sqrt()),
                    long_hand_error=float((outputs['long'][i,7:]-mapped[i,7:]).square().mean().sqrt()),
                    projected_nominal_hand_gap=float((nominal[i,7:]-mapped[i,7:]).square().mean().sqrt())))
        del models;torch.cuda.empty_cache()
    summary=[]
    for role in 'AB':
        chosen=[r for r in rows if r['role']==role]
        for phase in rows[0]['metrics']:
            result=dict(role=role,phase=phase,cases=len(chosen),models={})
            for model in ['short','long','projected_nominal_diagnostic']:
                values=[r['metrics'][phase][model] for r in chosen];n=sum(v['ticks'] for v in values)
                result['models'][model]=dict(ticks=n,**{part:float(np.sqrt(sum(v['ticks']*v[part]**2 for v in values if v['ticks'])/n)) if n else None for part in ['arm','hand']})
            summary.append(result)
    output.write_text(json.dumps(dict(complete=True,seconds=time.time()-began,rows=rows,summary=summary,updates=0,simulation_starts=0,
        scope='Exactly same real student sensor/plan/issued-action history for both rear models. Each rebuilds its own current-weight memory from reset. Only active model controls the robot. Expert targets are diagnostic suggestions, NOT executed labels or proof of recovery. A projected nominal is a mapping diagnostic, not a legitimate standalone A policy.'),indent=2)+'\n')
    with gzip.open(OUT/'common_student_ticks.jsonl.gz','wt') as f:
        for r in series:f.write(json.dumps(r)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for a,role in zip(ax,'AB'):
        v=[r for r in summary if r['role']==role];x=np.arange(len(v));
        for offset,k in [(-.18,'short'),(.18,'long')]:a.bar(x+offset,[r['models'][k]['hand'] for r in v],.35,label=k)
        a.set_xticks(x,[r['phase'] for r in v],rotation=30,ha='right');a.set_title(role+' exact same student input');a.set_ylabel('Hand RMSE vs diagnostic expert (rad)');a.legend()
    fig.suptitle('Direct rear parameter effect, with common actual inputs and separately recomputed memories');fig.tight_layout();fig.savefig(OUT/'common_student.png',dpi=150)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
