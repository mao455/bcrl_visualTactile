"""Matched short/long driven histories and exact common-input rear replay on CPU."""
import json,time
import numpy as np,torch
from scripts.tight_stability_20261116.replay import expert_model,diagnostic_expert,replay,rmse
from scripts.tight_stability_20261116.common_expert import load_model
from scripts.tight_stability_20261116.analyze_existing import OLD,OUT,read,first
from scripts.tight_stability_20261116.run import ROOT
from visuotactile.grasp_curriculum import holding_metrics

def main():
    torch.set_num_threads(2);d=ROOT/'b_feedback';plan=read(d/'plan.json');teacher=expert_model();rows=[];start=time.time()
    models={label:load_model(OLD/'B/backend/B/training'/name,device='cpu')[0] for label,name in [('short','best_through_epoch8.pt'),('long','best.pt')]}
    for case in plan['cases']:
        path=d/case['name'];result=read(path/'result.json')
        if not result['legal']:
            rows.append(dict(case=case['name'],legal=False));continue
        rec=torch.load(path/'rear_inputs.pt',map_location='cpu',weights_only=False,mmap=True)
        target,_=diagnostic_expert(teacher,path/'numeric.h5');pred={k:replay(m,rec) for k,m in models.items()}
        driven='short' if case['role']=='B_short' else 'long'
        gap=float((pred[driven]['u_exec']-rec['executed']).abs().max());assert gap<3e-5
        with np.load(path/'trace.npz') as f:
            fingers=(np.linalg.norm(f['object_contact'],axis=-1)>.1).sum(-1).min(-1)
            hold=holding_metrics(f['object'],f['object_contact'],window=60);lift=bool(f['lifted'].any())
        idx=np.arange(len(fingers));contact=first((fingers>0)&(idx>=48));end=(contact-1 if contact else len(fingers))
        masks=dict(first48=idx<48,precontact=idx<end,contact12=(idx>=end)&(idx<end+12),before150=idx<150)
        rows.append(dict(case=case['name'],seed=case['spec']['seed'],driven=driven,legal=True,replay_gap=gap,
            contact=contact,hold60=hold['sustained'],native_lift=lift,metrics={phase:{k:rmse((v['u_exec']-target)[mask]) for k,v in pred.items()} for phase,mask in masks.items()}))
    summary=[]
    for driven in ['short','long']:
        selected=[r for r in rows if r.get('driven')==driven]
        phases={}
        for phase in masks:
            phases[phase]={}
            for tested in ['short','long']:
                vals=[r['metrics'][phase][tested] for r in selected];n=sum(v['ticks'] for v in vals)
                phases[phase][tested]={part:float(np.sqrt(sum(v['ticks']*v[part]**2 for v in vals if v['ticks'])/n)) for part in ['arm','hand']}
        summary.append(dict(driven=driven,n=len(selected),hold60=sum(r['hold60'] for r in selected),native_lift=sum(r['native_lift'] for r in selected),metrics=phases))
    out=dict(rows=rows,summary=summary,seconds=time.time()-start,scope='Same 8 registered seeds per rear; 300-control diagnostic censor, not full success denominator. CPU common-input replay rebuilds each rear memory; expert labels are unexecuted diagnostic suggestions. Explicit B_short / B_long roles both use B supervision.')
    (OUT/'b_feedback.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
