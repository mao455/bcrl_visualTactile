"""Unit-separated, phase-stratified expert support with leave-family-out reference."""
import csv,json,time
from pathlib import Path
import h5py,numpy as np
from scipy.spatial import cKDTree
from scripts.tight_cycle_20261118.cycle import ROOT,DATA,OUT,read,write
from scripts.tight_balance_20261117.diagnose import phases
from visuotactile.geometry_sampling import STAGES
from visuotactile.grasp_curriculum import holding_metrics

def features(t,initial=False):
 b=t['diagnostics_before' if initial else 'diagnostics_after'];g=t['diagnostics_before'];obj=b['object_pose_w'][:];palm=b['palm_body_pose_w'][:];goal=g['goal_pose_w'][:];base=b['base_pose_w'][:]
 force=t['object_contact_force_w'][:];contacts=(np.linalg.norm(force,axis=-1)>.1).min(1).astype(float)
 # Native base orientation is fixed identity; subtract each lane's real base translation.
 assert np.max(abs(base[:,3:]-np.array([1,0,0,0])))<1e-5
 oq=obj[:,3:]/np.linalg.norm(obj[:,3:],axis=-1,keepdims=True);pq=palm[:,3:]/np.linalg.norm(palm[:,3:],axis=-1,keepdims=True);inv=pq.copy();inv[:,1:]*=-1
 delta=obj[:,:3]-palm[:,:3];cross=2*np.cross(inv[:,1:],delta);local=delta+inv[:,:1]*cross+np.cross(inv[:,1:],cross)
 relative=np.concatenate([inv[:,:1]*oq[:,:1]-(inv[:,1:]*oq[:,1:]).sum(-1,keepdims=True),inv[:,:1]*oq[:,1:]+oq[:,:1]*inv[:,1:]+np.cross(inv[:,1:],oq[:,1:])],axis=-1)
 v=b['object_velocity_w_m_s_rad_s'][:];cmd=t['target_after_rad'][:]
 lag=np.vstack([cmd[:1],cmd[:-1]]);hist=np.array([cmd[max(0,i-11):i+1].mean(0) for i in range(len(cmd))])
 return dict(object_base_m=obj[:,:3]-base[:,:3],object_goal_m=obj[:,:3]-goal[:,:3],object_palm_m=obj[:,:3]-palm[:,:3],object_rotation_rad=oq,palm_local_object_m=local,palm_local_rotation_rad=relative,joint_rms_rad=b['joint_position_rad'][:]/np.sqrt(29),joint_speed_rms_rad_s=b['joint_velocity_rad_s'][:]/np.sqrt(29),object_speed_m_s=v[:,:3],object_spin_rad_s=v[:,3:],finger_contact_hamming=contacts,issued_history12_rms_rad=hist/np.sqrt(29),issued_change_rms_rad=(cmd-lag)/np.sqrt(29)),phases(t)
def buildtree(x,key):return cKDTree(np.concatenate([x,-x]) if key in ('object_rotation_rad','palm_local_rotation_rad') else x)
def distance(tree,x,key):
 d=tree.query(x,k=1)[0]
 if key in ('object_rotation_rad','palm_local_rotation_rad'):d=4*np.arcsin(np.clip(d/2,0,1))
 if key=='finger_contact_hamming':d=d*d
 return d

def main(wave='capability'):
 data=read(DATA/'A/state_data/data.json');records=[];initial=[]
 for i in data['splits']['train']:
  e=data['entries'][i]
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];x,p=features(t);mask=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if e.get('trajectory_kind')=='expert_takeover_v1' else np.ones(len(p),bool)
   records.append((e['family'],{k:v[mask] for k,v in x.items()},p[mask]))
   if e.get('trajectory_kind')!='expert_takeover_v1':
    ix,_=features(t,initial=True);ix={k:v[:1] for k,v in ix.items() if k not in ('finger_contact_hamming','issued_history12_rms_rad','issued_change_rms_rad')};ix['actuator_target_rms_rad']=t['target_before_rad'][:1]/np.sqrt(29);initial.append((e['family'],ix))
 keys=list(records[0][1]);family=np.concatenate([np.repeat(f,len(p)) for f,x,p in records]);ph=np.concatenate([p for f,x,p in records]);allx={k:np.concatenate([x[k] for f,x,p in records]) for k in keys};refs={};trees={}
 for phase in sorted(set(ph)):
  m=ph==phase
  for k in keys:
   ref=np.full(int(m.sum()),np.nan);xx=allx[k][m];ff=family[m]
   for fam in set(ff):
    q=ff==fam
    if (~q).any():ref[q]=distance(buildtree(xx[~q],k),xx[q],k)
   finite=ref[np.isfinite(ref)];refs[int(phase),k]=float(np.percentile(finite,95)) if len(finite) else None;trees[int(phase),k]=buildtree(xx,k)
 directory=ROOT/wave;plan=read(directory/'plan.json');rows=[];initrows=[];summary=[]
 # Pre-action state and target; no lane-world translation in metrics.
 for c in plan['cases']:
  d=directory/c['name'];r=read(d/'result.json')
  if not r['legal']:continue
  with h5py.File(d/'numeric.h5') as f:
   t=f['episode/transition'];x,p=features(t)
   ix,_=features(t,initial=True);ix['actuator_target_rms_rad']=t['target_before_rad'][:1]/np.sqrt(29)
   for k in initial[0][1]:
    refx=np.concatenate([a[k] for fam,a in initial if fam!=c['spec']['family']]);dd=distance(buildtree(refx,k),ix[k][:1],k)[0]
    trainref=[distance(buildtree(np.concatenate([b[k] for ff,b in initial if ff!=fam]),k),a[k],k)[0] for fam,a in initial]
    initrows.append(dict(case=c['name'],cohort=c['diagnostic_cohort'],role=c['role'],metric=k,nearest=float(dd),train_leave_family_p95=float(np.percentile(trainref,95))))
   for phase in sorted(set(p)):
    m=p==phase
    for k in keys:
     if (int(phase),k) not in trees:continue
     dd=distance(trees[int(phase),k],x[k][m],k);thr=refs[int(phase),k]
     rows.append(dict(case=c['name'],cohort=c['diagnostic_cohort'],role=c['role'],phase=STAGES[int(phase)],metric=k,ticks=int(m.sum()),p50=float(np.median(dd)),p95=float(np.percentile(dd,95)),reference_p95=thr,far_fraction=float((dd>thr+1e-8).mean()) if thr is not None else None))
   farflags={}
   for k in ['object_palm_m','object_rotation_rad','joint_rms_rad']:
    flags=np.zeros(len(p),bool)
    for phase in set(p):
     key=(int(phase),k);m=p==phase
     if key in trees and refs[key] is not None:flags[m]=distance(trees[key],x[k][m],k)>refs[key]+1e-8
    farflags[k]=flags
   far=(farflags['object_palm_m']&farflags['object_rotation_rad'])|(farflags['object_palm_m']&farflags['joint_rms_rad'])
   first_far=next((i-11 for i in range(11,len(far)) if far[i-11:i+1].all()),None)
   contact=(np.linalg.norm(t['object_contact_force_w'][:],axis=-1)>.1).sum(-1).min(-1);lost=np.flatnonzero((contact[1:]==0)&(contact[:-1]>=1))+1
   hold60=holding_metrics(t['diagnostics_after/object_pose_w'][:],t['object_contact_force_w'][:]);hold120=holding_metrics(t['diagnostics_after/object_pose_w'][:],t['object_contact_force_w'][:],window=120)
   summary.append(dict(first_sustained_two_metric_support_deviation_tick=first_far,support_deviation_phase=STAGES[int(p[first_far])] if first_far is not None else None,hold60=hold60,hold120=hold120,case=c['name'],cohort=c['diagnostic_cohort'],role=c['role'],seed=c['spec']['seed'],family=c['spec']['family'],success=r['success'],steps=r['steps'],reasons=r['reasons'],subgoals=r['native_subgoals'],phase_ticks={STAGES[int(a)]:int((p==a).sum()) for a in set(p)},first_2finger_contact=next(iter(np.flatnonzero(contact>=2).tolist()),None),contact_loss_ticks=lost.tolist(),final_phase=STAGES[int(p[-1])]))
 out=OUT/wave;out.mkdir(parents=True,exist_ok=True)
 for name,rs in [('support',rows),('initial_preaction',initrows),('cases',summary)]:
  write(out/(name+'.json'),rs)
  if rs:
   with (out/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=rs[0]);w.writeheader();w.writerows(rs)
 write(out/'scope.json',dict(reference='All76 legal expert ticks; per-phase leave-family-out p95; independent metrics in their units, no mixed-unit score',initial='Actual diagnostics_before first action plus target_before; A/B actual pairing independently checked; native per-family RNG differs historical single-env collection so not exact training trajectory replay.',limitations='Near support is not recoverability. Contact is fingertip-only, not measured support transfer. Native stages are proxies, not a strict funnel. Historical issued commands remain real history, not supervision.'))
 print(json.dumps({co+ro:dict(n=sum(x['cohort']==co and x['role']==ro for x in summary),success=sum(x['cohort']==co and x['role']==ro and x['success'] for x in summary)) for co in ['training','new_development'] for ro in 'AB'}))
if __name__=='__main__':main(*__import__('sys').argv[1:])
