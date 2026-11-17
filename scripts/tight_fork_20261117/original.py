"""Resolve recorded policy, projection and physics differences at native ticks."""
import json
import h5py,numpy as np,torch
from .common import *
from scripts.tight_layout_20261116.publication import write,table
from visuotactile.position_control import project_processed_position_targets

def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def first(x,threshold,run=1):
 a=np.convolve(np.asarray(x)>threshold,np.ones(run,dtype=int),'valid')
 ids=np.flatnonzero(a==run);return int(ids[0]) if len(ids) else None

def load(p):
 z=dict(np.load(p/'trace.npz'));plans=np.load(p/'plans.npz')['plans']
 with h5py.File(p/'numeric.h5') as f:
  t=f['episode/transition'];o=f['episode/obs']
  z.update(q_before=t['diagnostics_before/joint_position_rad'][:],qd_before=t['diagnostics_before/joint_velocity_rad_s'][:],step=o['step_id'][:],time=o['time_s'][:],camera_frame=o['camera_frame'][:],camera_time=o['camera_time_s'][:],camera_valid=o['camera_valid'][:],episode=o['episode_id'][:],requested=t['position_requested_rad'][:],applied=t['position_applied_rad'][:],delay_enabled=t['control_trace/delay_enabled'][:],clip=t['position_clip_mask'][:],object_before=t['diagnostics_before/object_pose_w'][:],base=t['diagnostics_before/base_pose_w'][:],goal=t['diagnostics_before/goal_pose_w'][:],tactile=o['contact_substeps_w_N'][:],tactile_time=o['contact_time_s'][:],target_before=o['actuator_target_rad'][:])
 for k in ['object','object_before','goal']:
  z[k]=z[k].copy();z[k][:,:3]-=z['base'][:,:3]
 z['plans']=plans;return z

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];summaries=[];checks=[]
 for role,seed,part in CASES:
  variants={v:load(path(role+('_baseline' if v=='baseline' else ''),seed,part)) for v in ['baseline','candidate']}
  for variant,z in variants.items():
   n=len(z['command']);p=path(role+('_baseline' if variant=='baseline' else ''),seed,part)
   meta=read(p/'metadata.json');limits=torch.tensor(meta['joint_limits_rad']);adapter=read(p/'result.json')['complete_student_policy']['execution_adapter']
   raw=z['nominal']+z['residual_proposed_rad'];bounded=np.clip(raw,limits[0,:,0].numpy(),limits[0,:,1].numpy())
   recomputed=project_processed_position_targets(torch.from_numpy(bounded),torch.from_numpy(z['execution_previous_target_rad']),limits,adapter).numpy()
   plan_used=np.stack([z['plans'][t//12,t%12] for t in range(n)])
   plan_used=np.clip(plan_used,limits[0,:,0].numpy(),limits[0,:,1].numpy())
   ck=dict(role=role,seed=seed,variant=variant,steps=n,plan_count=len(z['plans']),expected_plan_count=(n+11)//12,nominal_plan_max=float(abs(plan_used-z['nominal']).max()),projection_recompute_max=float(abs(recomputed-z['command']).max()),bounded_recompute_max=float(abs(bounded-z['execution_preprojection_target_rad']).max()),submitted_vs_applied_max=float(abs(z['requested']-z['applied']).max()),command_vs_requested_max=float(abs(z['command']-z['requested']).max()),applied_vs_target_max=float(abs(z['applied']-z['target']).max()),substep_vs_target_max=float(abs(z['actuator_targets']-z['target'][:,None]).max()),history_target_max=float(abs(z['target_before'][1:]-z['target'][:-1]).max()),projection_previous_max=float(abs(z['execution_previous_target_rad']-z['target_before']).max()),q_continuity_max=float(abs(z['q_before'][1:]-z['joint'][:-1]).max()),qd_continuity_max=float(abs(z['qd_before'][1:]-z['velocity'][:-1]).max()),step_contiguous=bool(np.array_equal(z['step'],np.arange(n))),episode_count=len(np.unique(z['episode'])),min_camera_frame_increment=int(np.diff(z['camera_frame'],axis=0).min()),max_camera_age_s=float((z['time'][:,None]-z['camera_time']).max()),max_time_error_s=float(abs(np.diff(z['time'])-1/60).max()),delay_enabled_ticks=int(z['delay_enabled'].sum()),delay_index_values=np.unique(z['delay_index']).tolist())
   checks.append(ck)
  a,b=variants['baseline'],variants['candidate'];n=121
  signals={k:np.sqrt(np.mean((a[k][:n].astype(float)-b[k][:n])**2,axis=-1)) for k in ['nominal','residual_proposed_rad','execution_preprojection_target_rad','command','q_before','qd_before','joint','velocity','target_before']}
  signals['object_position']=np.linalg.norm(a['object'][:n,:3]-b['object'][:n,:3],axis=-1)
  qa=a['object'][:n,3:].astype(float);qb=b['object'][:n,3:].astype(float)
  qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True)
  signals['object_angle']=2*np.arccos(np.clip(abs(np.sum(qa*qb,axis=-1)),0,1))
  summaries.append(dict(role=role,seed=seed,firsts={k:{str(th):first(x,th) for th in ([1e-5,.001,.01] if k=='object_position' else [1e-5,.001,.02])} for k,x in signals.items()},tick0={k:float(x[0]) for k,x in signals.items()}))
  for t in range(n):
   row=dict(role=role,seed=seed,tick=t,plan_anchor=t-t%12,plan_cursor=t%12,**{k+'_pair_rms':float(x[t]) for k,x in signals.items()})
   for v,z in variants.items():
    for k in ['nominal','residual_proposed_rad','execution_preprojection_target_rad','command','joint']:
     row[v+'_'+k+'_jump']=rms(z[k][t]-z[k][t-1]) if t else 0.
    for label,sl in [('arm',slice(0,7)),('hand',slice(7,29))]:
     row[v+'_projection_'+label]=rms(z['execution_projection_rad'][t,sl]);row[v+'_tracking_'+label]=rms(z['joint'][t,sl]-z['target'][t,sl])
    row[v+'_contact_fingers']=int((np.linalg.norm(z['object_contact'][t],axis=-1)>.1).sum(-1).min())
    row[v+'_object_z']=float(z['object'][t,2]);row[v+'_object_goal_dist']=float(np.linalg.norm(z['object'][t,:3]-z['goal'][t,:3]))
   rows.append(row)
 write(OUT/'original_chain.json',dict(checks=checks,firsts=summaries,scope='0-based observation/action tick; after fields are the result of that action. Threshold crossings descriptive, not causal gates. Tracking lag is not command execution error.'))
 table(OUT/'original_ticks0_120.csv',rows);table(OUT/'original_ticks50_110.csv',[r for r in rows if 50<=r['tick']<=110]);table(OUT/'original_boundaries.csv',[r for r in rows if r['tick'] in [0,48,59,60,61,71,72,73,83,84,85,95,96,97,108]])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,3,figsize=(14,9),sharex=True)
 for i,(role,seed,_) in enumerate(CASES):
  rr=[r for r in rows if r['role']==role and r['seed']==seed];tt=[r['tick'] for r in rr]
  for k in ['nominal','residual_proposed_rad','command']:axs[i,0].plot(tt,[r[k+'_pair_rms'] for r in rr],label=k)
  for k in ['q_before','joint']:axs[i,1].plot(tt,[r[k+'_pair_rms'] for r in rr],label=k)
  axs[i,2].plot(tt,[r['object_position_pair_rms']*1000 for r in rr],label='Object displacement mm')
  for ax in axs[i]:
   for t in [60,72,84,96]:ax.axvline(t,color='gray',alpha=.2)
   ax.set_title(f'{role} {seed}');ax.legend(fontsize=7);ax.grid(alpha=.2)
 for ax in axs[-1]:ax.set_xlabel('Native control tick (0-based)')
 fig.tight_layout();fig.savefig(OUT/'original_first_forks.png',dpi=140)
 print(json.dumps(dict(firsts=summaries,checks=checks),indent=2))
if __name__=='__main__':main()
