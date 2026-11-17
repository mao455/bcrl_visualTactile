"""Paired trajectory outcome and first sustained geometric divergence, diagnostic only."""
import h5py,numpy as np
from scripts.tight_cycle_20261118.cycle import ROOT,OUT,read,write
from scripts.tight_layout_20261116.publication import table
from scripts.tight_balance_20261117.diagnose import phases
from visuotactile.geometry_sampling import STAGES

def first_run(mask,window=12):return next((i-window+1 for i in range(window-1,len(mask)) if mask[i-window+1:i+1].all()),None)
def main():
 rows=[];curves=[]
 for part in range(2):
  d=ROOT/f'comparison{part:02d}';plan=read(d/'plan.json');signals={}
  for c in plan['cases']:
   p=d/c['name'];z=dict(np.load(p/'trace.npz'))
   with h5py.File(p/'numeric.h5') as f:
    t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:];pose[:,:3]-=t['diagnostics_before/base_pose_w'][:,:3];z.update(pose=pose,phase=phases(t),keypoint=t['native_keypoints_max_dist'][:].reshape(-1))
   signals[c['spec']['seed'],c['role']]=(z,read(p/'result.json'),c)
  for seed in sorted({s for s,r in signals}):
   for role in 'AB':
    a,ar,case=signals[seed,role+'_baseline'];b,br,_=signals[seed,role+'_candidate'];n=min(len(a['pose']),len(b['pose']));pos=np.linalg.norm(a['pose'][:n,:3]-b['pose'][:n,:3],axis=-1);qa=a['pose'][:n,3:].astype(float);qb=b['pose'][:n,3:].astype(float);qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True);angle=2*np.arccos(np.clip(abs((qa*qb).sum(-1)),0,1));tick=first_run((pos>.01)|(angle>.15))
    rows.append(dict(seed=seed,role=role,cohort=case['diagnostic_cohort'],baseline_success=ar['success'],candidate_success=br['success'],baseline_subgoals=ar['native_subgoals'],candidate_subgoals=br['native_subgoals'],baseline_steps=ar['steps'],candidate_steps=br['steps'],baseline_termination=ar['reasons'],candidate_termination=br['reasons'],first_sustained_divergence_tick=tick,baseline_phase=STAGES[int(a['phase'][tick])] if tick is not None else None,candidate_phase=STAGES[int(b['phase'][tick])] if tick is not None else None,**{k+'_tick0_rms_rad':float(np.sqrt(np.mean((a[k][0].astype(float)-b[k][0])**2))) for k in ['nominal','residual_proposed_rad','command']}))
    for label,z in [('baseline',a),('candidate',b)]:
     for t in range(0,len(z['pose']),6):curves.append(dict(seed=seed,role=role,model=label,tick=t,object_z_m=float(z['pose'][t,2]),keypoint_m=float(z['keypoint'][t]),phase=STAGES[int(z['phase'][t])]))
 write(OUT/'evaluation/paired_changes.json',dict(rows=rows,definition='First12 consecutive native ticks with base-relative object position difference>1cm or quaternion geodesic>0.15rad. Descriptive model divergence, NOT first causal error or a timing bug. Existing deployment per-lane diffusion seed7093; real subsequent observations differ.',pairs=32));table(OUT/'evaluation/paired_changes.csv',rows);table(OUT/'evaluation/trajectory_curves.csv',curves)
 print({r:dict(gained=sum(x['candidate_success'] and not x['baseline_success'] for x in rows if x['role']==r),lost=sum(x['baseline_success'] and not x['candidate_success'] for x in rows if x['role']==r),retained=sum(x['baseline_success'] and x['candidate_success'] for x in rows if x['role']==r)) for r in 'AB'})
if __name__=='__main__':main()
