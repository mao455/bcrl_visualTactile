"""Opened final tests: descriptive first sustained A/B divergence, no training."""
from pathlib import Path
import json
import h5py,numpy as np
from scripts.tight_layout_20261116.publication import table,write
from scripts.tight_balance_20261117.diagnose import BASE,OUT,phases

def first_run(mask,length=12):
 for i in range(len(mask)-length+1):
  if mask[i:i+length].all():return i
 return None

def main():
 data={}
 for part in range(5):
  folder=BASE/'final_test'/f'part{part}';plan=json.loads((folder/'plan.json').read_text())
  paired=json.loads((folder/'paired_initial_check.json').read_text());assert paired['passed']
  for c in plan['cases']:
   p=folder/c['name'];r=json.loads((p/'result.json').read_text())
   with h5py.File(p/'numeric.h5') as f:
    t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:];palm=t['diagnostics_after/palm_body_pose_w'][:];goal=t['diagnostics_before/goal_pose_w'][:]
    base=t['diagnostics_before/base_pose_w'][:,:3]
    for value in (pose,palm,goal):value[:,:3]-=base
    force=t['object_contact_force_w'][:];contact=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
    data[c['spec']['seed'],c['role']]=dict(success=r['success'],pose=pose,palm=palm,goal=goal,phase=phases(t),contact=contact,target=t['target_after_rad'][:],q=t['diagnostics_after/joint_position_rad'][:],keypoint=t['native_keypoints_max_dist'][:].reshape(-1),subgoal=t['diagnostics_before/successes'][:].reshape(-1))
 rows=[]
 for seed in sorted({s for s,r in data}):
  a,b=data[seed,'A'],data[seed,'B'];n=min(len(a['pose']),len(b['pose']))
  pos=np.linalg.norm(a['pose'][:n,:3]-b['pose'][:n,:3],axis=-1)
  angle=2*np.arccos(np.clip(abs((a['pose'][:n,3:]*b['pose'][:n,3:]).sum(-1)),0,1))
  tick=first_run((pos>.01)|(angle>.15));cat='A_only' if a['success'] and not b['success'] else 'B_only' if b['success'] and not a['success'] else 'both_fail'
  row=dict(seed=seed,category=cat,first_sustained_divergence_tick=tick)
  for role,x in [('A',a),('B',b)]:
   held=np.isin(x['phase'],[2,3,4]);near=np.linalg.norm(x['pose'][:,:3]-x['goal'][:,:3],axis=-1)<.08
   lost=first_run((x['contact']==0)&near)
   row[role]=dict(success=x['success'],phase_at_divergence=int(x['phase'][tick]) if tick is not None else None,first_held12=first_run(held),first_near_no_finger_contact12=lost,min_keypoint=float(x['keypoint'].min()),angle_at_tick150=float(2*np.arccos(np.clip(abs(np.dot(x['pose'][min(150,len(x['pose'])-1),3:],x['goal'][min(150,len(x['pose'])-1),3:])),0,1))),max_subgoal=int(x['subgoal'].max()))
   if tick is not None:
    row[role].update(root_goal_at_divergence_m=float(np.linalg.norm(x['pose'][tick,:3]-x['goal'][tick,:3])),contact_at_divergence=int(x['contact'][tick]),hand_target_norm_at_divergence=float(np.linalg.norm(x['target'][tick,7:])))
  rows.append(row)
 table(OUT/'final_paired_divergence.csv',rows)
 write(OUT/'final_paired_divergence.json',dict(definition='Base-origin translation removed from all lane positions. First12 consecutive aligned control ticks with A/B object displacement>1cm OR relative orientation>0.15rad. Descriptive posthoc threshold, not acceptance, causal attribution or significance. Goal changes and different speeds can cause divergence. No finger contact does not prove no table/hole support.',rows=rows))
 print(json.dumps([r for r in rows if r['category']!='both_fail'],indent=2))
if __name__=='__main__':main()
