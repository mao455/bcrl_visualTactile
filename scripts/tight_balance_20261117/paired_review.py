"""Review all closed development pairs; posthoc descriptors are not success gates."""
import json
import h5py,numpy as np
from scripts.tight_balance_20261117.offline_phases import ROOT,OUT
from scripts.tight_balance_20261117.diagnose import phases
from scripts.tight_balance_20261117.pair_diagnosis import first_run
from scripts.tight_layout_20261116.publication import write,table

def main():
 data={};curves=[];rows=[]
 for part in range(4):
  folder=ROOT/f'comparison{part:02d}'/'development';plan=json.loads((folder/'plan.json').read_text())
  assert json.loads((folder/'paired_initial_check.json').read_text())['passed']
  for c in plan['cases']:
   p=folder/c['name'];r=json.loads((p/'result.json').read_text())
   with h5py.File(p/'numeric.h5') as f:
    t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:];goal=t['diagnostics_before/goal_pose_w'][:];base=t['diagnostics_before/base_pose_w'][:,:3]
    pose[:,:3]-=base;goal[:,:3]-=base
    force=t['object_contact_force_w'][:];contact=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
    distance=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1);angle=2*np.arccos(np.clip(abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
    x=dict(success=r['success'],pose=pose,goal=goal,phase=phases(t),contact=contact,target=t['target_after_rad'][:],q=t['diagnostics_after/joint_position_rad'][:],keypoint=t['native_keypoints_max_dist'][:].reshape(-1),subgoal=t['diagnostics_before/successes'][:].reshape(-1),angle=angle,distance=distance,result=r)
    data[c['spec']['seed'],c['role']]=x
    for tick in range(0,len(pose),6):curves.append(dict(seed=c['spec']['seed'],role=c['role'],tick=tick,object_z=float(pose[tick,2]),distance_m=float(distance[tick]),angle_rad=float(angle[tick]),contact_fingers=int(contact[tick]),phase=int(x['phase'][tick]),native_keypoint_m=float(x['keypoint'][tick]),native_subgoal=int(x['subgoal'][tick])))
 for role in 'AB':
  for seed in sorted({s for s,r in data}):
   a,b=data[seed,role+'_baseline'],data[seed,role];n=min(len(a['pose']),len(b['pose']))
   pos=np.linalg.norm(a['pose'][:n,:3]-b['pose'][:n,:3],axis=-1);angle=2*np.arccos(np.clip(abs((a['pose'][:n,3:]*b['pose'][:n,3:]).sum(-1)),0,1));tick=first_run((pos>.01)|(angle>.15))
   category=('both_success' if a['success'] and b['success'] else 'candidate_only' if b['success'] else 'baseline_only' if a['success'] else 'both_fail')
   row=dict(role=role,seed=seed,category=category,first_sustained_object_divergence=tick)
   for name,x in [('baseline72',a),('candidate76',b)]:
    held=np.isin(x['phase'],[2,3,4]);near=x['distance']<.08
    row[name]=dict(success=x['success'],steps=x['result']['steps'],first_two_finger12=first_run(x['contact']>=2),first_held12=first_run(held),first_near_without_finger12=first_run(near&(x['contact']==0)),min_keypoint_m=float(x['keypoint'].min()),maximum_native_subgoal=int(x['subgoal'].max()),phase_at_divergence=int(x['phase'][tick]) if tick is not None else None,angle_at150=float(x['angle'][min(150,len(x['angle'])-1)]),final_angle_rad=float(x['angle'][-1]),contact_at_divergence=int(x['contact'][tick]) if tick is not None else None)
   rows.append(row)
 table(OUT/'paired_development_deltas.csv',rows);table(OUT/'paired_development_curves.csv',curves)
 write(OUT/'paired_development_review.json',dict(rows=rows,definitions='Base-relative positions, first12 consecutive ticks displaced>1cm or rotated>0.15rad. Posthoc description only, not causal breakpoint, significance or a deployment gate. Native goal changes are retained. No finger contact does not establish absence of table/hole support.',categories={role:{c:sum(r['role']==role and r['category']==c for r in rows) for c in ['both_success','candidate_only','baseline_only','both_fail']} for role in 'AB'}))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for role in 'AB':
  fig,axes=plt.subplots(8,3,figsize=(12,16),sharex=True)
  for k,seed in enumerate(sorted({s for s,r in data})):
   for variant,label in [(role+'_baseline','Selected72'),(role,'Fixed76 onset')]:
    x=data[seed,variant];time=np.arange(len(x['pose']))/60
    for col,values in enumerate([x['pose'][:,2],x['angle'],x['keypoint']]):axes[k,col].plot(time,values,label=label)
   axes[k,0].set_ylabel(str(seed));
   for ax in axes[k]:ax.grid(alpha=.2)
  for j,title in enumerate(['Object z / m','Orientation to active goal / rad','Native keypoint distance / m']):axes[0,j].set_title(title);axes[-1,j].set_xlabel('Control time / s')
  axes[0,0].legend();fig.suptitle(role+' complete policies, paired development; goal switches retained');fig.tight_layout();fig.savefig(OUT/f'paired_{role}_traces.png',dpi=120);plt.close(fig)
if __name__=='__main__':main()
