"""Descriptive support distance; no learned estimator, admission or target labels."""
import json
import h5py,numpy as np
from scipy.spatial import cKDTree
from scripts.tight_balance_20261117.offline_phases import ROOT,OUT,BASE
from scripts.tight_balance_20261117.diagnose import phases
from scripts.tight_layout_20261116.publication import write,table

def describe(t):
 pose=t['diagnostics_after/object_pose_w'][:];palm=t['diagnostics_after/palm_body_pose_w'][:];goal=t['diagnostics_before/goal_pose_w'][:]
 angle=2*np.arccos(np.clip(abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
 return np.concatenate([pose[:,:3]-palm[:,:3],pose[:,:3]-goal[:,:3],angle[:,None],t['diagnostics_after/joint_position_rad'][:]],axis=-1),phases(t)

def main():
 data=json.loads((BASE/'round06/learning/A/state_data/data.json').read_text());items=[]
 for identity in data['splits']['train']:
  e=data['entries'][identity]
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];x,p=describe(t)
   mask=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if e.get('trajectory_kind')=='expert_takeover_v1' else np.ones(len(x),bool)
  items.append(dict(identity=identity,family=e['family'],original=e.get('trajectory_kind')!='expert_takeover_v1',x=x[mask],phase=p[mask]))
 std=np.concatenate([i['x'] for i in items if i['original']]).std(0).clip(.001)
 x=np.concatenate([i['x']/std for i in items]);phase=np.concatenate([i['phase'] for i in items]);family=np.concatenate([np.repeat(i['family'],len(i['x'])) for i in items]);dim=x.shape[-1]
 reference=np.full(len(x),np.inf)
 for f in set(family):
  for p in set(phase):
   train=(family!=f)&(phase==p);query=(family==f)&(phase==p)
   if train.any() and query.any():reference[query]=cKDTree(x[train]).query(x[query],k=1)[0]/np.sqrt(dim)
 trees={p:cKDTree(x[phase==p]) for p in set(phase)};threshold={int(p):float(np.percentile(reference[phase==p],95)) for p in set(phase)}
 rows=[]
 for part in range(4):
  folder=ROOT/f'comparison{part:02d}'/'development';plan=json.loads((folder/'plan.json').read_text())
  for c in plan['cases']:
   with h5py.File(folder/c['name']/'numeric.h5') as f:q,p=describe(f['episode/transition'])
   distance=np.full(len(q),np.inf)
   for stage,tree in trees.items():
    mask=p==stage
    if mask.any():distance[mask]=tree.query(q[mask]/std,k=1)[0]/np.sqrt(dim)
   for stage in set(p):
    mask=p==stage;far=distance[mask]>threshold[int(stage)]
    rows.append(dict(role=c['role'],seed=c['spec']['seed'],phase=int(stage),ticks=int(mask.sum()),median=float(np.median(distance[mask])),p95=float(np.percentile(distance[mask],95)),above_train_leave_family_out_p95=int(far.sum()),reference_p95=threshold[int(stage)]))
 table(OUT/'development_support_distance.csv',rows)
 write(OUT/'development_support_distance.json',dict(rows=rows,training_phase_p95=threshold,scope='36D object/palm/goal relative translations, goal angle,29 joints; original30 axis standard deviation floor0.001. Train reference nearest legal expert state of another family, phase-stratified. Development uses all76 legal expert states. Descriptive geometric distance only: omits velocity, sensory history and hidden simulator state; crossing p95 does not prove an unlearnable or expert-recoverable gap and is not a collection trigger. Post-action states, never fed as policy inputs.'))
 print(json.dumps({r:sum(v['above_train_leave_family_out_p95'] for v in rows if v['role']==r)/sum(v['ticks'] for v in rows if v['role']==r) for r in ['A','A_baseline','B','B_baseline']}))
if __name__=='__main__':main()
