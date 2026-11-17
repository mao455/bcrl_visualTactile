"""Descriptive train-only state overlap; thresholds do not drive admission."""
import json,collections
from pathlib import Path
import h5py,numpy as np
from scipy.spatial import cKDTree
from scripts.tight_balance_20261117.diagnose import BASE,OUT,phases
from scripts.tight_layout_20261116.publication import write,table

def main():
 d=json.loads((BASE/'round06/learning/A/state_data/data.json').read_text());items={};all_original=[]
 for i in d['splits']['train']:
  e=d['entries'][i]
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];pose=t['diagnostics_after/object_pose_w'][:];palm=t['diagnostics_after/palm_body_pose_w'][:];goal=t['diagnostics_before/goal_pose_w'][:];q=t['diagnostics_after/joint_position_rad'][:];ph=phases(t)
   # Translationally invariant descriptor; goal changes explicitly stratified by phase.
   a=2*np.arccos(np.clip(abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
   desc=np.concatenate([pose[:,:3]-palm[:,:3],pose[:,:3]-goal[:,:3],a[:,None],q],axis=-1)
   correction=e.get('trajectory_kind')=='expert_takeover_v1';valid=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if correction else np.ones(len(q),bool);ticks=np.flatnonzero(valid)
   items[i]=dict(desc=desc[valid],phase=ph[valid],ticks=ticks,original=not correction,family=e['family'])
   if not correction:all_original.append(desc[valid])
 original=np.concatenate(all_original);std=original.std(0).clip(.001);rows=[]
 for i,x in items.items():
  if x['original']:continue
  distances=np.full(len(x['ticks']),np.inf)
  for j,y in items.items():
   if x['family']==y['family']:continue
   for stage in set(x['phase']):
    xx=x['phase']==stage;yy=y['phase']==stage
    if not yy.any():continue
    near=cKDTree(y['desc'][yy]/std).query(x['desc'][xx]/std,k=1)[0]/np.sqrt(len(std))
    distances[xx]=np.minimum(distances[xx],near)
  for name,mask in [('onset24',np.arange(len(distances))<24),('remaining',np.arange(len(distances))>=24)]:
   v=distances[mask];rows.append(dict(identity=i,part=name,ticks=len(v),nearest_other_family_same_phase_median=float(np.median(v)),p90=float(np.percentile(v,90)),mean=float(v.mean())))
 table(OUT/'suffix_similarity.csv',rows)
 write(OUT/'suffix_similarity.json',dict(descriptor='36D object-palm translation, object-active-goal translation, orientation-to-goal angle,29 joint angles; scaled by original30 per-coordinate std floor0.001; RMS normalized distance; nearest same physical-phase state of another initial-state family. Descriptive only, not a physical equivalence or coverage threshold.',parts={part:dict(episodes=len([r for r in rows if r['part']==part]),median_episode_distance=float(np.median([r['nearest_other_family_same_phase_median'] for r in rows if r['part']==part]))) for part in ['onset24','remaining']}))
 print((OUT/'suffix_similarity.json').read_text())
if __name__=='__main__':main()
