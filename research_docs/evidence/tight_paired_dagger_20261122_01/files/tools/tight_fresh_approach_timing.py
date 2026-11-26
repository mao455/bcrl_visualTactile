"""Numeric-only timing of real approach/contact; no trajectory or label changes."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from visuotactile.training_resources import atomic_json

def numeric(path,identity,*,student_ticks=None):
 with h5py.File(path) as f:
  g=f['episode'] if 'episode' in f else f['episodes'][identity]
  t=g['transition'];d=t['diagnostics_after'];limit=150 if student_ticks is None else student_ticks
  obj=d['object_pose_w'][:limit,:3];palm=d['palm_body_pose_w'][:limit,:3]
  fingertips=d['finger_position_w'][:limit]
  force=t['object_contact_force_w'][:limit]
  finger_count=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
  root=np.linalg.norm(obj-palm,axis=-1);tip=np.linalg.norm(obj[:,None,:]-fingertips,axis=-1).min(-1)
  proposed=None
  if student_ticks is not None:
   from visuotactile.grasp_curriculum import APPROACH
   from visuotactile.takeover_trigger import make_trigger,observe_transition
   trigger=make_trigger(APPROACH,10**9)
   for tick in range(len(obj)):
    if observe_transition(trigger,tick,t,index=tick) is not None:break
   proposed=trigger.receipt()
 contact=np.flatnonzero(finger_count>0);first=int(contact[0])+1 if len(contact) else None
 samples={}
 for n in sorted(set([24,36,48]+([] if first is None else [first-20,first-10,first]))):
  if 0<n<=len(root):samples[str(n)]=dict(palm_root_distance_m=float(root[n-1]),nearest_finger_center_distance_m=float(tip[n-1]))
 return dict(identity=identity,path=str(path),student_ticks=student_ticks,observed_ticks=len(root),
  proposed_causal_trigger=proposed,new_rescue_demonstrated=False,
  first_contact_completed_tick=first,first_contact_sim_seconds=None if first is None else first/60,
  distances_at_completed_ticks=samples,
  palm_distance_crossings={str(x):next((n+1 for n in range(35,len(root)) if root[n]<=x),None) for x in [.18,.16,.14,.12,.10]},
  fingertip_distance_crossings={str(x):next((n+1 for n in range(35,len(tip)) if tip[n]<=x),None) for x in [.06,.05,.04,.03]})

def run(data,collections,output):
 assert not output.exists()
 d=json.loads(data.read_text());rows=[]
 for i in d['normalization_train_ids']:
  x=numeric(Path(d['entries'][i]['path']),i);x.update(cohort='original_success',family=d['entries'][i]['family']);rows.append(x)
 for directory in collections:
  for c in json.loads((directory/'plan.json').read_text())['cases']:
   result=json.loads((directory/c['name']/'result.json').read_text())
   if not result['legal']:continue
   tick=result['dagger']['recovery_coverage']['switch_tick']
   x=numeric(directory/c['name']/'numeric.h5',c['name'],student_ticks=tick)
   x.update(cohort='true_student_prefix',family=c['spec']['family'],role=c['role'],
    takeover_control_tick=tick,takeover_sim_seconds=tick/60);rows.append(x)
 atomic_json(output,dict(rows=rows,scope='Completed real transitions, control60Hz; distances use diagnostic truth only. Numeric history only, no policy input, target, success criterion or collection rule changed. Threshold crossings are descriptive retrospective screens, not recovered trajectories.'))

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--data',type=Path,required=True)
 p.add_argument('--collection',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();run(a.data,a.collection,a.output)
