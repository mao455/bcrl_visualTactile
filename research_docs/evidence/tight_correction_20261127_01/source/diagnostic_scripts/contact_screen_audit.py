from pathlib import Path
import json
import h5py,numpy as np
from visuotactile.adaptive_takeover import AdaptiveTrigger,profile
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01')
read=lambda p:json.loads(p.read_text())
rows=read(D/'development_before_cases.json');out=[]
for row in rows:
 if not row['complete']:continue
 path=Path(row['source']).parent/'numeric.h5'
 trigger=AdaptiveTrigger(profile('contact',observe_only=True),10**9)
 with h5py.File(path) as f:
  tr=f['episode/transition'];names=['diagnostics_after/object_pose_w','diagnostics_after/palm_body_pose_w','object_contact_force_w','native_keypoints_max_dist','diagnostics_before/successes','diagnostics_after/successes','diagnostics_before/goal_pose_w','diagnostics_after/goal_pose_w'];a={n:tr[n][:] for n in names}
  for t in range(len(a['native_keypoints_max_dist'])):
   trigger.observe(t,a['diagnostics_after/object_pose_w'][t],a['diagnostics_after/palm_body_pose_w'][t],a['object_contact_force_w'][t],keypoint_max_distance_m=float(a['native_keypoints_max_dist'][t]),goal_index_before=int(a['diagnostics_before/successes'][t]),goal_index_after=int(a['diagnostics_after/successes'][t]),goal_pose_before=a['diagnostics_before/goal_pose_w'][t],goal_pose_after=a['diagnostics_after/goal_pose_w'][t])
 out.append(dict(role=row['role'],seed=row['seed'],success=row['success'],category=row['category'],first_contact_eligible_completed_tick=trigger.first_eligible_completed_tick,first_held_completed_tick=trigger.first_held,source=str(path)))
atomic_json(D/'contact_screen_audit.json',out)
print(json.dumps(out))
