"""Train/development-only descriptive split of never-first-gate trajectories.

Replays the existing adaptive alignment screen without changing any threshold,
selection, policy, native state, or already-registered pilot seed.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.tight_premature_scope_20261122.trajectory_audit import PROJECT, OUT, PUBLIC, read, write, sha, first, first_run, runs, angle
from visuotactile.adaptive_takeover import AdaptiveTrigger, profile


def main():
    dest=OUT/'never_gate_stratification';dest.mkdir(exist_ok=True)
    pub=PUBLIC/'never_gate_stratification';pub.mkdir(exist_ok=True)
    write(dest/'registration.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),sources='frozen train180/development96 only',
        terminal_reads=0,pilot_seeds_unchanged=True,not_for_reselecting_current_pilot=True,
        existing_profile=profile('alignment'),existing_profile_code_sha256=sha(PROJECT/'visuotactile/adaptive_takeover.py'),
        held='>=2 contact fingers in both physics substeps for12ticks, object worldZ>=median(z[24:48])+20mm throughout12ticks; same as existing adaptive profile',
        near='current object-root to active goal-root<=90mm while held12; no new threshold',
        strata=['no_stable_held12','stable_held12_without_near90','stable_held12_and_near90'],
        ambiguity='never15mm alone does not identify grasp, transport, or alignment cause; adaptive alignment is a screen, not proven recoverability'))
    rows=read(OUT/'trajectories.json');out=[]
    for row in rows:
        assert row['cohort'] in ('train','development')
        with h5py.File(row['path'],'r') as f:
            tr=f[row['group']+'/transition'];n=len(tr['step_id']);ticks=np.arange(n)
            obj=tr['diagnostics_after/object_pose_w'][:];palm=tr['diagnostics_after/palm_body_pose_w'][:]
            force=tr['object_contact_force_w'][:];key=tr['native_keypoints_max_dist'][:]
            beforegoal=tr['diagnostics_before/goal_pose_w'][:];aftergoal=tr['diagnostics_after/goal_pose_w'][:]
            beforegoals=tr['diagnostics_before/successes'][:];aftergoals=tr['diagnostics_after/successes'][:]
            pre=ticks<(row['native_first_subgoal_tick'] if row['native_first_subgoal_tick'] is not None else n)
            count=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
            root=np.linalg.norm(obj[:,:3]-beforegoal[:,:3],axis=-1)
            orient=angle(obj[:,3:],beforegoal[:,3:])
            settled=float(np.median(obj[24:48,2])) if n>=48 else None
            held=np.zeros(n,bool)
            if settled is not None:
                good=(count>=2)&(obj[:,2]>=settled+.02)
                for rr in runs(good):
                    if rr['ticks']>=12:held[rr['start']+11:rr['end']+1]=True
                held[:47]=False
            near=held&(root<=.09)&pre
            stable=held&pre
            screen=AdaptiveTrigger(profile('alignment'),10**9)
            for t in range(n):
                hit=screen.observe(t,obj[t],palm[t],force[t],keypoint_max_distance_m=float(key[t]),
                    goal_index_before=int(beforegoals[t]),goal_index_after=int(aftergoals[t]),goal_pose_before=beforegoal[t],goal_pose_after=aftergoal[t])
                if hit is not None:break
            receipt=screen.receipt();kt=int(np.argmin(np.where(pre,key,np.inf)));rt=int(np.argmin(np.where(pre,root,np.inf)))
            level='stable_held12_and_near90' if near.any() else 'stable_held12_without_near90' if stable.any() else 'no_stable_held12'
            val=dict(identity=row['identity'],cohort=row['cohort'],policy=row['policy'],family=row['family'],seed=row.get('seed'),
                native_outcome=row['native_outcome'],never15mm=row['overlap_flags']['never15mm'],stratum=level,
                settled_reference_z_m=settled,stable_held12_ever=bool(stable.any()),first_stable_held12_completed_transition=first(stable),
                held12_near90_ever=bool(near.any()),first_held12_near90_completed_transition=first(near),
                held12_ticks_before_confirmation=int(stable.sum()),held12_near90_ticks_before_confirmation=int(near.sum()),
                minimum_keypoint_mm=float(key[kt]*1000),minimum_keypoint_tick=kt,root_at_minimum_keypoint_mm=float(root[kt]*1000),
                angle_at_minimum_keypoint_deg=float(orient[kt]),minimum_root_mm=float(root[rt]*1000),minimum_root_tick=rt,
                keypoint_at_minimum_root_mm=float(key[rt]*1000),angle_at_minimum_root_deg=float(orient[rt]),
                minimum_angle_deg=float(orient[pre].min()),alignment_screen_hit=receipt['trigger_completed_tick'] is not None,
                alignment_screen_scheduled_next_action_tick=receipt['trigger_completed_tick'],alignment_screen=receipt)
            out.append(val)
    write(dest/'trajectories.json',out)
    flat=[{k:v for k,v in r.items() if k!='alignment_screen'} for r in out]
    with (dest/'trajectories.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    summary={}
    pools={'train_all':out[:180],'development_all':out[180:]}
    for p in ('A_130_50','B_130_50','A_R2','B_R2','A_entry','B_entry'):
        pools['development/'+p]=[r for r in out if r['cohort']=='development' and r['policy']==p]
    for name,pool in pools.items():
        never=[r for r in pool if r['never15mm']]
        summary[name]=dict(n=len(pool),never15mm=len(never),never_strata=dict(Counter(r['stratum'] for r in never)),
            never_alignment_screen_hits=sum(r['alignment_screen_hit'] for r in never),
            all_alignment_screen_hits=sum(r['alignment_screen_hit'] for r in pool),
            native_success_screen_hits=sum(r['alignment_screen_hit'] for r in pool if r['native_outcome']=='complete_success'),
            native_success_denominator=sum(r['native_outcome']=='complete_success' for r in pool))
    write(dest/'summary.json',summary)
    write(dest/'receipt.json',dict(trajectories=len(out),terminal_reads=0,new_simulation=0,new_inference=0,optimizer_updates=0,new_dagger=0,pilot_seeds_changed=False,
         script_sha256=sha(__file__),primary_analyze_sha256=sha(PROJECT/'scripts/tight_premature_scope_20261122/trajectory_audit.py')))
    for p in dest.iterdir():shutil.copy2(p,pub/p.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
