"""Small measured reset/wrench report from audited episodes, not inferred noise."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .run_rl_trials import validate_completion
from .compact import EpisodeFile
from .trial_utils import write_json


def manifest_statistics(manifest, output):
    """Fast live statistics from values already extracted by the full audit.

    No repeated image scan. Missing old fields stay explicitly unavailable.
    Coordinates subtract the recorded env origin before combining batches.
    """
    from collections import Counter
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    attempts={r['id']:r for r in manifest['attempts']};ids={r['id'] for r in manifest['episodes']}
    rows=[];xyz=[];holes=[]
    for entry in manifest['episodes']:
        a=entry.get('initial') or attempts.get(entry['id'],{}).get('initial')
        if not a:continue
        s=entry['summary'];origin=np.asarray(entry.get('origin_w',[0.,0.,0.]))
        position=np.asarray(a['object_pose_w'])[:3]-origin
        hole=np.asarray(a['hole_pose_w'])[:3]-origin
        xyz.append(position);holes.append(hole)
        rows.append(dict(id=entry['id'],seed=entry['seed'],env_index=entry.get('env_index',0),
            steps=s['control_steps'],object_xyz_env_m=position.tolist(),hole_xyz_env_m=hole.tolist(),
            object_quat_wxyz=a['object_pose_w'][3:],q_arm_rad=a['joint_position_rad'][:7],
            qd_norm_rad_s=float(np.linalg.norm(a['joint_velocity_rad_s'])),
            force_probability=float(a['external_force_probability']),torque_probability=float(a['external_torque_probability']),
            force_physics_samples=s['applied_force']['nonzero_steps'],peak_force_N=s['applied_force']['peak'],
            torque_physics_samples=s['applied_torque']['nonzero_steps'],peak_torque_Nm=s['applied_torque']['peak'],
            delay_counts=entry.get('effective_delay_counts',attempts.get(entry['id'],{}).get('effective_delay_counts'))))
    temporary=output/'initial_states.csv.tmp'
    with temporary.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id']);writer.writeheader();writer.writerows(rows)
    temporary.replace(output/'initial_states.csv')
    outcomes=Counter(r.get('outcome','unknown') for r in manifest['attempts'])
    result=dict(successes=len(ids),success_rows_with_initial=len(rows),all_attempt_outcomes=dict(outcomes),
        excluded_historical_successes=sum(r.get('outcome')=='success' and r['id'] not in ids for r in manifest['attempts']),
        invalid_initializations=outcomes['invalid_initialization'],natural_task_failures=outcomes['failure'],
        startup_errors=sum(outcomes[k] for k in ('startup_error','software_startup_error')),
        worker_errors=outcomes['worker_error'],audit_errors=outcomes['audit_error'],
        source='Full per-episode audit outputs and native reset diagnostics; no imagined samples',
        object_xyz_env_quantiles_m=np.quantile(xyz,[0,.25,.5,.75,1],axis=0).tolist() if xyz else None,
        hole_xyz_env_quantiles_m=np.quantile(holes,[0,.25,.5,.75,1],axis=0).tolist() if holes else None,
        distinct_initial_object_positions=len({tuple(v) for v in xyz}),
        force_exposed_successes=sum(r['force_physics_samples']>0 for r in rows),
        limitation='Success-conditioned distribution. Illegal reset counts include retained historical attempts; error categories remain separate. Missing delay counts in older rows are null.')
    write_json(output/'summary.json',result)
    write_json(output/'attempts.json',dict(attempts=[{k:r.get(k) for k in ('id','seed','run_id','env_index','outcome','reason','initial','control_steps','error')} for r in manifest['attempts']]))
    return result


def report(manifest,output):
    manifest=Path(manifest);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    m=json.loads(manifest.read_text());rows=[];initials=[]
    for entry in m['episodes']:
        p=manifest.parent/entry['directory'];trial=validate_completion(p)
        a,s=trial['arrays']['initial'],trial['summary'];initials.append(a['object_pose_w'])
        with EpisodeFile(p/'data.h5') as f:
            g=f['episodes'][entry['episode']];lag=g['transition/control_trace/effective_lag_steps'][:]
        row=dict(id=entry['id'],seed=entry['seed'],outcome=s['outcome'],steps=s['control_steps'],
            object_xyz_w_m=a['object_pose_w'][:3].tolist(),object_quat_wxyz=a['object_pose_w'][3:].tolist(),
            hole_xyz_w_m=a['hole_pose_w'][:3].tolist(),q_arm_rad=a['joint_position_rad'][:7].tolist(),
            qd_norm_rad_s=float(np.linalg.norm(a['joint_velocity_rad_s'])),
            force_probability=float(a['external_force_probability']),torque_probability=float(a['external_torque_probability']),
            force_physics_samples=s['applied_force']['nonzero_steps'],force_duration_s=s['applied_force']['duration_s'],
            peak_force_N=s['applied_force']['peak'],torque_physics_samples=s['applied_torque']['nonzero_steps'],
            peak_torque_Nm=s['applied_torque']['peak'],force_steps_with_contact_before=s['force_steps_with_finger_contact_before'],
            force_steps_with_contact_after=s['force_steps_with_finger_contact_after'],
            effective_delay_counts=np.bincount(lag.astype(int),minlength=3).tolist())
        rows.append(row)
    with (output/'initial_states.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id']);writer.writeheader();writer.writerows(rows)
    xyz=np.array(initials)[:,:3] if initials else np.empty((0,3))
    outcomes={k:sum(a['outcome']==k for a in m['attempts']) for k in sorted({a['outcome'] for a in m['attempts']})}
    result=dict(source_manifest=str(manifest),episodes=rows,attempt_outcomes=outcomes,
        success_xyz_quantiles_m=np.quantile(xyz,[0,.25,.5,.75,1],axis=0).tolist() if len(xyz) else None,
        success_exact_initial_duplicates=len(initials)-len({tuple(x) for x in initials}),
        zero_force_episode_fraction=sum(r['force_physics_samples']==0 for r in rows)/len(rows) if rows else None,
        limitation='Success-conditioned sample, not native reset distribution; software startup without metadata has no initial state. No empirical noise distribution recorded.')
    write_json(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=report(a.manifest,a.output);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
