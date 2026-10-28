"""Serial, audited successful-demonstration collection under a frozen camera gate.

This is a new protocol, separate from the previous bounded offset experiment.
All attempted seeds/outcomes are accounted for; only audited natural successes
count toward the target. No training, seed retries or unknown-file deletion.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .compact import convert_episode, sha256
from .contracts import assert_policy_compatible
from .run_insertion_dataset import CHECKPOINT, CHECKPOINT_SHA, tree_bytes
from .run_rl_trials import validate_completion, utc
from .trial_utils import code_identity, write_json

GIB = 1024**3


def classify_pre_action_rejection(path):
    """Only the collector's explicit geometry rejection is an invalid start."""
    path=Path(path)
    if (path/'trajectories.h5').exists() or not (path/'metadata.json').exists(): return None
    if not (path/'FAILED.txt').exists(): return None
    if 'Initial object/table/hole collision overlap' not in (path/'FAILED.txt').read_text(): return None
    meta=json.loads((path/'metadata.json').read_text())
    if meta.get('initial_geometry',{}).get('valid_box_initialization') is not False: return None
    return meta


def run(root, gate_path, predecessor=None):
    gate=json.loads(gate_path.read_text())
    required=('camera_visibility','expert_pair','repeat_baseline','dynamic_pose','storage_roundtrip','model_interfaces','resource_budget','bootstrap_startup')
    if not all(gate.get('checks',{}).get(k) is True for k in required): raise ValueError('Camera gate incomplete')
    reference_path=Path(gate['reference_metadata'])
    if sha256(reference_path)!=gate['reference_metadata_sha256']: raise ValueError('Reference metadata changed')
    reference=json.loads(reference_path.read_text())
    if reference.get('kit_extension_watcher_enabled') is not False:
        raise ValueError('Headless production requires validated process-local extension watcher disabled')
    prior_attempts=[]
    if predecessor is not None:
        prior_manifest=json.loads((predecessor/'manifest.json').read_text())
        prior_status=json.loads((predecessor/'run_status/status.json').read_text())
        if prior_status['state']!='blocked' or prior_manifest['episodes'] or prior_status['successful_episodes']:
            raise ValueError('This continuation only handles a pre-episode blocked predecessor; never discards existing data')
        prior_attempts=[dict(r,predecessor=str(predecessor)) for r in prior_manifest['attempts']]
        if not prior_attempts or any(r['outcome']!='software_or_audit_error' for r in prior_attempts):
            raise ValueError('Inspect predecessor attempts before continuation')
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA: raise ValueError('Released expert checksum changed')
    root.mkdir(parents=True,exist_ok=False)
    for sub in ('episodes','temporary','logs','run_status'): (root/sub).mkdir()
    identity=code_identity()
    seed_start=max([929999,*[r['seed'] for r in prior_attempts]])+1
    plan=dict(protocol='triple400_native_seed_rejection_watchoff_v2',created_utc=utc(),identity=identity,
        target_successes=400,max_attempts=1600,seed_start=seed_start,max_steps=2400,wall_limit_s=900,
        predecessor=str(predecessor) if predecessor else None,prior_attempt_count=len(prior_attempts),
        kit_extension_watcher_enabled=False,
        max_bytes=95*GIB,min_free_bytes=40*GIB,attempt_reserve_bytes=12*GIB,
        checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,gate=gate,gate_sha256=sha256(gate_path),
        seed_rule='Continue from last attempted seed + 1 (930000 if no predecessor); original native reset, zero added offset; never retry a seed',
        selection='Original object/table/hole box overlap is pre-action rejection; all legal task failures/censored records retained. Only audited natural success counts.',
        software_rule='Stop on any startup/runtime/conversion/audit error; retain partials, no automatic retry',
        camera_rig='wrist_bracket',image_size=[448,252],render_mode='quality',render_aa='Off',
        wrench_frame_mode='world_hold',parameter_updates=0,cooldown_seconds=5)
    write_json(root/'run_status/plan.json',plan)
    manifest=dict(manifest_version=1,protocol=plan['protocol'],episodes=[],attempts=prior_attempts.copy())
    write_json(root/'manifest.json',manifest)
    status=dict(pid=os.getpid(),state='starting',successful_episodes=0,attempted_seeds=len(prior_attempts),started_utc=utc())
    def publish(**updates):
        status.update(updates,updated_utc=utc());write_json(root/'run_status/status.json',status)
    successes=0
    try:
        for i in range(plan['max_attempts']-len(prior_attempts)):
            if successes==plan['target_successes']: break
            if (root/'STOP').exists():
                publish(state='stopped_by_STOP_file');return
            if code_identity()['source_manifest_sha256']!=identity['source_manifest_sha256']:
                raise RuntimeError('Source changed during frozen batch')
            if tree_bytes(root)+plan['attempt_reserve_bytes']>plan['max_bytes'] or shutil.disk_usage(root).free<plan['min_free_bytes']+plan['attempt_reserve_bytes']:
                raise RuntimeError('Capacity guard before next attempt')
            seed=plan['seed_start']+i;name=f's{seed}';temporary=root/'temporary'/name
            row=dict(id=name,seed=seed,pair_id=f'native_initial_{seed}',condition='triple400_native',
                offset_xy=[0.,0.],wrench_frame_mode='world_hold',stage='production',attempt_index=i+len(prior_attempts))
            cmd=['bash','visuotactile/run_cpu.sh','-m','visuotactile.collect','--output',str(temporary),
                 '--camera-rig',plan['camera_rig'],'--image-size','448','252','--render-mode','quality',
                 '--seed',str(seed),'--pair-id',row['pair_id'],'--episodes','1','--max-steps',str(plan['max_steps']),
                 '--experiment-condition',row['condition'],'--wrench-frame-mode','world_hold','--disable-extension-watch']
            start=time.monotonic()
            publish(state='collecting',current_id=name,current_log=str(root/'logs'/f'{name}.log'),
                    successful_episodes=successes,attempted_seeds=i+1+len(prior_attempts))
            with (root/'logs'/f'{name}.log').open('x') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                publish(child_pid=child.pid)
                while child.poll() is None:
                    time.sleep(5)
                    if time.monotonic()-start>plan['wall_limit_s'] or tree_bytes(root)>plan['max_bytes'] or shutil.disk_usage(root).free<plan['min_free_bytes']:
                        child.terminate()
                        try: child.wait(timeout=15)
                        except subprocess.TimeoutExpired: child.kill();child.wait()
                        raise RuntimeError('Owned collector stopped by wall/disk bound; partial preserved')
            outcome=dict(**row,returncode=child.returncode,wall_seconds=time.monotonic()-start)
            rejected=classify_pre_action_rejection(temporary)
            if rejected is not None:
                outcome.update(outcome='invalid_initialization',reason=rejected['initial_geometry'],episode_started=False,
                               raw_pose_w=rejected['initial_offset']['raw_pose_w'])
                manifest['attempts'].append(outcome);write_json(root/'manifest.json',manifest)
                publish(state='rejected_initialization',child_pid=None)
                time.sleep(plan['cooldown_seconds']);continue
            trial=validate_completion(temporary,row,child.returncode)
            summary,meta=trial['summary'],trial['metadata']
            assert_policy_compatible(reference,meta)
            if meta.get('kit_extension_watcher_enabled') is not False: raise RuntimeError('Extension watcher setting drift')
            if meta['expert_model_sha256']!=CHECKPOINT_SHA: raise RuntimeError('Wrong expert')
            if not meta.get('initial_geometry',{}).get('valid_box_initialization'): raise RuntimeError('Missing geometry gate')
            for section in ('action','assets','domain_randomization','obs','reset','reward','termination','precise_assembly'):
                if reference['config'][section]!=meta['config'][section]: raise RuntimeError('Physical protocol drift: '+section)
            if summary['requested_vs_applied_peak_vector_error_N']>1e-5: raise RuntimeError('Wrench trace mismatch')
            publish(state='converting',child_pid=None)
            result=json.loads((temporary/'result.json').read_text())
            directory='episodes/'+name
            conversion=convert_episode(temporary/'trajectories.h5',root/directory,result=result)
            compact=validate_completion(root/directory,row)
            if compact['summary']['outcome']!=summary['outcome']: raise RuntimeError('Converted outcome differs')
            record=dict(**row,**conversion,directory=directory,outcome=summary['outcome'],valid=True,
                schema_version=2,trace_version=2,wrench_trace_version=3,summary=summary,parameter_updates=0,
                nominal_predictions_available=False,video_role='viewing_derivative',
                initial_object_pose_w=trial['arrays']['initial']['object_pose_w'].tolist(),
                initial_hole_pose_w=trial['arrays']['initial']['hole_pose_w'].tolist())
            manifest['episodes'].append(record)
            outcome.update(outcome=summary['outcome'],episode_started=True,control_steps=summary['control_steps'])
            manifest['attempts'].append(outcome);write_json(root/'manifest.json',manifest)
            # Exact full-field reconstruction and compact audit passed and manifest is durable.
            shutil.rmtree(temporary)
            successes+=int(summary['outcome']=='success')
            publish(state='committed',successful_episodes=successes,bytes=tree_bytes(root),last_episode=directory)
            print(json.dumps(dict(id=name,outcome=summary['outcome'],successful_episodes=successes,attempted_seeds=i+1+len(prior_attempts))),flush=True)
            time.sleep(plan['cooldown_seconds'])
        publish(state='complete' if successes==400 else 'attempt_limit',child_pid=None,successful_episodes=successes)
    except BaseException as exc:
        if 'row' in locals() and not any(a['id']==row['id'] for a in manifest['attempts']):
            manifest['attempts'].append(dict(**row,outcome='software_or_audit_error',error=repr(exc),
                returncode=child.returncode if 'child' in locals() else None))
            write_json(root/'manifest.json',manifest)
        publish(state='blocked',error=repr(exc),successful_episodes=successes,child_pid=None)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,help='Preserve an inspected pre-episode blocked batch in a new root; continue at next seed')
    a=p.parse_args();run(a.output,a.gate,a.predecessor)


if __name__=='__main__': main()
