"""Serial, audited successful-demonstration collection under a frozen camera gate.

This is a new protocol, separate from the previous bounded offset experiment.
All attempted seeds/outcomes are accounted for; only audited natural successes
count toward the target. No training, seed retries or unknown-file deletion.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .compact import sha256
from .run_insertion_dataset import CHECKPOINT, CHECKPOINT_SHA
from .run_rl_trials import validate_completion, utc
from .trial_utils import code_identity, write_json

MAX_BATCH_BYTES = 60_000_000_000  # user constraint: decimal GB, not GiB
MAX_WALL_SECONDS = 6 * 60 * 60


def classify_pre_action_rejection(path):
    """Only the collector's explicit geometry rejection is an invalid start."""
    path=Path(path)
    if (path/'trajectories.h5').exists() or not (path/'metadata.json').exists(): return None
    if not (path/'FAILED.txt').exists(): return None
    if 'Initial object/table/hole collision overlap' not in (path/'FAILED.txt').read_text(): return None
    meta=json.loads((path/'metadata.json').read_text())
    if meta.get('initial_geometry',{}).get('valid_box_initialization') is not False: return None
    return meta


def run(root, gate_path, predecessor=None, *, deadline_epoch, seed_start=930011):
    from .collection_resources import Supervisor
    from .manifest import ManifestReader
    gate=json.loads(gate_path.read_text())
    required=('camera_visibility','expert_pair','repeat_baseline','dynamic_pose',
              'storage_roundtrip','model_interfaces','resource_budget','bootstrap_startup')
    if not all(gate.get('checks',{}).get(k) is True for k in required):
        raise ValueError('Camera gate incomplete')
    reference_path=Path(gate['reference_metadata'])
    if sha256(reference_path)!=gate['reference_metadata_sha256']:raise ValueError('Reference changed')
    reference=json.loads(reference_path.read_text())
    if reference['camera_rig']['name']!='wrist_bracket_close' or reference['camera_ids']!=['fixed_high','fixed_side','wrist_rgbd']:
        raise ValueError('Production requires verified v4 triple camera gate')
    if reference.get('kit_extension_watcher_enabled') is not False:raise ValueError('Watcher not disabled')
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Released checkpoint changed')
    if not time.time()<deadline_epoch<=time.time()+MAX_WALL_SECONDS:raise ValueError('Invalid absolute six-hour deadline')
    prior_attempts=[];prior_episodes=[]
    if predecessor:
        previous=ManifestReader(predecessor/'manifest.json',verify=True)
        state=json.loads((predecessor/'run_status/status.json').read_text())
        if state['state']!='blocked':raise ValueError('Predecessor must be stopped and classified')
        prior_attempts=[dict(a,predecessor=str(predecessor)) for a in previous.manifest['attempts']]
        for key,row in previous.entries.items():
            trial=validate_completion(previous.path.parent/row['directory'],row)
            prior_episodes.append(dict(id=key,outcome=trial['summary']['outcome'],source=str(previous.path),
                counted_in_v4=False,reason='Separate v3 calibration; hashes and audits rechecked'))
    prior_attempts.extend(gate.get('excluded_attempts',[]))
    if seed_start<=max([929999,*[a['seed'] for a in prior_attempts]]):raise ValueError('Seed reuse')
    root.mkdir(parents=True,exist_ok=False)
    for sub in ('episodes','temporary','logs','run_status','audits'): (root/sub).mkdir()
    identity=code_identity()
    plan=dict(protocol='triple400_close_v4_budget2',identity=identity,created_utc=utc(),
        deadline_epoch=deadline_epoch,target_successes=400,max_attempts=1600,seed_start=seed_start,
        max_steps=2400,attempt_wall_s=900,checkpoint_sha256=CHECKPOINT_SHA,
        predecessor=str(predecessor),prior_episodes=prior_episodes,gate=gate,
        camera_rig='wrist_bracket_close',max_project_bytes=MAX_BATCH_BYTES,
        gpu_metric='time-weighted whole card; process pmon incomplete',gpu_limit_percent=50,
        allocator_environment={'MALLOC_ARENA_MAX':'2'},
        recovery='No automatic retries of unknown native errors; ordinary invalid/failure advances seed',
        disk='Global output inventory; 1 GB startup reserve; per-row raw+compact uncompressed reserve',
        time='Absolute deadline includes earlier probes, startup, audit, conversion, throttle, recovery')
    write_json(root/'run_status/plan.json',plan)
    manifest=dict(manifest_version=1,protocol=plan['protocol'],episodes=[],attempts=prior_attempts,
                  previous_camera_episodes=prior_episodes)
    write_json(root/'manifest.json',manifest)
    status=dict(pid=os.getpid(),state='starting',successful_episodes=0,started_utc=utc(),
                deadline_epoch=deadline_epoch,prior_version_successes=len(prior_episodes))
    def publish(**kw):
        status.update(kw,updated_utc=utc());write_json(root/'run_status/status.json',status)
    supervisor=Supervisor(root/'run_status',deadline_epoch)
    # All application and conversion/audit work is supervised; no telemetry gaps
    # hidden inside a long in-process conversion.
    successes=0;row=None
    try:
        for i in range(plan['max_attempts']):
            if successes>=400:break
            supervisor.check(1_000_000_000)
            if (root/'STOP').exists():publish(state='stopped_by_STOP_file');return
            if code_identity()['source_manifest_sha256']!=identity['source_manifest_sha256']:
                raise RuntimeError('Frozen source changed')
            seed=seed_start+i;name=f's{seed}';temporary=root/'temporary'/name
            row=dict(id=name,seed=seed,pair_id=f'native_initial_{seed}',condition='triple400_close',
                     offset_xy=[0.,0.],wrench_frame_mode='world_hold',stage='production')
            started=time.monotonic()
            log=root/'logs'/f'{name}.log'
            publish(state='collecting',current_id=name,current_log=str(log),attempted_new=i+1)
            cmd=['env','MALLOC_ARENA_MAX=2','bash','visuotactile/run_cpu.sh','-m','visuotactile.collect',
                 '--output',str(temporary),'--camera-rig','wrist_bracket_close','--image-size','448','252',
                 '--render-mode','quality','--seed',str(seed),'--pair-id',row['pair_id'],
                 '--episodes','1','--max-steps','2400','--experiment-condition',row['condition'],
                 '--wrench-frame-mode','world_hold','--disable-extension-watch','--global-output-guard']
            rc=supervisor.run(cmd,log,timeout=900,reserve=1_000_000_000)
            outcome=dict(**row,returncode=rc)
            rejected=classify_pre_action_rejection(temporary)
            if rejected is not None:
                outcome.update(outcome='invalid_initialization',reason=rejected['initial_geometry'],
                    initial_offset=rejected['initial_offset'],episode_started=False)
            else:
                if rc!=0:raise RuntimeError(f'Native collector exit {rc}; no blind retry')
                report=root/'audits'/f'{name}.json'
                directory='episodes/'+name
                publish(state='auditing_and_converting')
                cmd=['bash','visuotactile/run_cpu.sh','-m','visuotactile.finalize_collection',
                     '--source',str(temporary),'--destination',str(root/directory),
                     '--reference',str(reference_path),'--report',str(report)]
                rc=supervisor.run(cmd,root/'logs'/f'{name}_audit.log',timeout=900)
                if rc!=0:raise RuntimeError('Conversion/audit failed; preserve raw and partial')
                audit=json.loads(report.read_text());summary=audit['summary']
                outcome.update(outcome=summary['outcome'],episode_started=True,
                    control_steps=summary['control_steps'],initial=audit['initial'],summary=summary)
                if summary['outcome']=='success':
                    record=dict(**row,**audit['conversion'],directory=directory,outcome='success',valid=True,
                                summary=summary,parameter_updates=0,nominal_predictions_available=False)
                    manifest['episodes'].append(record);successes+=1
            outcome['full_cycle_seconds']=time.monotonic()-started
            manifest['attempts'].append(outcome);write_json(root/'manifest.json',manifest)
            # Classification and exact successful reconstruction are durable.
            # Only this attempt's generated payload is removed; never unknown data.
            deleted=[]
            for path in temporary.iterdir():
                if path.is_file() and path.suffix.lower() in ('.h5','.npy','.npz','.png','.mp4'):
                    deleted.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path)))
                    path.unlink()
            write_json(root/'audits'/f'{name}_cleanup.json',dict(reason=outcome['outcome'],files=deleted))
            supervisor.sample()
            publish(state='committed' if outcome['outcome']=='success' else 'classified',
                successful_episodes=successes,last_attempt=outcome['outcome'],last_episode=directory if outcome['outcome']=='success' else None,
                resources=supervisor.latest,attempted_new=i+1)
            print(json.dumps(dict(id=name,outcome=outcome['outcome'],successes=successes,
                                  full_cycle_seconds=outcome['full_cycle_seconds'])),flush=True)
        publish(state='complete' if successes==400 else 'attempt_limit',resources=supervisor.latest)
    except BaseException as exc:
        if row and not any(a['id']==row['id'] for a in manifest['attempts']):
            manifest['attempts'].append(dict(**row,outcome='software_or_budget_error',error=repr(exc)))
            write_json(root/'manifest.json',manifest)
        publish(state='blocked',error=repr(exc),successful_episodes=successes,resources=supervisor.latest)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--predecessor',type=Path)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--seed-start',type=int,default=930011)
    a=p.parse_args();run(a.output,a.gate,a.predecessor,deadline_epoch=a.deadline_epoch,seed_start=a.seed_start)

if __name__=='__main__':main()
