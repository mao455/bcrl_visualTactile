"""Registered scalable collection and autonomy using the same full student path."""
import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import traceback

from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def validate_inactive_render_request(profile,backend):
    if profile is None:return None
    from visuotactile.inactive_render_products import validate_profile
    from visuotactile.execution_backend import normalize_backend
    profile=validate_profile(profile);backend=normalize_backend(backend)
    if backend['physics_device']!='cpu' or backend['camera_backend']!='standard':
        raise ValueError('Inactive render products require explicit CPU standard execution')
    return profile


@contextmanager
def inactive_render_scope(env,profile,output,*,gate_limit_ticks=None):
    """Restore before publishing completion, including Python error paths."""
    if profile is None:
        yield None
        return
    from visuotactile.inactive_render_products import InactiveRenderProducts
    guard=None;primary=None;cleanup=None;before=None;after=None
    try:
        guard=InactiveRenderProducts.bind(env,profile,gate_limit_ticks=gate_limit_ticks)
        yield guard
    except BaseException as error:
        primary=repr(error)
        raise
    finally:
        if guard is not None:
            try:before=guard.snapshot()
            except BaseException as error:before=dict(snapshot_error=repr(error))
            try:guard.close()
            except BaseException as error:cleanup=error
            try:after=guard.snapshot()
            except BaseException as error:after=dict(snapshot_error=repr(error))
            if cleanup is None and (before.get('snapshot_error') or after.get('snapshot_error')
                    or before.get('callback_error') or after.get('callback_error')
                    or after.get('closed') is not True or not all(p['enabled'] is True for p in after.get('products',[]))):
                cleanup=RuntimeError('Inactive renderer final provenance or restoration receipt failed')
        atomic_json(Path(output)/'inactive_render_cleanup.json',dict(profile=profile,primary_error=primary,
            cleanup_error=repr(cleanup) if cleanup else None,before=before,after=after,
            restored=bool(guard is not None and cleanup is None),utc=time.time()))
        if cleanup is not None:raise cleanup


def prepare(root,output,*,lanes,count=8,seed=28142000,identities=None,delays=None,round_index=7,
            execution_backend=None,writer_bytes=512<<20,vision_microbatch=2,session_root=None,
            verification_workers=0,verification_max_pending=4,vision_tail_profile=None,inactive_render_profile=None):
    from visuotactile.models.native_vision import validate_vision_microbatch,validate_vision_tail_profile
    from visuotactile.async_recording import validate_verification_parameters
    vision_microbatch=validate_vision_microbatch(vision_microbatch)
    tail=validate_vision_tail_profile(vision_tail_profile)
    inactive=validate_inactive_render_request(inactive_render_profile,execution_backend)
    validate_verification_parameters(verification_workers,verification_max_pending)
    if type(lanes) is not int or lanes<1 or type(count) is not int or count<1:
        raise ValueError('Explicit positive lane and opportunity counts required')
    if type(writer_bytes) is not int or writer_bytes<32<<20:
        raise ValueError('Explicit bounded capture memory required')
    from visuotactile.execution_backend import normalize_backend
    backend=normalize_backend(execution_backend)
    if identities is None:
        if session_root is not None:
            raise ValueError('An independent resource session is supported for explicit training takeovers only')
        from tools.tight_fresh_rollouts import prepare as serial_prepare
        serial_prepare(root,output,count,seed)
    else:
        from tools.tight_fresh_stage_takeover import prepare as stage_prepare
        from visuotactile.takeover_trigger import CONTACT_HEIGHT_VERSION
        stage_prepare(root,output,identities,delays,round_index,seed+100000,
            trigger_version=CONTACT_HEIGHT_VERSION,session_root=session_root)
    plan=read(output/'plan.json');session=read(output/'session.json')
    plan.update(batch=dict(version='whole_wave_complete_student_v1',lanes=lanes,seed=seed,
        neural_batch=True,writer_bytes=writer_bytes,vision_microbatch=vision_microbatch,
        verification_workers=verification_workers,verification_max_pending=verification_max_pending,
        ordering='Role-grouped whole waves; no dynamic recycling; all opportunities retained'),
        execution_backend=backend,expert_sha256=session['expert_sha256'])
    if tail is not None:plan['batch']['vision_tail_profile']=tail
    if inactive is not None:plan['batch']['inactive_render_profile']=inactive
    atomic_json(output/'plan.json',plan)
    return plan


def validate_takeover_plan(plan):
    """The selected registration decides which opportunities are legal."""
    from visuotactile.training_expansion import COLLECTION_CONTRACT,validate_collection_plan
    from tools.tight_fresh_stage_takeover import validate_plan
    if plan.get('stage_takeover_contract')==COLLECTION_CONTRACT:
        validate_collection_plan(plan)
    else:
        validate_plan(plan)


def classify_capture(directory,case,result,plan):
    """Fixed schedules and causal stage schedules keep different evidence."""
    from visuotactile.training_expansion import COLLECTION_CONTRACT,classify_case
    from tools.tight_fresh_stage_takeover import CONTRACT,CONTACT_HEIGHT_CONTRACT,classify
    from tools.tight_fresh_report import audit_trigger_case
    contract=plan.get('stage_takeover_contract')
    if contract not in (CONTRACT,CONTACT_HEIGHT_CONTRACT,COLLECTION_CONTRACT):
        raise ValueError('Unknown registered collection contract')
    trigger=case.get('dagger',{}).get('trigger')
    audit=None
    if result.get('legal'):
        if result['dagger'].get('configuration',{}).get('trigger')!=trigger:
            raise ValueError('Recorded takeover trigger differs from registered opportunity')
        if trigger is not None:
            audit=audit_trigger_case(directory,result)
    if contract==COLLECTION_CONTRACT:
        category=classify_case(case,result,audit)
        name='training_expansion_audit.json'
    else:
        category=classify(result,audit)
        name='stable_stage_audit.json' if contract==CONTRACT else 'contact_height_stage_audit.json'
    return name,category,audit


def admit_capture(output,case,result,pair,plan,entries):
    """Publish only after existing source/label/camera checks on closed files."""
    from visuotactile.state_benchmark import audit_dagger_capture
    from visuotactile.narrow_data import dagger_record
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.anchor_sampling import supervision_population
    directory=output/case['name']
    result['accepted_for_training']=False
    if result.get('legal'):
        # Admission adds fields to result.json. Keep the exact execution bytes
        # that the provenance audit actually reads, so its hash stays resolvable.
        with (directory/'result_before_admission.json').open('xb') as stream:
            stream.write((directory/'result.json').read_bytes())
        audit=audit_dagger_capture(directory,pair)
        if not audit['passed']:raise ValueError('Actual batch takeover provenance failed')
        audit['result_binding_path']='result_before_admission.json'
        atomic_json(directory/'capture_audit.json',audit)
        audit_name,category,stage_audit=classify_capture(directory,case,result,plan)
        if result['dagger']['expert_valid_ticks']:
            path=Path(result['dataset_capture']['path'])
            entry=dict(path=str(path.resolve()),episode=case['spec']['identity'],length=result['steps'],
                family=case['spec']['family'],split='train',source_sha256=sha256(path),
                trajectory_kind='expert_takeover_v1',source_group=case['role'],round_index=plan['round_index'],
                seed=case['spec']['seed'],takeover_tick=result['dagger']['recovery_coverage']['switch_tick'],
                registered_takeover_tick=case['spec']['takeover_tick'],trigger_receipt=result['dagger'].get('trigger_receipt'),
                frontend_sha256=pair['frontend_sha256'],backend_sha256=pair['backend_sha256'])
            if case['spec'].get('training_expansion') is not None:
                entry.update({key:copy.deepcopy(case['spec'][key]) for key in
                    ('training_expansion','initial_configuration_sha256','source_identity')})
            if 'expansion_schedule' in case:
                entry['expansion_schedule']=case['expansion_schedule']
            record,_,_=dagger_record(entry,include_geometry=False)
            result['actual_exposure']=supervision_population({entry['episode']:record})
            if result['actual_exposure']['supervised_ticks']!=result['dagger']['expert_valid_ticks']:
                raise ValueError('Expert suffix denominator differs')
            if result['success'] and result['dagger']['expert_takeover_success']:
                visual=inspect_capture(path);atomic_json(directory/'camera_acceptance.json',visual)
                if not visual['passed']:raise ValueError('Actual batch camera content invalid')
                entries[case['spec']['identity']]=entry;result['accepted_for_training']=True
    else:
        audit_name,category,stage_audit=classify_capture(directory,case,result,plan)
    causal=case.get('dagger',{}).get('trigger') is not None
    from visuotactile.takeover_trigger import HELD_NEAR_GOAL_VERSION,HELD_ROOT_GOAL_VERSION
    near_goal=(case.get('dagger',{}).get('trigger') or {}).get('version')==HELD_NEAR_GOAL_VERSION
    root_goal=(case.get('dagger',{}).get('trigger') or {}).get('version')==HELD_ROOT_GOAL_VERSION
    atomic_json(directory/audit_name,dict(category=category,causal_trigger=stage_audit,
        source_plan_sha256=sha256(output/'plan.json'),no_expert_fallback=True if causal else None,
        expert_schedule=('causal_held_root_near_goal_window' if root_goal else 'causal_held_near_goal_window' if near_goal else
            'causal_contact_window' if causal else 'preregistered_fixed_tick'),
        expansion_schedule=case.get('expansion_schedule'),training_expansion=case['spec'].get('training_expansion'),
        not_autonomous_success=bool(result.get('dagger',{}).get('expert_ticks',0))))
    atomic_json(directory/'result.json',result)
    atomic_json(output/'correction_entries.json',entries)


def run(output):
    import torch
    from tools.tight_state import environment
    from visuotactile.batched_rollout import waves,run_wave
    from visuotactile.state_resources import lease_for
    from visuotactile.models.native_vision import NativeDinoSpatial,validate_vision_microbatch,validate_vision_tail_profile
    from visuotactile.async_recording import validate_verification_parameters
    from visuotactile.native_evaluation import exit_completed_kit_worker
    output=Path(output).resolve();plan=read(output/'plan.json');session=read(output/'session.json')
    vision_microbatch=validate_vision_microbatch(plan['batch'].get('vision_microbatch',2))
    tail=validate_vision_tail_profile(plan['batch'].get('vision_tail_profile'))
    inactive=validate_inactive_render_request(plan['batch'].get('inactive_render_profile'),plan['execution_backend'])
    validate_verification_parameters(plan['batch'].get('verification_workers',0),
        plan['batch'].get('verification_max_pending',4))
    if plan.get('test_used') is not False or plan['batch']['version']!='whole_wave_complete_student_v1':
        raise ValueError('Explicit non-test batch registration required')
    if plan['kind']=='takeover':
        validate_takeover_plan(plan)
        for role,digest in plan['source_data_sha256'].items():
            if sha256(Path(plan['source_root'])/role/'state_data/data.json')!=digest:
                raise ValueError('Source data registration changed')
        if (sha256(output/'session.json')!=plan['source_session_sha256']
                or sha256(Path(plan['resource_root'])/'session.json')!=plan['source_session_sha256']):
            raise ValueError('Registered session bytes changed')
    for path,digest in plan['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Frozen batch source changed: '+path)
    for pair in plan['pairs'].values():
        for key in ('frontend','backend'):
            if sha256(pair[key])!=pair[key+'_sha256']:raise ValueError('Frozen full pair changed')
    with (output/'run_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),utc=time.time(),plan_sha256=sha256(output/'plan.json')),stream)
    lease=lease_for(Path(plan['resource_root']),deadline=session['deadline'])
    started=time.perf_counter()
    app,env,player,lease=environment(output,evaluation=True,lease=lease,num_envs=plan['batch']['lanes'],
        camera_content_validation=True,execution_backend=plan['execution_backend'])
    with inactive_render_scope(env,inactive,output) as guard:
        atomic_json(output/'initialization.json',dict(seconds=time.perf_counter()-started,lanes=env.num_envs,
            execution_backend=plan['execution_backend'],vision_microbatch=vision_microbatch,
            **({} if tail is None else dict(vision_tail_profile=tail)),
            **({} if inactive is None else dict(inactive_render_profile=inactive))))
        with lease.block():dino=NativeDinoSpatial().cuda().eval()
        saved={r:torch.load(pair['backend'],map_location='cpu',weights_only=False) for r,pair in plan['pairs'].items()}
        results=[];entries={};profiles=[]
        for index,cases in enumerate(waves(plan['cases'],plan['batch']['lanes'])):
            role=cases[0]['role'];pair=plan['pairs'][role]
            finished,profile=run_wave(env,player,lease,output,cases,pair,saved[role],dino,plan,index,
                **({} if guard is None else dict(render_guard=guard)))
            profile.update(admission_s=0.,admission_scope='Sum of sequential admit_capture calls after wave execution; includes provenance, source-label/camera checks and admission receipt writes; excluded from wave seconds')
            for case,result in finished:
                if result.get('legal') and (result.get('residual_ticks')!=result['steps'] or not result.get('software_complete')):
                    raise ValueError('Incomplete full-student execution')
                if plan['kind']=='takeover':
                    before_admission=time.perf_counter()
                    admit_capture(output,case,result,pair,plan,entries)
                    profile['admission_s']+=time.perf_counter()-before_admission
                results.append(dict(role=role,case=case['name'],result=result,
                    timing=dict(batch_wave=index,total_seconds=result.get('wall_profile',{}).get('through_result_s'),
                        interpretation='Lane lifetimes overlap; use batch wave total for throughput')))
                atomic_json(output/'results.json',results)
            profiles.append(dict(wave=index,role=role,**profile))
            atomic_json(output/'wave_profiles.json',profiles)
            print(json.dumps(dict(wave=index,role=role,profile=profile,accepted=len(entries))),flush=True)
        if len(results)!=len(plan['cases']):raise ValueError('Incomplete registered wave population')
    elapsed=time.perf_counter()-started
    atomic_json(output/'result.json',dict(software_complete=True,worker_error=None,cases=len(results),utc=time.time(),
        elapsed_seconds=elapsed,accepted_training_trajectories=len(entries),
        accepted_trajectories_per_hour=len(entries)*3600/elapsed,
        attempts_per_hour=len(results)*3600/elapsed,lanes=env.num_envs))
    exit_completed_kit_worker(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path)
    p.add_argument('--session-root',type=Path,help='Independent active resource session for explicit training takeovers')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lanes',type=int,default=8)
    p.add_argument('--count',type=int,default=8);p.add_argument('--seed',type=int,default=28142000)
    p.add_argument('--identities',nargs='+');p.add_argument('--delays',nargs='+',type=int);p.add_argument('--round',type=int,default=7)
    p.add_argument('--backend-json',type=Path);p.add_argument('--writer-mib',type=int,default=512)
    p.add_argument('--vision-microbatch',type=int,default=2)
    p.add_argument('--vision-exact-tail',action='store_true',help='Explicit candidate: no padding of the final DINO lane group')
    p.add_argument('--inactive-render-products',action='store_true',help='Explicit CPU-standard candidate: retire render products only after actual lane termination')
    p.add_argument('--verification-workers',type=int,default=0)
    p.add_argument('--verification-max-pending',type=int,default=4)
    a=p.parse_args()
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    if a.stage=='run' and a.inactive_render_products:
        p.error('run uses the immutable prepared inactive-render profile; register it with prepare')
    if a.stage=='run' and a.vision_exact_tail:
        p.error('run uses the immutable prepared vision profile; --vision-exact-tail requires prepare')
    if a.stage=='run' and a.session_root is not None:
        p.error('run uses the immutable registered session; --session-root requires prepare')
    if a.stage=='run':run(a.output)
    else:
        if a.root is None:p.error('--root is required for prepare')
        inactive=None
        if a.inactive_render_products:
            from visuotactile.inactive_render_products import PROFILE
            inactive=PROFILE
        prepare(a.root.resolve(),a.output.resolve(),lanes=a.lanes,count=a.count,seed=a.seed,
            identities=a.identities,delays=a.delays,round_index=a.round,
            execution_backend=read(a.backend_json) if a.backend_json else None,writer_bytes=a.writer_mib<<20,
            vision_microbatch=a.vision_microbatch,session_root=a.session_root.resolve() if a.session_root else None,
            verification_workers=a.verification_workers,verification_max_pending=a.verification_max_pending,
            vision_tail_profile=EXACT_TAIL_PROFILE if a.vision_exact_tail else None,inactive_render_profile=inactive)


if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException as error:
        traceback.print_exc()
        import sys
        # Driver may abort inside Kit; the ordinary owner still leaves its own
        # returncode and partial-case census. Python failures are explicit too.
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
