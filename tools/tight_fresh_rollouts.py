"""Complete fresh policies: registered development or actual expert takeovers."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def completed_pair(root, role):
    import torch
    from visuotactile.state_benchmark import dagger_policy_identity
    base = root / role / 'backend' / role
    path = base / 'training/best.pt'
    finished = read(base / 'training/finished.json')
    offline = read(base / 'offline/result.json')
    if not finished.get('target_reached') or finished.get('disposable'):
        raise ValueError('Backend training budget is incomplete')
    if (not offline.get('complete') or offline.get('test_used') is not False
            or offline.get('checkpoint_sha256') != sha256(path)):
        raise ValueError('Selected complete pair lacks matching non-test offline evidence')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['contract'].get('visual_stride') != 1:
        raise ValueError('Every control frame required')
    return dagger_policy_identity(saved, str(path))


def prepare(root, output, count, seed):
    from visuotactile.narrow_data import direct_configuration
    if output.exists():
        raise FileExistsError('Existing rollout identity must be preserved')
    if not 1 <= count <= 32:
        raise ValueError('Register between one and 32 development opportunities per role')
    session=read(root/'session.json')
    pairs={role:completed_pair(root,role) for role in ('A','B')}
    cases=[]
    for index in range(count):
        s=seed+index
        spec=dict(identity=f'development_{s}',family=f'fresh_development_{s}',split='development',
            seed=s,event_seed=s+100000,cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(session['domain'],s))
        # Alternate order across cases; same registered reset/event seed.
        for role in (('A','B') if index%2==0 else ('B','A')):
            cases.append(dict(name=role+'_'+spec['identity'],role=role,spec=spec,video=index==0))
    output.mkdir(parents=True)
    atomic_json(output/'session.json',session)
    atomic_json(output/'plan.json',dict(kind='autonomous',pairs=pairs,cases=cases,
        resource_root=str(root),source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools'] for p in sorted(Path(folder).rglob('*.py'))},
        inference_scheduler_index_mode='host',
        test_used=False,cohort='registered development reset opportunities; same narrow domain; no exact hidden-state pairing claim',utc=time.time()))


def run(output):
    import torch
    from tools.tight_state import environment
    from visuotactile.state_resources import lease_for
    from visuotactile.native_evaluation import exit_completed_kit_worker
    from visuotactile.state_benchmark import trial
    plan=read(output/'plan.json');session=read(output/'session.json')
    for path,digest in plan['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Frozen rollout source changed: '+path)
    if (output/'run_started.json').exists():raise ValueError('No duplicate rollout worker')
    atomic_json(output/'run_started.json',dict(pid=os.getpid(),utc=time.time(),plan_sha256=sha256(output/'plan.json')))
    saved={}
    for role,pair in plan['pairs'].items():
        for key in ('frontend','backend'):
            if sha256(pair[key])!=pair[key+'_sha256']:raise ValueError('Policy pair changed')
        saved[role]=torch.load(pair['backend'],weights_only=False,map_location='cpu')
    lease=lease_for(Path(plan['resource_root']),deadline=session['deadline'])
    began=time.perf_counter();app,env,player,lease=environment(output,evaluation=True,lease=lease)
    atomic_json(output/'initialization.json',dict(seconds=time.perf_counter()-began))
    totals=defaultdict(float)
    def wrap(obj,name,key):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            t=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:totals[key]+=time.perf_counter()-t
        setattr(obj,name,call)
    wrap(env.sim,'step','physics_step_s');wrap(env.sim,'render','render_s');wrap(env.scene,'update','scene_update_s')
    results=[];entries={}
    for case in plan['cases']:
        pair=plan['pairs'][case['role']];before=dict(totals);start=time.perf_counter();wait=lease.wait_s
        directory=output/case['name']
        source=read(Path(case['source_data'])) if case.get('source_data') else {}
        row=trial(env,lease,directory,case['spec'],source,player,pair['backend'],saved=saved[case['role']],
            position=True,video=case.get('video',False),expert_checkpoint=session['expert_checkpoint'],
            expert_shadow=False,capture_dataset=plan['kind']=='takeover',dagger=case.get('dagger'),
            scheduler_index_mode=plan.get('inference_scheduler_index_mode','legacy'),
            video_title=case['role']+' fresh every-frame complete student')
        timing=dict(total_seconds=time.perf_counter()-start,resource_wait_seconds=lease.wait_s-wait,
            nested_environment_components={k:totals[k]-before.get(k,0) for k in totals},
            nesting='physics/render/scene are inside environment wall_profile; do not sum overlapping timings')
        atomic_json(directory/'runtime_profile.json',timing)
        if row.get('legal') and (row.get('residual_ticks')!=row['steps'] or not row.get('software_complete')):
            raise ValueError('Incomplete full-student control')
        if plan['kind']=='takeover' and row.get('legal') and row.get('software_complete'):
            from visuotactile.state_benchmark import audit_dagger_capture
            from visuotactile.narrow_data import dagger_record
            from visuotactile.camera_validity import inspect_capture
            from visuotactile.anchor_sampling import supervision_population
            audit=audit_dagger_capture(directory,pair)
            if not audit['passed']:raise ValueError('Actual takeover provenance failed')
            atomic_json(directory/'capture_audit.json',audit)
            row['accepted_for_training']=False
            if row['dagger']['expert_valid_ticks']:
                path=Path(row['dataset_capture']['path'])
                entry=dict(path=str(path.resolve()),episode=case['spec']['identity'],length=row['steps'],
                    family=case['spec']['family'],split='train',source_sha256=sha256(path),
                    trajectory_kind='expert_takeover_v1',source_group=case['role'],round_index=plan['round_index'],
                    seed=case['spec']['seed'],takeover_tick=row['dagger']['recovery_coverage']['switch_tick'],
                    registered_takeover_tick=case['spec']['takeover_tick'],
                    trigger_receipt=row['dagger'].get('trigger_receipt'),
                    frontend_sha256=pair['frontend_sha256'],backend_sha256=pair['backend_sha256'])
                record,_,_=dagger_record(entry,include_geometry=False)
                row['actual_exposure']=supervision_population({entry['episode']:record})
                if row['actual_exposure']['supervised_ticks']!=row['dagger']['expert_valid_ticks']:
                    raise ValueError('Expert suffix denominator differs')
                if row['success'] and row['dagger']['expert_takeover_success']:
                    visual=inspect_capture(path)
                    atomic_json(directory/'camera_acceptance.json',visual)
                    if not visual['passed']:raise ValueError('Actual correction camera content invalid')
                    entries[case['spec']['identity']]=entry;row['accepted_for_training']=True
            atomic_json(output/'correction_entries.json',entries)
        results.append(dict(role=case['role'],case=case['name'],result=row,timing=timing))
        atomic_json(output/'results.json',results)
        print(json.dumps(dict(role=case['role'],case=case['name'],legal=row['legal'],steps=row.get('steps'),
            success=row.get('success'),lift=row.get('native_lift'),subgoals=row.get('native_subgoals'),
            stable_grasp=row.get('contact',{}).get('stable_grasp_diagnostic'),seconds=timing['total_seconds'])),flush=True)
    atomic_json(output/'result.json',dict(software_complete=True,worker_error=None,cases=len(results),utc=time.time()))
    exit_completed_kit_worker(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=4);p.add_argument('--seed',type=int,default=28140000)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.root.resolve(),args.output.resolve(),args.count,args.seed)
    else:run(args.output.resolve())


if __name__=='__main__':
    try:main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        import sys
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
