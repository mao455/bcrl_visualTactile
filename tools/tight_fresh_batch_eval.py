"""Finite multi-environment complete-student development and actual throughput."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read, completed_pair


def prepare(root, output, count, seed, *, neural_batch=False):
    import subprocess
    from visuotactile.narrow_data import direct_configuration
    if not 1 <= count <= 4:raise ValueError('One to four registered lanes')
    if output.exists():raise FileExistsError('Preserve previous batch identity')
    session=read(root/'session.json')
    pairs={role:completed_pair(root,role) for role in ['A','B']}
    cases=[dict(identity=f'batch_development_{seed+i}',seed=seed+i,
        narrow_reset=direct_configuration(session['domain'],seed+i)) for i in range(count)]
    output.mkdir(parents=True)
    atomic_json(output/'session.json',session)
    atomic_json(output/'plan.json',dict(pairs=pairs,cases=cases,seed=seed,event_seed=seed+100000,
        resource_root=str(root),test_used=False,kind='engineering_batch_development',neural_batch=neural_batch,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ['tools','visuotactile'] for p in Path(folder).rglob('*.py')},
        limitation='Native batched event RNG differs from serial; same registered opportunities are not exact hidden-state pairs'))


def run(output):
    import numpy as np
    import torch
    from tools.tight_state import environment
    from visuotactile.state_resources import lease_for
    from visuotactile.batched_controller import BatchedResidualController
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.narrow_data import configure_direct_reset_lanes
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.live_collision import clearance
    from visuotactile.native_evaluation import stable_grasp,exit_completed_kit_worker
    from visuotactile.recording import BufferedNumericWriter
    import h5py
    plan=read(output/'plan.json');session=read(output/'session.json')
    for path,digest in plan['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Frozen source changed: '+path)
    if (output/'started.json').exists():raise ValueError('No duplicate starts')
    atomic_json(output/'started.json',dict(pid=os.getpid(),utc=time.time(),plan_sha256=sha256(output/'plan.json')))
    lease=lease_for(Path(plan['resource_root']),deadline=session['deadline']);count=len(plan['cases'])
    began=time.perf_counter();app,env,player,lease=environment(output,evaluation=True,lease=lease,num_envs=count)
    atomic_json(output/'startup_timing.json',dict(seconds=time.perf_counter()-began))
    configure_direct_reset_lanes(env,[c['narrow_reset'] for c in plan['cases']])
    isolated=IsolatedEnvironment(env,plan['seed'])
    with lease.block():dino=NativeDinoSpatial().cuda().eval()
    all_results=[]
    for role,pair in plan['pairs'].items():
        for key in ['frontend','backend']:
            if sha256(pair[key])!=pair[key+'_sha256']:raise ValueError('Frozen pair changed')
        directory=output/role;directory.mkdir()
        with lease.block():
            env.action_control_mode='native_command_clipped_pre_delay'
            teacher,_=isolated.reset(plan['seed']);isolated.begin_events(plan['event_seed'])
            env.action_control_mode='absolute_joint_position_target_rad'
            initial=env._diagnostics()
            atomic_json(directory/'initial_observable.json',{k:v.cpu().tolist() for k,v in initial.items()})
            admissions=[clearance(env,lane=i) for i in range(count)]
            atomic_json(directory/'initial_collision.json',admissions)
            active=[row['legal'] for row in admissions]
            saved=torch.load(pair['backend'],map_location='cpu',weights_only=False)
            metadata=env.research_metadata();metadata['config']=env.cfg.to_dict()
            controller=BatchedResidualController(saved,metadata,dino,count,neural_batch=plan.get('neural_batch',False))
        rows=[[] for _ in range(count)];outcomes=[None]*count
        files=[h5py.File(directory/f'lane_{i}_numeric.h5','w') for i in range(count)]
        writers=[BufferedNumericWriter(f.create_group('episode'),32) for f in files]
        profile=dict(controller_s=0.,environment_s=0.,recording_s=0.,resource_wait_s=0.)
        began=time.perf_counter();prior_wait=lease.wait_s
        try:
            bound=(int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
            for tick in range(bound):
                if not any(active):break
                packet=env.research_packet
                with lease.block():
                    start=time.perf_counter();command=controller.act(packet,env._diagnostics(),active)
                    profile['controller_s']+=time.perf_counter()-start
                    start=time.perf_counter();teacher,_,terminated,truncated,_=isolated.step(command)
                    profile['environment_s']+=time.perf_counter()-start
                    transition=env.last_transition;controller.observe_execution(transition,active)
                    start=time.perf_counter()
                    for lane,running in enumerate(active):
                        if not running:continue
                        from visuotactile.batched_controller import lane_tree
                        numeric=dict(obs={k:v for k,v in packet.items() if torch.is_tensor(v) and k not in ['image','depth_m','depth_valid']},transition=transition)
                        writers[lane].append(lane_tree(numeric,lane,count),len(rows[lane]))
                        after=transition['diagnostics_after']
                        take=lambda v:v[lane].cpu().numpy().copy()
                        rows[lane].append(dict(command=take(command),target=take(transition['target_after_rad']),
                            joint=take(after['joint_position_rad']),object=take(after['object_pose_w']),palm=take(after['palm_body_pose_w']),
                            object_contact=take(transition['object_contact_force_w']),lifted=take(after['lifted_object']),
                            alignment=take(transition['native_alignment_condition']),subgoals=take(after['successes'])))
                        reasons={key:bool(value[lane]) for key,value in transition['termination_reasons'].items()}
                        if reasons['max_successes'] and not bool(terminated[lane]):raise ValueError('Success/termination disagreement')
                        if bool((terminated|truncated)[lane]):
                            active[lane]=False;outcomes[lane]=reasons
                    profile['recording_s']+=time.perf_counter()-start
                if tick%100==0:atomic_json(directory/'progress.json',dict(tick=tick,active=active,utc=time.time()))
            if any(active):raise ValueError('Native natural-termination bound exhausted')
        finally:
            for writer,f in zip(writers,files):writer.flush();f.close()
        profile.update(total_s=time.perf_counter()-began,resource_wait_s=lease.wait_s-prior_wait,
            steps=sum(map(len,rows)),scope=('Current-frame DINO, DDIM and backend batches; per-lane histories and generators' if plan.get('neural_batch',False) else 'current-frame DINO batches; DP and backend still per lane; native dt unchanged'))
        profile['aggregate_control_ticks_per_second']=profile['steps']/profile['total_s']
        results=[]
        for lane in range(count):
            result=dict(lane=lane,identity=plan['cases'][lane]['identity'],legal=admissions[lane]['legal'],steps=len(rows[lane]),
                success=bool(outcomes[lane] and outcomes[lane]['max_successes']),reasons=outcomes[lane],pair=pair)
            if rows[lane]:
                arrays={key:np.stack([row[key] for row in rows[lane]]) for key in rows[lane][0]}
                np.savez_compressed(directory/f'lane_{lane}_trace.npz',**arrays)
                result.update(native_lift=bool(arrays['lifted'].any()),native_alignment=bool(arrays['alignment'].any()),
                    native_subgoals=int(arrays['subgoals'].max()),contact=stable_grasp(arrays,float(initial['object_pose_w'][lane,2])),
                    visual_frames=controller.visual_frames[lane],position_target_max_error=float(np.abs(arrays['command']-arrays['target']).max()))
                if result['visual_frames']!=result['steps']:raise ValueError('Dropped native visual frame')
                atomic_json(directory/f'lane_{lane}_inference.json',controller.inference_metrics(lane))
            results.append(result)
        atomic_json(directory/'result.json',dict(software_complete=True,results=results,profile=profile))
        all_results.append(dict(role=role,results=results,profile=profile))
        print(json.dumps(dict(role=role,profile=profile,successes=sum(r['success'] for r in results))),flush=True)
    atomic_json(output/'result.json',dict(software_complete=True,results=all_results,test_used=False))
    exit_completed_kit_worker(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=2);p.add_argument('--seed',type=int,default=28141000)
    p.add_argument('--neural-batch',action='store_true')
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.root.resolve(),args.output.resolve(),args.count,args.seed,neural_batch=args.neural_batch)
    else:run(args.output.resolve())


if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        traceback.print_exc()
        import sys
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
