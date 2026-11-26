"""Bounded, non-integrating reset/query diagnosis on the installed SDK."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from tools.runtime import CODE, RUN, read, write, kill_owned


def kinematic_reset_check(root, env):
    """Two resets and native goal-pose writes, with no control/physics steps.

    Advancing the goal index here only exercises the pose writer. It does not
    produce a task outcome, a recovered trajectory, or a new evaluation trial.
    """
    import torch
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics
    from isaaclab.utils.math import quat_apply, quat_mul
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.narrow_data import configure_direct_reset
    source = CODE/'outputs/visuotactile/run/tight_residual_20261110_01/G11/evaluation/palm_fresh_000/result.json'
    spec = read(source)
    configure_direct_reset(env, spec['narrow_reset'])
    bodies = {}
    for name in ('hole', 'goal_viz'):
        body = getattr(env, name)
        rigid = [p for start in sim_utils.find_matching_prims(body.cfg.prim_path)
                 for p in Usd.PrimRange(start) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(rigid) != 1 or not rigid[0].GetAttribute('physics:kinematicEnabled').Get():
            raise RuntimeError('Expected exactly one kinematic rigid body: '+name)
        bodies[name] = str(rigid[0].GetPath())
    started_steps = int(env._sim_step_counter)
    started_time = float(env.sim.current_time)
    rows = []
    previous_reset = previous_control = None
    for reset_index in range(2):
        IsolatedEnvironment(env, spec['seed']).reset(spec['seed'])
        reset_pose = {name: getattr(env, name).data.root_state_w[:, :7].clone()
                      for name in ('hole', 'goal_viz')}
        reset_control = {name: getattr(env, name).clone() for name in ('_prev_targets', '_cur_targets')}
        if previous_reset is not None and any(not torch.equal(reset_pose[k], previous_reset[k]) for k in reset_pose):
            raise RuntimeError('Same configured reset changed kinematic placement')
        if previous_control is not None and any(not torch.equal(reset_control[k], previous_control[k]) for k in reset_control):
            raise RuntimeError('Same configured reset changed native actuator targets')
        previous_reset = reset_pose
        previous_control = reset_control
        if env._num_prelude_goals != 0 or bool(env.is_random_goal_env.any()):
            raise RuntimeError('This focused check requires the existing two-goal insertion configuration')
        saved_successes = env._successes.clone()
        ids = torch.arange(env.num_envs, device=env.device)
        try:
            for goal in range(env._num_total_insertion_goals):
                env._successes[:] = goal
                env._write_goal_pose(ids)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(0.)
                expected = torch.cat((env.hole_pos + env.scene.env_origins +
                    quat_apply(env.hole_quat_wxyz, env._insert_pos_rel[goal].expand(env.num_envs, -1)),
                    quat_mul(env.hole_quat_wxyz, env._insert_quat_wxyz[goal].expand(env.num_envs, -1))), -1)
                raw_pose = env.goal_viz.root_physx_view.get_transforms().clone()
                actual = torch.cat((raw_pose[:, :3], raw_pose[:, 6:7], raw_pose[:, 3:6]), -1)
                error = float((actual-expected).abs().max())
                if error > 1e-6:
                    raise RuntimeError('Native goal placement differs from declared pose: '+str(error))
                if not torch.equal(env.hole.data.root_state_w[:, :7], reset_pose['hole']):
                    raise RuntimeError('Goal writing moved the hole')
                rows.append(dict(reset=reset_index, goal=goal, max_pose_error=error,
                    hole_pose=reset_pose['hole'].tolist(), goal_pose=actual.tolist()))
        finally:
            env._successes[:] = saved_successes
            env._write_goal_pose(ids)
    physics_steps = int(env._sim_step_counter)-started_steps
    physics_elapsed_s = float(env.sim.current_time)-started_time
    if physics_steps != 0 or physics_elapsed_s != 0.:
        raise RuntimeError('A non-integrating reset check advanced physics')
    write(root/'result.json', dict(complete=True, mode='kinematic_reset_only', resets=2,
        goal_pose_checks=len(rows), control_steps=0, physics_steps=physics_steps,
        physics_elapsed_s=physics_elapsed_s, body_paths=bodies, rows=rows,
        repeated_reset_native_targets_exact=True,
        seed=spec['seed'], source_spec=str(source),
        scope='API/placement acceptance only; temporary goal-index writes are diagnostic, not successful subgoals or historical physical replay'))


def worker(root, *, kinematic_reset_only=False):
    from visuotactile.native_evaluation import client_startup_controls, make_env
    from visuotactile.state_resources import lease_for
    write(root/'startup.json', client_startup_controls('kit_managed'))
    lease=lease_for(root)
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',fast_shutdown=False,
            kit_args='--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 --/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false').app
    import isaacsimenvs
    import numpy as np
    import omni.physx
    from visuotactile.state_benchmark import prepare, collision_evidence
    from visuotactile.environment_rng import IsolatedEnvironment
    env=make_env(True,lease,scene_queries=True,expert_speed=None if kinematic_reset_only else dict(kind='closed_loop',arm=1.,hand=1.,phase='all'))
    if kinematic_reset_only:
        with lease.block():
            kinematic_reset_check(root, env)
            from visuotactile.state_image_loss import live_geometry_probe
            geometry = live_geometry_probe(env)
            write(root/'geometry_live.json', geometry)
            write(root/'geometry_acceptance.json', dict(geometry,passed=geometry['live_collision_parity']))
        # The existing isolated evaluation worker uses OS cleanup after durable
        # results. Kit shutdown itself stalled in the preceding bounded probe;
        # do not label this path as successful SDK/application teardown.
        result=read(root/'result.json');result['software_complete']=True
        write(root/'result.json',result)
        from visuotactile.native_evaluation import exit_completed_kit_worker
        exit_completed_kit_worker(root)
    from visuotactile.expert import create_expert
    from visuotactile.state_benchmark import trial
    checkpoint=str(CODE/'pretrained_assembly/tight_insertion/model.pth')
    with lease.block():player,_=create_expert(env,checkpoint)
    families=read(CODE/'outputs/visuotactile/run/narrow_state_20261105_01/families.json')
    ids=['r931005_1793209376_e01_p000144','r931005_1793209376_e00_p000094']
    entries={f['identity']:f['entry'] for f in families}
    rows=[]
    for i,identity in enumerate([ids[0],ids[1],ids[0],ids[1],ids[0],ids[0]]):
        out=root/f'reset_{i}';out.mkdir()
        env.speed_config.update(arm=1. if i<2 else .58,hand=1. if i<2 else .58)
        isolated=IsolatedEnvironment(env,24110000+i)
        with lease.block():
            prepare(env,isolated,dict(seed=24110000+i,cohort='historical',identity=identity),dict(entries=entries),out)
            initial={k:v.clone() for k,v in env._diagnostics().items()}
            def capture(label):
                evidence=collision_evidence(env)
                from visuotactile.live_collision import clearance
                geometry=clearance(env)
                tensor=env.robot.root_physx_view.get_link_transforms()[0].clone().cpu().numpy()
                rigid={}
                paths=env.robot.root_physx_view.link_paths[0]
                for path,pose in zip(paths,tensor):
                    actual=omni.physx.get_physx_interface().get_rigidbody_transformation(path)
                    rigid[path]=dict(tensor=pose.tolist(),physx={k:list(v) if hasattr(v,'__len__') else v for k,v in actual.items()})
                row=dict(reset=i,identity=identity,label=label,collision=evidence,mesh_clearance=geometry,rigid=rigid,
                    observable_change={k:float((v.double()-initial[k].double()).abs().max()) for k,v in env._diagnostics().items()})
                write(out/(label+'.json'),row);rows.append(row)
            capture('baseline')
            omni.physx.get_physx_simulation_interface().flush_changes()
            capture('flush')
            env.sim.physics_sim_view.update_articulations_kinematic()
            capture('kinematic')
            omni.physx.get_physx_interface().update_transformations(False,True,True)
            capture('usd_output')
            env.robot.write_root_pose_to_sim(env.robot.data.root_pose_w.clone())
            env.sim.forward();env.scene.update(0.)
            capture('root_refresh')
        if i in (0,1):
            result=trial(env,lease,root/f'rollout_{i}',dict(seed=24110000+i,cohort='historical',identity=identity),
                dict(entries=entries),player,checkpoint,expert_speed=env.speed_config)
            if not result.get('success'):raise RuntimeError('Bounded native diagnostic did not succeed')
    write(root/'result.json',dict(complete=True,resets=6,physics_steps=0,
        rows=[dict(reset=r['reset'],label=r['label'],legal=r['collision']['legal'],hits=r['collision']['external_hits']) for r in rows]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--kinematic-reset-only',action='store_true',help='One Kit, two resets, native goal-pose writes, zero control/physics steps')
    a=p.parse_args();root=a.root.resolve()
    if a.worker:
        try:worker(root,kinematic_reset_only=a.kinematic_reset_only)
        except BaseException:
            write(root/'worker_error.json',dict(traceback=traceback.format_exc()));sys.stdout.flush();sys.stderr.flush();os._exit(1)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    root.mkdir(parents=True,exist_ok=False)
    write(root/'session.json',dict(resource_policy='full_gpu_v1' if a.kinematic_reset_only else 'shared_recent_v1',resource_target_percent=None if a.kinematic_reset_only else 40.,
        resources=dict(min_gpu_free_mib=6144,min_ram_gib=14,min_disk_gib=60),reserve_bytes=60*1024**3,training_deadline=time.time()+900))
    from tools.kit_resource_preflight import inspect
    pre=inspect(parent=root);write(root/'inotify.json',pre)
    if not pre['resource_admitted']:raise RuntimeError('Inotify preflight failed')
    monitor=child=None
    try:
        with (root/'monitor.log').open('w') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'worker.log').open('w') as log:
            child=subprocess.Popen(RUN+['-m','tools.reset_query_probe','--worker','--root',str(root)]+(['--kinematic-reset-only'] if a.kinematic_reset_only else []),cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait(timeout=900)
        write(root/'exit.json',dict(returncode=code,ended=time.time()))
        if a.kinematic_reset_only:
            errors=[line for line in (root/'worker.log').read_text(errors='replace').splitlines() if 'PhysX error:' in line]
            result=read(root/'result.json') if (root/'result.json').exists() else {}
            census=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            gpu_pids={int(x.strip()) for x in census.stdout.splitlines() if x.strip().isdigit()}
            reaped=not Path('/proc',str(child.pid)).exists()
            gpu_absent=census.returncode==0 and child.pid not in gpu_pids
            passed=(code==0 and result.get('complete') is True and result.get('software_complete') is True
                and not errors and (root/'process_exit_requested.json').exists() and reaped and gpu_absent)
            write(root/'acceptance.json',dict(passed=passed,worker_returncode=code,physx_errors=errors,
                worker_pid=child.pid,worker_reaped=reaped,worker_gpu_absent=gpu_absent,
                sdk_graceful_shutdown=False,exit_method='existing isolated-worker durable-results OS exit',
                source_result='result.json',scope='Entire new worker log; old162 errors remain preserved'))
            if not passed:
                raise RuntimeError('Kinematic reset acceptance failed; inspect original worker log/result')
    finally:
        if child:kill_owned(child)
        (root/'STOP_MONITOR').touch()
        if monitor:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(monitor)


if __name__=='__main__':main()
