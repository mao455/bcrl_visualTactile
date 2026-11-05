"""Bounded, non-integrating reset/query diagnosis on the installed SDK."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from tools.runtime import CODE, RUN, read, write, kill_owned


def worker(root):
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
    env=make_env(True,lease,scene_queries=True,expert_speed=dict(kind='closed_loop',arm=1.,hand=1.,phase='all'))
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
    a=p.parse_args();root=a.root.resolve()
    if a.worker:
        try:worker(root)
        except BaseException:
            write(root/'worker_error.json',dict(traceback=traceback.format_exc()));sys.stdout.flush();sys.stderr.flush();os._exit(1)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    root.mkdir(parents=True,exist_ok=False)
    write(root/'session.json',dict(resource_policy='shared_recent_v1',resource_target_percent=40.,training_deadline=time.time()+900))
    from tools.kit_resource_preflight import inspect
    pre=inspect(parent=root);write(root/'inotify.json',pre)
    if not pre['resource_admitted']:raise RuntimeError('Inotify preflight failed')
    monitor=child=None
    try:
        with (root/'monitor.log').open('w') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'worker.log').open('w') as log:
            child=subprocess.Popen(RUN+['-m','tools.reset_query_probe','--worker','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait(timeout=900)
        write(root/'exit.json',dict(returncode=code,ended=time.time()))
    finally:
        if child:kill_owned(child)
        (root/'STOP_MONITOR').touch()
        if monitor:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(monitor)


if __name__=='__main__':main()
