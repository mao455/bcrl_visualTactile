"""Frozen state diagnostics in one persistent Kit, with source history shadow comparisons."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch

from .compact import sha256
from .evaluate_baselines import RolloutVideo
from .evaluate_recorded_replay import run_trial
from .evaluate_training_initials import restore_observable_state
from .native_data import CAMERAS
from .native_evaluation import disable_omniclient_watches, make_env, stable_grasp
from .native_v2_resources import LeaseV2
from .native_v2_state import StateController, StateWindows, condition_windows
from .training_resources import atomic_json


def trajectory_errors(arrays,source,origin,metadata):
    n=min(len(arrays['command']),len(source['transition/command_clipped_pre_delay']))
    mapping={'command':'command_clipped_pre_delay','delayed':'control_trace/delayed_command',
        'target':'target_after_rad','joint':'diagnostics_after/joint_position_rad','object':'diagnostics_after/object_pose_w'}
    result={}
    for name,key in mapping.items():
        expected=np.asarray(source['transition/'+key][:n]).copy()
        if name=='object': expected[:,:3]+=np.asarray(origin)-np.asarray(metadata['env_origins_w'][0])
        diff=arrays[name][:n]-expected
        if name=='object':
            error=np.linalg.norm(diff[:,:3],axis=-1);threshold=.001
            result['object_position_m']=error
        else:
            error=np.sqrt(np.mean(diff**2,axis=-1));threshold=.1 if name in ('command','delayed') else .01
            result[name+'_arm_rmse']=np.sqrt(np.mean(diff[:,:7]**2,axis=-1))
            result[name+'_hand_rmse']=np.sqrt(np.mean(diff[:,7:]**2,axis=-1))
        result[name+'_first_exceeded']=int(np.flatnonzero(error>threshold)[0]) if np.any(error>threshold) else -1
    return result


@torch.no_grad()
def student_trial(env,lease,output,entry,identity,saved,checkpoint_hash,seed,cohort,source_root,video_enabled=False):
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'started.json',dict(identity=identity,checkpoint_sha256=checkpoint_hash,sampler_seed=seed,cohort=cohort,utc=time.time()))
    windows=StateWindows(source_root,[identity])
    with h5py.File(entry['path'],'r') as stream:
        source=stream['episodes'][entry['episode']];metadata=json.loads(stream.attrs['metadata_json'])
        with lease.block():
            env.clear_recorded_events();env.reset(seed=20261102)
            restoration=restore_observable_state(env,source,metadata)
            if max(restoration['saved_observable_errors'].values())>1e-4: raise ValueError('Initial observable mismatch')
            controller=StateController(saved,metadata);controller.generator.manual_seed(seed)
        atomic_json(output/'restoration.json',restoration)
        initial=float(source['transition/diagnostics_before/object_pose_w'][0,2])
        video=RolloutVideo(output/'student.mp4','S-NativeDP' if saved.get('contract',{}).get('diagnostic_model') is None else 'StateBC-diagnostic',
                           seed,saved.get('epoch',0),CAMERAS) if video_enabled else None
        if video: video.title=f"{cohort} | {identity} | sampler {seed} | ckpt {checkpoint_hash[:12]} | autonomous"
        rows=[];plans=[];teacher_plans=[];boundary=[];previous=None;planning=[]
        timing=dict(simulation_s=0.,inference_s=0.,shadow_inference_s=0.,video_s=0.,serialization_s=0.)
        started=time.monotonic()
        try:
            bound=(int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
            for tick in range(bound):
                packet=env.research_packet
                if video:
                    began=time.perf_counter();video.append(packet,tick);timing['video_s']+=time.perf_counter()-began
                with lease.block():
                    before=env._diagnostics()
                    rng=controller.generator.get_state() if tick%12==0 else None
                    began=time.perf_counter();command=controller.act(packet,before);torch.cuda.synchronize()
                    infer=time.perf_counter()-began;timing['inference_s']+=infer
                    if not torch.isfinite(command).all():raise FloatingPointError('Nonfinite native student command')
                    if tick%12==0:
                        planning.append(infer);plans.append(controller.plan[0].numpy().copy())
                        if previous is not None: boundary.append(float((command-previous).square().mean().sqrt()))
                        # Shadow uses only saved expert observations at the same tick, same noise.
                        # It never controls the robot and is explicitly teacher forced.
                        if tick//12<len(windows):
                            began=time.perf_counter();robot,geom,_,_=windows.batch([tick//12],controller.device)
                            cond=condition_windows(controller.model,robot,geom)
                            generator=torch.Generator(device=controller.device);generator.set_state(rng)
                            teacher=controller.model.predict(cond,generator).clamp(-1,1)[0].cpu().numpy()
                            torch.cuda.synchronize();timing['shadow_inference_s']+=time.perf_counter()-began
                        else: teacher=np.full((64,29),np.nan,dtype=np.float32)
                        teacher_plans.append(teacher)
                    began=time.perf_counter();_,_,terminated,truncated,_=env.step(command)
                    timing['simulation_s']+=time.perf_counter()-began
                    transition=env.last_transition;controller.observe_execution(transition);previous=command.clone()
                    after=transition['diagnostics_after']
                    def take(value):return value[0].detach().cpu().numpy().copy()
                    rows.append(dict(command=take(command),delayed=take(transition['control_trace']['delayed_command']),
                        delay_index=take(transition['control_trace']['delay_index']),target=take(transition['target_after_rad']),
                        target_before=take(transition['target_before_rad']),joint=take(after['joint_position_rad']),
                        object=take(after['object_pose_w']),palm=take(after['palm_body_pose_w']),
                        object_contact=take(transition['object_contact_force_w']),lifted=take(after['lifted_object']),
                        native_alignment=take(transition['native_alignment_condition']),
                        subgoals=take(after['successes']),native_keypoints=take(transition['native_keypoints_max_dist']),inference_s=infer))
                if tick%100==0:atomic_json(output/'progress.json',dict(tick=tick,utc=time.time()))
                if bool(terminated[0] or truncated[0]):break
            reasons={k:bool(v[0]) for k,v in transition['termination_reasons'].items()}
            if not any(reasons.values()):raise RuntimeError('No native termination at safety bound')
            arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
            errors=trajectory_errors(arrays,source,env.scene.env_origins[0].tolist(),metadata)
            began=time.perf_counter()
            np.savez_compressed(output/'trace.npz',**arrays)
            np.savez_compressed(output/'expert_comparison.npz',**errors)
            np.savez_compressed(output/'plans.npz',student=np.stack(plans),expert_history_shadow=np.stack(teacher_plans))
            timing['serialization_s']+=time.perf_counter()-began
            result=dict(identity=identity,cohort=cohort,checkpoint_sha256=checkpoint_hash,updates=saved['updates'],
                sampler_seed=seed,software_complete=True,native_termination=True,success=reasons['max_successes'],
                reasons=reasons,steps=len(rows),native_lift=bool(arrays['lifted'].any()),native_subgoals=int(arrays['subgoals'].max()),
                native_alignment=bool(arrays['native_alignment'].any()),contact=stable_grasp(arrays,initial),
                first_divergence={k:int(v) for k,v in errors.items() if k.endswith('first_exceeded')},
                divergence_thresholds=dict(command_rmse=.1,joint_rmse_rad=.01,target_rmse_rad=.01,object_position_m=.001),
                planning_p50_p95_max_s=[float(np.percentile(planning,q)) for q in (50,95,100)],
                planning_over_200ms=int(np.sum(np.asarray(planning)>.2)),chunk_boundary_rmse=boundary,
                timing=timing,wall_s=time.monotonic()-started,exact_historical_physics_restoration=False,
                comparison_scope='Same observable initial fields; missing simulator state. Shadow expert history is offline teacher forcing, never autonomous success.')
            if video:video.close(complete=True);result['video']=str((output/'student.mp4').resolve())
            atomic_json(output/'result.json',result)
            return result
        finally:
            if video:video.close()


def run(root,output,job_path):
    output.mkdir(parents=True,exist_ok=False)
    session=json.loads((root/'session.json').read_text());job=json.loads(job_path.read_text())
    source_root=Path(session['source_root']);data=json.loads((source_root/'data.json').read_text())
    lease=LeaseV2(root,deadline=session['deadline']);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    atomic_json(output/'runtime_controls.json',disable_omniclient_watches())
    atomic_json(output/'kit_lifecycle.json',dict(phase='initializing',utc=time.time()))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env=None;results=[]
    try:
        import isaacsimenvs
        env=make_env(True,lease)
        atomic_json(output/'kit_lifecycle.json',dict(phase='environment_ready',utc=time.time()))
        saved={}
        for spec in job['trials']:
            identity=spec['identity'];entry=data['entries'][identity];trial=output/spec['name']
            if trial.exists():raise RuntimeError('Trial already attempted; never silently repeat')
            if spec['kind']=='replay':
                video=RolloutVideo(trial.parent/(spec['name']+'.mp4'),'expert native-command replay',spec['seed'],0,CAMERAS) if spec.get('video') else None
                original=env.step
                if video:
                    video.title='Recorded expert commands | '+identity+' | historical replay outcome unselected'
                    def wrapped(command):
                        video.append(env.research_packet,int(env.research_packet['step_id'][0]));return original(command)
                    env.step=wrapped
                try:
                    with h5py.File(entry['path'],'r') as stream:
                        r=run_trial(env,lease,trial,stream['episodes'][entry['episode']],json.loads(stream.attrs['metadata_json']),
                            mode='delay_wrench_fixed',line='V2_historical_replay',controller=None,seed=spec['seed'],
                            wrench_reconstruction='native_request')
                    if video:video.close(complete=True);r['video']=str((trial.parent/(spec['name']+'.mp4')).resolve())
                finally:
                    env.step=original
                    if video:video.close()
                r.update(identity=identity,software_complete=True)
            else:
                checkpoint=Path(spec['checkpoint'])
                if sha256(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint identity changed')
                if str(checkpoint) not in saved:saved[str(checkpoint)]=torch.load(checkpoint,map_location='cpu',weights_only=False)
                r=student_trial(env,lease,trial,entry,identity,saved[str(checkpoint)],spec['checkpoint_sha256'],spec['seed'],
                                spec['cohort'],source_root,spec.get('video',False))
            results.append(dict(**r,trial_name=spec['name'],kind=spec['kind']))
            atomic_json(output/'result.json',dict(software_complete=len(results)==len(job['trials']),trials=results,
                        completed=len(results),required=len(job['trials'])))
        atomic_json(output/'environment.json',dict(metadata=env.research_metadata(),config=env.cfg.to_dict(),
                    physical_initial_restore='observable only',parallel_envs=1))
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True);a=p.parse_args();run(a.root,a.output,a.job)
