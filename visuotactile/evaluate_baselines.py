"""Paired B0/B1 command-controller evaluation on prespecified fresh legal resets.

One rendered CPU physics instance, unchanged native action chain. Expert
reference requires explicit development mode. Invalid starts precede actions.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from .training_resources import atomic_json, ResourceGuard


def checkpoint_identity(path, variant, development=False):
    """Bind evaluation to the completed validation-selected checkpoint, not last."""
    from .compact import sha256
    import torch
    path=Path(path)
    digest=sha256(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if sha256(path)!=digest:
        raise RuntimeError('Checkpoint changed during evaluation admission')
    if saved.get('variant')!=variant:
        raise ValueError('Wrong checkpoint variant')
    if not development:
        report=json.loads((path.parent/'report.json').read_text())
        selected=Path(report['checkpoint']).resolve()
        if (not report.get('training_complete') or report.get('variant')!=variant
                or path.resolve()!=selected or path.name!='best.pt'
                or report['selected_epoch']!=saved['epoch']
                or report['manifest_sha256']!=saved['manifest_sha256']
                or saved.get('diagnostic_only')):
            raise ValueError('Evaluation requires the completed validation-selected best identity')
    return dict(variant=variant,epoch=saved['epoch'],sha256=digest,
                manifest_sha256=saved['manifest_sha256'],path=str(path.resolve()))


def publish_report(output, report):
    """Only a complete cohort publishes the runner's completion marker.

    Kit shutdown may mask a nonzero Python exit, and the already-running
    supervisor checks report existence. Preserve incomplete evidence separately.
    """
    output=Path(output)
    if not report['complete']:
        atomic_json(output/'incomplete_report.json',report)
        atomic_json(output/'status.json',dict(state='incomplete',report=report))
        raise RuntimeError('Candidate budget exhausted before required legal pairs; incomplete report retained')
    atomic_json(output/'report.json',report)
    atomic_json(output/'status.json',dict(state='complete',report=report))


class RolloutVideo:
    """Bounded, CPU-encoded viewing derivative of every pre-action RGB packet."""
    def __init__(self, path, variant, seed, epoch, camera_ids):
        import av
        from PIL import ImageFont
        self.path=Path(path)
        self.partial=self.path.with_suffix('.mp4.partial')
        self.container=av.open(str(self.partial),'w',format='mp4')
        self.stream=None;self.frames=0
        self.title=f'{variant.upper()} | epoch {epoch} | seed {seed} | rerun for viewing | 1x simulation time'
        self.camera_ids=camera_ids
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)

    def append(self, packet, step, *, status=None):
        import av
        import numpy as np
        from PIL import Image,ImageDraw
        rgb=np.rint(packet['image'][0,:,:3].cpu().numpy().clip(0,1)*255).astype('uint8').transpose(0,2,3,1)
        n,h,w,_=rgb.shape
        if n!=len(self.camera_ids):raise ValueError('Video camera identity/count mismatch')
        canvas=Image.new('RGB',(n*w,h+64),(22,22,26));draw=ImageDraw.Draw(canvas)
        draw.text((10,3),self.title if status is None else status,font=self.font,fill='white')
        for i,name in enumerate(self.camera_ids):
            draw.text((i*w+10,32),f'{name} | step {step} | t={step/60:.3f}s',font=self.font,fill='white')
            canvas.paste(Image.fromarray(rgb[i]),(i*w,64))
        if self.stream is None:
            self.stream=self.container.add_stream('libx264',rate=60)
            self.stream.width,self.stream.height=canvas.size
            self.stream.pix_fmt='yuv420p';self.stream.codec_context.thread_count=2
            self.stream.options={'crf':'18','preset':'fast'}
        frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');frame.pts=self.frames
        for encoded in self.stream.encode(frame):self.container.mux(encoded)
        self.frames+=1

    def close(self, complete=False):
        if self.container is None:return
        if self.stream is not None:
            for encoded in self.stream.encode():self.container.mux(encoded)
        self.container.close();self.container=None
        if complete:
            self.partial.replace(self.path)
            return dict(file=self.path.name,frames=self.frames,fps=60,duration_s=self.frames/60,
                        camera_ids=self.camera_ids,frame_source='RGB observation before each native action',
                        terminal_image_available=False,reset_frames_included=False,
                        encoding='CPU libx264 CRF18 yuv420p; viewing only, not lossless training data')


def main():
    process_started=time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument('--b0',type=Path,required=True);p.add_argument('--b1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=50);p.add_argument('--seed-start',type=int,default=981000)
    p.add_argument('--max-steps',type=int,default=2400)
    p.add_argument('--development',action='store_true')
    p.add_argument('--policy-device',choices=('cpu','cuda:0'),default='cuda:0')
    p.add_argument('--record-video',action='store_true',help='Every pre-action frame; separate viewing run, not new formal trials')
    p.add_argument('--step-sleep-ms',type=float,default=0,help='Wall-only pacing; native dt and actions unchanged')
    p.add_argument('--diagnostic-inference-steps',type=int,choices=(8,20,50,100))
    p.add_argument('--diagnostic-plans',action='store_true')
    p.add_argument('--expert-reference',type=Path,help='Add released privileged expert only in bounded development runs')
    args=p.parse_args()
    if args.step_sleep_ms<0:p.error('step-sleep-ms must be nonnegative')
    if (args.diagnostic_inference_steps or args.expert_reference or args.diagnostic_plans) and not args.development:
        p.error('Diagnostic sampler/plans/expert require --development')
    if args.expert_reference and args.pairs>2:
        p.error('Expert reference is bounded to at most two starts')
    args.output.mkdir(parents=True,exist_ok=False)
    candidates=list(range(args.seed_start,args.seed_start+max(30,args.pairs*10)))
    identities={v:checkpoint_identity(getattr(args,v),v,args.development) for v in ('b0','b1')}
    plan=dict(protocol_version=2,candidate_seeds=candidates,target_legal_pairs=args.pairs,
        development=args.development,max_steps=args.max_steps,
        policy_device=args.policy_device,network_dtype='float32',num_envs=1,env_spacing_m=8.,
        pairing='compare recorded initial states before actions; first legal starts, no outcome selection',
        checkpoints=identities,record_video=args.record_video,step_sleep_ms=args.step_sleep_ms,
        diagnostic_inference_steps=args.diagnostic_inference_steps,diagnostic_plans=args.diagnostic_plans,
        expert_reference=bool(args.expert_reference),
        purpose='viewing_rerun' if args.record_video else 'diagnostic' if args.development else 'evaluation')
    from .compact import sha256
    import subprocess
    source_root=Path(__file__).resolve().parent
    source_files=[source_root/name for name in ('evaluate_baselines.py','rollout.py','history.py',
        'observations.py','action_history.py','expert_reference.py','contracts.py','training_resources.py','sim/environment.py')]
    source_files+=sorted((source_root/'models').glob('*.py'))
    plan['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    plan['source_sha256']={str(path.relative_to(source_root.parent)):sha256(path) for path in source_files}
    atomic_json(args.output/'plan.json',plan)
    # Admission/waits affect wall time only; never alter the native simulation dt.
    if shutil.disk_usage(args.output).free < 12_000_000_000:
        raise RuntimeError('Evaluation requires 12 GB free disk reserve')
    guard=ResourceGuard(args.output/'resources',wait_timeout_s=180 if args.development else 3600)
    guard.before_chunk()
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    from isaaclab.app import AppLauncher
    tick=time.perf_counter()
    app=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=(
        '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
        '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    timing=dict(isaac_app_s=time.perf_counter()-tick,scene_s=0.,checkpoint_load_s=0.,
                reset_s=0.,loop_s=0.,trace_write_s=0.,demo_write_s=0.)
    env=None;video=None;rows=[];rejections=[]
    try:
        guard.before_chunk()
        import numpy as np
        import torch
        import isaacsimenvs
        from evaluation.eval_isaacsim import _load_env_cfg
        from .sim.environment import InstrumentedAssemblyEnv
        from .observations import TaskCommand
        from .rollout import load_command_controller
        from .contracts import assert_policy_compatible, assert_execution_compatible
        from .initial_geometry import inspect
        from .analyze_rl_trials import wilson
        from .compact import sha256
        from PIL import Image
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        cfg=_load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
        cfg.seed=args.seed_start;cfg.scene.num_envs=1;cfg.scene.env_spacing=8.
        cfg.sim.device='cpu';cfg.precise_assembly.problem='tight_insertion'
        cfg.student_obs.image_modality='rgbd';cfg.student_obs.camera_backend='standard'
        cfg.student_obs.camera_update_period_s=0.
        cfg.student_obs.image_width=cfg.student_obs.image_input_width=448
        cfg.student_obs.image_height=cfg.student_obs.image_input_height=252
        cfg.sim.render.rendering_mode='quality';cfg.sim.render.antialiasing_mode='Off'
        tick=time.perf_counter()
        env=InstrumentedAssemblyEnv(cfg,perception_version=2,task_command=TaskCommand('tight_insertion'),
            wrench_frame_mode='world_hold',camera_rig='wrist_bracket_close')
        env.configure_camera_views((-.7,-.35,1.15),(0.,-.03,.68))
        timing['scene_s']=time.perf_counter()-tick
        tick=time.perf_counter()
        policies={v:load_command_controller(getattr(args,v),1,device=args.policy_device) for v in ('b0','b1')}
        timing['checkpoint_load_s']=time.perf_counter()-tick
        for v,policy in policies.items():
            if sha256(getattr(args,v))!=identities[v]['sha256'] or policy.checkpoint_epoch!=identities[v]['epoch']:
                raise RuntimeError('Checkpoint changed between admission and controller load')
            if args.diagnostic_inference_steps:
                policy.model.cfg.inference_steps=args.diagnostic_inference_steps
        plan.update(native_stage_timeout_steps=int(env.max_episode_length),
                    native_episode_length_s=cfg.episode_length_s,
                    physics_dt=env.physics_dt,control_dt=env.step_dt,
                    max_goals=int(env.env_max_goals.max()),inference_steps=policies['b0'].model.cfg.inference_steps)
        if not args.development and args.max_steps < (plan['max_goals']+1)*plan['native_stage_timeout_steps']:
            raise ValueError('External budget would cut the native per-stage timeout allowance')
        atomic_json(args.output/'plan.json',plan)
        if any(x.smoke_only for x in policies.values()) and not args.development:
            raise ValueError('Diagnostic weights cannot enter the formal evaluation')
        if [policies[v].variant for v in ('b0','b1')]!=['b0','b1']:
            raise ValueError('Variant/checkpoint ordering mismatch')
        if args.expert_reference:
            from .expert_reference import ExpertReference
            if sha256(args.expert_reference)!=policies['b0'].metadata['expert_model_sha256']:
                raise ValueError('Expert reference must match the recorded demonstration checkpoint')
            policies['expert']=ExpertReference(env,args.expert_reference,policies['b0'].metadata)
            identities['expert']=dict(variant='expert',epoch=None,sha256=sha256(args.expert_reference))
            atomic_json(args.output/'plan.json',plan)
        def initial():
            result={k:v[0].detach().cpu().numpy().copy() for k,v in env._diagnostics().items()}
            result.update(actuator_target_rad=env.research_packet['actuator_target_rad'][0].numpy().copy(),
                force_probability=env._random_force_prob[0].numpy().copy(),
                torque_probability=env._random_torque_prob[0].numpy().copy(),
                object_mass=env._object_mass[0].numpy().copy())
            return result
        meta=None;legal=0;started=time.time()
        for seed in candidates:
            if legal>=args.pairs:break
            if shutil.disk_usage(args.output).free < 12_000_000_000:
                raise RuntimeError('Evaluation disk reserve exhausted')
            reset_tick=time.perf_counter()
            teacher_observation,_=env.reset(seed=seed)
            reset_seconds=time.perf_counter()-reset_tick;timing['reset_s']+=reset_seconds
            meta=env.research_metadata();meta.update(problem='tight_insertion',camera_mode='oblique',config=cfg.to_dict())
            if not (args.output/'environment.json').exists():atomic_json(args.output/'environment.json',meta)
            for policy in policies.values():
                assert_policy_compatible(policy.metadata,meta)
                assert_execution_compatible(policy.metadata,meta)
            reference=initial();geometry=inspect(meta,reference)
            if not geometry['valid_box_initialization']:
                rejections.append(dict(seed=seed,geometry=geometry))
                atomic_json(args.output/'invalid_initializations.json',rejections);continue
            pair_rows=[]
            for variant,policy in policies.items():
                if variant!='b0':
                    reset_tick=time.perf_counter();teacher_observation,_=env.reset(seed=seed)
                    reset_seconds=time.perf_counter()-reset_tick;timing['reset_s']+=reset_seconds
                current=initial()
                differences={k:float(np.max(np.abs(current[k].astype(float)-reference[k].astype(float)))) for k in reference}
                if max(differences.values())>1e-5:
                    atomic_json(args.output/'pairing_error.json',dict(seed=seed,differences=differences))
                    raise RuntimeError('Actual initial states do not match; evaluation stopped')
                # Same independent policy-noise seed in each pair. No env RNG mutation.
                policy.generator.manual_seed(seed+1000000)
                trace={k:[] for k in ('command','target_rad','joint_before','joint_after','object_after',
                    'time_before_s','force_w_N','torque_w_Nm','contact_after_w_N',
                    'submitted_action','command_clipped_pre_delay','delayed_command','selected_delay',
                    'native_stage_step','successes')}
                if args.diagnostic_plans:
                    trace.update(target_before_rad=[],lifted_object=[],mean_finger_link_object_distance_m=[])
                    if variant!='expert':
                        trace.update(predicted_chunk=[],chunk_cursor=[],plan_generation_step=[])
                history_fields=()
                if args.diagnostic_plans and getattr(policy.model.cfg,'action_history_mode','none')=='issued':
                    history_fields=('previous_issued_command','previous_delayed_command',
                                    'previous_command_time_s','previous_delayed_time_s','action_history_valid')
                    trace.update({name:[] for name in history_fields})
                latencies=[];rng_unchanged=True;done=False
                if args.record_video:
                    video=RolloutVideo(args.output/f'{variant}_s{seed}.mp4',variant,seed,
                                       policy.checkpoint_epoch,meta['camera_ids'])
                costs=dict(memory_s=0.,replan_s=0.,policy_s=0.,env_step_s=0.,trace_pack_s=0.,
                           demo_s=0.,resource_gate_s=0.,pacing_s=0.,reset_s=reset_seconds)
                loop_tick=time.perf_counter()
                for step in range(args.max_steps):
                    tick=time.perf_counter();guard.before_chunk();costs['resource_gate_s']+=time.perf_counter()-tick
                    packet=env.research_packet
                    if variant=='expert':
                        policy.teacher_observation=teacher_observation
                    rng=torch.get_rng_state().clone();tick=time.perf_counter()
                    action=policy.act(packet);latencies.append((time.perf_counter()-tick)*1000)
                    costs['policy_s']+=latencies[-1]/1000
                    costs['memory_s']+=policy.memory_ms[-1]/1000
                    if policy.replanned.any():costs['replan_s']+=policy.inference_ms[-1]/1000
                    rng_unchanged &= torch.equal(rng,torch.get_rng_state())
                    if not rng_unchanged and variant!='expert':raise RuntimeError('Policy polluted environment RNG')
                    if video is not None:
                        tick=time.perf_counter();video.append(packet,step)
                        costs['demo_s']+=time.perf_counter()-tick
                    if legal==0 and step%120==0:
                        tick=time.perf_counter()
                        im=packet['image'][0,:,:3].numpy().transpose(0,2,3,1)
                        Image.fromarray((np.concatenate(im,axis=1).clip(0,1)*255).astype('uint8')).save(
                            args.output/f'{variant}_first_step{step:04d}.png')
                        costs['demo_s']+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    teacher_observation,_,terminated,truncated,_=env.step(action)
                    costs['env_step_s']+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    tr=env.last_transition;diag=tr['diagnostics_after']
                    policy.observe_execution(tr)
                    values=dict(command=action,target_rad=tr['target_after_rad'],
                        joint_before=tr['diagnostics_before']['joint_position_rad'],joint_after=diag['joint_position_rad'],
                        object_after=diag['object_pose_w'],time_before_s=tr['time_before_s'],
                        force_w_N=tr['applied_wrench']['force_w_N'],torque_w_Nm=tr['applied_wrench']['torque_w_Nm'],
                        contact_after_w_N=tr['contact_substeps_after_w_N'],submitted_action=tr['submitted_action'],
                        command_clipped_pre_delay=tr['command_clipped_pre_delay'],
                        delayed_command=tr['control_trace']['delayed_command'],
                        selected_delay=tr['control_trace']['delay_index'],
                        native_stage_step=diag['native_stage_step'],successes=diag['successes'])
                    if args.diagnostic_plans:
                        values.update(target_before_rad=tr['target_before_rad'],lifted_object=diag['lifted_object'],
                            mean_finger_link_object_distance_m=(diag['finger_position_w']-
                                diag['object_pose_w'][:,None,:3]).norm(dim=-1).mean(-1))
                        if variant!='expert':
                            values.update(predicted_chunk=policy.plan,chunk_cursor=policy.cursor-1,
                                          plan_generation_step=policy.plan_generation_step)
                        values.update({name:policy.last_packet[name] for name in history_fields})
                    for k,v in values.items():trace[k].append(v[0].detach().numpy().copy())
                    done=bool((terminated|truncated)[0])
                    costs['trace_pack_s']+=time.perf_counter()-tick
                    if args.step_sleep_ms:
                        tick=time.perf_counter();time.sleep(args.step_sleep_ms/1000)
                        costs['pacing_s']+=time.perf_counter()-tick
                    if done:break
                video_info=None
                if video is not None:
                    tick=time.perf_counter();video_info=video.close(complete=True);video=None
                    costs['demo_s']+=time.perf_counter()-tick
                costs['loop_s']=time.perf_counter()-loop_tick
                timing['loop_s']+=costs['loop_s'];timing['demo_write_s']+=costs['demo_s']
                success=bool(done and diag['successes'][0]>=diag['max_goals'][0] and diag['retract_succeeded'][0])
                reasons=[k for k,v in tr['termination_reasons'].items() if bool(v[0])]
                arrays={k:np.stack(v) for k,v in trace.items()}
                path=args.output/f'{variant}_s{seed}.npz';tick=time.perf_counter();np.savez_compressed(path,**arrays)
                costs['trace_write_s']=time.perf_counter()-tick;timing['trace_write_s']+=costs['trace_write_s']
                row=dict(seed=seed,variant=variant,steps=step+1,success=success,
                    outcome=('success' if success else 'failure' if done else 'censored'),reasons=reasons,
                    initial={k:v.tolist() for k,v in current.items()},pair_initial_max_error=max(differences.values()),
                    inference_ms_median=float(np.median(latencies)),inference_ms_p95=float(np.percentile(latencies,95)),
                    inference_ms_max=max(latencies),rng_unchanged=rng_unchanged,
                    peak_force_N=float(np.linalg.norm(arrays['force_w_N'],axis=-1).max()),
                    nonzero_force_physical_samples=int(np.count_nonzero(np.linalg.norm(arrays['force_w_N'],axis=-1)>1e-8)),
                    trace=path.name,checkpoint_sha256=identities[variant]['sha256'],
                    checkpoint_epoch=identities[variant]['epoch'],timing=costs,
                    devices=dict(simulation=str(env.device),model=str(policy.device),images=str(policy.device),
                                 recurrence=str(policy.device),diffusion=str(policy.device),command=str(action.device)),
                    inference_steps=policy.model.cfg.inference_steps,
                    action_history_mode=getattr(policy.model.cfg,'action_history_mode','none'))
                if video_info is not None:row['video']=video_info
                rows.append(row);pair_rows.append(row)
                atomic_json(args.output/'trials.json',dict(trials=rows,invalid=rejections))
                print('TRIAL '+json.dumps({k:v for k,v in row.items() if k!='initial'}),flush=True)
            legal+=1
            elapsed=time.time()-started
            # First two valid pairs belong to the final cohort; update ETA with
            # their complete measured cost, not a second selected-outcome pilot.
            atomic_json(args.output/'status.json',dict(state='running',legal_pairs=legal,pid=os.getpid(),
                elapsed_s=elapsed,invalid_initializations=len(rejections),
                eta_seconds=elapsed/legal*(args.pairs-legal) if legal>=2 else None))
        report=dict(development=args.development,legal_pairs=legal,target_pairs=args.pairs,
            invalid_initializations=len(rejections),complete=legal==args.pairs,elapsed_s=time.time()-started,
            variants={},paired=dict(both=0,b0_only=0,b1_only=0,neither=0),
            limitation='Single training seed; same starts, policy-dependent disturbance gates/trajectories',
            success_metric='success within the declared control-step budget / all legal trials; censoring reported separately',
            resources=guard.summary(),timing=timing,process_wall_s=time.perf_counter()-process_started,
            protocol_version=2,checkpoints=identities,purpose=plan['purpose'])
        for variant in policies:
            selected=[r for r in rows if r['variant']==variant];n=len(selected);success=sum(r['success'] for r in selected)
            report['variants'][variant]=dict(successes=success,attempted_legal=n,censored=sum(r['outcome']=='censored' for r in selected),
                success_fraction=success/n if n else None,wilson_95=wilson(success,n))
        policy_rows=[row for row in rows if row['variant']!='expert']
        for a,b in zip(policy_rows[::2],policy_rows[1::2]):
            name='both' if a['success'] and b['success'] else 'b0_only' if a['success'] else 'b1_only' if b['success'] else 'neither'
            report['paired'][name]+=1
        publish_report(args.output,report)
    except BaseException as error:
        import traceback
        (args.output/'error.txt').write_text(traceback.format_exc())
        atomic_json(args.output/'status.json',dict(state='failed',error=repr(error),completed_trials=len(rows)))
        raise
    finally:
        if video is not None:video.close()
        if env is not None:env.close()
        app.close()

if __name__=='__main__':main()
