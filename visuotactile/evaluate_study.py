"""Preregistered native DP comparisons and matched local/autonomous residual trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .compact import sha256
from .study_resources import StudyLease
from .training_resources import atomic_json


def longest_streak(values):
    longest = current = 0
    for value in values:
        current = current+1 if value else 0
        longest = max(longest,current)
    return longest


def allocation(root, phase):
    session = json.loads((root/'session.json').read_text())
    selected = json.loads((root/'selected.json').read_text())['selected']
    if phase=='dp':
        return [dict(seed=seed,line=line,mode='autonomous',variant='dp')
                for seed in session['seeds'] for line in session['dp_lines']]
    report = json.loads((root/'dp_closed_loop/report.json').read_text())
    history = json.loads((root/selected['G00']/'metrics.json').read_text())
    validation = min(history,key=lambda row:row['validation']['selection_mse'])['validation']
    offline = all(validation['metrics']['command'][group]['prefix12_rmse'] <
                  validation['metrics']['zero'][group]['prefix12_rmse'] for group in ('arm','hand'))
    opportunity = any(row['line']==selected['G00'] and row.get('max_contact_streak',0)>=3 for row in report['trials'])
    local = not (offline and opportunity)
    atomic_json(root/'residual_admission.json',dict(validation_prefix_below_zero=offline,
        autonomous_observed_contact_opportunity=opportunity,mode='expert_rollin_local' if local else 'autonomous',
        base_line=selected['G00'],base_history='private_proposals_with_native_delay_timing_v1',
        switch='first3 consecutive true-gate observations at step<=180; no state writes',
        interpretation='local diagnostic authorization does not imply the autonomous base gate passed'))
    if local:
        return [dict(seed=seed,line=selected['G00'],mode='expert_rollin',variant=variant)
                for seed in session['seeds'][:2] for variant in ('base','force','masked')]
    return [dict(seed=seed,line=selected['G00'],mode='autonomous',variant=variant)
            for seed in session['seeds'] for variant in ('force','masked')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('dp','residual'),required=True)
    args = parser.parse_args()
    root,phase = args.root,args.phase
    output = root/(phase+'_closed_loop')
    output.mkdir(exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    selected = json.loads((root/'selected.json').read_text())['selected']
    trials_plan = allocation(root,phase)
    lease = StudyLease(root,deadline=session['deadline'])
    lines = session['dp_lines'] if phase=='dp' else [selected['G00']]
    sources = {line:root/line/'best.pt' for line in lines}
    for line,path in sources.items():
        if json.loads((path.parent/'test.json').read_text())['checkpoint_sha256']!=sha256(path):
            raise ValueError('DP deployed checkpoint differs from once-tested selected weights')
    if phase=='residual':
        for variant in ('force','masked'):
            path = root/'transformer_training'/variant/'best.pt'
            if json.loads((path.parent/'test.json').read_text())['checkpoint_sha256']!=sha256(path):
                raise ValueError('Residual deployed checkpoint differs from once-tested selected weights')
    atomic_json(output/'plan.json',dict(trials=trials_plan,allocated=len(trials_plan),max_total_starts=18,
        retry='none; preserve invalid and software-failed attempts',unmatched='label and continue; never paired evidence',
        checkpoint_sha256={line:sha256(path) for line,path in sources.items()},
        native_bound='(max_goals+2)*(native max_episode_length+1)',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={str(path):sha256(path) for path in (Path(__file__),Path('visuotactile/transformer_controller.py'),
            Path('visuotactile/private_history.py'),Path('visuotactile/models/transformer_residual.py'),
            Path('visuotactile/long_controller.py'),Path('visuotactile/sim/environment.py'))}))
    atomic_json(output/'status.json',dict(state='initializing',pid=os.getpid(),utc_epoch=time.time()))
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env = video = None
    trace,rows = [],[]
    attempt = None
    try:
        import numpy as np
        import torch
        import isaacsimenvs
        from evaluation.eval_isaacsim import _load_env_cfg
        from .contracts import assert_policy_compatible, assert_execution_compatible
        from .evaluate_baselines import RolloutVideo
        from .expert_reference import ExpertReference
        from .initial_geometry import inspect
        from .long_controller import LongController
        from .models.long_horizon import FrozenDinoSpatial
        from .models.residual import observed_contact
        from .observations import TaskCommand
        from .sim.environment import InstrumentedAssemblyEnv
        from .transformer_controller import TransformerController
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = _load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
        cfg.seed = session['seeds'][0]
        cfg.scene.num_envs,cfg.scene.env_spacing = 1,8.
        cfg.sim.device = 'cpu'
        cfg.precise_assembly.problem = 'tight_insertion'
        cfg.student_obs.image_modality,cfg.student_obs.camera_backend = 'rgbd','standard'
        cfg.student_obs.camera_update_period_s = 0.
        cfg.student_obs.image_width = cfg.student_obs.image_input_width = 448
        cfg.student_obs.image_height = cfg.student_obs.image_input_height = 252
        cfg.sim.render.rendering_mode,cfg.sim.render.antialiasing_mode = 'quality','Off'
        with lease.block():
            env = InstrumentedAssemblyEnv(cfg,perception_version=2,task_command=TaskCommand('tight_insertion'),
                wrench_frame_mode='world_hold',camera_rig='wrist_bracket_close')
            env.configure_camera_views((-.7,-.35,1.15),(0.,-.03,.68))
            dino = FrozenDinoSpatial().to('cuda:0').eval()
            if phase=='dp':
                controllers = {line:LongController(torch.load(path,map_location='cpu',weights_only=False),dino)
                               for line,path in sources.items()}
            else:
                controllers = {variant:TransformerController(sources[selected['G00']],
                    root/'transformer_training'/('force' if variant=='base' else variant)/'best.pt',dino,
                    bypass=variant=='base') for variant in ('base','force','masked')}
            dino_hash = sha256(Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth')
            for controller in controllers.values():
                base = controller if phase=='dp' else controller.base
                if base.vision['dino_weights_sha256']!=dino_hash:
                    raise ValueError('Frozen DINO identity differs')
            expert = None
            if phase=='residual' and trials_plan[0]['mode']=='expert_rollin':
                path = Path('pretrained_assembly/tight_insertion/model.pth')
                expert = ExpertReference(env,path,controllers['base'].base.metadata)
                if sha256(path)!=controllers['base'].base.metadata['expert_model_sha256']:
                    raise ValueError('Released expert identity differs')
        initials,switches = {},{}
        for ordinal,trial in enumerate(trials_plan):
            seed,line,variant,mode = (trial[name] for name in ('seed','line','variant','mode'))
            attempt = f'{ordinal+1:02d}_{line}_{variant}_{mode}_s{seed}'
            trace,plans,delta_windows = [],[],[]
            atomic_json(output/'status.json',dict(state='running',attempt=attempt,completed=len(rows),pid=os.getpid()))
            atomic_json(output/(attempt+'_started.json'),dict(**trial,utc_epoch=time.time(),ordinal=ordinal+1))
            controller = controllers[line if phase=='dp' else variant]
            base = controller if phase=='dp' else controller.base
            with lease.block():
                teacher_observation,_ = env.reset(seed=seed)
            metadata = env.research_metadata()
            metadata.update(problem='tight_insertion',camera_mode='oblique',config=cfg.to_dict())
            assert_policy_compatible(base.metadata,metadata)
            assert_execution_compatible(base.metadata,metadata)
            if not (output/'environment.json').exists():
                atomic_json(output/'environment.json',metadata)
            initial = {name:value[0].detach().cpu().numpy().copy() for name,value in env._diagnostics().items()}
            initial.update(actuator_target_rad=env.research_packet['actuator_target_rad'][0].numpy().copy(),
                force_probability=env._random_force_prob[0].numpy().copy(),torque_probability=env._random_torque_prob[0].numpy().copy(),
                object_mass=env._object_mass[0].numpy().copy())
            np.savez_compressed(output/(attempt+'_initial.npz'),**initial)
            geometry = inspect(metadata,initial)
            if not geometry['valid_box_initialization']:
                rows.append(dict(**trial,attempt=attempt,outcome='invalid_before_action',geometry=geometry,success=None))
                atomic_json(output/'trials.json',rows)
                continue
            if seed not in initials:
                if phase=='residual' and mode=='autonomous':
                    previous = next(row for row in json.loads((root/'dp_closed_loop/trials.json').read_text())
                                    if row['seed']==seed and row['line']==line)
                    with np.load(root/'dp_closed_loop'/(previous['attempt']+'_initial.npz')) as saved_initial:
                        initials[seed] = dict(saved_initial)
                else:
                    initials[seed] = initial
            initial_error = max(float(np.max(np.abs(value.astype(float)-initials[seed][name].astype(float))))
                                for name,value in initial.items())
            base.generator.manual_seed(seed+1000000)
            make_video = ((phase=='dp' and line==selected['G10'] and seed==981004) or
                          (phase=='residual' and seed==session['seeds'][0] and variant in ('base','force')))
            if make_video:
                video = RolloutVideo(output/(attempt+'.mp4'),line+' '+variant,seed,base.checkpoint_epoch,metadata['camera_ids'])
            switched = mode=='autonomous'
            switch_step = 0 if switched else None
            switch_match = None
            switch_error = None
            prefix = hashlib.sha256()
            contact_streak = 0
            done = False
            began = time.monotonic()
            native_bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
            for step in range(native_bound):
                packet = env.research_packet
                cpu_rng = torch.get_rng_state().clone()
                with lease.block():
                    inference_began = time.perf_counter()
                    command = controller.act(packet)
                    torch.cuda.synchronize()
                    policy_seconds = time.perf_counter()-inference_began
                    if not torch.equal(cpu_rng,torch.get_rng_state()):
                        raise RuntimeError('Student policy changed native CPU RNG')
                    gate = observed_contact(packet['contact_w_N'],packet['contact_valid'])
                    contact_streak = contact_streak+1 if bool(gate[0]) else 0
                    if not switched and contact_streak>=3 and step<=180:
                        switched,switch_step = True,step
                        switch = {name:value[0].detach().cpu().numpy().copy() for name,value in env._diagnostics().items()}
                        switch.update(target=packet['actuator_target_rad'][0].numpy().copy(),contact=packet['contact_w_N'][0].numpy().copy())
                        np.savez_compressed(output/(attempt+'_switch.npz'),**switch)
                        if seed not in switches:
                            switches[seed] = (step,switch,prefix.hexdigest())
                        reference_step,reference,prefix_hash = switches[seed]
                        switch_error = max(float(np.max(np.abs(value.astype(float)-reference[name].astype(float))))
                                           for name,value in switch.items())
                        switch_match = bool(step==reference_step and switch_error<=1e-5 and prefix.hexdigest()==prefix_hash)
                    if not switched:
                        expert.teacher_observation = teacher_observation
                        command = expert.act(packet)
                        prefix.update(command.numpy().tobytes())
                    if step%12==0:
                        plans.append(base.plan[0].numpy().copy())
                    if phase=='residual' and step%6==0:
                        delta_windows.append(controller.delta_window.numpy().copy())
                    if video is not None:
                        video.append(packet,step,status=f'{line} {variant} | {mode} | gate={int(gate[0])} | seed{seed}')
                    teacher_observation,_,terminated,truncated,_ = env.step(command)
                transition = env.last_transition
                if phase=='dp':
                    controller.observe_execution(transition)
                else:
                    controller.acknowledge(transition,command,source='student' if switched else 'released_expert_rollin')
                diagnostic = transition['diagnostics_after']
                last = dict(base=command,delta=torch.zeros_like(command),residual_generation_step=-1,residual_cursor=-1,
                    residual_forward_s=0.,residual_refresh=False) if phase=='dp' else controller.last
                values = dict(command=command,base=last['base'],delta=last['delta'],gate=gate,
                    reference=base.reference,submitted=transition['submitted_action'],clipped=transition['command_clipped_pre_delay'],
                    delayed=transition['control_trace']['delayed_command'],selected_time=transition['control_trace']['selected_command_time_s'],
                    time_before=transition['time_before_s'],time_after=transition['time_after_s'],
                    target_before=transition['target_before_rad'],target_after=transition['target_after_rad'],
                    joint_before=transition['diagnostics_before']['joint_position_rad'],joint_after=diagnostic['joint_position_rad'],
                    object_pose=diagnostic['object_pose_w'],force=packet['contact_substeps_w_N'],
                    force_after=transition['contact_substeps_after_w_N'],successes=diagnostic['successes'],lifted=diagnostic['lifted_object'],
                    finger_distance=(diagnostic['finger_position_w']-diagnostic['object_pose_w'][:,None,:3]).norm(dim=-1).mean(-1),
                    requested_force=transition['external_force_after_dr_w_N'],requested_torque=transition['external_torque_after_dr_w_Nm'])
                for name,value in transition['applied_wrench'].items():
                    values['applied_'+name] = value
                row = {name:value[0].detach().cpu().numpy().copy() for name,value in values.items()}
                row.update(student=np.asarray(switched),step=np.asarray(step),nominal_cursor=np.asarray(step%12),
                    nominal_generation=np.asarray(step-step%12),residual_cursor=np.asarray(last['residual_cursor']),
                    residual_generation=np.asarray(last['residual_generation_step']),policy_s=np.asarray(policy_seconds),
                    residual_s=np.asarray(last['residual_forward_s']),residual_refresh=np.asarray(last['residual_refresh']))
                if phase=='residual':
                    row.update(private_previous=controller.last['private_previous'][0].numpy().copy(),
                               actual_previous=controller.last['actual_previous'][0].numpy().copy())
                trace.append(row)
                if step%60==0:
                    atomic_json(output/'status.json',dict(state='running',attempt=attempt,step=step,completed=len(rows),pid=os.getpid()))
                done = bool((terminated|truncated)[0])
                if done:
                    break
            arrays = {name:np.stack([row[name] for row in trace]) for name in trace[0]}
            path = output/(attempt+'.npz')
            extras = dict(plans=np.stack(plans))
            if delta_windows:
                extras['delta_windows'] = np.stack(delta_windows)
            np.savez_compressed(path,**arrays,**extras)
            video_info = video.close(complete=done) if video is not None else None
            video = None
            student = arrays['student'].astype(bool)
            contact = arrays['gate'].astype(bool)&student
            height = arrays['object_pose'][:,2]-float(initial['object_initial_z_m'])
            peak = int(np.argmax(height))
            drop = np.flatnonzero((np.arange(len(height))>peak)&(height<.02)) if height[peak]>.05 else []
            success = bool(done and diagnostic['successes'][0]>=diagnostic['max_goals'][0] and diagnostic['retract_succeeded'][0])
            no_contact = student&~arrays['gate'].astype(bool)
            bypass = bool(np.array_equal(arrays['command'][no_contact],arrays['base'][no_contact]))
            if not bypass:
                raise RuntimeError('Executed no-contact residual differs from base')
            actual_force = np.linalg.norm(arrays['applied_force_w_N'],axis=-1)
            actual_torque = np.linalg.norm(arrays['applied_torque_w_Nm'],axis=-1)
            delta = arrays['command']-arrays['base']
            result = dict(**trial,attempt=attempt,steps=len(trace),student_steps=int(student.sum()),
                success=success if switched else None,outcome='no_contact_switch' if not switched else 'success' if success else 'failure' if done else 'censored',
                reasons=[name for name,value in transition['termination_reasons'].items() if bool(value[0])],
                initial_match_max_error=initial_error,paired=initial_error<=1e-5 and (mode=='autonomous' or bool(switch_match)),
                switch_step=switch_step,switch_match=switch_match,switch_max_error=switch_error,prefix_sha256=prefix.hexdigest(),
                checkpoint_sha256=sha256(sources[line]),residual_sha256=None if phase=='dp' else controller.checkpoint_sha256,
                trace=path.name,trace_sha256=sha256(path),video=video_info,native_lift=bool(arrays['lifted'].any()),
                max_subgoals=int(arrays['successes'].max()),max_object_height_change_m=float(height.max()),
                height_peak_step=peak,first_drop_below_2cm_after_5cm=int(drop[0]) if len(drop) else None,
                longest_height_above_5cm_s=longest_streak(height>.05)/60.,contact_steps=int(contact.sum()),
                max_contact_streak=longest_streak(contact),first_contact=int(np.flatnonzero(contact)[0]) if contact.any() else None,
                student_delta_mean_abs=float(np.abs(delta[student]).mean()) if student.any() else None,
                bypass_exact=bypass,peak_finger_force_N=float(np.linalg.norm(arrays['force'],axis=-1).max()),
                initial_finger_distance_m=float(arrays['finger_distance'][0]),minimum_finger_distance_m=float(arrays['finger_distance'].min()),
                final_finger_distance_m=float(arrays['finger_distance'][-1]),actual_force_peak_N=float(actual_force.max()),
                actual_torque_peak_Nm=float(actual_torque.max()),actual_force_norm_impulse_Ns=float(actual_force.sum()/120),
                policy_ms_median=float(np.median(arrays['policy_s'])*1000),policy_ms_p95=float(np.quantile(arrays['policy_s'],.95)*1000),
                residual_refresh_ms_p95=float(np.quantile(arrays['residual_s'][arrays['residual_refresh']],.95)*1000) if phase=='residual' else None,
                wall_s=time.monotonic()-began,contact_partner='unknown; five aggregate sensors do not identify object grasp',
                loss_of_hold_cause='not causally identified; inspect force/torque/contact/base/delta/replan traces')
            rows.append(result)
            atomic_json(output/'trials.json',rows)
            print('TRIAL',json.dumps(result),flush=True)
        atomic_json(output/'report.json',dict(complete=len(rows)==len(trials_plan),trials=rows,
            formal_counts_unchanged=True,resources=json.loads((root/'resources/resources.json').read_text())))
        atomic_json(output/'status.json',dict(state='complete',trials=len(rows),pid=os.getpid()))
    except BaseException as error:
        if trace and attempt:
            np.savez_compressed(output/(attempt+'_interrupted.npz'),
                **{name:np.stack([row[name] for row in trace]) for name in trace[0]})
        atomic_json(output/'interruption.json',dict(error=repr(error),attempt=attempt,recorded_steps=len(trace),completed_trials=len(rows)))
        atomic_json(output/'status.json',dict(state='interrupted',error=repr(error),attempt=attempt,pid=os.getpid()))
        raise
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()
        app.close()


if __name__=='__main__':
    main()
