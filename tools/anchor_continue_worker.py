"""Independent Kit rollout, lifecycle/execution audit, video and optional Image input capture."""
import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.runtime import read,write,sha
from visuotactile.training_resources import atomic_json


def run_lifecycle(root, output, job_path):
    job=json.loads(job_path.read_text());case=job.get('lifecycle',{})
    from tools.rollout_timing import Profile
    profile=Profile(output) if job.get('profile_timing') else None
    from isaaclab import app as launcher
    original_launcher=launcher.AppLauncher
    fast=case.get('fast_shutdown',True)
    def tracked_launcher(*args,**kwargs):
        kwargs['fast_shutdown']=fast
        launched=(profile.call('kit_startup',original_launcher,*args,**kwargs) if profile else original_launcher(*args,**kwargs));close=launched.app.close
        def closing(*a,**kw):
            atomic_json(output/'shutdown_phase.json',dict(phase='app_close_entered',fast_shutdown=fast,utc=time.time()))
            close(*a,**kw)
            atomic_json(output/'shutdown_phase.json',dict(phase='app_close_returned',fast_shutdown=fast,utc=time.time()))
        launched.app.close=closing
        return launched
    launcher.AppLauncher=tracked_launcher
    from visuotactile import state_benchmark as base
    if profile:profile.patch(base)
    original_make=base.make_env
    def make_env(*args,**kwargs):
        env=original_make(*args,**kwargs);close=env.close
        dynamics={}
        for name in ('joint_stiffness','joint_damping','joint_effort_limits','joint_vel_limits','joint_armature','joint_friction_coeff'):
            value=getattr(env.robot.data,name,None)
            dynamics[name]=value[:,env._perm_lab_to_canon].cpu().tolist() if torch.is_tensor(value) else None
        write(output/'actuator_configuration.json',dict(canonical_order=True,values=dynamics,
            physics_dt=env.physics_dt,control_dt=env.step_dt,decimation=env.cfg.decimation,
            missing_values_are_not_inferred=True))
        def closing():
            atomic_json(output/'shutdown_phase.json',dict(phase='env_close_entered',utc=time.time()))
            if case.get('disarm_stop',True):
                sim=env.sim
                sim._disable_app_control_on_stop_handle=True
                handle=sim._app_control_on_stop_handle
                if handle is not None:handle.unsubscribe();sim._app_control_on_stop_handle=None
            # No timed faulthandler or other added in-process diagnostic thread.
            close()
            atomic_json(output/'shutdown_phase.json',dict(phase='env_close_returned',utc=time.time()))
        env.close=closing
        return env
    base.make_env=make_env
    # Match State's registered latent seed while retaining the original image
    # observer/action path and parameter tensors.
    from visuotactile import native_controller
    original_image=native_controller.NativeController
    class SeededImage(original_image):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.generator.manual_seed(job.get('inference',{}).get('sampler_seed',7091))
    native_controller.NativeController=SeededImage
    base.run(root,output,job.get('mode','dp'),job_path)

def run_audited_episode(root, output, job_path):
    import torch
    from isaaclab import app as launcher
    from visuotactile import state_benchmark as base, native_controller
    job = read(job_path)
    assert job['mode'] in ('dp','expert','expert_slow','expert_retime') and len(job['specs']) == 1
    assert not any(k in job for k in ('takeover_steps', 'replay'))
    session = read(root/'session.json')
    original_launcher = launcher.AppLauncher

    def isolated_launcher(*args, **kwargs):
        # Preserve the exact prior persistent values, but never write shared paths.
        # Fresh simulator processes share only this run's seeded shader cache.
        # No physical/environment/controller state is reused between workers.
        private = root/'private_runtime'
        private.mkdir(parents=True, exist_ok=True)
        kwargs['kit_args'] = kwargs.get('kit_args', '') + (
            f' --portable-root {private}'
            f' --/app/userConfigPath={root / "freeze/user.config.json"}'
            ' --/app/settings/persistent=false')
        app = original_launcher(*args, **kwargs)
        import carb.tokens
        tokens = carb.tokens.get_tokens_interface()
        write(output/'isolated_paths.json', dict(
            tokens={k:tokens.resolve('${'+k+'}') for k in ('data', 'cache', 'logs')},
            copied_user_config_sha256=sha(root/'freeze/user.config.json')))
        return app
    launcher.AppLauncher = isolated_launcher
    audit = dict(model=job['model'], updates=job.get('updates',10000), checkpoint_sha256=job['checkpoint_sha256'],
        act_calls=0, execution_acks=0, episode_ids=[], ticks=[], model_output_only=True,
        teacher_forcing=False, expert_takeover=False, created_utc=time.time())

    def instrument(cls, state):
        original_act, original_ack = cls.act, cls.observe_execution
        def act(self, *args, **kwargs):
            packet = args[1] if state else args[0]
            tick, episode = int(packet['step_id'][0]), int(packet['episode_id'][0])
            if not audit['act_calls']:
                assert tick == 0 and self.episode is None
                assert self.plan is None
                audit['history_initially_empty'] = True
                audit['model_class'] = type(self.model).__name__
                audit['model_contract'] = self.model.contract
            assert tick == audit['act_calls'], 'Skipped/duplicated native controller tick'
            if episode not in audit['episode_ids']: audit['episode_ids'].append(episode)
            assert len(audit['episode_ids']) == 1, 'Unexpected episode reset'
            result = original_act(self, *args, **kwargs)
            assert torch.isfinite(result).all()
            audit['act_calls'] += 1
            if tick % 100 == 0: write(output/'controller_audit.json', audit)
            return result
        def ack(self, *args, **kwargs):
            result = original_ack(self, *args, **kwargs)
            audit['execution_acks'] += 1
            return result
        cls.act, cls.observe_execution = act, ack
    instrument(base.HistoryController, True)
    instrument(native_controller.NativeController, False)
    original_trial = base.trial
    def trial(*args, **kwargs):
        result = original_trial(*args, **kwargs)
        if job['mode'] in ('expert','expert_slow','expert_retime'):
            audit.update(model_output_only=False,expert_reference=True)
            audit['completed']=bool(result.get('software_complete') and (result.get('legal') or result.get('not_started')))
            audit['initialization_rejected']=bool(result.get('not_started'))
        elif not result.get('legal'):
            audit['completed']=audit['act_calls']==audit['execution_acks']==0
            audit['initialization_rejected']=True
        else:
            audit['completed'] = bool(audit['act_calls'] == result['steps']
                and audit['execution_acks'] == result['steps'])
        write(output/'controller_audit.json', audit)
        assert audit['completed'], 'Controller execution/history audit incomplete'
        return result
    base.trial = trial
    original_video = base.RolloutVideo
    class Video(original_video):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            execution='open-loop retimed targets' if job['mode']=='expert_retime' else 'autonomous'
            weights='frozen released RL' if job['mode'].startswith('expert') else f"updates {job.get('updates',10000)}"
            self.title = f"{job['model']} | {weights} | {job['specs'][0]['identity']} | {execution}"
    base.RolloutVideo = Video
    run_lifecycle(root, output, job_path)

def run(root,output,path):
    job=read(path);capture=job.get('capture_pair',False);rows={}
    from visuotactile import native_controller,state_benchmark as base
    cls=native_controller.NativeController;old_init,old_act=cls.__init__,cls.act
    def stash(key,value):
        if torch.is_tensor(value):rows[key]=value.detach().cpu().numpy().copy()
    def init(self,*args,**kwargs):
        # Same numerical settings for the Image duplicate gate and later Image evaluation.
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        old_init(self,*args,**kwargs)
        if not capture:return
        observe=self.model.observe;predict=self.model.predict
        def observed(robot,stamp,tick,frame,memory=None):
            if tick<12:
                stash(f'{tick:02d}_robot',robot);stash(f'{tick:02d}_normalized_robot',self.model.normalized_robot(robot))
                if frame is not None:
                    for k,v in frame.items():stash(f'{tick:02d}_frame_{k}',v)
            result=observe(robot,stamp,tick,frame,memory)
            if tick<12:
                stash(f'{tick:02d}_gru_native',result.native);stash(f'{tick:02d}_gru_visual',result.visual)
            return result
        def predicted(condition,generator,*a,**kw):
            first=self.next_step==0
            if first:
                stash('00_condition',condition);stash('00_rng_before',generator.get_state())
                count=[0]
                def hook(module,inputs):
                    stash(f'00_diffusion_{count[0]:02d}_input',inputs[0]);stash(f'00_diffusion_{count[0]:02d}_time',inputs[1]);count[0]+=1
                handle=self.model.denoiser.register_forward_pre_hook(hook)
            try:result=predict(condition,generator,*a,**kw)
            finally:
                if first:handle.remove()
            if first:stash('00_plan',result);stash('00_rng_after',generator.get_state())
            return result
        self.model.observe=observed;self.model.predict=predicted
    def act(self,packet,*a,**kw):
        tick=int(packet['step_id'][0])
        if capture and tick<12:
            for k in ('proprio','actuator_target_rad','tracking_error_rad','time_s','camera_time_s','camera_valid'):
                stash(f'{tick:02d}_packet_{k}',packet[k])
            if tick==0:
                stash('00_raw_image',packet['image']);stash('00_raw_depth_valid',packet['depth_valid'])
                stash('00_rgb_u8',(packet['image'][...,:3,:,:]*255).round().clamp(0,255).to(torch.uint8))
        result=old_act(self,packet,*a,**kw)
        if capture and tick<12:stash(f'{tick:02d}_command',result)
        if capture and tick==11:
            np.savez_compressed(output/'image_pair_inputs.npz',**rows)
            write(output/'image_pair_capture.json',dict(complete=True,fields=list(rows),ticks=12,read_only=True,expert_takeover=False))
        return result
    cls.__init__,cls.act=init,act
    run_audited_episode(root,output,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;run(a.root,a.output,a.job)
