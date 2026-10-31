"""A separate naturally executed expert sample and same-scene recorded replay."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .compact import sha256
from .diagnostic_recording import snapshot_before_policy, write_policy_step, write_tensors
from .recording import flatten, numpy_tree
from .training_resources import atomic_json


def append_numeric(group, values, step):
    for name,value in flatten(numpy_tree(values)).items():
        array=np.asarray(value)[0]
        if step==0:
            parent,_,leaf=name.rpartition('/')
            group.require_group(parent).create_dataset(leaf,shape=(0,*array.shape),maxshape=(None,*array.shape),
                dtype=array.dtype,chunks=(1,*array.shape),compression='lzf')
        dataset=group[name]
        dataset.resize(step+1,axis=0)
        dataset[step]=array


def fresh_expert_and_replay(env,lease,output,*,wrench_reconstruction='native_request'):
    from evaluation.eval_isaacsim import _configure_agent
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights
    from .evaluate_recorded_replay import run_trial
    task='Isaacsimenvs-PreciseAssembly-Direct-v0'
    checkpoint=Path('pretrained_assembly/tight_insertion/model.pth')
    with lease.block():
        config=_configure_agent(task,'rl_games_sapg_cfg_entry_point',rl_device='cpu',num_envs=1,
            deterministic=True,games=1,extra_overrides={})
        register_rlgames_env(env,rl_device='cpu',clip_obs=config['params']['env']['clip_observations'],clip_actions=1.)
        runner=Runner()
        runner.load(config)
        runner.reset()
        player=runner.create_player()
        player.set_weights(_load_checkpoint_weights(player,str(checkpoint)))
        player.has_batch_dimension=True
        player.reset()
        if player.intr_reward_coef_embd is not None:
            player.intr_reward_coef_embd.fill_(50.)
        env.clear_recorded_events()
        teacher,_=env.reset(seed=17)
        env.configure_camera_views((-.7,-.35,1.15),(0.,-.03,.68))
        metadata=env.research_metadata()
        metadata.update(config=env.cfg.to_dict(),env_origins_w=env.scene.env_origins.cpu().tolist(),
            source_env_index=0,num_parallel_envs=1,expert_checkpoint_sha256=sha256(checkpoint),seed=17,
            purpose='independent diagnostic success sample; never merged into original400')
        tensors={name:value.detach().cpu().clone() for name,value in vars(env).items() if torch.is_tensor(value)}
        physics={}
        for name in ('robot','object','table','hole','goal_viz'):
            body=getattr(env,name)
            physics[name]={'root_state_w':body.data.root_state_w.clone()}
            view=body.root_physx_view
            for label,method in (('masses','get_masses'),('inertias','get_inertias'),('materials','get_material_properties'),
                                 ('dof_stiffness','get_dof_stiffnesses'),('dof_damping','get_dof_dampings')):
                if hasattr(view,method):
                    physics[name][label]=getattr(view,method)().clone()
        torch.save(dict(env_tensors=tensors,physics=physics,actor_states=player.states,
            cpu_rng=torch.get_rng_state(),missing=['PhysX contact/solver internal warm-start and broadphase state']),output/'initial_export.pt')
    source_path=output/'expert_numeric.h5'
    bound=(int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
    with h5py.File(source_path,'w') as stream,h5py.File(output/'expert_inputs.h5','w') as inputs:
        stream.attrs['metadata_json']=json.dumps(metadata)
        group=stream.create_group('episode')
        for step in range(bound):
            packet=env.research_packet
            with lease.block():
                snapshot=snapshot_before_policy(env)
                before_inference=torch.get_rng_state().clone()
                with torch.no_grad():
                    actor=player.obs_to_torch(teacher['policy'])
                    if player.intr_reward_coef_embd is not None:
                        actor=torch.cat((actor,player.intr_reward_coef_embd),dim=1)
                    command=player.get_action(actor,is_deterministic=True)
                before_step=torch.get_rng_state().clone()
                saved=write_policy_step(inputs,step,packet,snapshot,command)
                teacher,_,terminated,truncated,_=env.step(command)
                transition=env.last_transition
                transition.update(torch_rng_before_step=before_step[None],torch_rng_before_inference=before_inference[None],
                    expert_actor_input_diagnostic=actor.clone())
                write_tensors(saved.create_group('transition'),transition)
                numeric_packet={name:value for name,value in packet.items() if torch.is_tensor(value) and name not in ('image','depth_m','depth_valid')}
                append_numeric(group,dict(obs=numeric_packet,transition=transition),step)
            if step%100==0:
                atomic_json(output/'expert_progress.json',dict(step=step))
            if bool(terminated[0] or truncated[0]):
                break
        reasons={name:bool(value[0]) for name,value in transition['termination_reasons'].items()}
        expert=dict(success=reasons['max_successes'],steps=step+1,reasons=reasons,
            lift=bool(transition['diagnostics_after']['lifted_object'][0]),
            subgoals=int(transition['diagnostics_after']['successes'][0]),checkpoint_sha256=sha256(checkpoint))
        atomic_json(output/'expert_result.json',expert)
        stream.flush()
    if not expert['success']:
        atomic_json(output/'report.json',dict(expert=expert,replay=None,complete=False,reason='Fresh expert did not succeed; no replacement seed'))
        return
    with h5py.File(source_path,'r') as stream:
        replay=run_trial(env,lease,output/'replay',stream['episode'],metadata,mode='delay_wrench_fixed',line='T0_fresh',
            controller=None,seed=17,exported_state=torch.load(output/'initial_export.pt',map_location='cpu',weights_only=False),
            wrench_reconstruction=wrench_reconstruction)
    atomic_json(output/'report.json',dict(expert=expert,replay=replay,complete=True,
        scope='Single lane, same instantiated scene, observable restoration and recorded events; exported state does not imply full PhysX restoration'))
