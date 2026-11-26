"""Unchanged released expert creation and observable initial-state provenance."""
import copy
import hashlib
import json
import torch

def create_expert(env, checkpoint):
    from evaluation.eval_isaacsim import _configure_agent
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights
    config = _configure_agent('Isaacsimenvs-PreciseAssembly-Direct-v0', 'rl_games_sapg_cfg_entry_point',
        rl_device='cpu', num_envs=env.num_envs, deterministic=True, games=1, extra_overrides={})
    register_rlgames_env(env, rl_device='cpu', clip_obs=config['params']['env']['clip_observations'], clip_actions=1.)
    runner = Runner(); runner.load(config); runner.reset()
    player = runner.create_player(); player.set_weights(_load_checkpoint_weights(player, str(checkpoint)))
    player.has_batch_dimension = True; player.reset()
    if player.intr_reward_coef_embd is not None:
        player.intr_reward_coef_embd.fill_(50.)
    return player, config


def export_initial(env, player):
    physics = {}
    for name in ('robot', 'object', 'table', 'hole', 'goal_viz'):
        body = getattr(env, name); view = body.root_physx_view
        physics[name] = dict(root_state_w=body.data.root_state_w.clone())
        for label, method in (('masses', 'get_masses'), ('inertias', 'get_inertias'),
                              ('materials', 'get_material_properties'), ('dof_stiffness', 'get_dof_stiffnesses'),
                              ('dof_damping', 'get_dof_dampings'), ('coms', 'get_coms')):
            if hasattr(view, method):
                physics[name][label] = getattr(view, method)().clone()
    return dict(env_tensors={name: value.detach().cpu().clone() for name, value in vars(env).items()
                            if torch.is_tensor(value)}, physics=physics, actor_states=copy.deepcopy(player.states),
                cpu_rng=torch.get_rng_state(), missing=['PhysX contact/solver/broadphase internal state'])


def environment_identity(env):
    config = json.loads(json.dumps(env.cfg.to_dict(), default=str))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config, digest
def restore_observable_state(env, group, metadata):
    import numpy as np
    import torch
    initial={name:torch.from_numpy(np.asarray(value[0]).copy()).reshape(1,*value.shape[1:])
             for name,value in group['transition/diagnostics_before'].items()}
    original_origin=torch.tensor(metadata['env_origins_w'][0])
    target_origin=env.scene.env_origins[0]
    for name,key,velocity in (('object','object_pose_w','object_velocity_w_m_s_rad_s'),
        ('table','table_pose_w',None),('hole','hole_pose_w',None),('goal_viz','goal_pose_w','goal_velocity_w_m_s_rad_s')):
        pose=initial[key].clone()
        pose[:,:3]+=target_origin-original_origin
        body=getattr(env,name)
        body.write_root_pose_to_sim(pose)
        if name=='object':
            body.write_root_velocity_to_sim(initial[velocity])
        elif velocity and bool(initial[velocity].abs().max()>0):
            raise ValueError('Nonzero recorded kinematic velocity cannot be restored through a dynamic-body API')
        if name in ('table','hole'):
            env._kinematic_reset_poses[name]=pose[:,:3].clone()
    permutation=env._perm_canon_to_lab
    env.robot.write_joint_state_to_sim(initial['joint_position_rad'][:,permutation],initial['joint_velocity_rad_s'][:,permutation])
    target=torch.from_numpy(group['obs/actuator_target_rad'][0:1])[:,permutation]
    env._prev_targets[:]=target
    env._cur_targets[:]=target
    env.robot.set_joint_position_target(target)
    env._table_z_per_env[:]=initial['table_pose_w'][:,2]-original_origin[2]
    env.hole_pos[:]=initial['hole_pose_w'][:,:3]-original_origin
    env.hole_quat_wxyz[:]=initial['hole_pose_w'][:,3:]
    for destination,key in (('_successes','successes'),('retract_phase','retract_phase'),
        ('retract_succeeded','retract_succeeded'),('env_max_goals','max_goals'),('_lifted_object','lifted_object'),
        ('_object_init_z','object_initial_z_m'),('is_random_goal_env','is_random_goal'),('episode_length_buf','native_stage_step')):
        getattr(env,destination)[:]=initial[key].reshape_as(getattr(env,destination))
    for destination,key in (('_random_force_prob','external_force_probability'),('_random_torque_prob','external_torque_probability')):
        getattr(env,destination)[:]=torch.from_numpy(group['transition/'+key][0:1]).reshape_as(getattr(env,destination))
    if env._num_prelude_goals!=0:
        raise ValueError('Saved prelude hidden state unavailable; no approximate reconstruction implemented')
    env._fk_samples=[]
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.)
    env.configure_camera_views((-.7,-.35,1.15),(0.,-.03,.68))
    for camera in env.research_cameras:
        camera.reset()
    env.sim.render()
    env._get_observations()
    torch.set_rng_state(torch.from_numpy(group['transition/torch_rng_before_step'][0]))
    current=env._diagnostics()
    errors={}
    for name in ('joint_position_rad','joint_velocity_rad_s','object_pose_w','object_velocity_w_m_s_rad_s',
                 'table_pose_w','hole_pose_w','goal_pose_w'):
        expected=initial[name].clone()
        if name.endswith('pose_w'):
            expected[:,:3]+=target_origin-original_origin
        errors[name]=float((current[name]-expected).abs().max())
    errors['target']=float((env.research_packet['actuator_target_rad']-torch.from_numpy(group['obs/actuator_target_rad'][0:1])).abs().max())
    return dict(saved_observable_errors=errors,exact_simulator_restore=False,
        missing=['PhysX solver/contact warm-start','sensor histories','other persistent environment states',
                 'native multi-environment RNG consumption sequence','expert observation-delay queues'],
        queue='native reset clears action queue; first native action fills it; source step0 verified',
        rng='single initial CPU RNG restored; subsequent native randomization untouched; multi-environment draw schedule differs')
