"""Behavior-neutral snapshots for future approximate-state diagnostics."""
import numpy as np
import torch


def snapshot_before_policy(env, controller=None):
    snapshot = dict(cpu_rng_before_policy=torch.get_rng_state().clone(),
        action_queue_lab=env._action_queue.clone(),target_before_lab=env._prev_targets.clone(),
        stage_step=env.episode_length_buf.clone(),successes=env._successes.clone())
    if controller is not None:
        snapshot['sampling_rng_before_policy'] = controller.generator.get_state().clone()
        same_episode = controller.episode == int(env.research_packet['episode_id'][0])
        if same_episode and controller.state is not None:
            snapshot['memory_before_policy'] = controller.state.detach().clone()
    return snapshot


def write_tensors(group, values):
    for name,value in values.items():
        if isinstance(value,dict):
            write_tensors(group.create_group(name),value)
        elif torch.is_tensor(value):
            tensor = value.detach().cpu()
            array = (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()
            dataset = group.create_dataset(name,data=array,compression='lzf' if array.ndim else None)
            dataset.attrs['torch_dtype'] = str(value.dtype)


def write_policy_step(stream, step, packet, snapshot, command, controller=None):
    stream.attrs['diagnostic_input_version'] = 2
    stream.attrs['image_scope'] = 'RGB-D on each consumed DINO refresh (step % 12 == 0); all numeric packet fields every native tick'
    stream.attrs['restoration_scope'] = 'observable and RNG diagnostics; not complete PhysX state'
    stream.attrs['torch_version'] = str(torch.__version__)
    group = stream.create_group(str(step))
    for name,value in packet.items():
        if not torch.is_tensor(value) or name == 'depth_m':
            continue
        if name == 'image':
            if step%12 == 0:
                image = value[0].detach().cpu().numpy()
                group.create_dataset('rgb_u8',data=np.rint(image[:,:3]*255).astype(np.uint8),compression='lzf')
                group.create_dataset('depth_m',data=image[:,3:],compression='lzf')
        elif name != 'depth_valid' or step%12 == 0:
            array = value[0].detach().cpu().numpy()
            group.create_dataset(name,data=array,compression='lzf' if array.ndim else None)
    write_tensors(group.create_group('before_policy'),snapshot)
    write_tensors(group,dict(command=command,cpu_rng_before_step=torch.get_rng_state()))
    if controller is not None:
        write_tensors(group.create_group('policy'),dict(condition=controller.last_condition,
            reference=controller.reference,memory=controller.state,action_history=controller.records['action'][-1],
            sampling_rng_after_policy=controller.generator.get_state()))
        if step%12 == 0:
            group.create_dataset('plan',data=controller.plan[0].detach().cpu().numpy(),compression='lzf')
            group.create_dataset('condition',data=controller.last_condition[0].detach().cpu().numpy())
    return group
