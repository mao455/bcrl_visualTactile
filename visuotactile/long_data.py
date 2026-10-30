"""Versioned frozen feature cache and identical causal offline/online windows."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as functional

from .action_reference import command_to_reference, reference_to_command
from .compact import sha256
from .long_resources import Lease
from .models.long_horizon import FrozenDinoSpatial, DINO_REVISION
from .training_resources import atomic_json


BASE = Path('outputs/visuotactile/run/dp_command_20261029/b0/best.pt')
MANIFEST = Path('outputs/visuotactile/data/close_v4/manifest.json')
CAMERAS = ['fixed_high','fixed_side','wrist_rgbd']


def depth_grid(depth, valid):
    depth = torch.as_tensor(depth).float()
    valid = torch.as_tensor(valid).bool()
    leading = depth.shape[:-3]
    metric = torch.where(valid,depth,0.).reshape(-1,1,252,448)
    mask = valid.float().reshape_as(metric)
    coverage = functional.avg_pool2d(mask,14,14)
    average = functional.avg_pool2d(metric,14,14)/coverage.clamp_min(1e-6)
    return torch.cat((average/2.,coverage),1).reshape(*leading,2,18,32).half()


def proprio_radians(proprio, limits):
    result = proprio.clone()
    result[..., :29] = limits[:,0]+.5*(proprio[..., :29]+1)*(limits[:,1]-limits[:,0])
    return result


def tactile_values(world, local, stamps, observed, valid):
    values = torch.cat((world,local),-1).flatten(-2)
    values = values.sign()*torch.log1p(values.abs())
    age = (observed[...,None]-stamps).float()
    mask = valid[...,None].expand_as(stamps)
    if ((age < -1e-6)&mask).any():
        raise ValueError('Future tactile sample')
    return torch.where(mask[...,None],torch.cat((values,age[...,None],mask[...,None].float()),-1),0.)


def shift_actions(commands, delayed, time_before, source_time, observation_time):
    length = len(commands)
    previous = (torch.arange(length)-1).clamp_min(0)
    valid = torch.arange(length)>0
    ages = torch.stack((observation_time-time_before[previous],observation_time-source_time[previous]),-1).float()
    if (ages[valid,0] <= 0).any() or (ages[valid,1] < ages[valid,0]-1e-6).any():
        raise ValueError('Action feedback must strictly precede observation')
    value = torch.cat((commands[previous],delayed[previous],ages,valid[:,None].float()),-1)
    return torch.where(valid[:,None],value,0.)


def window(values, anchors, width):
    indices = anchors[:,None]+torch.arange(1-width,1)
    valid = indices >= 0
    selected = values[indices.clamp_min(0)]
    mask = valid.reshape(*valid.shape,*([1]*(selected.ndim-2)))
    return torch.where(mask,selected,0.),valid


def inputs_for(record, start, stop, tactile=False, anchors=None):
    if anchors is None:
        anchors = torch.arange(start,stop)
        anchors = anchors[anchors%12 == 0]
    visual_at = anchors//12
    visual_indices = visual_at[:,None]+torch.arange(-10,1)
    visual_valid = visual_indices >= 0
    first = max(0,start//12-10)
    last = (stop-1)//12+1
    visual_indices = visual_indices.clamp_min(0)
    held = torch.arange(start,stop)//12
    camera_time = record['camera_time']
    times = record['time']
    visual_age = times[anchors,None,None]-camera_time[visual_indices]
    held_age = times[start:stop,None]-camera_time[held]
    if (held_age < -1e-6).any() or (visual_age[visual_valid] < -1e-6).any():
        raise ValueError('Future visual observation')
    q_history,native_valid = window(record['proprio'],anchors,121)
    action_history,action_valid = window(record['action'],anchors,121)
    time_history,_ = window(times,anchors,121)
    action_valid &= action_history[...,-1].bool()
    result = dict(patches=record['patches'][first:last],depth=record['depth'][first:last],
        camera_valid=record['camera_valid'][first:last],visual_indices=visual_indices-first,
        visual_valid=visual_valid,visual_age=torch.where(visual_valid[...,None],visual_age,0.),
        held_indices=held-first,held_age=held_age,proprio=record['proprio'][start:stop],
        action=record['action'][start:stop],time=times[start:stop],window_proprio=q_history,
        window_action=action_history,window_valid=native_valid,window_action_valid=action_valid,
        window_age=torch.where(native_valid,times[anchors,None]-time_history,0.),anchor_indices=anchors-start)
    if tactile:
        force_history,force_valid = window(record['tactile'],anchors,121)
        force_history = force_history.flatten(1,2)
        force_valid = force_valid.repeat_interleave(2,1)&force_history[...,-1].bool()
        result.update(tactile=record['tactile'][start:stop],tactile_valid=record['tactile'][start:stop,...,-1].bool(),
                      window_tactile=force_history,window_tactile_valid=force_valid)
    return result


def supervision(record, anchors):
    indices = anchors[:,None]+torch.arange(64)
    valid = indices < len(record['time'])
    clipped = indices.clamp_max(len(record['time'])-1)
    return {name:torch.where(valid[...,None],record[name][clipped],0.)
            for name in ('reference','command','target_before','joint_after','target_after')}|{'valid':valid}


def load_record(root, identity):
    record = torch.load(Path(root)/'cache'/f'{identity}.pt',map_location='cpu',weights_only=False)
    if record.get('version') != 2:
        raise ValueError('Reject invalid preliminary cache')
    return record


def cache(root):
    root = Path(root)
    lease = Lease(root)
    directory = root/'cache'
    directory.mkdir(exist_ok=False)
    saved = torch.load(BASE,map_location='cpu',weights_only=False)
    metadata = saved['metadata']
    manifest = json.loads(MANIFEST.read_text())
    splits = {name:saved[key] for name,key in (('train','train_episodes'),('validation','validation_episodes'),('test','test_episodes'))}
    if set(sum(splits.values(),[])) != {row['id'] for row in manifest['episodes']}:
        raise ValueError('Original fixed split does not cover manifest')
    atomic_json(root/'splits.json',splits)
    atomic_json(root/'metadata.json',metadata)
    limits = torch.tensor(metadata['joint_limits_rad'][0])
    speed = metadata['config']['action']['dof_speed_scale']
    dt = metadata['control_dt']
    if metadata['joint_order'][:7] != metadata['joint_order_lab'][:7]:
        raise ValueError('Native arm-first mapping differs')
    with lease.block():
        dino = FrozenDinoSpatial().to('cuda:0')
    index = dict(version=2,manifest_sha256=sha256(MANIFEST),source_split_sha256=sha256(BASE),
        dino_revision=DINO_REVISION,dino_weights_sha256=sha256(Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'),
        dino_preprocess='RGB/255; original 252x448; ImageNet mean/std; no crop/resize; full 18x32 patch tokens',
        camera_order=CAMERAS,splits=splits,entries={},roundtrip_max=0.,native_formula_max=0.,
        arm_references_outside_limits=0,delay_indices={},started=time.time())
    expected_bytes=sum((row['length']+11)//12 for row in manifest['episodes'])*3*384*18*32*2
    atomic_json(root/'cache_estimate.json',dict(frozen_patch_bytes=expected_bytes,peak_rgb_read_frames=2,
        one_episode_patch_bytes_max=max((row['length']+11)//12 for row in manifest['episodes'])*3*384*18*32*2))
    for number,row in enumerate(manifest['episodes'],1):
        lease.check()
        path = MANIFEST.parent/row['directory']/'data.h5'
        with h5py.File(path,'r') as stream:
            group = stream['episodes'][row['episode']]
            length = int(group.attrs['length'])
            meta = json.loads(stream.attrs['metadata_json'])
            if meta['camera_ids'] != CAMERAS or row['length'] != length:
                raise ValueError('Source camera/length mismatch')
            def read(name):
                return torch.from_numpy(group[name][:length])
            time_values = read('obs/time_s')
            command = read('transition/command_clipped_pre_delay')
            delayed = read('transition/control_trace/delayed_command')
            before = read('transition/target_before_rad')
            if not torch.equal(before,read('obs/actuator_target_rad')):
                raise ValueError('Submission target identity differs')
            reference = command_to_reference(command,before,limits,speed_scale=speed,control_dt=dt)
            recovered = reference_to_command(reference,before,limits,speed_scale=speed,control_dt=dt)
            error = float((recovered-command).abs().max())
            index['roundtrip_max'] = max(index['roundtrip_max'],error)
            if error > 1e-5:
                raise ValueError('Actual oracle roundtrip failed')
            raw = command_to_reference(delayed,before,limits,speed_scale=speed,control_dt=dt)
            raw[:,:7] = raw[:,:7].clamp(limits[:7,0],limits[:7,1])
            mix = torch.tensor([meta['config']['action']['arm_moving_average']]*7+
                               [meta['config']['action']['hand_moving_average']]*22)
            expected = (mix*raw+(1-mix)*before).clamp(limits[:,0],limits[:,1])
            formula_error = float((expected-read('transition/target_after_rad')).abs().max())
            index['native_formula_max'] = max(index['native_formula_max'],formula_error)
            if formula_error > 1e-6:
                raise ValueError('Native delay/smoothing formula differs')
            index['arm_references_outside_limits'] += int(((reference[:,:7]<limits[:7,0])|
                                                         (reference[:,:7]>limits[:7,1])).sum())
            delay_ids,counts = read('transition/control_trace/delay_index').unique(return_counts=True)
            for delay,count in zip(delay_ids.tolist(),counts.tolist()):
                index['delay_indices'][str(delay)] = index['delay_indices'].get(str(delay),0)+count
            if not read('transition/control_trace/valid').all():
                raise ValueError('Native execution telemetry missing')
            record = dict(version=2,identity=row['id'],time=time_values,
                proprio=proprio_radians(read('obs/proprio'),limits),command=command,reference=reference,
                target_before=before,target_after=read('transition/target_after_rad'),
                joint_after=read('transition/diagnostics_after/joint_position_rad'),
                action=shift_actions(command,delayed,read('transition/time_before_s'),
                    read('transition/control_trace/selected_command_time_s'),time_values),
                tactile=tactile_values(read('obs/contact_substeps_w_N'),read('obs/contact_substeps_local_N'),
                    read('obs/contact_time_s'),time_values,read('obs/contact_valid')))
            anchors = torch.arange(0,length,12)
            record['camera_time'] = read('obs/camera_time_s')[anchors]
            record['camera_valid'] = read('obs/camera_valid')[anchors]
            patches,depths = [],[]
            for start in range(0,len(anchors),2):
                selection = anchors[start:start+2].numpy()
                rgb = torch.from_numpy(group['frames/rgb_u8'][selection])
                depths.append(depth_grid(group['frames/depth_m'][selection],group['frames/depth_valid'][selection]))
                with lease.block():
                    patches.append(dino(rgb.to('cuda:0')).cpu())
            record['patches'] = torch.cat(patches)
            record['depth'] = torch.cat(depths)
        target = directory/(row['id']+'.pt')
        torch.save(record,target)
        index['entries'][row['id']] = dict(bytes=target.stat().st_size,sha256=sha256(target),
            native_rows=length,anchors=len(anchors),future_targets=int(supervision(record,anchors)['valid'].sum()),
            source_file_sha256=row['files']['data.h5']['sha256'])
        atomic_json(root/'cache_progress.json',dict(complete=number,total=400,last=row['id']))
        if number%20 == 0:
            print('CACHE',number,flush=True)
    index['completed'] = time.time()
    atomic_json(root/'cache_index.json',index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache(args.root)


if __name__ == '__main__':
    main()
