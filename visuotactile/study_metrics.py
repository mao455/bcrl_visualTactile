"""Matched x0 objectives and sampled native-command metrics by group/horizon."""
import json
import math

import torch

from .action_reference import reference_to_command
from .long_data import inputs_for, load_record, supervision
from .train_long import move


def joint_groups(joint_order):
    if len(joint_order) != 29 or not all(name.startswith('iiwa14_joint_') for name in joint_order[:7]):
        raise ValueError('Expected canonical7 arm and22 hand joints')
    groups = {'arm':list(range(7)),'hand':list(range(7,29))}
    for finger in ('thumb','index','middle','ring','pinky'):
        groups[finger] = [index for index,name in enumerate(joint_order) if finger in name]
    if sorted(sum([groups[name] for name in ('thumb','index','middle','ring','pinky')],[])) != groups['hand']:
        raise ValueError('Finger names must partition canonical hand joints')
    return groups


def masked_objectives(prediction, target, valid):
    error = (prediction.float()-target.float()).square()
    error = torch.where(valid[...,None],error,0.)
    denominator = valid.sum().clamp_min(1)
    arm = error[...,:7].mean(-1).sum()/denominator
    hand = error[...,7:].mean(-1).sum()/denominator
    return dict(A=error.mean(-1).sum()/denominator,B=.5*(arm+hand),arm=arm,hand=hand)


def denoising_objectives(model, condition, target, valid, generator=None):
    if model.cfg.prediction_type != 'sample' or model.noise.config.prediction_type != 'sample':
        raise ValueError('This comparison requires identical sample/x0 targets')
    normalized = torch.where(valid[...,None],target*model.target_scale+model.target_offset,0.)
    noise = torch.randn(normalized.shape,device=target.device,generator=generator)
    timestep = torch.randint(100,(len(target),),device=target.device,generator=generator)
    noisy = model.noise.add_noise(normalized,noise,timestep)
    prediction = model.denoiser(noisy,timestep,global_cond=condition)
    return masked_objectives(prediction,normalized,valid)


@torch.no_grad()
def evaluate(model, root, identities, lease, seed=7091, trajectories=False):
    metadata = json.loads((root/'metadata.json').read_text())
    groups = joint_groups(metadata['joint_order'])
    limits = torch.tensor(metadata['joint_limits_rad'][0])
    speed,dt = metadata['config']['action']['dof_speed_scale'],metadata['control_dt']
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    loss_generator = torch.Generator(device=device).manual_seed(seed+1)
    sums = {name:torch.zeros(64,29,dtype=torch.float64) for name in ('reference','normalized','command','zero')}
    counts = torch.zeros(64,dtype=torch.float64)
    loss_sums = {name:0. for name in ('A','B','arm','hand')}
    saturated = anchors_count = 0
    representative = []
    model.eval()
    for ordinal,identity in enumerate(identities):
        record = load_record(root,identity)
        state = None
        for start in range(0,len(record['time']),96):
            stop = min(start+96,len(record['time']))
            anchors = torch.arange(start,stop,12)
            inputs_cpu = inputs_for(record,start,stop,model.cfg.tactile)
            labels_cpu = supervision(record,anchors)
            with lease.block():
                inputs,labels = move(inputs_cpu,device),move(labels_cpu,device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    condition,state = model.encode(inputs,state)
                    losses = denoising_objectives(model,condition,labels['reference'],labels['valid'],loss_generator)
                    prediction = model.predict_from_condition(condition,generator).float()
                command = reference_to_command(prediction,labels['target_before'],limits,speed_scale=speed,control_dt=dt)
                valid = labels['valid']
                count = int(valid.sum())
                for name,value in losses.items():
                    loss_sums[name] += float(value)*count
                errors = dict(reference=prediction-labels['reference'],
                              normalized=(prediction-labels['reference'])*model.target_scale,
                              command=command-labels['command'],zero=labels['command'])
                for name,error in errors.items():
                    sums[name] += torch.where(valid[...,None],error.square(),0.).sum(0).double().cpu()
                counts += valid.sum(0).double().cpu()
                saturated += int(((command.abs()>=.99999)&valid[...,None]).sum())
                anchors_count += len(anchors)
                if trajectories and ordinal == 0:
                    for index,anchor in enumerate(anchors):
                        length = int(valid[index,:12].sum())
                        representative.append(dict(step=int(anchor),expert=labels['command'][index,:length].cpu().tolist(),
                                                   predicted=command[index,:length].cpu().tolist()))
    result = dict(episodes=len(identities),anchors=anchors_count,valid_future_positions=int(counts.sum()),
                  horizon_counts=counts.tolist(),diffusion_loss={name:value/max(1,float(counts.sum())) for name,value in loss_sums.items()},
                  command_saturation_fraction=saturated/max(1,29*float(counts.sum())),metrics={},horizon={})
    for space,values in sums.items():
        result['metrics'][space] = {}
        result['horizon'][space] = {}
        for name,indices in groups.items():
            squared = values[:,indices].mean(-1)
            result['metrics'][space][name] = dict(
                full_rmse=math.sqrt(float(squared.sum()/counts.sum().clamp_min(1))),
                prefix12_rmse=math.sqrt(float(squared[:12].sum()/counts[:12].sum().clamp_min(1))))
            result['horizon'][space][name] = torch.sqrt(squared/counts.clamp_min(1)).tolist()
    result['selection_mse'] = .5*sum(result['metrics']['command'][name]['prefix12_rmse']**2 for name in ('arm','hand'))
    result['scope'] = 'teacher-forced expert controller targets for future conversion; not predicted future targets'
    if trajectories:
        result['representative'] = dict(episode=identities[0],blocks=representative)
    return result
