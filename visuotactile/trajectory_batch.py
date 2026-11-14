"""Experimental independent full-trajectory training batches.

This module does not change the formal trainer. Each trajectory retains every
real observation and its own recurrent state. The objective is the mean of the
original per-trajectory objectives, not a pooled valid-tick objective. One such
batch is ONE optimizer/EMA update; it is not equivalent to sequential updates.
"""
import torch

from .continuous_vision import frame_features, causal_visual_readouts
from .anchor_sampling import planning_anchors, labels


IDENTITY = 'independent_full_trajectory_batch_v1'


def encode_trajectories(model, records, episodes):
    """Recompute complete histories and batch only independent GRU rows.

    Frame feature extraction retains the original per-trajectory blocks of 32.
    There are no fabricated padding observations, zero-state windows, cross-lane
    attention, or persistent learned caches. An ended trajectory is never fed
    through another recurrent update.
    """
    if (model.contract.get('visual_stride') != 1 or model.modality != 'G00'
            or model.group != 'I-HistoryDP' or not 1 <= len(records) <= 8
            or len(records) != len(episodes)):
        raise ValueError('One to eight complete G00 continuous trajectories required')
    device = next(model.parameters()).device
    prepared = []
    for record, episode in zip(records, episodes):
        n = len(record['time'])
        if not n or episode.visual_stride != 1:
            raise ValueError('Nonempty every-tick episode required')
        robot = record['robot'].to(device)
        time = record['time'].to(device)
        ct = episode.record['camera_time'].to(device)
        cv = episode.record['camera_valid'].to(device)
        if (robot.shape != (n, 178) or ct.shape != (n, 3)
                or cv.shape != (n, 3) or cv.dtype != torch.bool or not cv.all()
                or (ct > time[:, None] + 1e-6).any()
                or not torch.isfinite(ct).all() or not torch.isfinite(robot).all()
                or not torch.isfinite(time).all() or (time[1:] <= time[:-1]).any()
                or len(episode.perception['patches']) != n
                or len(episode.perception['depth']) != n):
            raise ValueError('Complete aligned causal trajectory observations required')
        tokens, summaries = [], []
        for start in range(0, n, 32):
            end = min(start + 32, n)
            token, summary = frame_features(model,
                episode.perception['patches'][start:end].to(device),
                episode.perception['depth'][start:end].to(device),
                time[start:end, None] - ct[start:end], cv[start:end])
            tokens.append(token)
            summaries.append(summary)
        normalized = model.normalized_robot(robot)
        prepared.append(dict(n=n, robot=normalized, time=time, ct=ct, cv=cv,
            tokens=torch.cat(tokens), summaries=torch.cat(summaries),
            inputs=model.native_input(normalized), native=[], visual=[]))
    for tick in range(max(part['n'] for part in prepared)):
        active = [part for part in prepared if tick < part['n']]
        native = model.native_gru(torch.cat([part['inputs'][tick:tick+1] for part in active]),
            None if tick == 0 else torch.cat([part['native'][-1] for part in active]))
        visual = model.visual_gru(torch.cat([part['summaries'][tick:tick+1] for part in active]),
            None if tick == 0 else torch.cat([part['visual'][-1] for part in active]))
        for lane, part in enumerate(active):
            part['native'].append(native[lane:lane+1])
            part['visual'].append(visual[lane:lane+1])
    results = []
    for part in prepared:
        time = part['time']
        anchors = torch.arange(0, part['n'], 12, device=device)
        index = anchors[:, None] + torch.arange(-12, 1, device=device)
        valid = index >= 0
        selected = index.clamp_min(0)
        rows = torch.cat((part['robot'][selected],
            (time[anchors, None] - time[selected]).float()[..., None],
            valid.float()[..., None]), -1)
        rows = torch.where(valid[..., None], rows, 0.).flatten(1)
        readout = causal_visual_readouts(model.vision, part['tokens'], time,
            part['ct'], part['cv'], anchors)
        condition = model.condition_projection(torch.cat((model.robot_encoder(rows),
            readout, torch.cat(part['native'])[anchors], torch.cat(part['visual'])[anchors]), -1))
        results.append((condition, anchors))
    return results


def supervised_trajectories(model, records, episodes):
    results = []
    for record, (conditions, anchors) in zip(records, encode_trajectories(model, records, episodes)):
        keep = torch.isin(anchors, planning_anchors(record).to(anchors.device))
        if not keep.any():
            raise ValueError('Every trajectory needs legal expert planning windows')
        conditions, anchors = conditions[keep], anchors[keep]
        truth, valid = labels(record, anchors, model.horizon)
        results.append(dict(condition=conditions, anchors=anchors, truth=truth, valid=valid))
    return results


def trajectory_objective(model, objective, batches, generator, contexts):
    """One denoiser call; original per-visit RNG draws and geometry allocation.

    Each context must come from that trajectory's original projection_batch.
    In particular, A keeps its original up-to-two hand-geometry selections PER
    trajectory. Different lengths do not reweight trajectories or stages.
    """
    if (not batches or len(batches) != len(contexts) or len(batches) > 8
            or model.horizon != 24 or model.noise.config.prediction_type != 'sample'):
        raise ValueError('Explicit H24 sample trajectory batch and per-episode contexts required')
    targets, noises, ticks = [], [], []
    for batch in batches:
        command, valid = batch['truth'], batch['valid']
        target = torch.where(valid[..., None], command * model.action_scale + model.action_offset, 0.)
        if target.shape != (len(batch['condition']), 24, 29) or not valid.any():
            raise ValueError('Each trajectory requires aligned nonempty legal windows')
        targets.append(target)
        # Preserve randn -> randint for each visit before advancing to the next.
        noises.append(torch.randn(target.shape, device=target.device, generator=generator))
        ticks.append(torch.randint(100, (len(target),), device=target.device, generator=generator))
    prediction = model.denoiser(model.noise.add_noise(torch.cat(targets), torch.cat(noises), torch.cat(ticks)),
        torch.cat(ticks), global_cond=torch.cat([batch['condition'] for batch in batches]))
    losses, reports, offset = [], [], 0
    for batch, context in zip(batches, contexts):
        end = offset + len(batch['condition'])
        values = objective.components(prediction[offset:end], batch['truth'], batch['valid'],
            model.action_scale, model.action_offset, context)
        loss = values['base'] + objective.gain * sum(objective.weights[key] * values[key] for key in objective.weights)
        losses.append(loss)
        reports.append(dict(parts=values, geometry=objective.last_geometry_metrics,
            anchors=len(batch['condition']), target_ticks=int(batch['valid'].sum())))
        offset = end
    return torch.stack(losses).mean(), reports, prediction
