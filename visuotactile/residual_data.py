"""Immutable, causal B0 replay caches over full expert observation sequences."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compact import EpisodeFile, sha256
from .correction import remaining_nominal_chunk
from .history import recurrent_scan
from .manifest import ManifestSequenceDataset
from .models.residual import observed_contact
from .residual_session import SessionGate
from .rollout import load_command_controller
from .training_resources import atomic_json


BASE = Path('outputs/visuotactile/run/dp_command_20261029/b0/best.pt')
MANIFEST = Path('outputs/visuotactile/data/close_v4/manifest.json')
BASE_HASH = 'fd1bd263de38ea6c6e0082d2ba55121adf3908b31c78b55d1cf39477ad6ff9d5'


def load_base(device='cpu'):
    if sha256(BASE) != BASE_HASH:
        raise ValueError('Formal B0 identity changed')
    controller = load_command_controller(BASE, 1, device=device)
    if controller.variant != 'b0' or controller.model.cfg.action_history_mode != 'none':
        raise ValueError('Residual requires tactile-free formal B0 v2')
    controller.model.requires_grad_(False).eval()
    return controller


def splits():
    saved = torch.load(BASE, map_location='cpu', weights_only=False)
    return {name: saved[field] for name, field in (
        ('train', 'train_episodes'), ('validation', 'validation_episodes'), ('test', 'test_episodes'))}


def cache_episode(dataset, episode, model, device, gate):
    row = dataset.reader.entries[episode]
    path = dataset.reader.path.parent / row['directory'] / 'data.h5'
    if sha256(path) != row['files']['data.h5']['sha256']:
        raise ValueError('Source demonstration checksum mismatch')
    seed = int(hashlib.sha256(episode.encode()).hexdigest()[:8], 16)
    generator = torch.Generator(device=device).manual_seed(seed)
    with EpisodeFile(path) as stream:
        group = stream['episodes'][row['episode']]
        length = int(group.attrs['length'])
        actions = torch.from_numpy(group['transition/command_clipped_pre_delay'][:length]).float()
        before = torch.from_numpy(group['transition/time_before_s'][:length]).double()
        after = torch.from_numpy(group['transition/time_after_s'][:length]).double()
        times = torch.from_numpy(group['obs/time_s'][:length]).double()
        if not torch.allclose(after[:-1], times[1:], atol=1e-6, rtol=0) or not torch.equal(before, times):
            raise ValueError('Teacher-forced execution feedback is misaligned')
        if not group['transition/control_trace/valid'][:length].all():
            raise ValueError('Native execution feedback is missing')
        force = torch.from_numpy(group['obs/contact_w_N'][:length]).float()
        contact_valid = torch.from_numpy(group['obs/contact_valid'][:length]).bool()
        motion = {name: torch.from_numpy(group['obs/'+name][:length]).float()
                  for name in ('proprio', 'actuator_target_rad', 'tracking_error_rad')}
        previous = torch.cat((torch.zeros_like(actions[:1]), actions[:-1]))
        motion.update(previous_command=previous, command_valid=torch.arange(length)>0)
        tactile = {target: torch.from_numpy(group['obs/'+source][:length]).flatten(0, 1)
            for source, target in (('contact_substeps_w_N', 'force_w_N'),
                ('contact_substeps_local_N', 'force_local_N'), ('finger_position_base_m', 'position_base_m'),
                ('finger_rotation_base', 'rotation_base'), ('contact_time_s', 'time_s'))}
        if ((tactile['time_s'].reshape(length, 2)>times[:, None]+1e-6) & contact_valid[:, None]).any():
            raise ValueError('Future tactile input')
    memory = None
    plan = None
    features, plans, masks, bases, cursors, generations = [], [], [], [], [], []
    with torch.no_grad():
        for chunk in dataset.chunks(episode, 8):
            gate.before()
            began = time.monotonic()
            obs = {name: value.to(device) for name, value in chunk['obs'].items()}
            if obs['prior_plan_valid'].any() or obs['previous_nominal_valid'].any():
                raise ValueError('B0 no-own-plan observations must stay masked')
            encoded, valid = model.encoder.encode_frames(obs)
            condition, memory = recurrent_scan(model.encoder.temporal, encoded, valid, memory)
            features.append(encoded[0].cpu())
            for offset in range(condition.shape[1]):
                step = chunk['start']+offset
                cursor = step % model.cfg.replan_steps
                if cursor == 0:
                    plan = model.predict_from_condition(condition[:, offset], generator=generator).cpu()
                remaining, plan_valid = remaining_nominal_chunk(plan, torch.tensor([cursor]))
                plans.append(remaining[0]); masks.append(plan_valid[0])
                bases.append(plan[0, cursor].clamp(-1, 1)); cursors.append(cursor)
                generations.append(step-cursor)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            gate.pace(began)
    if any(parameter.grad is not None or parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('Base freeze violated')
    return dict(cache_version=1, episode=episode, steps=length, base_sha256=BASE_HASH,
        source_sha256=row['files']['data.h5']['sha256'], policy_noise_seed=seed,
        features=torch.cat(features), plans=torch.stack(plans), plan_valid=torch.stack(masks),
        base=torch.stack(bases), cursor=torch.tensor(cursors), generation_step=torch.tensor(generations),
        time_s=times, motion=motion, tactile=tactile,
        tactile_valid=contact_valid.repeat_interleave(2), gate=observed_contact(force, contact_valid),
        target=actions, history_source='past_executed_expert_teacher_forcing',
        plan_source='frozen_formal_B0_predictions_on_expert_observations',
        precision='float32', encoder_cache='frozen_trained_frame_features; independent trainable temporal GRU')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'full', 'test'), default='pilot')
    args = parser.parse_args()
    gate = SessionGate(args.root, gpu_stage=True)
    gate.before(reserve_s=60)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    controller = load_base('cuda:0')
    split = splits()
    selected = (split['train'][:16]+split['validation'][:8] if args.stage == 'pilot' else
                split['train']+split['validation'] if args.stage == 'full' else split['test'])
    cache = args.root / 'cache'
    cache.mkdir(exist_ok=True)
    identity_path = cache / 'index.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else dict(
        cache_version=1, base_sha256=BASE_HASH, manifest_sha256=sha256(MANIFEST), splits=split,
        config=asdict(controller.model.cfg), entries={}, source_sha256={
            str(path): sha256(path) for path in (Path(__file__), Path('visuotactile/models/residual.py'),
                                                Path('visuotactile/rollout.py'))})
    dataset = ManifestSequenceDataset(MANIFEST, ids=selected, stage='production')
    try:
        for episode in selected:
            if episode in identity['entries']:
                if sha256(cache / (episode+'.pt')) != identity['entries'][episode]['sha256']:
                    raise ValueError('Cache changed after generation')
                continue
            gate.before(reserve_s=60)
            record = cache_episode(dataset, episode, controller.model, 'cuda:0', gate)
            temporary = cache / (episode+'.partial.pt')
            torch.save(record, temporary)
            temporary.replace(cache / (episode+'.pt'))
            identity['entries'][episode] = dict(sha256=sha256(cache / (episode+'.pt')),
                steps=record['steps'], contact_steps=int(record['gate'].sum()),
                source_sha256=record['source_sha256'], policy_noise_seed=record['policy_noise_seed'])
            atomic_json(identity_path, identity)
            print(json.dumps(dict(cached=episode, count=len(identity['entries']), steps=record['steps'],
                                  gate_steps=int(record['gate'].sum()), wait_s=gate.wait_s)), flush=True)
    except InterruptedError as error:
        atomic_json(args.root / ('cache_'+args.stage+'_stopped.json'), dict(reason=str(error),
                    complete_episodes=len(identity['entries']), wait_s=gate.wait_s))
    finally:
        if sha256(BASE) != BASE_HASH:
            raise RuntimeError('Base checkpoint mutated')


if __name__ == '__main__':
    main()
