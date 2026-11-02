"""Frozen released Actor recurrence interventions on recorded true Actor inputs."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from .compact import sha256
from .training_resources import atomic_json


def load_actor(checkpoint):
    from rl_games.algos_torch.network_builder import A2CBuilder
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)[0]['model']
    config = yaml.safe_load(Path('isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml').read_text())
    builder = A2CBuilder(); builder.load(config['params']['network'])
    network = builder.build('actor_audit', actions_num=29, input_shape=(141,), num_seqs=1,
        value_size=1, type='extra_param', coef_ids=torch.linspace(50., 0., 6), coef_id_idx=140)
    network.load_state_dict({key.removeprefix('a2c_network.'): value for key, value in saved.items()
                             if key.startswith('a2c_network.')})
    network.eval().requires_grad_(False)
    return network, saved


def normalized(actor, saved):
    value = actor.clone()
    value[:, :140] = ((value[:, :140]-saved['running_mean_std.running_mean'].float()) /
                     torch.sqrt(saved['running_mean_std.running_var'].float()+1e-5)).clamp(-5, 5)
    return value


@torch.no_grad()
def predictions(network, actor, reset_every=None):
    states = tuple(torch.zeros(1, 1, 1024) for _ in range(2))
    rows = []
    for tick in range(len(actor)):
        if reset_every is not None and tick % reset_every == 0:
            states = tuple(torch.zeros_like(value) for value in states)
        mean, _, _, states = network(dict(obs=actor[tick:tick+1], rnn_states=states))
        rows.append(mean[0].clamp(-1, 1))
    return torch.stack(rows)


@torch.no_grad()
def window_predictions(network, actor, width):
    rows = []
    for tick in range(0, len(actor), 12):
        states = tuple(torch.zeros(1, 1, 1024) for _ in range(2))
        for index in range(max(0, tick-width+1), tick+1):
            mean, _, _, states = network(dict(obs=actor[index:index+1], rnn_states=states))
        rows.append(mean[0].clamp(-1, 1))
    return torch.stack(rows)


def audit(source, previous, checkpoint, output):
    output.mkdir(parents=True, exist_ok=False)
    data = json.loads((source/'data.json').read_text())
    session = json.loads((previous/'session.json').read_text())
    network, weights = load_actor(checkpoint)
    rows = []
    for identity in session['cohorts']['replay']:
        entry = data['entries'][identity]
        with h5py.File(entry['path'], 'r') as stream:
            group = stream['episodes'][entry['episode']]
            raw = torch.from_numpy(group['transition/expert_actor_input_diagnostic'][:])
            target = torch.from_numpy(group['transition/command_clipped_pre_delay'][:])
            native_stage = torch.from_numpy(group['transition/diagnostics_before/native_stage_step'][:])
            metadata = json.loads(stream.attrs['metadata_json'])
            actor = normalized(raw, weights)
            full = predictions(network, actor)
            every_tick = predictions(network, actor, reset_every=1)
            every_chunk = predictions(network, actor, reset_every=12)
            recent13 = window_predictions(network, actor, 13)
            recent25 = window_predictions(network, actor, 25)
            def errors(prediction):
                difference = prediction-target
                return dict(maximum=float(difference.abs().max()), arm_rmse=float(difference[:, :7].square().mean().sqrt()),
                            hand_rmse=float(difference[:, 7:].square().mean().sqrt()))
            row = dict(identity=identity, source_lane=metadata['source_env_index'], steps=len(raw),
                persistent=errors(full), reset_each_tick=errors(every_tick), reset_each_chunk=errors(every_chunk),
                recent13_anchor_arm_rmse=float((recent13[:, :7]-target[::12, :7]).square().mean().sqrt()),
                recent13_anchor_hand_rmse=float((recent13[:, 7:]-target[::12, 7:]).square().mean().sqrt()),
                recent25_anchor_arm_rmse=float((recent25[:, :7]-target[::12, :7]).square().mean().sqrt()),
                recent25_anchor_hand_rmse=float((recent25[:, 7:]-target[::12, 7:]).square().mean().sqrt()),
                subgoal_boundaries=torch.where(native_stage[1:] < native_stage[:-1])[0].add(1).tolist(),
                group_embedding_values=raw[:, -1].unique().tolist(),
                scope='Frozen expert weights; only its LSTM history changes; offline command evidence, no optimizer or rollout')
            rows.append(row)
            np.savez_compressed(output/(identity+'.npz'), expert=target.numpy(), persistent=full.numpy(),
                                reset_each_tick=every_tick.numpy(), reset_each_chunk=every_chunk.numpy(),
                                recent13_at_anchors=recent13.numpy(), recent25_at_anchors=recent25.numpy())
            atomic_json(output/'progress.json', dict(completed=len(rows), rows=rows))
    atomic_json(output/'result.json', dict(software_complete=True, rows=rows, checkpoint_sha256=sha256(checkpoint),
        actor_obs_dim=140, stored_actor_dim=141, learned_group_embedding_dim=32, lstm_input_dim=172,
        lstm_hidden_and_cell=[1024, 1024], actor_mlp=[1024, 1024, 512, 512],
        actor_normalization='checkpoint running mean/variance, eps1e-5, clip5; group ID not normalized',
        stage_is_not_actor_input=True, reset_scope='full episode only; never native subgoal', optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args(); torch.set_num_threads(2)
    audit(arguments.source, arguments.previous, arguments.checkpoint, arguments.output)
