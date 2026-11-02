"""State-only Actor parity and causal, current-weight episode recurrence.

No expert recurrent state is copied. Original image policies/checkpoints keep
their old contract. Geometry is explicitly privileged and sampled at 60 Hz.
"""
import json
from pathlib import Path

import h5py
import torch
from torch import nn

from .action_history import ExecutionHistory
from .long_data import proprio_radians
from .native_data import CONTRACT, Episode, geometry, object_dimensions, robot_rows
from .models.native_dp import NativePolicy, mlp


STATE_CONTRACT = dict(CONTRACT, version=2, observation='actor141_robot178_geometry85_v2',
    persistent_history='own_GRU256_every_native_tick_true_episode_reset_only',
    actor_preprocess='frozen_expert_RMS140_clip5_group_embedding32',
    geometry_hz=60, history_training='current_weight_full_prefix_no_grad_then_96_tick_TBPTT')


class PersistentState(NativePolicy):
    def __init__(self, kind='dp'):
        super().__init__('S-NativeDP')
        if kind not in ('bc', 'dp'):
            raise ValueError('State-only BC or DP required')
        self.kind = kind
        self.actor_encoder = mlp(172, 128)
        self.native_encoder = mlp(178, 128)
        self.geometry_encoder = mlp(85, 128)
        self.history = nn.GRU(384, 256, batch_first=True)
        self.condition_projection = mlp(896, 512)
        self.register_buffer('actor_mean', torch.zeros(140))
        self.register_buffer('actor_var', torch.ones(140))
        self.register_buffer('group_ids', torch.linspace(50., 0., 6))
        self.register_buffer('group_embedding', torch.zeros(6, 32))
        if kind == 'bc':
            del self.denoiser, self.noise, self.sampler
            self.regressor = nn.Sequential(nn.Linear(512, 256), nn.Mish(),
                nn.Linear(256, 256), nn.Mish(), nn.Linear(256, 64*29))

    def set_statistics(self, stats, expert):
        for key, value in stats.items():
            getattr(self, key).copy_(value)
        self.actor_mean.copy_(expert['running_mean_std.running_mean'].float())
        self.actor_var.copy_(expert['running_mean_std.running_var'].float())
        self.group_embedding.copy_(expert['a2c_network.extra_params'])

    def actor_input(self, actor):
        if actor.shape[-1] != 141 or not torch.isfinite(actor).all():
            raise ValueError('Actual recorded/live Actor141 required')
        matches = actor[..., -1, None] == self.group_ids
        if not matches.any(-1).all():
            raise ValueError('Unknown expert task/group identifier')
        normalized = ((actor[..., :140]-self.actor_mean)/torch.sqrt(self.actor_var+1e-5)).clamp(-5, 5)
        return torch.cat((normalized, self.group_embedding[matches.long().argmax(-1)]), -1)

    def native_features(self, actor, robot, geo):
        a = self.actor_encoder(self.actor_input(actor))
        r = self.native_encoder(self.normalized_robot(robot))
        g = self.geometry_encoder((geo-self.geometry_mean)/self.geometry_std)
        return torch.cat((a, r, g), -1), a

    def encode(self, record, start=0, stop=None):
        """Recompute prefix with current weights; only [start,stop) has gradients.

        A mid-episode window never starts with a fabricated zero hidden state.
        Short windows retain the old 13 robot/3 geometry anchors and timestamps.
        """
        stop = len(record['time']) if stop is None else stop
        if not 0 <= start < stop <= len(record['time']):
            raise ValueError('Invalid episode segment')
        hidden = None
        if start:
            with torch.no_grad():
                features, _ = self.native_features(record['actor'][:start], record['robot'][:start], record['geometry'][:start])
                _, hidden = self.history(features[None])
        features, actors = self.native_features(record['actor'][start:stop], record['robot'][start:stop], record['geometry'][start:stop])
        states, hidden = self.history(features[None], hidden)
        anchors = torch.arange(start, stop, device=features.device)
        anchors = anchors[anchors % 12 == 0]
        if not len(anchors):
            raise ValueError('Segment contains no plan anchor')
        robots, geos = short_windows(record, anchors, self)
        conditions = self.condition_projection(torch.cat((self.robot_encoder(robots), self.state_encoder(geos),
            actors[anchors-start], states[0, anchors-start]), -1))
        return conditions, anchors, hidden

    def loss(self, condition, command, valid, generator=None):
        if self.kind == 'dp':
            return super().loss(condition, command, valid, generator)
        from .dp_objectives import masked_objectives
        return masked_objectives(self.regressor(condition).reshape(-1, 64, 29),
            command*self.action_scale+self.action_offset, valid)['B']

    @torch.no_grad()
    def predict(self, condition, generator=None):
        if self.kind == 'dp':
            return super().predict(condition, generator)
        return (self.regressor(condition).reshape(-1, 64, 29)-self.action_offset)/self.action_scale


def short_windows(record, anchors, model):
    robot_indices = anchors[:, None]+torch.arange(-12, 1, device=anchors.device)
    geo_indices = anchors[:, None]+torch.tensor([-24, -12, 0], device=anchors.device)
    robots = model.normalized_robot(record['robot'])[robot_indices.clamp_min(0)]
    geos = ((record['geometry']-model.geometry_mean)/model.geometry_std)[geo_indices.clamp_min(0)]
    def pack(values, indices):
        valid = indices >= 0
        age = record['time'][anchors, None]-record['time'][indices.clamp_min(0)]
        rows = torch.cat((values, age[..., None].float(), valid[..., None].float()), -1)
        return torch.where(valid[..., None], rows, 0.).flatten(1)
    return pack(robots, robot_indices), pack(geos, geo_indices)


def load_records(source, identities):
    source = Path(source)
    records = {}
    for identity in identities:
        episode = Episode(source, identity, 'S-NativeDP')
        record = dict(episode.record)
        with h5py.File(episode.index['path'], 'r') as stream:
            group = stream['episodes'][episode.index['episode']]
            metadata = json.loads(stream.attrs['metadata_json'])
            before = group['transition/diagnostics_before']
            names = ('base_pose_w', 'object_pose_w', 'goal_pose_w', 'hole_pose_w', 'table_pose_w',
                'palm_body_pose_w', 'object_velocity_w_m_s_rad_s', 'goal_velocity_w_m_s_rad_s',
                'finger_position_w', 'finger_quat_wxyz')
            record['geometry'] = geometry({key: torch.from_numpy(before[key][:]) for key in names}, object_dimensions(metadata))
            record['actor'] = torch.from_numpy(group['transition/expert_actor_input_diagnostic'][:])
            if record['actor'].shape != (len(record['time']), 141):
                raise ValueError('Missing actual Actor observations')
            if not torch.equal(torch.from_numpy(group['obs/step_id'][:]), torch.arange(len(record['time']))):
                raise ValueError('Incomplete episode prefix')
            if not torch.equal(record['command'], torch.from_numpy(group['transition/command_clipped_pre_delay'][:])):
                raise ValueError('Supervision source mismatch')
        records[identity] = record
    return records


class HistoryBank:
    """Independent per-environment lifecycle; native subgoal counts never enter it."""
    def __init__(self):
        self.lanes = {}

    def observe(self, lane, episode, tick, model, actor, robot, geo, stamp):
        state = self.lanes.get(lane)
        if state is None or state['episode'] != episode:
            if tick != 0:
                raise ValueError('New episode requires full prefix')
            state = dict(episode=episode, next_tick=0, hidden=None, robot=[], geometry=[], time=[])
            self.lanes[lane] = state
        if tick != state['next_tick']:
            raise ValueError('Skipped or repeated native history tick')
        features, encoded_actor = model.native_features(actor[None], robot[None], geo[None])
        _, state['hidden'] = model.history(features[None], state['hidden'])
        # At most25 native rows are needed by the retained short windows.
        for key, value in (('robot', robot), ('geometry', geo), ('time', stamp)):
            state[key] = (state[key]+[value])[-25:]
        state['next_tick'] += 1
        if tick % 12:
            return None
        record = {key: torch.stack(state[key]) for key in ('robot', 'geometry', 'time')}
        anchors = torch.tensor([len(record['time'])-1], device=actor.device)
        r, g = short_windows(record, anchors, model)
        return model.condition_projection(torch.cat((model.robot_encoder(r), model.state_encoder(g),
            encoded_actor, state['hidden'][0]), -1))


class HistoryController:
    def __init__(self, saved, metadata, device='cuda:0'):
        if saved.get('contract') != STATE_CONTRACT or saved.get('updates', 0) < 1:
            raise ValueError('Trained persistent State v2 checkpoint required')
        with torch.random.fork_rng(devices=[]):
            self.model = PersistentState(saved['kind']).to(device)
        self.model.load_state_dict(saved['ema']); self.model.eval()
        self.device = device
        self.dimensions = object_dimensions(metadata)
        self.limits = torch.tensor(metadata['joint_limits_rad'][0])
        self.bank = HistoryBank(); self.execution = ExecutionHistory(1)
        self.episode = None
        self.generator = torch.Generator(device=device).manual_seed(7091)

    @torch.no_grad()
    def act(self, actor, packet, diagnostics):
        episode, tick = int(packet['episode_id'][0]), int(packet['step_id'][0])
        stamp = packet['time_s'][0].cpu()
        if episode != self.episode:
            self.execution = ExecutionHistory(1); self.started = stamp; self.episode = episode
        history = self.execution.context(packet)
        past = torch.cat((history['previous_issued_command'][0], history['previous_delayed_command'][0],
            torch.stack((stamp-history['previous_command_time_s'][0], stamp-history['previous_delayed_time_s'][0])).float(),
            history['action_history_valid'][0, None].float()))
        past = torch.where(history['action_history_valid'][0], past, 0.)
        proprio = proprio_radians(packet['proprio'][0].cpu(), self.limits)
        robot = robot_rows(proprio[:29], proprio[29:], packet['actuator_target_rad'][0].cpu(), past, stamp-self.started)
        geo = geometry(diagnostics, self.dimensions)[0]
        condition = self.bank.observe(0, episode, tick, self.model, actor[0].to(self.device),
            robot.to(self.device), geo.to(self.device), stamp.to(self.device))
        self.last_inputs = dict(actor=actor[0].cpu(), robot=robot, geometry=geo.cpu())
        if condition is not None:
            self.last_condition = condition
            self.plan = self.model.predict(condition, self.generator).cpu()
        self.command = self.plan[:, tick % 12].clamp(-1, 1)
        self.last_packet = packet
        return self.command

    def observe_execution(self, transition):
        self.execution.observe(transition, self.command, self.last_packet['episode_id'], self.last_packet['step_id'])
