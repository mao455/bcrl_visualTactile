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
from .native_data import proprio_radians
from .native_data import CONTRACT, Episode, geometry, object_dimensions, robot_rows
from .models.native_dp import NativePolicy, mlp


STATE_CONTRACT = dict(CONTRACT, version=2, observation='actor141_robot178_geometry85_v2',
    persistent_history='own_GRU256_every_native_tick_true_episode_reset_only',
    actor_preprocess='frozen_expert_RMS140_clip5_group_embedding32',
    geometry_hz=60, history_training='current_weight_full_prefix_no_grad_then_96_tick_TBPTT')


class PersistentState(NativePolicy):
    def __init__(self, kind='dp', action_contract=None):
        self.contract = STATE_CONTRACT if action_contract is None else action_contract
        if self.contract != STATE_CONTRACT:
            from .position_control import contract, POSITION_ACTION
            if kind != 'dp' or self.contract != contract(self.contract['horizon'], self.contract['action']==POSITION_ACTION,
                    self.contract.get('prediction_type','sample')):
                raise ValueError('Only explicit short-horizon DP contracts supported')
        super().__init__('S-NativeDP',horizon=self.contract['horizon'],prediction_type=self.contract['prediction_type'])
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
            return super().loss(condition, command, valid, generator,
                temporal_weight=getattr(self, 'temporal_weight', 0.))
        from .dp_objectives import masked_objectives
        return masked_objectives(self.regressor(condition).reshape(-1, 64, 29),
            command*self.action_scale+self.action_offset, valid)['B']

    @torch.no_grad()
    def predict(self, condition, generator=None, **sampling):
        if self.kind == 'dp':
            return super().predict(condition, generator, **sampling)
        if sampling:
            raise ValueError('Diffusion settings cannot be applied to BC')
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


def load_records(source, identities, action_contract=None):
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
            if action_contract and action_contract['action']=='absolute_joint_position_target_rad':
                from .position_control import position_robot_history
                target = torch.from_numpy(group['transition/target_after_rad'][:])
                record['robot'] = position_robot_history(record,target)
                record['command'] = target
                record['joint_limits'] = torch.tensor(metadata['joint_limits_rad'][0])
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


def controller_proprio(packet, diagnostics, limits, source):
    """Exact canonical encoder values for new runs; legacy weights retain I/O.

    Inverting the packet's normalized float32 q discards bits before small train
    standard deviations amplify the error. HDF5 labels/history use the canonical
    observed q/qd directly, so the new online constructor must use the same values.
    No object state, target or future observation is introduced into proprio.
    """
    if source=='legacy_packet_inverse_v1':
        return proprio_radians(packet['proprio'][0].cpu(),limits)
    if source!='canonical_joint_state_v1':raise ValueError('Unknown proprio input builder')
    q=diagnostics['joint_position_rad'][0].cpu();qd=diagnostics['joint_velocity_rad_s'][0].cpu()
    if q.shape!=(29,) or qd.shape!=(29,) or not torch.isfinite(q).all() or not torch.isfinite(qd).all():
        raise ValueError('Finite canonical29D observed joint position/velocity required')
    return torch.cat((q,qd))


class HistoryController:
    def __init__(self, saved, metadata, device='cuda:0', inference=None):
        stored_contract = saved.get('contract')
        from .position_control import contract, POSITION_ACTION
        allowed = [STATE_CONTRACT]+[contract(h,p) for h in (24,32) for p in (False,True)]+[contract(24,True,'epsilon')]
        if stored_contract not in allowed or saved.get('updates', 0) < 1:
            raise ValueError('Trained persistent State v2 checkpoint required')
        self.position_mode = stored_contract['action']==POSITION_ACTION
        with torch.random.fork_rng(devices=[]):
            self.model = PersistentState(saved['kind'],stored_contract).to(device)
        self.model.load_state_dict(saved['ema']); self.model.eval()
        self.device = device
        self.dimensions = object_dimensions(metadata)
        self.limits = torch.tensor(metadata['joint_limits_rad'][0])
        self.input_builder=saved.get('configuration',{}).get('input_builder','legacy_packet_inverse_v1')
        if self.input_builder not in ('legacy_packet_inverse_v1','canonical_joint_state_v1'):
            raise ValueError('Unknown checkpoint input builder')
        self.bank = HistoryBank(); self.execution = ExecutionHistory(1)
        self.episode = None
        self.generator = torch.Generator(device=device).manual_seed(7091)
        self.inference = dict(inference or {})
        allowed_settings = {'inference_steps', 'clip_sample', 'overlap_steps', 'sampler_seed'}
        if not set(self.inference) <= allowed_settings:
            raise ValueError('Unknown inference setting')
        self.overlap_steps = self.inference.get('overlap_steps', 0)
        if self.overlap_steps not in (0, 4) or (self.inference and not self.position_mode):
            raise ValueError('Overlap ablation requires explicit position StateDP')
        self.generator.manual_seed(self.inference.get('sampler_seed', 7091))
        self.plan = None
        self.plan_tick = None

    @torch.no_grad()
    def act(self, actor, packet, diagnostics):
        episode, tick = int(packet['episode_id'][0]), int(packet['step_id'][0])
        stamp = packet['time_s'][0].cpu()
        if episode != self.episode:
            self.execution = ExecutionHistory(1); self.started = stamp; self.episode = episode
            self.position_previous = None
            self.plan = None; self.plan_tick = None
        if self.position_mode:
            past = torch.zeros(61)
            if self.position_previous is not None:
                requested,applied,last_stamp = self.position_previous
                past = torch.cat((requested,applied,(stamp-last_stamp).float().repeat(2),torch.ones(1)))
        else:
            history = self.execution.context(packet)
            past = torch.cat((history['previous_issued_command'][0], history['previous_delayed_command'][0],
                torch.stack((stamp-history['previous_command_time_s'][0], stamp-history['previous_delayed_time_s'][0])).float(),
                history['action_history_valid'][0, None].float()))
            past = torch.where(history['action_history_valid'][0], past, 0.)
        proprio = controller_proprio(packet,diagnostics,self.limits,self.input_builder)
        robot = robot_rows(proprio[:29], proprio[29:], packet['actuator_target_rad'][0].cpu(), past, stamp-self.started)
        geo = geometry(diagnostics, self.dimensions)[0]
        condition = self.bank.observe(0, episode, tick, self.model, actor[0].to(self.device),
            robot.to(self.device), geo.to(self.device), stamp.to(self.device))
        self.last_inputs = dict(actor=actor[0].cpu(), robot=robot, geometry=geo.cpu())
        if condition is not None:
            self.last_condition = condition
            fresh = self.model.predict(condition, self.generator,
                **{k:v for k,v in self.inference.items() if k in ('inference_steps','clip_sample')}).cpu()
            self.raw_plan = fresh.clone()
            from .position_control import overlap_transition
            self.plan = overlap_transition(self.plan, fresh, tick-self.plan_tick if self.plan_tick is not None else 0,
                self.overlap_steps)
            self.plan_tick = tick
        if self.position_mode:
            from .position_control import bounded_targets
            self.command = bounded_targets(self.plan[:,tick%12],self.limits)
        else:
            self.command = self.plan[:, tick % 12].clamp(-1, 1)
        self.last_packet = packet
        return self.command

    def observe_execution(self, transition):
        if self.position_mode:
            self.position_previous=(self.command[0].clone(),transition['position_applied_rad'][0].cpu().clone(),self.last_packet['time_s'][0].cpu())
        else:
            self.execution.observe(transition, self.command, self.last_packet['episode_id'], self.last_packet['step_id'])


class PersistentImage(NativePolicy):
    """Existing visual architectures under the shared V2 trainer; no State branch."""
    def __init__(self, group, action_contract):
        from collections import OrderedDict
        from .position_control import image_contract
        if action_contract != image_contract(group,action_contract['horizon'],action_contract.get('modality'),action_contract.get('supervision')):
            raise ValueError('Image position contract mismatch')
        super().__init__(group,action_contract['horizon'],modality=action_contract.get('modality'))
        self.contract=action_contract
        self._episodes=OrderedDict()

    def encode(self, record, start=0, stop=None):
        from .models.native_dp import encode_segment
        allowed={'robot','command','time','joint_limits','identity','source_root'}
        if self.contract.get('supervision') is not None:allowed.add('clean_sample')
        if self.modality == 'G10':allowed.add('tactile')
        supervision_fields={'supervision_mask','control_segment','supervision','expert_command','target_window_contract'}
        if not allowed<=set(record) or set(record)-allowed-supervision_fields:
            raise ValueError('Image record contains non-allowlisted fields')
        if 'supervision' in record and record['supervision']!=self.contract.get('supervision','u_expert'):
            raise ValueError('Image label semantics differ from checkpoint contract')
        key=record['identity']
        if key not in self._episodes:
            episode=Episode(record['source_root'],key,self.group)
            # Keep only image acquisition metadata, never State/Actor arrays.
            episode.record={k:episode.record[k] for k in ('camera_time','camera_valid')}
            if getattr(self,'raw_frame_device_cache',False):
                # Only immutable DINO/depth and timestamps move once. Learned
                # projections and all recurrent updates are still recomputed at
                # current weights in the original tick order below. LRU stays 2.
                device=next(self.parameters()).device
                episode.record={k:v.to(device) for k,v in episode.record.items()}
                episode.perception={k:v.to(device) for k,v in episode.perception.items()}
            self._episodes[key]=episode
            while len(self._episodes)>2:self._episodes.popitem(last=False)
        episode=self._episodes[key]
        # Episode.frame uses camera timestamps; encoder consumes robot/time only.
        episode.record.update(robot=record['robot'],time=record['time'])
        if self.modality == 'G10':episode.record['tactile']=record['tactile']
        return encode_segment(self,episode,start,len(record['time']) if stop is None else stop)


def load_image_warm_start(saved, action_contract, statistics):
    """Initialize a new q_reached G00 policy from the historical command EMA.

    This is weight transfer, never optimizer/schedule/checkpoint continuation.
    Every learned tensor, including the existing conditional U-Net denoiser,
    is copied strictly. Only the four explicitly supplied observation/action
    normalizer buffers are replaced. Q00 and QG00 call this same helper with
    the same source and train-only statistics; geometry adds a training loss,
    not a different policy module. The caller creates fresh optimizer, EMA,
    sampler and counters and records the source file path/SHA separately.
    """
    from .position_control import image_contract
    source_contract=image_contract('I-HistoryDP',24,'G00')
    target_contract=image_contract('I-HistoryDP',24,'G00','q_reached')
    if saved.get('group')!='I-HistoryDP' or saved.get('contract')!=source_contract:
        raise ValueError('Reached warm start requires the explicit historical command G00 contract')
    if action_contract!=target_contract:
        raise ValueError('Warm start target must explicitly declare q_reached supervision and affine identity')
    if saved.get('updates',0)<1 or 'ema' not in saved:
        raise ValueError('A trained historical EMA source is required')
    replaced=('robot_mean','robot_std','action_scale','action_offset')
    if set(statistics)!=set(replaced):
        raise ValueError('Warm start requires exactly the four explicit train-only normalizers')
    # Construction must not consume the caller's fresh experiment RNG stream.
    with torch.random.fork_rng(devices=[]):
        model=PersistentImage('I-HistoryDP',action_contract)
    state=saved['ema']
    model.load_state_dict(state,strict=True)
    for name,value in state.items():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite historical EMA tensor: '+name)
    changed=[]
    with torch.no_grad():
        for name in replaced:
            value=statistics[name].detach().cpu()
            current=getattr(model,name)
            if value.shape!=current.shape or value.dtype!=current.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid replacement normalizer: '+name)
            if name in ('robot_std','action_scale') and not (value>0).all():
                raise ValueError('Normalizer scale must be positive: '+name)
            if not torch.equal(current,value):changed.append(name)
            current.copy_(value)
    parameters=dict(model.named_parameters())
    if not all(torch.equal(value.detach(),state[name]) for name,value in parameters.items()):
        raise ValueError('Warm start changed a learned parameter')
    audit=dict(version='command_ema_to_reached_warm_start_v1',source='ema',
        source_contract=source_contract,target_contract=target_contract,
        source_updates=saved['updates'],new_optimizer_updates=0,completed_reached_policy=False,
        learned_tensors_reused=len(parameters),learned_parameters_reused=sum(p.numel() for p in parameters.values()),
        learned_parameter_names=list(parameters),normalizers_replaced=list(replaced),
        normalizers_numerically_changed=changed,
        other_buffers_reused=[name for name,_ in model.named_buffers() if name not in replaced],
        denoiser_class=type(model.denoiser).__name__,new_modules=[],
        optimizer_state_restored=False,schedule_state_restored=False,ema_schedule_restored=False,
        rng_state_restored=False,requires_fresh_optimizer_ema_sampler_and_counters=True)
    return model,audit


def trained_continuation_updates(saved):
    """Return inherited training only for a hash-bound, nondisposable update0.

    A selected initialization candidate has zero updates in this round, but its
    weights came from a trained checkpoint. Missing provenance is never treated
    as training. The referenced parent remains an immutable local dependency.
    """
    from .compact import sha256
    cfg=saved.get('configuration',{})
    if saved.get('disposable') or cfg.get('disposable'):return 0
    updates=saved.get('updates',0)
    if isinstance(updates,int) and not isinstance(updates,bool) and updates>0:return updates
    if updates!=0 or saved.get('new_optimizer_updates')!=0:return 0
    audit=cfg.get('continuation_audit',{});source=cfg.get('continuation_from',{})
    inherited=audit.get('source_updates',0);checkpoint_updates=audit.get('source_checkpoint_updates')
    if not isinstance(inherited,int) or isinstance(inherited,bool) or inherited<=0:return 0
    if not isinstance(checkpoint_updates,int) or isinstance(checkpoint_updates,bool) or checkpoint_updates<0:return 0
    if not isinstance(source,dict) or not isinstance(source.get('path'),str):return 0
    digest=source.get('sha256','')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):return 0
    path=Path(source['path'])
    if audit.get('source_sha256',digest)!=digest:return 0
    if audit.get('source') and Path(audit['source']).resolve()!=path.resolve():return 0
    if not path.is_file() or sha256(path)!=digest:return 0
    return inherited


def load_image_continuation(saved, action_contract, statistics, supervision=None):
    """Fresh optimizer, identical EMA and affine; never relabel q as command.

    Historical v4 command contracts may explicitly acquire the v5 u_expert
    declaration. This is a same-label continuation, unlike the Q warm start.
    A supplied tactile affine is mandatory for a G10 checkpoint.
    """
    from .position_control import image_contract
    from .native_data import target_supervision
    target=target_supervision(supervision or action_contract.get('supervision','u_expert'))
    source=saved.get('contract',{})
    modality=action_contract.get('modality')
    expected=image_contract('I-HistoryDP',24,modality,target)
    legacy=image_contract('I-HistoryDP',24,modality)
    if action_contract!=expected or source not in (expected,legacy):
        raise ValueError('Continuation requires identical architecture and supervision')
    if source==legacy and target!='u_expert':
        raise ValueError('Legacy command EMA cannot be continued as an already-trained configuration policy')
    source_updates=trained_continuation_updates(saved)
    if saved.get('group')!='I-HistoryDP' or source_updates<1 or 'ema' not in saved:
        raise ValueError('Trained selected image EMA required')
    names={'robot_mean','robot_std','action_scale','action_offset'}
    if modality=='G10':names|={'tactile_mean','tactile_std'}
    if set(statistics)!=names:raise ValueError('All and only fixed observation/action affine buffers required')
    with torch.random.fork_rng(devices=[]):model=PersistentImage('I-HistoryDP',action_contract)
    model.load_state_dict(saved['ema'],strict=True)
    for key,value in model.state_dict().items():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite continuation EMA tensor: '+key)
    for key in names:
        value=statistics[key].detach().cpu()
        if not torch.equal(getattr(model,key),value):raise ValueError('Continuation affine changed: '+key)
        if key.endswith('std') or key=='action_scale':
            if not (value>0).all():raise ValueError('Invalid continuation affine scale: '+key)
    return model,dict(version='same_supervision_ema_continuation_v1',source='ema',
        source_contract=source,target_contract=action_contract,supervision=target,
        source_updates=source_updates,source_checkpoint_updates=saved['updates'],new_optimizer_updates=0,all_state_tensors_reused=True,
        affine_replaced=False,affine_buffers_verified=sorted(names),
        legacy_command_contract_declared=source==legacy,optimizer_state_restored=False,
        requires_fresh_optimizer_ema_sampler_and_counters=True)
