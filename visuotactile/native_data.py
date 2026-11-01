"""Native-command v1: causal numeric records and read-only perception reuse."""
import json
from pathlib import Path
import xml.etree.ElementTree as element_tree

import h5py
import numpy as np
import torch

from .compact import sha256
from .long_data import BASE, CAMERAS, MANIFEST, shift_actions
from .models.long_horizon import DINO_REVISION
from .training_resources import atomic_json


CACHE_SOURCE = Path('outputs/visuotactile/run/long_horizon_20261030_v2')
ROBOT_DIM = 178
GEOMETRY_DIM = 85
GROUPS = ('S-NativeDP', 'I-OfficialDP', 'I-HistoryDP')
CONTRACT = dict(version=1, action='native_command_clipped_pre_delay', horizon=64,
    commit=12, native_history=13, visual_history=3, native_hz=60, physics_hz=120,
    observation='native_robot178_geometry85_rgbd_v1', prediction_type='sample',
    loss='B', train_noise_steps=100, inference_steps=16,
    history_training='current_weight_causal_prefix_no_grad_then_96_ticks',
    history_source='past_expert_executed_teacher_forcing; student_executed_at_deployment')


def multiply_quaternion(left, right):
    scalar = left[..., :1]*right[..., :1]-(left[..., 1:]*right[..., 1:]).sum(-1, keepdim=True)
    vector = left[..., :1]*right[..., 1:]+right[..., :1]*left[..., 1:]
    return torch.cat((scalar, vector+torch.cross(left[..., 1:], right[..., 1:], dim=-1)), -1)


def rotate(quaternion, vector):
    imaginary = quaternion[..., 1:].expand_as(vector)
    return vector+2*torch.cross(imaginary, torch.cross(imaginary, vector, dim=-1)
                              +quaternion[..., :1]*vector, dim=-1)


def geometry(diagnostics, dimensions):
    base = diagnostics['base_pose_w']
    inverse = base[..., 3:].clone()
    inverse[..., 1:] *= -1
    parts = []
    for name in ('object_pose_w', 'goal_pose_w', 'hole_pose_w', 'table_pose_w', 'palm_body_pose_w'):
        pose = diagnostics[name]
        orientation = multiply_quaternion(inverse, pose[..., 3:])
        orientation = torch.where(orientation[..., :1] < 0, -orientation, orientation)
        parts.extend((rotate(inverse, pose[..., :3]-base[..., :3]), orientation))
    for name in ('object_velocity_w_m_s_rad_s', 'goal_velocity_w_m_s_rad_s'):
        velocity = diagnostics[name]
        parts.extend((rotate(inverse, velocity[..., :3]), rotate(inverse, velocity[..., 3:])))
    position = rotate(inverse[..., None, :], diagnostics['finger_position_w']-base[..., None, :3])
    orientation = multiply_quaternion(inverse[..., None, :], diagnostics['finger_quat_wxyz'])
    orientation = torch.where(orientation[..., :1] < 0, -orientation, orientation)
    parts.extend((position.flatten(-2), orientation.flatten(-2), dimensions.expand(*base.shape[:-1], 3)))
    result = torch.cat(parts, -1).float()
    if result.shape[-1] != GEOMETRY_DIM or not torch.isfinite(result).all():
        raise ValueError('Invalid explicitly allowlisted state geometry')
    return result


def object_dimensions(metadata):
    path = Path(metadata['config']['assets']['object_urdf'])
    lower, upper = [], []
    for collision in element_tree.parse(path).getroot().findall('.//collision'):
        box, origin = collision.find('geometry/box'), collision.find('origin')
        if box is None:
            raise ValueError('Object dimensions require the audited box asset')
        position = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        angles = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        if np.any(angles):
            raise ValueError('Rotated collider dimensions require explicit implementation')
        half = np.fromstring(box.get('size'), sep=' ')/2
        lower.append(position-half)
        upper.append(position+half)
    return torch.tensor(np.max(upper, axis=0)-np.min(lower, axis=0), dtype=torch.float32)


def robot_rows(position, velocity, target, action, elapsed):
    result = torch.cat((position, velocity, target, target-position, action,
                        elapsed[..., None].float()), -1).float()
    if result.shape[-1] != ROBOT_DIM or not torch.isfinite(result).all():
        raise ValueError('Invalid native robot observation')
    return result


def labels(record, anchors):
    indices = anchors[:, None]+torch.arange(64)
    valid = indices < len(record['time'])
    commands = record['command'][indices.clamp_max(len(record['time'])-1)]
    return torch.where(valid[..., None], commands, 0.), valid


def limits_affine(values):
    minimum, maximum = values.amin(0), values.amax(0)
    span = maximum-minimum
    constant = span < 1e-4
    scale = 2/torch.where(constant, torch.full_like(span, 2.), span)
    offset = -1-scale*minimum
    offset = torch.where(constant, -minimum, offset)
    return scale, offset


def prepare(root, source=CACHE_SOURCE):
    root, source = Path(root), Path(source)
    index = json.loads((source/'cache_index.json').read_text())
    saved = torch.load(BASE, map_location='cpu', weights_only=False)
    splits = {name: saved[key] for name, key in (('train', 'train_episodes'),
        ('validation', 'validation_episodes'), ('test', 'test_episodes'))}
    manifest = json.loads(MANIFEST.read_text())
    identities = sum(splits.values(), [])
    if ([len(splits[name]) for name in ('train', 'validation', 'test')] != [320, 40, 40]
            or len(set(identities)) != 400 or set(identities) != {row['id'] for row in manifest['episodes']}
            or splits != index['splits'] or index['manifest_sha256'] != sha256(MANIFEST)
            or index['dino_revision'] != DINO_REVISION or index['camera_order'] != CAMERAS
            or index['source_split_sha256'] != sha256(BASE)):
        raise ValueError('Frozen data/split/perception identity mismatch')
    directory = root/'numeric'
    directory.mkdir(exist_ok=True)
    entries, train_records = {}, []
    for ordinal, row in enumerate(manifest['episodes']):
        path = MANIFEST.parent/row['directory']/'data.h5'
        cached = source/'cache'/(row['id']+'.pt')
        if sha256(path) != row['files']['data.h5']['sha256']:
            raise ValueError('Original trajectory hash mismatch: '+row['id'])
        if sha256(cached) != index['entries'][row['id']]['sha256']:
            raise ValueError('Frozen DINO cache hash mismatch: '+row['id'])
        with h5py.File(path, 'r') as stream:
            group = stream['episodes'][row['episode']]
            metadata = json.loads(stream.attrs['metadata_json'])
            length = row['length']
            def read(name):
                value = torch.from_numpy(np.asarray(group[name][:length]).copy())
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite source '+name)
                return value
            command = read('transition/command_clipped_pre_delay')
            submitted = read('transition/submitted_action')
            target = read('obs/actuator_target_rad')
            times = read('obs/time_s')
            steps = read('obs/step_id')
            if (command.shape != (length, 29) or not torch.equal(command, submitted.clamp(-1, 1))
                    or not torch.equal(target, read('transition/target_before_rad'))
                    or not torch.equal(steps, torch.arange(length))
                    or not read('transition/control_trace/valid').all()
                    or metadata['camera_ids'] != CAMERAS
                    or not torch.allclose(times, read('transition/time_before_s'), atol=1e-8, rtol=0)
                    or len(read('obs/episode_id').unique()) != 1):
                raise ValueError('Native data alignment failed: '+row['id'])
            if not torch.allclose(times[1:]-times[:-1], torch.full_like(times[1:], 1/60), atol=2e-5, rtol=0):
                raise ValueError('Non-native observation clock')
            for start in range(0, length, 8):
                depth = np.asarray(group['frames/depth_m'][start:start+8])
                valid_depth = np.asarray(group['frames/depth_valid'][start:start+8], dtype=bool)
                if not np.isfinite(depth).all() or np.any(depth[valid_depth] < 0):
                    raise ValueError('Nonfinite or negative measured depth')
            if group['frames/rgb_u8'].shape != (length, 3, 3, 252, 448) or group['frames/rgb_u8'].dtype != np.uint8:
                raise ValueError('RGB field or camera dimensions differ')
            action = shift_actions(command, read('transition/control_trace/delayed_command'),
                read('transition/time_before_s'), read('transition/control_trace/selected_command_time_s'), times)
            diagnostics = {name: read('transition/diagnostics_before/'+name) for name in
                ('base_pose_w', 'object_pose_w', 'goal_pose_w', 'hole_pose_w', 'table_pose_w',
                 'palm_body_pose_w', 'object_velocity_w_m_s_rad_s', 'goal_velocity_w_m_s_rad_s',
                 'finger_position_w', 'finger_quat_wxyz')}
            position = read('transition/diagnostics_before/joint_position_rad')
            velocity = read('transition/diagnostics_before/joint_velocity_rad_s')
            if not torch.allclose(target-position, read('obs/tracking_error_rad'), atol=1e-6, rtol=0):
                raise ValueError('Tracking-error alignment mismatch')
            record = dict(version=1, identity=row['id'], command=command, time=times,
                robot=robot_rows(position, velocity, target, action, times-times[0]),
                geometry=geometry(diagnostics, object_dimensions(metadata))[::12],
                camera_time=read('obs/camera_time_s')[::12], camera_valid=read('obs/camera_valid')[::12])
            old = torch.load(cached, map_location='cpu', weights_only=False)
            if (old['version'] != 2 or not torch.equal(old['command'], command)
                    or not torch.equal(old['time'], times)
                    or not torch.equal(old['camera_time'], record['camera_time'])
                    or old['patches'].shape != (len(times[::12]), 3, 384, 18, 32)):
                raise ValueError('Perception cache is not aligned with source')
            del old
        target_path = directory/(row['id']+'.pt')
        if target_path.exists():
            previous = torch.load(target_path, map_location='cpu', weights_only=False)
            if set(previous) != set(record) or any(not torch.equal(previous[name], value)
                    if torch.is_tensor(value) else previous[name] != value for name, value in record.items()):
                raise ValueError('Existing numeric cache differs; preserve failed evidence')
        else:
            temporary = target_path.with_suffix('.partial.pt')
            if temporary.exists():
                temporary.rename(temporary.with_name(temporary.name+'.interrupted'))
            torch.save(record, temporary)
            temporary.replace(target_path)
        entries[row['id']] = dict(path=str(path.resolve()), episode=row['episode'], length=length,
            source_sha256=row['files']['data.h5']['sha256'], numeric_sha256=sha256(target_path),
            perception=str(cached.resolve()), perception_sha256=index['entries'][row['id']]['sha256'])
        if row['id'] in splits['train']:
            train_records.append(record)
        atomic_json(root/'data_progress.json', dict(completed=ordinal+1, total=400))
    statistics = {}
    for name in ('robot', 'geometry'):
        values = torch.cat([record[name] for record in train_records])
        statistics[name+'_mean'] = values.mean(0)
        statistics[name+'_std'] = values.std(0).clamp_min(1e-4)
    commands = torch.cat([record['command'] for record in train_records])
    statistics['action_scale'], statistics['action_offset'] = limits_affine(commands)
    recovered = ((commands*statistics['action_scale']+statistics['action_offset'])
                 -statistics['action_offset'])/statistics['action_scale']
    error = float((recovered-commands).abs().max())
    if error > 1e-6:
        raise ValueError('Action normalizer roundtrip failed')
    torch.save(statistics, root/'statistics.pt')
    atomic_json(root/'data.json', dict(contract=CONTRACT, splits=splits, entries=entries,
        manifest_sha256=sha256(MANIFEST), split_checkpoint_sha256=sha256(BASE),
        perception_index_sha256=sha256(source/'cache_index.json'), statistics_sha256=sha256(root/'statistics.pt'),
        dino_revision=index['dino_revision'], dino_weights_sha256=index['dino_weights_sha256'],
        normalizer_roundtrip_max=error, rows=sum(row['length'] for row in manifest['episodes']),
        camera_acquisition='native camera_update_period_s=0', policy_visual_consumption_hz=5))
    atomic_json(root/'data_gate.json', dict(passed=True, episodes=400, train=320, validation=40, test=40,
        labels='identical future-only commands for every model', normalizer_roundtrip_max=error))


class Episode:
    def __init__(self, root, identity, group):
        self.group = group
        root = Path(root)
        index = json.loads((root/'data.json').read_text())['entries'][identity]
        self.record = torch.load(root/'numeric'/(identity+'.pt'), map_location='cpu', weights_only=False)
        self.index = index
        self.perception = None
        if group == 'I-HistoryDP':
            old = torch.load(index['perception'], map_location='cpu', weights_only=False)
            self.perception = {name: old[name] for name in ('patches', 'depth')}

    def __len__(self):
        return len(self.record['time'])

    def frame(self, tick, device):
        ordinal = tick//12
        result = dict(time=self.record['camera_time'][ordinal].to(device),
                      valid=self.record['camera_valid'][ordinal].to(device))
        if self.group == 'S-NativeDP':
            result['geometry'] = self.record['geometry'][ordinal].to(device)
        elif self.group == 'I-HistoryDP':
            result.update({name: value[ordinal].to(device) for name, value in self.perception.items()})
        else:
            with h5py.File(self.index['path'], 'r') as stream:
                frames = stream['episodes'][self.index['episode']]['frames']
                result.update(rgb=torch.from_numpy(frames['rgb_u8'][tick]).to(device),
                    depth=torch.from_numpy(frames['depth_m'][tick]).to(device),
                    depth_valid=torch.from_numpy(frames['depth_valid'][tick]).to(device))
        return result
