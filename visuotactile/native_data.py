"""Current immutable train320/validation40/test40 data and causal image cache reader."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as functional
from .compact import sha256
from .models.native_vision import DINO_REVISION
CAMERAS=["fixed_high","fixed_side","wrist_rgbd"]
MANIFEST=Path("outputs/visuotactile/data/close_v4/manifest.json")
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
            perception_index = json.loads((root/'native_perception.json').read_text())
            overlay = perception_index['entries'][identity].get('tail')
            if overlay:
                self.perception['patches'][-1] = torch.load(overlay['path'], map_location='cpu', weights_only=False)

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
