"""Three independently initialized policies sharing the native-command contract."""
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint
from diffusers import DDPMScheduler, DDIMScheduler

from ..native_data import CONTRACT, GEOMETRY_DIM, GROUPS, ROBOT_DIM
from ..dp_objectives import masked_objectives
from .vendor.conditional_unet1d import ConditionalUnet1D


@dataclass
class Memory:
    robot: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    native: torch.Tensor | None = None
    visual: torch.Tensor | None = None
    next_tick: int = 0


def mlp(source, target):
    return nn.Sequential(nn.Linear(source, target), nn.LayerNorm(target), nn.SiLU())


class OfficialVision(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(channels//16, channels))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.keypoints = nn.Conv2d(512, 32, 1)
        self.depth = nn.Sequential(nn.Conv2d(2, 16, 5, stride=2, padding=2),
                                   nn.GroupNorm(4, 16), nn.SiLU(),
                                   nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU())
        self.fusion = mlp(32*20, 128)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, rgb, depth, valid):
        images = rgb.float()/255. if rgb.dtype == torch.uint8 else rgb.float()
        if images.shape != (3, 3, 252, 448):
            raise ValueError('Official encoder requires all three full-resolution views')
        features = self.backbone((images-self.mean)/self.std)
        logits = self.keypoints(features)
        height, width = logits.shape[-2:]
        vertical, horizontal = torch.meshgrid(torch.linspace(-1, 1, height, device=rgb.device),
            torch.linspace(-1, 1, width, device=rgb.device), indexing='ij')
        locations = torch.stack((horizontal, vertical), -1).reshape(-1, 2)
        weights = logits.flatten(-2).softmax(-1)
        points = weights@locations
        metric = torch.where(valid.bool(), depth, 0.).float()
        depth_input = torch.cat((metric/2., valid.float()), 1)
        depth_features = functional.interpolate(self.depth(depth_input), size=(height, width), mode='bilinear', align_corners=False)
        geometry = functional.interpolate(depth_input, size=(height, width), mode='bilinear', align_corners=False)
        associated = weights@torch.cat((geometry, depth_features), 1).flatten(-2).transpose(1, 2)
        return self.fusion(torch.cat((points, associated), -1).flatten(1)).flatten()


class HistoryVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(384, 112, 1), nn.GroupNorm(8, 112), nn.SiLU())
        self.depth = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.position = nn.Linear(2, 128)
        self.camera = nn.Embedding(3, 128)
        self.age = mlp(1, 128)
        self.queries = nn.Parameter(torch.randn(4, 128)*.02)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        vertical, horizontal = torch.meshgrid(torch.linspace(-1, 1, 6), torch.linspace(-1, 1, 8), indexing='ij')
        self.register_buffer('coordinates', torch.stack((horizontal, vertical), -1).reshape(48, 2))

    def forward(self, patches, depth):
        if patches.shape != (3, 384, 18, 32) or depth.shape != (3, 2, 18, 32):
            raise ValueError('Frozen DINO patch source must retain its 18x32 grid')
        fused = torch.cat((self.rgb(patches.float()), self.depth(depth.float())), 1)
        tokens = functional.adaptive_avg_pool2d(fused, (6, 8)).flatten(-2).transpose(1, 2)
        return tokens+self.position(self.coordinates)[None]+self.camera.weight[:, None]

    def read(self, tokens, ages, valid):
        values = tokens+self.age(ages[..., None].float())[..., None, :]
        values = values.reshape(1, -1, 128)
        padding = ~valid[..., None].expand(*valid.shape, 48).reshape(1, -1)
        if padding.all():
            raise ValueError('No valid visual samples')
        result, _ = self.attention(self.queries[None], values, values, key_padding_mask=padding, need_weights=False)
        return result.flatten(1)


class NativePolicy(nn.Module):
    def __init__(self, group):
        super().__init__()
        if group not in GROUPS:
            raise ValueError('Unknown native policy group')
        self.group = group
        self.robot_encoder = mlp(13*(ROBOT_DIM+2), 256)
        if group == 'S-NativeDP':
            self.state_encoder = mlp(3*(GEOMETRY_DIM+2), 256)
            condition_width = 512
        elif group == 'I-OfficialDP':
            self.vision = OfficialVision()
            self.recent_encoder = mlp(3*(384+6), 256)
            condition_width = 512
        else:
            self.vision = HistoryVision()
            self.native_input = mlp(ROBOT_DIM, 256)
            self.native_gru = nn.GRUCell(256, 256)
            self.visual_gru = nn.GRUCell(512, 256)
            condition_width = 256+512+256+256
        self.condition_projection = mlp(condition_width, 512)
        self.denoiser = ConditionalUnet1D(input_dim=29, global_cond_dim=512,
            diffusion_step_embed_dim=128, down_dims=(256, 512, 1024), kernel_size=5,
            n_groups=8, cond_predict_scale=True)
        self.noise = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                                   clip_sample=False, prediction_type='sample')
        self.sampler = DDIMScheduler.from_config(self.noise.config)
        for name, width in (('robot', ROBOT_DIM), ('geometry', GEOMETRY_DIM)):
            self.register_buffer(name+'_mean', torch.zeros(width))
            self.register_buffer(name+'_std', torch.ones(width))
        self.register_buffer('action_scale', torch.ones(29))
        self.register_buffer('action_offset', torch.zeros(29))

    def normalized_robot(self, robot):
        value = (robot-self.robot_mean)/self.robot_std
        mask = robot[..., 176:177].bool()
        value = torch.cat((value[..., :116], torch.where(mask, value[..., 116:176], 0.),
                           robot[..., 176:177], value[..., 177:]), -1)
        return value

    def observe(self, robot, time, tick, frame, memory=None):
        if memory is None and self.group == 'I-HistoryDP' and tick != 0:
            raise ValueError('History memory requires the complete causal episode prefix')
        memory = Memory(next_tick=tick) if memory is None else memory
        if tick != memory.next_tick or (frame is not None) != (tick % 12 == 0):
            raise ValueError('Skipped native observation or incorrect visual update')
        if robot.shape != (ROBOT_DIM,):
            raise ValueError('Strict robot input contract differs')
        normalized = self.normalized_robot(robot)
        memory.robot = (memory.robot+[(normalized, time)])[-13:]
        if self.group == 'I-HistoryDP':
            with torch.autocast(device_type=robot.device.type, enabled=False):
                memory.native = self.native_gru(self.native_input(normalized.float())[None], memory.native)
        if frame is not None:
            expected = {'time', 'valid'} | ({'geometry'} if self.group == 'S-NativeDP' else
                {'rgb', 'depth', 'depth_valid'} if self.group == 'I-OfficialDP' else {'patches', 'depth'})
            if set(frame) != expected or (time-frame['time'] < -1e-6).any():
                raise ValueError('Frame allowlist or acquisition time differs')
            if self.group == 'S-NativeDP':
                feature = (frame['geometry']-self.geometry_mean)/self.geometry_std
            elif self.group == 'I-OfficialDP':
                arguments = (frame['rgb'], frame['depth'], frame['depth_valid'])
                feature = checkpoint(self.vision, *arguments, use_reentrant=False) if self.training else self.vision(*arguments)
            else:
                feature = self.vision(frame['patches'], frame['depth'])
                with torch.autocast(device_type=robot.device.type, enabled=False):
                    summary = self.vision.read(feature.float()[None], (time-frame['time'])[None], frame['valid'][None])
                    memory.visual = self.visual_gru(summary.float(), memory.visual)
            memory.frames = (memory.frames+[(feature, frame['time'], frame['valid'])])[-3:]
        memory.next_tick = tick+1
        return memory

    def condition(self, memory, time):
        device = self.robot_mean.device
        rows = [torch.cat((value, (time-stamp).float().reshape(1), torch.ones(1, device=device)))
                for value, stamp in memory.robot]
        rows = [torch.zeros(ROBOT_DIM+2, device=device)]*(13-len(rows))+rows
        common = self.robot_encoder(torch.stack(rows).flatten()[None])
        if self.group == 'I-HistoryDP':
            tokens = torch.stack([frame[0] for frame in memory.frames])
            ages = torch.stack([time-frame[1] for frame in memory.frames])
            valid = torch.stack([frame[2] for frame in memory.frames])
            visual = self.vision.read(tokens, ages, valid)
            parts = (common, visual, memory.native, memory.visual)
        else:
            rows = []
            for feature, stamp, valid in memory.frames:
                if self.group == 'S-NativeDP':
                    rows.append(torch.cat((feature, (time-stamp[0]).float().reshape(1), torch.ones(1, device=device))))
                else:
                    masked = torch.where(valid[:, None], feature.reshape(3, 128), 0.).flatten()
                    rows.append(torch.cat((masked, (time-stamp).float(), valid.float())))
            rows = [torch.zeros_like(rows[0])]*(3-len(rows))+rows
            encoder = self.state_encoder if self.group == 'S-NativeDP' else self.recent_encoder
            parts = (common, encoder(torch.stack(rows).flatten()[None]))
        return self.condition_projection(torch.cat(parts, -1))

    def loss(self, condition, command, valid, generator=None):
        target = torch.where(valid[..., None], command*self.action_scale+self.action_offset, 0.)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        timestep = torch.randint(100, (len(target),), device=target.device, generator=generator)
        prediction = self.denoiser(self.noise.add_noise(target, noise, timestep), timestep, global_cond=condition)
        return masked_objectives(prediction, target, valid)['B']

    @torch.no_grad()
    def predict(self, condition, generator):
        sample = torch.randn((len(condition), 64, 29), device=condition.device, generator=generator)
        self.sampler.set_timesteps(16, device=condition.device)
        for timestep in self.sampler.timesteps:
            prediction = self.denoiser(sample, timestep, global_cond=condition)
            sample = self.sampler.step(prediction, timestep, sample, eta=0.).prev_sample
        return (sample-self.action_offset)/self.action_scale


def encode_segment(model, episode, start, stop):
    device = next(model.parameters()).device
    record = episode.record
    first = 0 if model.group == 'I-HistoryDP' else max(0, start//12*12-24)
    memory, conditions, anchors = None, [], []
    for tick in range(first, stop):
        with torch.set_grad_enabled(model.training and tick >= start):
            robot = record['robot'][tick].to(device)
            stamp = record['time'][tick].to(device)
            frame = episode.frame(tick, device) if tick % 12 == 0 else None
            memory = model.observe(robot, stamp, tick, frame, memory)
            if tick >= start and tick % 12 == 0:
                conditions.append(model.condition(memory, stamp))
                anchors.append(tick)
    return torch.cat(conditions), torch.tensor(anchors), memory


def load_policy(payload, device='cpu', ema=True):
    if payload.get('contract') != CONTRACT or payload.get('group') not in GROUPS:
        raise ValueError('Not a native-command v1 checkpoint; old weights cannot be reinterpreted')
    model = NativePolicy(payload['group']).to(device)
    model.load_state_dict(payload['ema'] if ema else payload['model'])
    return model.eval()
