"""H64 reference policy with explicit recent histories and native-rate memory."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional
from diffusers import DDPMScheduler, DDIMScheduler

from .vendor.conditional_unet1d import ConditionalUnet1D


DINO_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


@dataclass
class LongHorizonConfig:
    group: str = 'G00-new'
    version: int = 2
    horizon: int = 64
    replan_steps: int = 12
    visual_history: int = 11
    native_history: int = 121
    tactile_history: int = 242
    condition_dim: int = 512
    inference_steps: int = 16
    prediction_type: str = 'epsilon'

    def __post_init__(self):
        if self.group not in ('G00-new', 'G10-new') or self.version != 2:
            raise ValueError('Unknown candidate group/version')
        if self.prediction_type not in ('epsilon', 'sample'):
            raise ValueError('Unsupported diffusion prediction type')
        if (self.horizon, self.replan_steps, self.visual_history, self.native_history,
                self.tactile_history) != (64, 12, 11, 121, 242):
            raise ValueError('Fixed H64/5Hz/two-second history contract required')

    @property
    def tactile(self):
        return self.group == 'G10-new'


class FrozenDinoSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.backbone = torch.hub.load('facebookresearch/dinov2:'+DINO_REVISION,
                'dinov2_vits14', pretrained=True, trust_repo=True).eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.shape[-3:] != (3,252,448):
            raise ValueError('Expected unresized 448x252 RGB')
        images = rgb.float()/(255. if rgb.dtype == torch.uint8 else 1.)
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('DINO RGB must be finite [0,1]')
        leading = images.shape[:-3]
        images = (images.reshape(-1,3,252,448)-self.mean)/self.std
        features = self.backbone.forward_features(images)['x_norm_patchtokens']
        return features.transpose(1,2).reshape(*leading,384,18,32).half()


class MaskedHistory(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gru = nn.GRU(width, hidden, batch_first=True)

    def forward(self, values, valid):
        lengths = valid.sum(1).long()
        offsets = valid.shape[1]-lengths
        indices = (torch.arange(valid.shape[1], device=values.device)[None]+offsets[:,None]).clamp_max(valid.shape[1]-1)
        aligned = values.gather(1, indices[...,None].expand_as(values))
        packed = nn.utils.rnn.pack_padded_sequence(aligned, lengths.clamp_min(1).cpu(),
                                                  batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return torch.where(lengths[:,None] > 0, hidden[0], 0.)


class LongHorizonPolicy(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or LongHorizonConfig()
        self.rgb_projection = nn.Sequential(nn.Conv2d(384,32,1), nn.GroupNorm(4,32), nn.SiLU())
        self.depth_projection = nn.Sequential(nn.Conv2d(2,16,3,padding=1), nn.GroupNorm(4,16), nn.SiLU())
        self.spatial = nn.Sequential(nn.Linear(48*3*5,128),nn.LayerNorm(128),nn.SiLU())
        self.visual_history = MaskedHistory(384+6,128)
        self.proprio_history = MaskedHistory(59,64)
        self.action_history = MaskedHistory(61,64)
        self.tactile_history = MaskedHistory(32,64) if self.cfg.tactile else None
        self.native_memory = nn.GRU(384+6+58+61+(64 if self.cfg.tactile else 0),128,batch_first=True)
        self.condition = nn.Sequential(nn.Linear(384+(64 if self.cfg.tactile else 0),512),
                                        nn.LayerNorm(512),nn.SiLU())
        self.denoiser = ConditionalUnet1D(input_dim=29, global_cond_dim=512,
            diffusion_step_embed_dim=128, down_dims=(256,512,1024), kernel_size=5,
            n_groups=8, cond_predict_scale=True)
        self.noise = DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',
                                   clip_sample=False,prediction_type=self.cfg.prediction_type)
        self.sampler = DDIMScheduler.from_config(self.noise.config)
        self.register_buffer('target_scale',torch.ones(29))
        self.register_buffer('target_offset',torch.zeros(29))
        self.register_buffer('proprio_mean',torch.zeros(58))
        self.register_buffer('proprio_std',torch.ones(58))

    def project_frames(self, patches, depth, valid):
        count, views = patches.shape[:2]
        rgb = self.rgb_projection(patches.float().reshape(-1,384,18,32))
        dep = self.depth_projection(depth.float().reshape(-1,2,18,32))
        grid = functional.adaptive_avg_pool2d(torch.cat((rgb,dep),1),(3,5)).flatten(1)
        features = self.spatial(grid).reshape(count,views,128)
        return torch.where(valid[...,None],features,0.).flatten(1)

    def encode(self, inputs, state=None):
        expected = {'patches','depth','camera_valid','visual_indices','visual_valid','visual_age',
            'held_indices','held_age','proprio','action','time','window_proprio','window_action',
            'window_valid','window_action_valid','window_age','anchor_indices'}
        if self.cfg.tactile:
            expected |= {'tactile','tactile_valid','window_tactile','window_tactile_valid'}
        if set(inputs) != expected:
            raise ValueError('Strict group input allowlist differs')
        frame = self.project_frames(inputs['patches'],inputs['depth'],inputs['camera_valid'])
        visual = frame[inputs['visual_indices']]
        visual_context = torch.cat((visual, inputs['visual_age'].float(),
            inputs['camera_valid'][inputs['visual_indices']].float()),-1)
        recent_visual = self.visual_history(visual_context,inputs['visual_valid'])
        recent_q = self.proprio_history(torch.cat(((inputs['window_proprio']-self.proprio_mean)/self.proprio_std,
            inputs['window_age'][...,None].float()),-1),inputs['window_valid'])
        recent_action = self.action_history(inputs['window_action'],inputs['window_action_valid'])
        held = frame[inputs['held_indices']]
        current = [held, inputs['held_age'].float(), inputs['camera_valid'][inputs['held_indices']].float(),
                   (inputs['proprio']-self.proprio_mean)/self.proprio_std, inputs['action']]
        recent = [recent_visual,recent_q,recent_action]
        if self.cfg.tactile:
            current_force = inputs['tactile'].flatten(1)
            current.append(current_force)
            recent.append(self.tactile_history(inputs['window_tactile'],inputs['window_tactile_valid']))
        memory, state = self.native_memory(torch.cat(current,-1)[None],state)
        recent.append(memory[0,inputs['anchor_indices']])
        return self.condition(torch.cat(recent,-1)),state

    def loss_from_condition(self, condition, target, valid, generator=None):
        normalized = target*self.target_scale+self.target_offset
        normalized = torch.where(valid[...,None],normalized,0.)
        noise = torch.randn(normalized.shape,device=target.device,generator=generator)
        timestep = torch.randint(100,(len(target),),device=target.device,generator=generator)
        noisy = self.noise.add_noise(normalized,noise,timestep)
        prediction = self.denoiser(noisy,timestep,global_cond=condition)
        target_value = noise if self.cfg.prediction_type == 'epsilon' else normalized
        error = (prediction-target_value).float().square().mean(-1)
        return (error*valid).sum()/valid.sum().clamp_min(1)

    @torch.no_grad()
    def predict_from_condition(self, condition, generator=None):
        sample = torch.randn((len(condition),64,29),device=condition.device,generator=generator)
        self.sampler.set_timesteps(self.cfg.inference_steps,device=condition.device)
        for timestep in self.sampler.timesteps:
            prediction = self.denoiser(sample,timestep,global_cond=condition)
            sample = self.sampler.step(prediction,timestep,sample,eta=0.).prev_sample
        return (sample-self.target_offset)/self.target_scale
