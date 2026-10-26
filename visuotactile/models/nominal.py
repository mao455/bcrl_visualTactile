"""Tactile-free history-conditioned action diffusion, not preload nominalization."""
from dataclasses import dataclass
import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
from .vendor.conditional_unet1d import ConditionalUnet1D


@dataclass
class PolicyConfig:
    views: int = 1
    channels: int = 4
    proprio_dim: int = 58
    history: int = 4
    stride: int = 3
    horizon: int = 12
    replan_steps: int = 6
    hidden: int = 128
    diffusion_steps: int = 100
    inference_steps: int = 8


class VisualHistoryEncoder(nn.Module):
    """Shared per-view CNN, ordered view features, masked finite-window GRU."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision = nn.Sequential(
            nn.Conv2d(cfg.channels, 16, 5, 2, 2), nn.GroupNorm(4,16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4,32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.GroupNorm(4,32), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Linear(512,64), nn.SiLU(),
        )
        self.proprio = nn.Sequential(nn.Linear(cfg.proprio_dim,64), nn.SiLU())
        self.temporal = nn.GRUCell(64*cfg.views+64, cfg.hidden)
        self.register_buffer("proprio_mean", torch.zeros(cfg.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(cfg.proprio_dim))

    def forward(self, obs):
        if set(obs) != {"image", "proprio", "valid"}:
            raise ValueError("Nominal encoder accepts ONLY image, proprio, valid history")
        image, proprio, valid = obs["image"], obs["proprio"], obs["valid"]
        if image.ndim != 6:
            raise ValueError("image must be (B,T,V,C,H,W)")
        b,t,v,c,h,w = image.shape
        if (v,c) != (self.cfg.views,self.cfg.channels) or proprio.shape != (b,t,self.cfg.proprio_dim):
            raise ValueError("Observation dimensions differ from checkpoint contract")
        if valid.shape != (b,t) or not valid[:, -1].all():
            raise ValueError("Valid history must include the current observation")
        visual = self.vision(image.reshape(b*t*v,c,h,w)).reshape(b,t,v*64)
        q = self.proprio((proprio-self.proprio_mean)/self.proprio_std)
        x = torch.cat((visual,q),dim=-1)
        hidden = x.new_zeros(b,self.cfg.hidden)
        for i in range(t):
            update = self.temporal(x[:,i],hidden)
            hidden = torch.where(valid[:,i,None],update,hidden)
        return hidden


class NominalDiffusionPolicy(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        if self.cfg.horizon % 2 or not 1 <= self.cfg.replan_steps <= self.cfg.horizon:
            raise ValueError("Two-level U-Net requires even horizon; replan must fit chunk")
        self.encoder = VisualHistoryEncoder(self.cfg)
        self.denoiser = ConditionalUnet1D(input_dim=29,global_cond_dim=self.cfg.hidden,
            diffusion_step_embed_dim=64,down_dims=(64,128),kernel_size=3,cond_predict_scale=True)
        self.noise = DDPMScheduler(num_train_timesteps=self.cfg.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
        self.sampler = DDIMScheduler.from_config(self.noise.config)

    def loss(self, obs, action_target, action_valid):
        if action_target.shape[1:] != (self.cfg.horizon,29):
            raise ValueError("Action supervision must be (B,H,29)")
        cond = self.encoder(obs)
        eps = torch.randn_like(action_target)
        t = torch.randint(self.cfg.diffusion_steps,(len(action_target),),device=action_target.device)
        noisy = self.noise.add_noise(action_target,eps,t)
        prediction = self.denoiser(noisy,t,global_cond=cond)
        squared = (prediction-eps).square().mean(-1)
        return (squared*action_valid).sum()/action_valid.sum().clamp_min(1)

    @torch.no_grad()
    def predict(self, obs, generator=None):
        cond = self.encoder(obs)
        x = torch.randn((len(cond),self.cfg.horizon,29),device=cond.device,generator=generator)
        self.sampler.set_timesteps(self.cfg.inference_steps, device=cond.device)
        for t in self.sampler.timesteps:
            pred = self.denoiser(x,t,global_cond=cond)
            x = self.sampler.step(pred,t,x,eta=0.0).prev_sample
        return x.clamp(-1,1)
