"""Causal native-command correction with explicit current and remaining-plan tokens."""
from dataclasses import dataclass

import torch
from torch import nn


HISTORY_PROTOCOL = 'past_actual_mixed_commands_v1'


@dataclass
class TransformerResidualConfig:
    version: int = 1
    hidden: int = 128
    layers: int = 2
    heads: int = 4
    horizon: int = 6
    recent_ticks: int = 13
    force_masked: bool = False
    amplitude_weight: float = 1e-4

    def __post_init__(self):
        if (self.version,self.hidden,self.layers,self.heads,self.horizon,self.recent_ticks) != (1,128,2,4,6,13):
            raise ValueError('Unsupported residual architecture or native clock')


def combine(base, delta, gate):
    if base.shape != delta.shape or base.shape[:-1] != gate.shape:
        raise ValueError('Residual/base/gate alignment differs')
    if not torch.isfinite(base).all() or not torch.isfinite(delta).all():
        raise ValueError('Nonfinite base or residual command')
    return torch.where(gate[...,None],(base+delta).clamp(-1,1),base)


def residual_loss(prediction, target, valid, amplitude_weight=1e-4):
    error = torch.where(valid[...,None],(prediction.float()-target.float()).square(),0.)
    count = valid.sum().clamp_min(1)
    arm = error[...,:7].mean(-1).sum()/count
    hand = error[...,7:].mean(-1).sum()/count
    amplitude = torch.where(valid[...,None],prediction.float().square(),0.).mean(-1).sum()/count
    return dict(loss=.5*(arm+hand)+amplitude_weight*amplitude,arm=arm,hand=hand,amplitude=amplitude)


class TransformerResidual(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or TransformerResidualConfig()
        hidden = self.cfg.hidden
        self.current_action = nn.Linear(29,hidden)
        self.plan = nn.Linear(29,hidden)
        self.motion = nn.Sequential(nn.Linear(177,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.visual = nn.Linear(384,hidden)
        self.touch = nn.Sequential(nn.Linear(20,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.finger_id = nn.Embedding(5,hidden)
        self.token_type = nn.Embedding(7,hidden)
        self.plan_time = nn.Linear(1,hidden)
        self.queries = nn.Parameter(torch.randn(6,hidden)*.02)
        self.native_memory = nn.GRU(177+384+200,hidden,batch_first=True)
        layer = nn.TransformerEncoderLayer(hidden,self.cfg.heads,hidden*4,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(layer,self.cfg.layers,enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,29))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer('proprio_mean',torch.zeros(58))
        self.register_buffer('proprio_std',torch.ones(58))
        self.register_buffer('target_scale',torch.ones(29))
        self.register_buffer('target_offset',torch.zeros(29))

    def force_features(self, tactile):
        force = tactile[...,:6]
        if self.cfg.force_masked:
            force = torch.zeros_like(force)
        force = force.sign()*torch.log1p(force.abs())
        features = torch.cat((force,tactile[...,6:]),-1)
        return torch.where(tactile[...,-1:].bool(),features,0.)

    def forward(self, inputs, state=None):
        expected = {'proprio','action','target','tracking','visual','tactile','anchors',
                    'base','plan','plan_valid','recent_tactile'}
        if set(inputs) != expected:
            raise ValueError('Residual input allowlist differs')
        anchors = inputs['anchors']
        motion = torch.cat(((inputs['proprio']-self.proprio_mean)/self.proprio_std,
            inputs['target']*self.target_scale+self.target_offset,inputs['tracking'],inputs['action']),-1)
        current_touch = self.force_features(inputs['tactile'])
        native = torch.cat((motion,inputs['visual'],current_touch.flatten(1)),-1)
        memory,state = self.native_memory(native[None],state)
        count = len(anchors)
        if count==0:
            return native.new_empty((0,6,29)),state
        tokens = [self.queries[None].expand(count,-1,-1)+self.token_type.weight[0],
            self.current_action(inputs['base'][anchors])[:,None]+self.token_type.weight[1],
            self.motion(motion[anchors])[:,None]+self.token_type.weight[2],
            self.visual(inputs['visual'][anchors])[:,None]+self.token_type.weight[3],
            memory[0,anchors,None]+self.token_type.weight[4]]
        plan_valid = inputs['plan_valid']
        normalized_plan = inputs['plan']*self.target_scale+self.target_offset
        normalized_plan = torch.where(plan_valid[...,None],normalized_plan,0.)
        relative_time = torch.arange(64,device=motion.device,dtype=motion.dtype)[None,:,None]/60.
        tokens.append(self.plan(normalized_plan)+self.plan_time(relative_time)+self.token_type.weight[5])
        recent = self.force_features(inputs['recent_tactile'])
        recent_tokens = self.touch(recent)+self.finger_id.weight[None,None]+self.token_type.weight[6]
        tokens.append(recent_tokens.flatten(1,2))
        valid = torch.cat((torch.ones(count,10,dtype=torch.bool,device=motion.device),plan_valid,
                           inputs['recent_tactile'][...,-1].bool().flatten(1)),1)
        encoded = self.transformer(torch.cat(tokens,1),src_key_padding_mask=~valid)
        return self.output(encoded[:,:6]).float(),state
