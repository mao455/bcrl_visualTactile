"""Three independently initialized policies sharing the native-command contract."""
from dataclasses import dataclass, field
import math

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
    tactile: list = field(default_factory=list)


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
    def __init__(self, group, horizon=64, prediction_type='sample', modality=None):
        super().__init__()
        if group not in GROUPS:
            raise ValueError('Unknown native policy group')
        self.group = group
        if horizon not in (24,32,64):
            raise ValueError('Unsupported prediction horizon')
        self.horizon = horizon
        if prediction_type not in ('sample', 'epsilon'):
            raise ValueError('Unsupported diffusion prediction identity')
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
                                   clip_sample=False, prediction_type=prediction_type)
        self.sampler = DDIMScheduler.from_config(self.noise.config)
        for name, width in (('robot', ROBOT_DIM), ('geometry', GEOMETRY_DIM)):
            self.register_buffer(name+'_mean', torch.zeros(width))
            self.register_buffer(name+'_std', torch.ones(width))
        self.register_buffer('action_scale', torch.ones(29))
        self.register_buffer('action_offset', torch.zeros(29))
        self.modality = modality
        if modality not in (None, 'G00', 'G10') or (modality and group != 'I-HistoryDP'):
            raise ValueError('Unsupported visual modality')
        if modality == 'G10':
            # Existing G10 convention: 242 real 120Hz samples, signed-log
            # world/local five-finger forces + age/valid, GRU32->64.
            # Additive fusion keeps every shared parameter shape/initial value.
            self.tactile_history = nn.GRU(32, 64, batch_first=True)
            self.tactile_fusion = nn.Linear(64, 512)
            self.register_buffer('tactile_mean', torch.zeros(30))
            self.register_buffer('tactile_std', torch.ones(30))

    def normalized_robot(self, robot):
        value = (robot-self.robot_mean)/self.robot_std
        mask = robot[..., 176:177].bool()
        value = torch.cat((value[..., :116], torch.where(mask, value[..., 116:176], 0.),
                           robot[..., 176:177], value[..., 177:]), -1)
        return value

    def observe(self, robot, time, tick, frame, memory=None, tactile=None):
        if (tactile is not None) != (self.modality == 'G10'):
            raise ValueError('G00 cannot consume tactile; G10 requires real tactile')
        if memory is None and self.group == 'I-HistoryDP' and tick != 0:
            raise ValueError('History memory requires the complete causal episode prefix')
        memory = Memory(next_tick=tick) if memory is None else memory
        if tick != memory.next_tick or (frame is not None) != (tick % 12 == 0):
            raise ValueError('Skipped native observation or incorrect visual update')
        if robot.shape != (ROBOT_DIM,):
            raise ValueError('Strict robot input contract differs')
        normalized = self.normalized_robot(robot)
        memory.robot = (memory.robot+[(normalized, time)])[-13:]
        if self.modality == 'G10':
            if tactile.shape != (2, 32) or not torch.isfinite(tactile).all():
                raise ValueError('Two finite causal tactile samples per native tick required')
            memory.tactile = (memory.tactile+[tactile])[-121:]
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
        condition = self.condition_projection(torch.cat(parts, -1))
        if self.modality == 'G10':
            touch = torch.cat(memory.tactile)
            touch = touch[touch[:, -1].bool()]
            if len(touch):
                touch = torch.cat(((touch[:, :30]-self.tactile_mean)/self.tactile_std, touch[:, 30:]), -1)
                _, hidden = self.tactile_history(touch[None])
                condition = condition+self.tactile_fusion(hidden[-1])
        return condition

    def loss(self, condition, command, valid, generator=None, *, temporal_weight=0.):
        if command.shape[1:] != (self.horizon,29):
            raise ValueError('Loss horizon differs from checkpoint contract')
        target = torch.where(valid[..., None], command*self.action_scale+self.action_offset, 0.)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        timestep = torch.randint(100, (len(target),), device=target.device, generator=generator)
        prediction = self.denoiser(self.noise.add_noise(target, noise, timestep), timestep, global_cond=condition)
        objective = noise if self.noise.config.prediction_type == 'epsilon' else target
        base = masked_objectives(prediction, objective, valid)['B']
        if temporal_weight:
            if self.noise.config.prediction_type != 'sample':
                raise ValueError('Temporal objective requires clean action predictions')
            from ..dp_objectives import arm_temporal_objective
            temporal = arm_temporal_objective(prediction, target, valid)
            self.last_loss_components = dict(base=base.detach(), temporal=temporal.detach())
            return base+temporal_weight*temporal
        return base

    @torch.no_grad()
    def predict(self, condition, generator, *, inference_steps=16, clip_sample=False, scheduler_index_mode='legacy'):
        if inference_steps not in (16, 32, 100) or not isinstance(clip_sample, bool):
            raise ValueError('Explicit registered DDIM sampling configuration required')
        if scheduler_index_mode not in ('legacy','host'):
            raise ValueError('Unknown DDIM scheduler index implementation')
        sample = torch.randn((len(condition), self.horizon, 29), device=condition.device, generator=generator)
        # Clip predicted x0 in train-normalized coordinates, before inverse affine.
        # A fresh scheduler prevents one ablation leaking settings into another.
        sampler = DDIMScheduler.from_config(self.noise.config, clip_sample=clip_sample)
        sampler.set_timesteps(inference_steps, device=condition.device)
        # Diffusers' alpha table is on CPU. Passing a CUDA scalar to step()
        # synchronizes for each index and branch repeatedly. Host indices retain
        # that table and its arithmetic exactly; denoising still receives the
        # identical device timestep. Opt-in only after measured cache parity.
        indices=sampler.timesteps.cpu().tolist() if scheduler_index_mode=='host' else sampler.timesteps
        for timestep,index in zip(sampler.timesteps,indices):
            prediction = self.denoiser(sample, timestep, global_cond=condition)
            sample = sampler.step(prediction, index, sample, eta=0.).prev_sample
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
            touch = record['tactile'][tick].to(device) if model.modality == 'G10' else None
            memory = model.observe(robot, stamp, tick, frame, memory, tactile=touch)
            if tick >= start and tick % 12 == 0:
                conditions.append(model.condition(memory, stamp))
                anchors.append(tick)
    return torch.cat(conditions), torch.tensor(anchors), memory


def load_policy(payload, device='cpu', ema=True):
    stored=payload.get('contract');group=payload.get('group')
    if payload.get('updates')==0:
        from ..state_history import trained_continuation_updates
        if trained_continuation_updates(payload)<1:
            raise ValueError('Update-zero policy requires a trained hash-bound continuation source')
    from ..position_control import image_contract
    allowed=[CONTRACT]+[image_contract(g,h) for g in ('I-OfficialDP','I-HistoryDP') for h in (24,32)]
    allowed += [image_contract('I-HistoryDP',24,m) for m in ('G00','G10')]
    allowed += [image_contract('I-HistoryDP',24,m,s) for m in ('G00','G10') for s in ('u_expert','q_reached')]
    if stored not in allowed or group not in GROUPS or (stored != CONTRACT and stored!=image_contract(group,stored['horizon'],stored.get('modality'),stored.get('supervision'))):
        raise ValueError('Not a native-command v1 checkpoint; old weights cannot be reinterpreted')
    model = NativePolicy(group,stored['horizon'],modality=stored.get('modality')).to(device)
    model.contract=stored
    model.load_state_dict(payload['ema'] if ema else payload['model'])
    return model.eval()


# Restored from this repository 3e11d62:models/tactile.py; 120Hz recurrence unchanged.
class FingerTactileEncoder(nn.Module):
    def __init__(self, hidden=64, contact_threshold_N=.1):
        super().__init__()
        self.threshold = contact_threshold_N
        self.finger_id = nn.Embedding(5,8)
        # world/local force, base xyz/rotation6, velocity, force change,
        # magnitude, contact/onset/release, time interval, pair validity + ID.
        self.spatial = nn.Sequential(nn.Linear(27+8,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.per_finger = nn.GRUCell(32,32)
        self.fusion = nn.Sequential(nn.Linear(5*32,128),nn.SiLU(),nn.Linear(128,hidden),nn.LayerNorm(hidden))

    def forward(self,tactile,valid,state=None,*,return_state=False,return_sequence=False):
        if set(tactile) != {"force_w_N","force_local_N","position_base_m","rotation_base","time_s"}:
            raise ValueError("Unexpected tactile field; no privileged inputs accepted")
        # Retain the previous physical sample for boundary derivatives/events.
        # Its features are NOT fed through the recurrence a second time.
        if state is not None:
            previous_valid = state["previous_valid"].bool()
            tactile = {k:torch.cat((state["previous"][k][:,None],v),1) for k,v in tactile.items()}
            valid = torch.cat((previous_valid[:,None],valid),1)
        force = tactile["force_w_N"]
        b,t,f,c = force.shape
        if (f,c) != (5,3) or valid.shape != (b,t):
            raise ValueError("Expected B,T,5,3 force and B,T validity")
        local, pos, rot = tactile["force_local_N"],tactile["position_base_m"],tactile["rotation_base"]
        if rot.shape != (b,t,5,3,3) or pos.shape != force.shape or local.shape != force.shape:
            raise ValueError("Finger calibration/force shape mismatch")
        valid = valid.bool()
        pair_valid = valid & torch.roll(valid,1,1)
        pair_valid[:,0] = False
        dt = tactile["time_s"]-torch.roll(tactile["time_s"],1,1)
        if ((dt<=0)&pair_valid).any():
            raise ValueError("Tactile timestamps must strictly increase within valid history")
        dt = torch.where(pair_valid,dt,torch.ones_like(dt)).to(force.dtype)
        velocity = (pos-torch.roll(pos,1,1))/dt[...,None,None]
        delta = local-torch.roll(local,1,1)
        velocity = torch.where(pair_valid[...,None,None],velocity,0.)
        delta = torch.where(pair_valid[...,None,None],delta,0.)
        norm = force.norm(dim=-1,keepdim=True)
        contact = norm > self.threshold
        previous = torch.roll(contact,1,1)
        onset = contact & ~previous & pair_valid[...,None,None]
        release = ~contact & previous & pair_valid[...,None,None]
        signed_log = lambda x: x.sign()*torch.log1p(x.abs())
        identity = self.finger_id(torch.arange(5,device=force.device))[None,None].expand(b,t,-1,-1)
        features = torch.cat((signed_log(force),signed_log(local),pos,rot[...,:,:2].transpose(-1,-2).flatten(-2),
                              velocity,signed_log(delta),torch.log1p(norm),contact.float(),onset.float(),release.float(),
                              dt[...,None,None].expand(b,t,5,1),pair_valid[...,None,None].expand(b,t,5,1).float(),identity),-1)
        features = self.spatial(torch.where(valid[...,None,None],features,0.))
        hidden = force.new_zeros(b*5,32) if state is None else state["hidden"].reshape(b*5,32)
        seen = torch.zeros(b,dtype=torch.bool,device=force.device) if state is None else state["seen"].bool()
        start = 0 if state is None else 1
        encoded_steps=[]
        for i in range(t):
            if i >= start:
                m = valid[:,i,None].expand(b,5).reshape(b*5,1)
                hidden = torch.where(m,self.per_finger(features[:,i].reshape(b*5,32),hidden),hidden)
                seen = seen | valid[:,i]
                encoded_i = self.fusion(hidden.reshape(b,5*32))
                encoded_steps.append(torch.where(seen[:,None],encoded_i,0.))
        if return_sequence:
            result = torch.stack(encoded_steps,1) if encoded_steps else force.new_zeros(b,0,self.fusion[-1].normalized_shape[0])
        else:
            encoded = self.fusion(hidden.reshape(b,5*32))
            result = torch.where(seen[:,None],encoded,0.)
        # Carry the last actually valid physical sample.  A padded tail must
        # never become the derivative reference for the next chunk.
        last = torch.where(valid, torch.arange(t, device=force.device), -1).amax(1).clamp_min(0)
        rows = torch.arange(b,device=force.device)
        next_state = dict(hidden=hidden.reshape(b,5,32),seen=seen,
                          previous={k:v[rows,last] for k,v in tactile.items()},
                          previous_valid=valid[rows,last])
        return (result,next_state) if return_state else result


class LegacyTargetResidual(nn.Module):
    """3e11d62 ResidualCorrection's 60Hz GRUs, adapted to H24 target radians.

    The frozen current DP supplies its existing 512D visual readout instead of
    the retired B0 frame encoder. Hidden sizes, contact gate, tanh residual and
    equal arm/hand objective are retained; there is no 10Hz residual window.
    """
    version='target_residual_gru60_v1'

    def __init__(self, statistics, scales):
        super().__init__()
        self.visual=nn.GRUCell(512,128)
        self.tactile=FingerTactileEncoder()
        self.intent=nn.GRUCell(29,64)
        self.intent_memory=nn.GRUCell(64,64)
        self.motion=nn.GRUCell(146,64)
        self.policy=nn.Sequential(nn.Linear(320,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,29))
        nn.init.zeros_(self.policy[-1].weight);nn.init.zeros_(self.policy[-1].bias)
        for k in ('robot_mean','robot_std','action_scale','action_offset'):
            self.register_buffer(k,statistics[k].detach().cpu().clone())
        self.register_buffer('scales',torch.as_tensor(scales,dtype=torch.float32))

    def forward(self, inputs, state=None, *, mask_force=False):
        if set(inputs)!={'visual','robot','tactile','tactile_valid','plans','plan_valid','valid'}:
            raise ValueError('Residual sensor/nominal allowlist differs')
        state={} if state is None else state
        robot=inputs['robot'];batch,steps=robot.shape[:2];valid=inputs['valid']
        touch=inputs['tactile']
        if mask_force:
            touch={k:torch.zeros_like(v) if k in ('force_w_N','force_local_N') else v for k,v in touch.items()}
        tactile,ts=self.tactile(touch,inputs['tactile_valid'],state.get('tactile'),return_state=True,return_sequence=True)
        tactile=tactile[:,1::2]
        plan=inputs['plans']*self.action_scale+self.action_offset
        intent=robot.new_zeros(batch*steps,64)
        for cursor in range(24):
            intent=torch.where(inputs['plan_valid'][...,cursor].reshape(-1,1),
                self.intent(plan[:,:,cursor].detach().reshape(-1,29),intent),intent)
        intent=intent.reshape(batch,steps,64)
        previous=torch.where(robot[...,176:177].bool(),robot[...,116:145]*self.action_scale+self.action_offset,0.)
        motion=torch.cat(((robot[...,:58]-self.robot_mean[:58])/self.robot_std[:58],previous,
            robot[...,176:177],robot[...,58:87]*self.action_scale+self.action_offset,robot[...,87:116]),-1)
        def scan(cell,x,key,width):
            h=state.get(key,x.new_zeros(batch,width));rows=[]
            for t in range(steps):
                h=torch.where(valid[:,t,None],cell(x[:,t],h),h);rows.append(h)
            return torch.stack(rows,1),h
        visual,vs=scan(self.visual,inputs['visual'],'visual',128)
        intent,ins=scan(self.intent_memory,intent,'intent',64)
        motion,ms=scan(self.motion,motion,'motion',64)
        raw=self.policy(torch.cat((visual,tactile,intent,motion),-1))
        return raw,dict(visual=vs,tactile=ts,intent=ins,motion=ms)

    def compose(self, base, raw, contact, limits):
        delta=self.scales*raw.tanh()
        # Residual units are the existing train-only affine action coordinates.
        # Inverse scale only: zero delta and noncontact preserve base bitwise.
        proposal=base+delta/self.action_scale
        proposal=torch.maximum(torch.minimum(proposal,limits[...,1]),limits[...,0])
        return torch.where(contact[...,None],proposal,base),delta

    def objective(self, action, target, delta, contact):
        error=((action-target)*self.action_scale).square()
        fit=.5*error[...,:7].mean(-1)+.5*error[...,7:].mean(-1)
        penalty=.5*delta[...,:7].square().mean(-1)+.5*delta[...,7:].square().mean(-1)
        return ((fit+1e-4*penalty)*contact).sum()/contact.sum().clamp_min(1)


PRECLIP_RESIDUAL_LOSS_VERSION = 'exec_preclip_equal_arm_hand_v1'
CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION = 'current_weight_prefix_v1'


def validate_residual_loss_contract(value):
    """Absent identity preserves historical exec-only fitting, including its math."""
    if value is None:
        return None
    from ..execution_learning import PROJECTION_FY_LOSS,is_projection_fy_loss
    if is_projection_fy_loss(value):
        weight=value.get('regularizer_weight')
        if (set(value)!=set(PROJECTION_FY_LOSS) or value.get('multiplier')!=2.
                or value.get('idealization_tolerance_rad')!=1e-6
                or isinstance(weight,bool) or not isinstance(weight,(int,float))
                or not math.isfinite(weight) or not 0<=weight<=1e-4):
            raise ValueError('Invalid explicit projection loss weights or rad tolerance')
        return dict(value)
    if (not isinstance(value, dict) or set(value) != {'version', 'lambda_pre'}
            or value['version'] != PRECLIP_RESIDUAL_LOSS_VERSION):
        raise ValueError('Unknown residual loss contract')
    weight = value['lambda_pre']
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Preclip command supervision needs a finite positive weight')
    return dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=float(weight))


def validate_residual_history_contract(value):
    """Historical checkpoints retain carry-and-detach TBPTT unless explicit."""
    if value is None:
        return None
    if value != dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION):
        raise ValueError('Unknown residual history training contract')
    return dict(value)


class TargetResidual(nn.Module):
    """60 Hz Transformer plan/feedback fusion with persistent sensor memory.

    The Transformer query is current collected feedback, including separately
    identified previous nominal and executed commands. Its memory is the
    currently available H24 nominal plan, 32 past feedback tokens, and a
    persistent recurrent summary. Future *predicted plan* tokens are legal;
    future measured feedback or expert targets are never supplied. All new
    groups retain the same frozen visual readout bypass. No contact hard gate
    removes correction during free motion or temporary loss of contact.
    """
    version = 'target_residual_transformer60_v1'
    history_ticks = 32
    width = 128
    input_keys = {'visual', 'robot', 'tactile', 'tactile_valid', 'plans',
                  'plan_valid', 'plan_cursor', 'previous_nominal',
                  'previous_nominal_valid', 'valid'}

    def __init__(self, statistics, scales, *, execution_adapter=None, loss_contract=None, history_contract=None, output_contract=None):
        super().__init__()
        from ..position_control import validate_execution_adapter
        self.execution_adapter = validate_execution_adapter(execution_adapter)
        self.loss_contract = validate_residual_loss_contract(loss_contract)
        self.history_contract = validate_residual_history_contract(history_contract)
        from ..residual_output import validate_output_contract
        self.output_contract=validate_output_contract(output_contract)
        from ..execution_learning import is_projection_fy_loss
        if (is_projection_fy_loss(self.loss_contract) or self.output_contract is not None) and self.execution_adapter is None:
            raise ValueError('Projection loss requires an explicit execution adapter')
        self.visual = nn.GRUCell(512, 64)
        self.tactile = FingerTactileEncoder()
        self.motion = nn.GRUCell(ROBOT_DIM, 64)
        self.feedback = mlp(64*3+30, self.width)
        self.memory = nn.GRUCell(self.width, self.width)
        self.intent = nn.Linear(29, self.width)
        self.plan_position = nn.Embedding(24, self.width)
        self.plan_cursor = nn.Embedding(12, self.width)
        self.history_position = nn.Embedding(self.history_ticks, self.width)
        self.token_type = nn.Embedding(3, self.width)
        layer = nn.TransformerDecoderLayer(self.width, 4, dim_feedforward=256,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(self.width))
        self.policy = nn.Sequential(nn.Linear(self.width, 128), nn.SiLU(), nn.Linear(128, 29))
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        for name in ('robot_mean', 'robot_std', 'action_scale', 'action_offset'):
            self.register_buffer(name, statistics[name].detach().cpu().clone())
        scale = torch.as_tensor(scales, dtype=torch.float32)
        if scale.shape != (29,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Residual scales must be 29 finite positive normalized amplitudes')
        self.register_buffer('scales', scale)

    def architecture(self):
        return dict(version=self.version, transformer='decoder_cross_attention', layers=2,
            width=self.width, heads=4, feedforward=256, dropout=0.,
            plan_tokens=24, past_feedback_tokens=self.history_ticks,
            persistent_memory='visual64, tactile64 with per-finger32 at120Hz, motion64, feedback128; episode reset only',
            previous_nominal='separate previous-tick base prediction in radians',
            direct_frozen_visual_bypass=True, control_hz=60,
            gate='none; all valid native ticks supervised including noncontact',
            parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, inputs, state=None, *, mask_force=False, state_only=False):
        if set(inputs) != self.input_keys:
            raise ValueError('Transformer residual sensor/plan allowlist differs')
        state = {} if state is None else state
        robot = inputs['robot']
        batch, steps = robot.shape[:2]
        valid = inputs['valid'].bool()
        cursor = inputs['plan_cursor']
        if robot.shape != (batch, steps, ROBOT_DIM) or valid.shape != (batch, steps) or steps < 1:
            raise ValueError('Expected nonempty B,T,178 causal robot observations')
        if cursor.shape != valid.shape or cursor.is_floating_point() or ((cursor < 0) | (cursor >= 12)).any():
            raise ValueError('Expected explicit H24/commit12 integer plan cursor')
        if inputs['plans'].shape != (batch, steps, 24, 29) or inputs['plan_valid'].shape != (batch, steps, 24):
            raise ValueError('Expected current remaining H24 plan and validity mask')
        if inputs['previous_nominal'].shape != (batch, steps, 29) or inputs['previous_nominal_valid'].shape != valid.shape:
            raise ValueError('Previous nominal must have its own value and validity')
        if not inputs['plan_valid'][valid].any(-1).all():
            raise ValueError('Every valid control tick needs a valid current nominal plan')
        touch = inputs['tactile']
        if mask_force:
            touch = {k: torch.zeros_like(v) if k in ('force_w_N', 'force_local_N') else v for k, v in touch.items()}
        touch_valid = inputs['tactile_valid'].bool() & valid.repeat_interleave(2, dim=1)
        tactile, ts = self.tactile(touch, touch_valid, state.get('tactile'), return_state=True, return_sequence=True)
        tactile = tactile[:, 1::2]
        normalized = (robot-self.robot_mean)/self.robot_std
        # Unavailable strict-past execution slots must remain exactly zero.
        normalized = torch.cat((normalized[..., :116],
            torch.where(robot[..., 176:177].bool(), normalized[..., 116:176], 0.),
            robot[..., 176:177], normalized[..., 177:]), -1)
        previous_valid = inputs['previous_nominal_valid'].bool()
        previous_nominal = torch.where(previous_valid[..., None],
            inputs['previous_nominal']*self.action_scale+self.action_offset, 0.)
        visual_state = state.get('visual', robot.new_zeros(batch, 64))
        motion_state = state.get('motion', robot.new_zeros(batch, 64))
        memory_state = state.get('memory', robot.new_zeros(batch, self.width))
        history = state.get('history', robot.new_zeros(batch, self.history_ticks, self.width))
        history_valid = state.get('history_valid', torch.zeros(batch, self.history_ticks, dtype=torch.bool, device=robot.device))
        queries, summaries, histories, history_masks = [], [], [], []
        for tick in range(steps):
            active = valid[:, tick, None]
            visual_state = torch.where(active, self.visual(inputs['visual'][:, tick], visual_state), visual_state)
            motion_state = torch.where(active, self.motion(normalized[:, tick], motion_state), motion_state)
            feedback = self.feedback(torch.cat((visual_state, tactile[:, tick], motion_state,
                previous_nominal[:, tick], previous_valid[:, tick, None].to(robot.dtype)), -1))
            memory_state = torch.where(active, self.memory(feedback, memory_state), memory_state)
            if not state_only:
                queries.append(feedback + memory_state + self.plan_cursor(cursor[:, tick]))
                summaries.append(memory_state)
                histories.append(history)
                history_masks.append(history_valid)
            # The query sees only preceding feedback tokens. Current feedback
            # becomes history only for the next native observation.
            history = torch.where(active[..., None], torch.cat((history[:, 1:], feedback[:, None]), 1), history)
            history_valid = torch.where(active, torch.cat((history_valid[:, 1:], valid[:, tick, None]), 1), history_valid)
        next_state = dict(visual=visual_state, tactile=ts, motion=motion_state,
            memory=memory_state, history=history, history_valid=history_valid)
        if state_only:
            # The decoder/output head never feed the recurrent state. Recompute
            # all current-parameter sensor/history modules, then skip only the
            # unused plan/query readout during a causal prefix replay.
            return None, next_state
        plan = inputs['plans'].detach()*self.action_scale+self.action_offset
        positions = (torch.arange(24, device=robot.device)[None, None]+cursor[..., None]).clamp_max(23)
        plan_tokens = self.intent(plan)+self.plan_position(positions)+self.token_type.weight[0]
        history_tokens = torch.stack(histories, 1)+self.history_position.weight+self.token_type.weight[1]
        summary_tokens = torch.stack(summaries, 1)[:, :, None]+self.token_type.weight[2]
        tokens = torch.cat((plan_tokens, history_tokens, summary_tokens), 2).reshape(batch*steps, 57, self.width)
        padding = torch.cat((~inputs['plan_valid'].bool(), ~torch.stack(history_masks, 1),
            torch.zeros(batch, steps, 1, dtype=torch.bool, device=robot.device)), -1).reshape(batch*steps, 57)
        query = torch.stack(queries, 1).reshape(batch*steps, 1, self.width)
        fused = self.transformer(query, tokens, memory_key_padding_mask=padding).reshape(batch, steps, self.width)
        raw = torch.where(valid[..., None], self.policy(fused), 0.)
        return raw, next_state

    def compose_chain(self, base, raw, contact, limits, *, previous=None):
        """Expose the real command chain without changing final execution math.

        u_pre is an absolute command in rad, before *both* hard clips. delta is
        the bounded correction in command-normalizer units, not a command.
        Observed contact remains diagnostic and never becomes a hard switch.
        """
        from ..residual_output import proposal_delta
        from ..execution_learning import is_projection_fy_loss,ideal_execution_bounds
        bounds=None
        if getattr(self,'output_contract',None) is not None or is_projection_fy_loss(self.loss_contract):
            if previous is None:raise ValueError('Conditional proposal requires actual previous target')
            bounds=ideal_execution_bounds(previous,limits,self.execution_adapter)
        delta = proposal_delta(raw,self.scales,self.action_scale,bounds,getattr(self,'output_contract',None))
        u_pre = base+delta/self.action_scale
        u_joint = torch.maximum(torch.minimum(u_pre, limits[..., 1]), limits[..., 0])
        proposal = u_joint
        if self.execution_adapter is not None:
            from ..position_control import project_processed_position_targets
            if previous is None:
                raise ValueError('Reachable residual composition requires real pre-action targets')
            proposal = project_processed_position_targets(proposal, previous, limits, self.execution_adapter)
        result = dict(raw=raw, delta=delta, u_pre=u_pre, u_joint=u_joint, u_exec=proposal)
        if bounds is not None:result['execution_bounds']=bounds
        return result

    def compose(self, base, raw, contact, limits, *, previous=None):
        chain = self.compose_chain(base, raw, contact, limits, previous=previous)
        return chain['u_exec'], chain['delta']

    def objective_components(self, action, target, delta, valid, *, u_pre=None, execution_bounds=None):
        """Separate main fits from the historical residual regularizer.

        ``valid`` is the legal supervision mask or its recovery-weighted form.
        The same weights/denominator and command affine apply to both fits.
        An absent loss contract retains the exact legacy total arithmetic.
        """
        error = ((action-target)*self.action_scale).square()
        fit = .5*error[..., :7].mean(-1)+.5*error[..., 7:].mean(-1)
        penalty = .5*delta[..., :7].square().mean(-1)+.5*delta[..., 7:].square().mean(-1)
        denominator = valid.sum().clamp_min(1)
        loss_contract = getattr(self, 'loss_contract', None)
        pre_fit = torch.zeros_like(fit)
        from ..execution_learning import is_projection_fy_loss, projection_fy_elements
        if is_projection_fy_loss(loss_contract):
            if u_pre is None or execution_bounds is None:
                raise ValueError('Projection loss needs real precommand and conditional execution bounds')
            elements, roundoff = projection_fy_elements(u_pre, target, execution_bounds, valid,
                tolerance=loss_contract['idealization_tolerance_rad'])
            elements = elements*self.action_scale.square()
            fy = .5*elements[..., :7].mean(-1)+.5*elements[..., 7:].mean(-1)
            main = (fy*valid).sum()/denominator
            regularizer = (loss_contract['regularizer_weight']*penalty*valid).sum()/denominator
            return dict(exec_fit=(fit*valid).sum()/denominator, fy_fit=main,
                label_roundoff_max_rad=roundoff.max(), regularizer=regularizer, total=main+regularizer)
        if loss_contract is not None:
            if u_pre is None or u_pre.shape != action.shape:
                raise ValueError('Preclip loss requires the unclipped absolute composed command')
            error_pre = ((u_pre-target)*self.action_scale).square()
            pre_fit = .5*error_pre[..., :7].mean(-1)+.5*error_pre[..., 7:].mean(-1)
            total = ((fit+loss_contract['lambda_pre']*pre_fit+1e-4*penalty)*valid).sum()/denominator
        else:
            total = ((fit+1e-4*penalty)*valid).sum()/denominator
        return dict(exec_fit=(fit*valid).sum()/denominator,
            pre_fit=(pre_fit*valid).sum()/denominator,
            regularizer=(1e-4*penalty*valid).sum()/denominator, total=total)

    def objective(self, action, target, delta, valid, *, u_pre=None, execution_bounds=None):
        """Equal Arm/Hand command fitting; optional explicit preclip supervision."""
        if getattr(self, 'loss_contract', None) is None:
            # Keep the historical training path's reductions and arithmetic;
            # extra component reductions are opt-in diagnostics/new training.
            error = ((action-target)*self.action_scale).square()
            fit = .5*error[..., :7].mean(-1)+.5*error[..., 7:].mean(-1)
            penalty = .5*delta[..., :7].square().mean(-1)+.5*delta[..., 7:].square().mean(-1)
            return ((fit+1e-4*penalty)*valid).sum()/valid.sum().clamp_min(1)
        return TargetResidual.objective_components(self, action, target, delta, valid, u_pre=u_pre, execution_bounds=execution_bounds)['total']


def load_residual(payload, device='cpu'):
    """Dispatch historical evidence explicitly; GRU weights never become Transformer."""
    if payload.get('updates')==0:
        from ..state_history import trained_continuation_updates
        if trained_continuation_updates(payload)<1:
            raise ValueError('Update-zero residual requires a trained hash-bound continuation source')
    versions = {LegacyTargetResidual.version: LegacyTargetResidual, TargetResidual.version: TargetResidual}
    if payload.get('residual_version') not in versions:
        raise ValueError('Unknown residual architecture identity')
    state = payload['model']
    cls = versions[payload['residual_version']]
    adapter = payload.get('configuration', {}).get('execution_adapter')
    loss_contract = payload.get('configuration', {}).get('loss_contract')
    history_contract = payload.get('configuration', {}).get('history_contract')
    output_contract = payload.get('configuration', {}).get('output_contract')
    if payload.get('residual_output_version') != (output_contract or {}).get('version'):
        raise ValueError('Residual checkpoint output identity differs from configuration')
    if payload.get('residual_loss_version') != (loss_contract or {}).get('version'):
        raise ValueError('Residual checkpoint loss identity differs from configuration')
    if payload.get('residual_history_version') != (history_contract or {}).get('version'):
        raise ValueError('Residual checkpoint history identity differs from configuration')
    if cls is LegacyTargetResidual and adapter is not None:
        raise ValueError('Legacy GRU residual cannot acquire a new execution adapter')
    if cls is LegacyTargetResidual and (loss_contract is not None or history_contract is not None or output_contract is not None):
        raise ValueError('Legacy GRU residual cannot acquire Transformer training semantics')
    kwargs = dict(execution_adapter=adapter, loss_contract=loss_contract,
        history_contract=history_contract,output_contract=output_contract) if cls is TargetResidual else {}
    model = cls(state, state['scales'], **kwargs).to(device)
    model.load_state_dict(state)
    return model.eval()
