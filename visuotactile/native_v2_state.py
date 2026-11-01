"""State-only cached causal windows and a deterministic diagnostic BC control."""
import json
from pathlib import Path
import time
import torch
from torch import nn

from .action_history import ExecutionHistory
from .compact import sha256
from .native_controller import NativeController
from .native_data import CONTRACT, Episode, object_dimensions, labels
from .models.native_dp import NativePolicy, load_policy


BC_CONTRACT = dict(CONTRACT, diagnostic_model='StateBC_chunk_mlp_v1')


def normalize_robot(robot, stats):
    value=(robot-stats['robot_mean'])/stats['robot_std']
    return torch.cat((value[...,:116],torch.where(robot[...,176:177].bool(),value[...,116:176],0.),
                      robot[...,176:177],value[...,177:]),-1)


class StateWindows:
    """~tens of MB, loaded once. Every row is a native replan anchor (0,12,...)."""
    def __init__(self, source, identities):
        began=time.monotonic();source=Path(source)
        self.identities=list(identities);self.source=source
        self.stats=torch.load(source/'statistics.pt',map_location='cpu',weights_only=False)
        self.records={};self.samples=[];robots=[];geometry=[];commands=[];valids=[]
        for identity in identities:
            ep=Episode(source,identity,'S-NativeDP');r=ep.record;self.records[identity]=r
            normalized=normalize_robot(r['robot'],self.stats)
            geo=(r['geometry']-self.stats['geometry_mean'])/self.stats['geometry_std']
            anchors=torch.arange(0,len(ep),12)
            targets,valid=labels(r,anchors)
            for j,t in enumerate(anchors.tolist()):
                history=torch.zeros(13,180);frames=torch.zeros(3,87)
                for i,k in enumerate(range(max(0,t-12),t+1), start=max(0,12-t)):
                    history[i]=torch.cat((normalized[k],(r['time'][t]-r['time'][k]).float().reshape(1),torch.ones(1)))
                for i,k in enumerate(range(max(0,j-2),j+1), start=max(0,2-j)):
                    frames[i]=torch.cat((geo[k],(r['time'][t]-r['camera_time'][k,0]).float().reshape(1),torch.ones(1)))
                robots.append(history.flatten());geometry.append(frames.flatten());commands.append(targets[j]);valids.append(valid[j])
                self.samples.append((identity,t))
        self.robot=torch.stack(robots);self.geometry=torch.stack(geometry)
        self.command=torch.stack(commands);self.valid=torch.stack(valids)
        self.load_s=time.monotonic()-began
        self.bytes=sum(x.numel()*x.element_size() for x in (self.robot,self.geometry,self.command,self.valid))

    def __len__(self): return len(self.samples)

    def batch(self, indices, device):
        return tuple(x[indices].to(device) for x in (self.robot,self.geometry,self.command,self.valid))


def condition_windows(model, robot, geometry):
    if isinstance(model,StateBC): return torch.cat((robot,geometry),-1)
    return model.condition_projection(torch.cat((model.robot_encoder(robot),model.state_encoder(geometry)),-1))


class StateBC(NativePolicy):
    """Same causal 13xrobot + 3xgeometry; deterministic 64x29 native-command regression."""
    def __init__(self):
        nn.Module.__init__(self)
        self.group='S-NativeDP'
        self.regressor=nn.Sequential(nn.Linear(2601,256),nn.Mish(),nn.Linear(256,256),nn.Mish(),nn.Linear(256,64*29))
        for name,width in (('robot',178),('geometry',85)):
            self.register_buffer(name+'_mean',torch.zeros(width));self.register_buffer(name+'_std',torch.ones(width))
        self.register_buffer('action_scale',torch.ones(29));self.register_buffer('action_offset',torch.zeros(29))

    def condition(self,memory,time):
        device=self.robot_mean.device
        rows=[torch.cat((value,(time-stamp).float().reshape(1),torch.ones(1,device=device))) for value,stamp in memory.robot]
        rows=[torch.zeros(180,device=device)]*(13-len(rows))+rows
        frames=[torch.cat((feature,(time-stamp[0]).float().reshape(1),torch.ones(1,device=device))) for feature,stamp,_ in memory.frames]
        frames=[torch.zeros(87,device=device)]*(3-len(frames))+frames
        return torch.cat((torch.stack(rows).flatten(),torch.stack(frames).flatten()))[None]

    def loss(self,condition,command,valid,generator=None):
        from .dp_objectives import masked_objectives
        return masked_objectives(self.regressor(condition).reshape(-1,64,29),
            command*self.action_scale+self.action_offset,valid)['B']

    @torch.no_grad()
    def predict(self,condition,generator=None):
        return (self.regressor(condition).reshape(-1,64,29)-self.action_offset)/self.action_scale


def load_state_policy(saved,device='cpu',ema=True):
    if saved.get('contract')==BC_CONTRACT:
        model=StateBC().to(device);model.load_state_dict(saved['ema'] if ema else saved['model']);return model.eval()
    return load_policy(saved,device,ema)


class StateController(NativeController):
    def __init__(self,saved,metadata,device='cuda:0'):
        if saved.get('updates',0)<1: raise ValueError('Untrained controller')
        with torch.random.fork_rng(devices=[]): self.model=load_state_policy(saved,device)
        self.device,self.dino=torch.device(device),None
        self.limits=torch.tensor(metadata['joint_limits_rad'][0]);self.dimensions=object_dimensions(metadata)
        self.generator=torch.Generator(device=device).manual_seed(7093)
        self.episode,self.memory,self.plan=None,None,None
        self.execution=ExecutionHistory(1);self.next_step=0
