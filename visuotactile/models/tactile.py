"""Five aggregate normal-force sensors; per-finger identity, kinematics and time.

Features are trainable but UNTRAINED until a justified correction objective is
established. No taxel/pressure/friction/slip observability is implied.
"""
import torch
from torch import nn


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

    def forward(self,tactile,valid):
        if set(tactile) != {"force_w_N","force_local_N","position_base_m","rotation_base","time_s"}:
            raise ValueError("Unexpected tactile field; no privileged inputs accepted")
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
        dt = torch.where(pair_valid,dt,1.).to(force.dtype)
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
        hidden = force.new_zeros(b*5,32)
        for i in range(t):
            m = valid[:,i,None].expand(b,5).reshape(b*5,1)
            hidden = torch.where(m,self.per_finger(features[:,i].reshape(b*5,32),hidden),hidden)
        encoded = self.fusion(hidden.reshape(b,5*32))
        return torch.where(valid.any(-1,keepdim=True),encoded,0.)
