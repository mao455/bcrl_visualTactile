"""Shared multi-view RGB/depth spatial encoder; see PERCEPTION.md for sources.

No pretrained RGB weights are reinterpreted as four-channel weights. RGB uses
torchvision's ResNet18 through layer2 (stride 8); metric depth + validity have
their own stem. Small spatial grids and soft keypoint moments retain location.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from ..observations import NOMINAL_KEYS_V2, TASK_IDS
from ..history import recurrent_scan


class RGBDSpatialEncoder(nn.Module):
    def __init__(self, feature_dim=128, keypoints=16):
        super().__init__()
        net = resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(max(1,c//16), c))
        self.rgb = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
        self.depth = nn.Sequential(nn.Conv2d(2,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),
                                   nn.Conv2d(32,64,3,2,1),nn.GroupNorm(4,64),nn.SiLU(),
                                   nn.Conv2d(64,128,3,2,1),nn.GroupNorm(8,128),nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(256,64,1), nn.GroupNorm(8,64), nn.SiLU())
        self.keypoints = nn.Conv2d(64,keypoints,1)
        self.grid = nn.Conv2d(64,16,1)
        self.output = nn.Sequential(nn.Linear(keypoints*6+16*3*5,feature_dim),nn.LayerNorm(feature_dim),nn.SiLU())

    def forward(self, image, depth_valid):
        if image.shape[1] != 4 or depth_valid.shape != (len(image),1,*image.shape[-2:]):
            raise ValueError("Expected RGB + metric depth and a separate depth validity mask")
        if not torch.isfinite(image).all():
            raise ValueError("Use zero invalid depth and its mask, not NaN/Inf in policy images")
        mask = depth_valid.bool()
        metric = torch.where(mask,image[:,3:4],0.)
        # Meter scaling is fixed, not fitted on held-out trajectories.
        depth = torch.cat((metric/2.,mask.float()),1)
        feature = self.fuse(torch.cat((self.rgb(image[:,:3]*2-1),self.depth(depth)),1))
        logits = self.keypoints(feature)
        b,k,h,w = logits.shape
        probability = logits.flatten(2).softmax(-1)
        y,x = torch.meshgrid(torch.linspace(-1,1,h,device=image.device,dtype=image.dtype),
                             torch.linspace(-1,1,w,device=image.device,dtype=image.dtype),indexing="ij")
        coordinates = torch.stack((x,y),-1).reshape(h*w,2)
        mean = probability @ coordinates
        variance = (probability @ coordinates.square()-mean.square()).clamp_min(0)
        # Weighted valid metric depth, with confidence; invalid returns don't look like surfaces.
        valid_grid = F.adaptive_avg_pool2d(mask.float(),(h,w)).flatten(2)
        depth_grid = F.adaptive_avg_pool2d(metric,(h,w)).flatten(2)
        confidence = (probability*valid_grid).sum(-1,keepdim=True)
        z = (probability*depth_grid).sum(-1,keepdim=True)/confidence.clamp_min(1e-6)/2.
        spatial = torch.cat((mean,variance,z,confidence),-1).flatten(1)
        grid = F.adaptive_avg_pool2d(self.grid(feature),(3,5)).flatten(1)
        return self.output(torch.cat((spatial,grid),-1))


class MultimodalHistoryEncoder(nn.Module):
    """Exact non-tactile allowlist; own-plan conditioning is causal and masked."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.vision = RGBDSpatialEncoder()
        self.proprio = nn.Sequential(nn.Linear(cfg.proprio_dim,64),nn.SiLU())
        self.task = nn.Embedding(len(TASK_IDS),16)
        self.plan = nn.GRUCell(29,32)
        self.context = nn.Sequential(nn.Linear(16+10+30+32+cfg.views*2,64),nn.SiLU())
        self.temporal = nn.GRUCell(128*cfg.views+128,cfg.hidden)
        self.register_buffer("proprio_mean",torch.zeros(cfg.proprio_dim))
        self.register_buffer("proprio_std",torch.ones(cfg.proprio_dim))

    def encode_frames(self,obs):
        """Frame-local features, for bounded image batches and full recurrent scans."""
        if set(obs) != set(NOMINAL_KEYS_V2)|{"valid"}:
            raise ValueError("Perception v2 nominal inputs differ from the explicit non-tactile allowlist")
        im, q, valid = obs["image"],obs["proprio"],obs["valid"].bool()
        if im.ndim != 6:
            raise ValueError("Image must be B,T,V,4,H,W")
        b,t,v,c,h,w = im.shape
        if (v,c) != (self.cfg.views,4) or q.shape != (b,t,self.cfg.proprio_dim):
            raise ValueError("Image, proprio or history contract mismatch")
        if valid.shape != (b,t) or not valid.any():
            raise ValueError("The batch must contain at least one valid observation")
        # A packed batch keeps a finished, shorter episode in its slot until
        # the longest slot reaches the end.  Such a row is entirely padding;
        # recurrent_scan intentionally carries its previous state unchanged.
        # Active rows still have to contain their current observation, which
        # is checked by the dataset/batch masks and by the caller's loss mask.
        # Padding is allowed after an episode ends.  The recurrent scan keeps
        # the last valid state and the trainer masks all corresponding losses.
        if obs["prior_plan"].shape != (b,t,self.cfg.horizon,29):
            raise ValueError("Prior predicted plan horizon mismatch")
        vision = self.vision(im.reshape(b*t*v,c,h,w),obs["depth_valid"].reshape(b*t*v,1,h,w)).reshape(b,t,v,128)
        camera_valid = obs["camera_valid"].bool()
        vision = torch.where(camera_valid[...,None],vision,0.).flatten(2)
        age = (obs["time_s"][...,None]-obs["camera_time_s"]).to(q.dtype)
        if ((age < -1e-6)&camera_valid&valid[...,None]).any():
            raise ValueError("Camera observation comes from the future")
        plan_h = q.new_zeros(b*t,32)
        for i in range(self.cfg.horizon):
            a = obs["prior_plan"][:,:,i].reshape(b*t,29).detach()
            m = obs["prior_plan_valid"][:,:,i].reshape(b*t,1).bool()
            plan_h = torch.where(m,self.plan(torch.where(m,a,0.),plan_h),plan_h)
        pose_valid = obs["goal_valid"].bool()[...,None]
        previous_valid = obs["previous_nominal_valid"].bool()[...,None]
        context = torch.cat((self.task(obs["task_id"].long()),
            torch.where(pose_valid,obs["goal_command"],0.),pose_valid.float(),
            torch.where(previous_valid,obs["previous_nominal_action"].detach(),0.),previous_valid.float(),
            plan_h.reshape(b,t,32),torch.where(camera_valid,age,0.),camera_valid.float()),-1)
        features = torch.cat((vision,self.proprio((q-self.proprio_mean)/self.proprio_std),self.context(context)),-1)
        return features,valid

    def forward(self,obs,state=None,*,return_state=False,return_sequence=False,reset=None):
        features,valid = self.encode_frames(obs)
        sequence, hidden = recurrent_scan(self.temporal,features,valid,state,reset)
        output = sequence if return_sequence else hidden
        return (output,hidden) if return_state else output
