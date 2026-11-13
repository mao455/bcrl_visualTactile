"""Shared frozen weights and batched current-frame vision, isolated lane histories.

DP/Transformer calls remain per lane in this first implementation. Every active
lane consumes its own real RGB-D at each native tick; no temporal frame skipping.
"""
import copy
import torch
from .native_controller import ResidualController


def lane_tree(value, lane, count):
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == count:
            return value[lane:lane+1]
        return value
    if isinstance(value, dict):
        return {key:lane_tree(item,lane,count) for key,item in value.items()}
    return value


class CurrentFrame:
    def set(self, patches):
        self.patches = patches
        self.consumed = False

    def __call__(self, rgb):
        if self.consumed:
            raise ValueError('A lane attempted to reuse a consumed visual frame')
        self.consumed = True
        return self.patches


class BatchedResidualController:
    def __init__(self, saved, metadata, dino, count, device='cuda:0'):
        if count < 1 or saved['contract'].get('visual_stride') != 1:
            raise ValueError('Positive lanes and explicit every-native-frame policy required')
        if saved['contract'].get('modality') != 'G00':
            raise ValueError('Only complete fresh G00 frontends are admitted')
        self.count, self.dino, self.device = count, dino, torch.device(device)
        first = ResidualController(saved, metadata, dino=None, device=device)
        self.lanes = [first]
        for lane in range(1,count):
            controller = copy.copy(first)
            controller.base = copy.copy(first.base)
            controller.base.generator = torch.Generator(device=device).manual_seed(
                first.model.contract.get('sampler_seed',7093))
            # No rollout has begun: all learned recurrence/state are still None.
            self.lanes.append(controller)
        for lane, controller in enumerate(self.lanes):
            controller.base.limits = torch.tensor(metadata['joint_limits_rad'][lane])
            controller.base.dino = CurrentFrame()
        self.visual_frames = [0]*count

    @torch.no_grad()
    def act(self, packet, diagnostics, active):
        active = torch.as_tensor(active, dtype=torch.bool, device='cpu')
        if active.shape != (self.count,) or len(packet['episode_id']) != self.count:
            raise ValueError('Lane mask/observation count differs')
        ids = active.nonzero().flatten().tolist()
        # Inactive lanes have already terminated; their next reset targets are
        # held solely to let still-active native lanes complete naturally.
        commands = packet['actuator_target_rad'].detach().cpu().clone()
        if not ids:
            return commands
        rgb = packet['image'][ids,:,:3].to(self.device)
        features = self.dino.forward_lanes(rgb)
        if len(features) != len(ids):
            raise ValueError('DINO batch lost lane identity')
        for local, lane in enumerate(ids):
            controller = self.lanes[lane]
            controller.base.dino.set(features[local])
            commands[lane:lane+1] = controller.act(lane_tree(packet,lane,self.count),
                lane_tree(diagnostics,lane,self.count))
            if not controller.base.dino.consumed:
                raise ValueError('Every current visual frame must enter the student')
            self.visual_frames[lane] += 1
        return commands

    def observe_execution(self, transition, active):
        for lane, running in enumerate(active):
            if running:
                self.lanes[lane].observe_execution(lane_tree(transition,lane,self.count))
