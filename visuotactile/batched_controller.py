"""Shared frozen weights and batched current-frame vision, isolated lane histories.

Optional neural batching shares DDIM and backend kernels while retaining each
lane's generator, recurrence and reset. Every active lane consumes its own real RGB-D at each native tick; no temporal frame skipping.
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


def stack_lane_trees(values):
    """Join lane-first state; a freshly reset lane gets zero/invalid past state."""
    template=next((value for value in values if value is not None),None)
    if template is None:return None
    if isinstance(template,dict):
        if any(value is not None and set(value)!=set(template) for value in values):
            raise ValueError('Inconsistent lane-state fields')
        return {key:stack_lane_trees([None if value is None else value[key] for value in values]) for key in template}
    if not torch.is_tensor(template) or template.ndim<1 or template.shape[0]!=1:
        raise ValueError('Expected explicit single-lane state tensors')
    return torch.cat([torch.zeros_like(template) if value is None else value for value in values])


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
    def __init__(self, saved, metadata, dino, count, device='cuda:0', *, neural_batch=False,
                 scheduler_index_mode='legacy'):
        if count < 1 or saved['contract'].get('visual_stride') != 1:
            raise ValueError('Positive lanes and explicit every-native-frame policy required')
        if saved['contract'].get('modality') != 'G00':
            raise ValueError('Only complete fresh G00 frontends are admitted')
        self.neural_batch = neural_batch
        self.count, self.dino, self.device = count, dino, torch.device(device)
        first = ResidualController(saved, metadata, dino=None, device=device,
            scheduler_index_mode=scheduler_index_mode)
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
            if not getattr(self,'neural_batch',False):
                commands[lane:lane+1] = controller.act(lane_tree(packet,lane,self.count),
                    lane_tree(diagnostics,lane,self.count))
        if getattr(self,'neural_batch',False):
            self._act_neural_batch(packet,diagnostics,ids,commands)
        for lane in ids:
            controller=self.lanes[lane]
            if not controller.base.dino.consumed:
                raise ValueError('Every current visual frame must enter the student')
            self.visual_frames[lane] += 1
        return commands

    def _act_neural_batch(self,packet,diagnostics,ids,commands):
        replanners=[];conditions=[];packets={};diags={}
        for lane in ids:
            packets[lane]=lane_tree(packet,lane,self.count)
            diags[lane]=lane_tree(diagnostics,lane,self.count)
            condition=self.lanes[lane].base.prepare_observation(packets[lane],diags[lane])
            if condition is not None:
                replanners.append(lane);conditions.append(condition)
        plans={}
        if replanners:
            bases=[self.lanes[lane].base for lane in replanners]
            measurements=[b.timings.begin('ddim_plan',int(packets[lane]['step_id'][0]))
                for lane,b in zip(replanners,bases)]
            predicted=bases[0].model.predict(torch.cat(conditions),[b.generator for b in bases],
                scheduler_index_mode=bases[0].scheduler_index_mode).cpu()
            for i,(lane,b,measurement) in enumerate(zip(replanners,bases,measurements)):
                plans[lane]=predicted[i:i+1];b.timings.finish(measurement)
        inputs=[]
        for lane in ids:
            controller=self.lanes[lane]
            nominal=controller.base.finish_plan(plans.get(lane))
            inputs.append(controller.prepare_feedback(packets[lane],diags[lane],nominal=nominal))
        combined=stack_lane_trees(inputs)
        state=stack_lane_trees([self.lanes[lane].state for lane in ids])
        measurements=[self.lanes[lane].timings.begin('residual_forward_compose',int(packets[lane]['step_id'][0])) for lane in ids]
        raw,next_state=self.lanes[ids[0]].residual(combined,state)
        for i,lane in enumerate(ids):
            controller=self.lanes[lane]
            commands[lane:lane+1]=controller.finish_feedback(inputs[i],raw[i:i+1],lane_tree(next_state,i,len(ids)))
            controller.timings.finish(measurements[i])

    def inference_metrics(self,lane):
        report=self.lanes[lane].inference_metrics()
        if self.neural_batch:
            report['batch_timing_scope']='DDIM/backend durations overlap across lanes and must not be summed; combined_controller begins after batched base inference. Aggregate wall throughput is in the batch profile.'
        return report

    def observe_execution(self, transition, active):
        for lane, running in enumerate(active):
            if running:
                self.lanes[lane].observe_execution(lane_tree(transition,lane,self.count))
