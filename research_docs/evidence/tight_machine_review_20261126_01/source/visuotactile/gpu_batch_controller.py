"""Batch current-frame history, planning inputs and target composition on CUDA.

Same frozen parameters and causal histories as BatchedResidualController. Lane
objects retain independent episode state; GPU tensors never alias another lane's
mutable history. The only host control ticket is episode/step/task validity.
"""
import torch

from .batched_controller import BatchedResidualController, lane_tree, stack_lane_trees
from .continuous_vision import frame_features, read_batch
from .native_controller import InferenceTimings
from .native_data import depth_grid, robot_rows
from .models.native_dp import Memory, TargetResidual

PROFILE = 'gpu_observation_history_control_v1'


class MultiPolicyController:
    """Reuse A/B weights while each assigned lane retains independent memory."""
    def __init__(self,saved,metadata,dino,count,**kwargs):
        self.models={role:GPUBatchedResidualController(value,metadata,dino,count,**kwargs)
                     for role,value in saved.items()}
        self.roles=[None]*count; self.visual_frames=[0]*count; self.count=count
        self.dino=dino;self.vision_microbatch=kwargs.get('vision_microbatch',2)
        self.vision_tail_profile=kwargs.get('vision_tail_profile')
        self.shared_vision_work=dict(calls=0,real_lanes=0,processed_images=0,padding_lanes=0)

    @property
    def lanes(self):
        return [self.models[role].lanes[i] if role is not None else None for i,role in enumerate(self.roles)]

    def assign(self,lane,role):
        self.roles[lane]=role; self.visual_frames[lane]=0

    def act(self,packet,diagnostics,active):
        commands=packet['actuator_target_rad'].clone()
        ids=[i for i,running in enumerate(active) if running]
        if not ids:return commands
        first=next(iter(self.models.values()))
        kwargs=dict(vision_microbatch=self.vision_microbatch)
        if self.vision_tail_profile is not None:kwargs['vision_tail_profile']=self.vision_tail_profile
        features=self.dino.forward_lanes(packet['image'][ids,:,:3].to(first.device),**kwargs)
        from .models.native_vision import vision_lane_budget
        work=vision_lane_budget(len(ids),self.vision_microbatch,vision_tail_profile=self.vision_tail_profile)
        for key in ('real_lanes','processed_images','padding_lanes'):self.shared_vision_work[key]+=work[key]
        self.shared_vision_work['calls']+=work['forward_features_calls']
        positions={lane:i for i,lane in enumerate(ids)}
        for role,controller in self.models.items():
            mask=[running and assigned==role for running,assigned in zip(active,self.roles)]
            if any(mask):
                selected=[positions[i] for i,running in enumerate(mask) if running]
                proposed=controller.act(packet,diagnostics,mask,current_features=features[selected])
                commands[mask]=proposed[mask].to(commands.device)
        for lane,running in enumerate(active):
            if running:self.visual_frames[lane]+=1
        return commands

    def vision_execution_report(self):
        reports={role:c.vision_execution_report() for role,c in self.models.items()}
        if sum(r['real_lanes'] for r in reports.values())!=self.shared_vision_work['real_lanes']:
            raise ValueError('Shared DINO and per-role consumed frames differ')
        return dict(version=PROFILE,roles=reports,
            **self.shared_vision_work)


class GPUBatchedResidualController(BatchedResidualController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.lanes[0].residual.version != TargetResidual.version:
            raise ValueError('GPU batch path requires the trained complete Transformer')
        self.gpu_pipeline = True
        for c in self.lanes:
            c.gpu_pipeline = c.base.gpu_pipeline = True
            c.base.limits = c.base.limits.to(self.device)

    def _act_neural_batch(self, packet, diagnostics, ids, commands):
        device = self.device
        model = self.lanes[0].model
        rear = self.lanes[0].residual
        n = len(ids)
        # Copy/select each sensor field once for the complete active batch.
        fields=('episode_id','step_id','task_id','goal_valid','time_s','camera_time_s','camera_valid','camera_frame',
            'actuator_target_rad','contact_valid','contact_time_s','contact_substeps_w_N','contact_substeps_local_N',
            'contact_w_N','finger_position_base_m','finger_rotation_base','depth_m','depth_valid')
        p = {k:packet[k][ids].to(device) for k in fields}
        ticket = torch.stack((p['episode_id'], p['step_id'], p['task_id'], p['goal_valid'].long()), -1).cpu().tolist()
        q = diagnostics['joint_position_rad'][ids].to(device)
        qd = diagnostics['joint_velocity_rad_s'][ids].to(device)
        if not torch.isfinite(torch.cat((q, qd), -1)).all():
            raise ValueError('Nonfinite canonical observed joint states')
        if not p['camera_valid'].all():raise ValueError('Invalid current RGB-D camera content')
        if (p['camera_time_s'] > p['time_s'][:, None]+1e-6).any() or (
                p['contact_valid'][:, None] & (p['contact_time_s'] > p['time_s'][:, None]+1e-6)).any():
            raise ValueError('Future sensor sample')
        clocks, past, memories, bases, controllers = [], [], [], [], []
        for i, lane in enumerate(ids):
            c = self.lanes[lane]; b = c.base
            episode, tick, task, goal = ticket[i]
            if task or goal: raise ValueError('Unexpected external task command')
            if b.episode != episode:
                if tick != 0: raise ValueError('Complete history requires episode start')
                b.episode, b.next_step, b.memory, b.plan = episode, 0, Memory(), None
                b.position_previous = None; b.started = p['time_s'][i].clone()
                b.timings = InferenceTimings(device)
                c.episode, c.state, c.previous_nominal = episode, None, None
                c.timings = InferenceTimings(device)
                c.batch_observed_counts=dict(history_updates=0,robot_observations=0,plans=0,backend_updates=0)
            if tick != b.next_step or tick != b.memory.next_tick:
                raise ValueError('Skipped or duplicated native observation')
            if tick and not (p['camera_frame'][i]>b.last_packet['camera_frame'][0].to(device)).all():
                raise ValueError('Repeated camera exposure')
            clocks.append(tick); bases.append(b); controllers.append(c); memories.append(b.memory)
            prior = torch.zeros(61, device=device)
            if b.position_previous is not None:
                requested, applied, stamp = b.position_previous
                prior = torch.cat((requested.to(device), applied.to(device),
                    (p['time_s'][i]-stamp.to(device)).float().repeat(2), torch.ones(1, device=device)))
            past.append(prior)
        measured = [c.timings.begin('gpu_batched_complete_controller', tick) for c,tick in zip(controllers,clocks)]
        stamps = p['time_s']
        robot = robot_rows(q, qd, p['actuator_target_rad'], torch.stack(past),
                          stamps-torch.stack([b.started for b in bases]))
        normalized = model.normalized_robot(robot)
        patches = torch.stack([b.dino.patches for b in bases])
        tokens, summary = frame_features(model, patches,
            depth_grid(p['depth_m'], p['depth_valid']),
            stamps[:, None]-p['camera_time_s'], p['camera_valid'])
        native_state = stack_lane_trees([m.native for m in memories])
        visual_state = stack_lane_trees([m.visual for m in memories])
        native_state = model.native_gru(model.native_input(normalized.float()), native_state)
        visual_state = model.visual_gru(summary.float(), visual_state)
        for i, (b,m,tick) in enumerate(zip(bases,memories,clocks)):
            m.robot = (m.robot+[(normalized[i], stamps[i])])[-13:]
            m.frames = (m.frames+[(tokens[i],p['camera_time_s'][i],p['camera_valid'][i])])[-3:]
            m.native, m.visual, m.next_tick = native_state[i:i+1], visual_state[i:i+1], tick+1
            b.last_robot = robot[i]; b.dino.consumed = True
        # Padded history masks only nonexistent past frames, never a real frame.
        frame_tokens=[]; frame_ages=[]; frame_valid=[]
        for i,m in enumerate(memories):
            pad=3-len(m.frames)
            frame_tokens.append(torch.stack([torch.zeros_like(tokens[i])]*pad+[f[0] for f in m.frames]))
            frame_ages.append(torch.stack([torch.zeros_like(stamps[i]-p['camera_time_s'][i])]*pad+
                                          [stamps[i]-f[1] for f in m.frames]))
            frame_valid.append(torch.stack([torch.zeros_like(p['camera_valid'][i])]*pad+[f[2] for f in m.frames]))
        visual = read_batch(model.vision,torch.stack(frame_tokens),torch.stack(frame_ages),torch.stack(frame_valid))
        replanners=[i for i,tick in enumerate(clocks) if tick%12==0]
        if replanners:
            rows=[]
            for i in replanners:
                history=[torch.cat((r,(stamps[i]-t).float().reshape(1),torch.ones(1,device=device)))
                         for r,t in memories[i].robot]
                rows.append(torch.stack([torch.zeros(180,device=device)]*(13-len(history))+history).flatten())
            condition=model.condition_projection(torch.cat((model.robot_encoder(torch.stack(rows)),
                visual[replanners],native_state[replanners],visual_state[replanners]),-1))
            prediction=model.predict(condition,[bases[i].generator for i in replanners],scheduler_index_mode='host')
            for j,i in enumerate(replanners):
                bases[i].last_condition=condition[j:j+1]; bases[i].plan=prediction[j:j+1]
                controllers[i].batch_observed_counts['plans']+=1
        limits=torch.stack([b.limits for b in bases])
        full_plans=torch.cat([b.plan for b in bases])
        full_plans=torch.maximum(torch.minimum(full_plans,limits[:,None,:,1]),limits[:,None,:,0])
        cursor=torch.tensor([t%12 for t in clocks],device=device)
        nominal=full_plans[torch.arange(n,device=device),cursor]
        indices=cursor[:,None]+torch.arange(24,device=device)[None]
        remaining=full_plans.gather(1,indices.clamp_max(23)[...,None].expand(-1,-1,29))
        previous_nominal=torch.cat([torch.zeros_like(nominal[i:i+1]) if c.previous_nominal is None
                                    else c.previous_nominal for i,c in enumerate(controllers)])
        inputs=dict(visual=visual[:,None],robot=robot[:,None],plans=remaining[:,None],
            plan_valid=(indices<24)[:,None],plan_cursor=cursor[:,None],previous_nominal=previous_nominal[:,None],
            previous_nominal_valid=torch.tensor([[c.previous_nominal is not None] for c in controllers],device=device),
            valid=torch.ones(n,1,dtype=torch.bool,device=device),
            tactile_valid=p['contact_valid'][:,None].expand(-1,2),
            tactile={target:p[source] for source,target in (
                ('contact_substeps_w_N','force_w_N'),('contact_substeps_local_N','force_local_N'),
                ('finger_position_base_m','position_base_m'),('finger_rotation_base','rotation_base'),('contact_time_s','time_s'))})
        raw, next_state=rear(inputs,stack_lane_trees([c.state for c in controllers]))
        gate=p['contact_valid'] & (p['contact_w_N'].norm(dim=-1).amax(-1)>.1)
        command,delta=rear.compose(nominal,raw[:,0],gate,limits,previous=robot[:,58:87])
        proposal=delta/rear.action_scale; unbounded=nominal+proposal
        bounded=torch.maximum(torch.minimum(unbounded,limits[...,1]),limits[...,0])
        diag=dict(proposed_rad=proposal,applied_rad=command-nominal,
            clip_mask=(unbounded<limits[...,0])|(unbounded>limits[...,1]),
            tanh_fraction=delta.abs()/rear.scales,contact_valid=p['contact_valid'],
            execution_projection_rad=command-bounded,execution_preprojection_target_rad=bounded,
            execution_previous_target_rad=robot[:,58:87])
        gates=gate.cpu().tolist()
        for i,(lane,c,b,tick) in enumerate(zip(ids,controllers,bases,clocks)):
            c.state=lane_tree(next_state,i,n); c.visual=visual[i:i+1]
            c.nominal=nominal[i:i+1]; c.delta=delta[i:i+1]; c.contact_gate=gates[i]
            c.previous_nominal=c.nominal.clone(); c.command=command[i:i+1]; c.plan=b.plan
            c.residual_diagnostics=lane_tree(diag,i,n)
            b.command=c.command; b.last_packet=lane_tree(packet,lane,self.count); b.next_step+=1
            commands[lane:lane+1]=c.command.to(commands.device)
            for key in ('history_updates','robot_observations','backend_updates'):c.batch_observed_counts[key]+=1
            c.timings.finish(measured[i])

    def inference_metrics(self,lane):
        value=super().inference_metrics(lane)
        value['batch_observed_counts']=dict(self.lanes[lane].batch_observed_counts)
        return value
