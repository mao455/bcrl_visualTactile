"""One native expert mapping for the complete batch, independent delay queues.

Calls the unchanged native action pipeline on an N-lane delay-disabled proxy;
uses the position executor's existing single N-lane draw, never draws again.
"""
from types import SimpleNamespace
import torch
from .position_control import ExpertPositionBridge,POSITION_ACTION,processed_expert_target

class VectorExpertPositionBridge:
    version='vector_expert_takeover_position_v1'
    speed_scheme=ExpertPositionBridge.speed_scheme
    def __init__(self,env,takeover_ticks,speed,*,pipeline=None):
        if env.action_control_mode!=POSITION_ACTION or dict(speed)!=self.speed_scheme:
            raise ValueError('Unchanged expert speed and position executor required')
        if pipeline is None:
            from isaacsimenvs.tasks.play.utils.action_utils import apply_action_pipeline
            pipeline=apply_action_pipeline
        self.env=env;self.pipeline=pipeline;self.speed=dict(speed);self.num_envs=env.num_envs
        self.takeover=list(takeover_ticks);self.ticks=[0]*self.num_envs;self.segments=[-1]*self.num_envs
        self.phases=[None]*self.num_envs;self.rescheduled=[False]*self.num_envs
        dr=env.cfg.domain_randomization
        self.delay_enabled=bool(dr.use_action_delay and dr.action_delay_max>0)
        self.queue_length=env._action_queue.shape[1] if self.delay_enabled else 1
        self.queue=torch.zeros(self.num_envs,self.queue_length,29,device=env.device)
        self.source_steps=torch.zeros(self.num_envs,self.queue_length,dtype=torch.long,device=env.device)
        self.initialized=torch.zeros(self.num_envs,dtype=torch.bool,device=env.device)
        self.pending=None;self.tick=0;self.active=[False]*self.num_envs;self.dynamic=True
        keys=('device','num_envs','step_dt','_perm_canon_to_lab','_arm_joint_ids','_hand_joint_ids',
              '_arm_lower','_arm_upper','_hand_lower','_hand_upper')
        self.proxy=SimpleNamespace(**{k:getattr(env,k) for k in keys},cfg=SimpleNamespace(action=env.cfg.action,
            domain_randomization=SimpleNamespace(use_action_delay=False,action_delay_max=0)))
    def replace_lane(self,lane,takeover_tick):
        if self.pending is not None or type(takeover_tick) is not int or takeover_tick<1:raise ValueError('Invalid queue reset')
        self.takeover[lane]=takeover_tick;self.ticks[lane]=0;self.segments[lane]=-1;self.phases[lane]=None
        self.initialized[lane]=False;self.rescheduled[lane]=False
    def schedule_takeover(self,lane,tick):
        if (self.pending is not None or self.rescheduled[lane] or self.ticks[lane]>tick
                or tick>self.takeover[lane] or (self.phases[lane] is not None and self.phases[lane][0])):
            raise ValueError('Invalid causal takeover scheduling')
        self.takeover[lane]=tick;self.rescheduled[lane]=True
    def submit(self,raw_action,clipped_action,tick,active):
        if tick!=self.tick or self.pending is not None or raw_action.shape!=(self.num_envs,29):raise ValueError('Invalid batch clock')
        self.active=list(active)
        ids=[i for i,v in enumerate(active) if v]
        if not torch.isfinite(raw_action[ids]).all() or not torch.equal(raw_action[ids].clamp(-1,1),clipped_action[ids]):
            raise ValueError('Expert batch raw/clipped mismatch')
        self.pending=(raw_action.to(self.env.device),clipped_action.to(self.env.device))
    def select(self,env,student_targets,event_draw):
        if env is not self.env or self.pending is None or env.action_control_mode!=POSITION_ACTION:raise ValueError('Missing batch expert')
        raw,clipped=self.pending;device=env.device
        active=torch.tensor(self.active,device=device,dtype=torch.bool)
        ticks=torch.tensor(self.ticks,device=device);takeover=torch.tensor(self.takeover,device=device)
        new=active & ~self.initialized
        self.queue[new]=clipped[new,None];self.source_steps[new]=0
        shifted=torch.roll(self.queue,1,1);shifted[:,0]=clipped
        steps=torch.roll(self.source_steps,1,1);steps[:,0]=ticks
        self.queue=torch.where(active[:,None,None],shifted,self.queue)
        self.source_steps=torch.where(active[:,None],steps,self.source_steps);self.initialized|=active
        index=event_draw if self.delay_enabled else torch.zeros_like(event_draw)
        if ((index[active]<0)|(index[active]>=self.queue_length)).any():raise ValueError('Expert delay outside queue')
        index=index.clamp(0,self.queue_length-1)
        rows=torch.arange(self.num_envs,device=device)
        delayed=self.queue[rows,index];selected=self.source_steps[rows,index]
        previous=env._prev_targets[:,env._perm_lab_to_canon].clone()
        self.proxy._prev_targets=env._prev_targets.clone();self.proxy._cur_targets=env._cur_targets.clone()
        self.pipeline(self.proxy,delayed)
        native=self.proxy._cur_targets[:,env._perm_lab_to_canon].clone()
        progressed,mapped=processed_expert_target(previous,native,self.speed,env.step_dt)
        expert=active & (ticks>=takeover);clean=expert & (ticks>=takeover+self.queue_length-1)
        source_expert=selected>=takeover;valid=clean & source_expert
        for i,running in enumerate(self.active):
            if running:
                phase=(self.ticks[i]>=self.takeover[i],self.ticks[i]>=self.takeover[i]+self.queue_length-1)
                if phase!=self.phases[i]:self.segments[i]+=1;self.phases[i]=phase
        requested=torch.where(active[:,None],torch.where(expert[:,None],mapped,student_targets.to(device)),previous)
        self.trace=dict(expert_raw_action=raw,expert_clipped_action=clipped,expert_delayed_action=delayed,
            expert_source_step=selected,expert_delay_index=index,
            expert_delay_enabled=torch.full_like(active,self.delay_enabled),expert_queue_length=torch.full_like(ticks,self.queue_length),
            expert_source_is_expert_issued=source_expert,control_source=expert.long(),segment_id=torch.tensor(self.segments,device=device),
            switch_clean=clean,expert_label_valid=valid,expert_native_target_rad=native,
            expert_progressed_target_rad=progressed,expert_mapped_target_rad=mapped,
            student_proposed_target_rad=student_targets.to(device).clone(),previous_issued=previous,
            previous_issued_valid=ticks>0,takeover_tick=takeover,bridge_active=active)
        self.requested=requested.clone();self.pending=None;self.tick+=1
        self.ticks=[t+int(active) for t,active in zip(self.ticks,self.active)]
        return requested
