"""Causal, privileged collection screens. Never inputs, labels, or task criteria."""
import numpy as np
VERSION='layout_multistage_recovery_v1'
STAGES=('transport','near','alignment','weakening','post_subgoal')

def profile(stage):
 if stage not in STAGES:raise ValueError('Unknown recovery stage')
 result=dict(version=VERSION,stage=stage,transport_xy_m=.07,near_root_m=.07,
  alignment_keypoint_m=.035,held_window=12,confirm_ticks=2,weak_relative_drift_m=.015,weak_root_m=.08)
 if stage=='post_subgoal':result['minimum_completed_subgoals']=1
 return result

def validate(configuration,fallback):
 if configuration!=profile(configuration.get('stage')) or fallback!=10**9:
  raise ValueError('Exact phase profile and never-fallback required')
 return dict(configuration)

class PhaseTrigger:
 def __init__(self,configuration,fallback):
  self.configuration=validate(configuration,fallback);self.scheduled_tick=fallback;self.fallback_tick=fallback
  self.poses=[];self.forces=[];self.relative=[];self.height=None;self.origin=None
  self.first_held=None;self.held_reference=None;self.confirm=0;self.trigger_completed_tick=None
  self.reason='stage_not_reached';self.at_trigger=None;self.last=None
 def observe(self,tick,pose,palm,force,*,keypoint_max_distance_m,goal_index_before,goal_index_after,goal_pose_before,goal_pose_after):
  if tick!=len(self.poses) or tick>=self.scheduled_tick:raise ValueError('Noncausal trigger observation')
  pose,palm,force,goal=map(lambda x:np.asarray(x).copy(),(pose,palm,force,goal_pose_before))
  if [x.shape for x in (pose,palm,force,goal)]!=[(7,),(7,),(2,5,3),(7,)]:raise ValueError('Actual transition shapes required')
  distance=float(keypoint_max_distance_m)
  if not all(np.isfinite(x).all() for x in (pose,palm,force,goal)) or not np.isfinite(distance):raise ValueError('Nonfinite phase truth')
  inverse=palm[3:].copy();inverse[1:]*=-1
  v=pose[:3]-palm[:3];cross=2*np.cross(inverse[1:],v)
  relative=v+inverse[0]*cross+np.cross(inverse[1:],cross)
  self.poses.append(pose);self.forces.append(force);self.relative.append(relative);n=tick+1
  if n==48:
   self.height=float(np.median(np.asarray(self.poses)[24:48,2]));self.origin=np.median(np.asarray(self.poses)[24:48,:2],axis=0)
  if self.height is None:return None
  two=(np.linalg.norm(np.asarray(self.forces[-12:]),axis=-1)>.1).sum(-1).min(-1)>=2
  held=len(two)==12 and bool(two.all()) and bool((np.asarray(self.poses[-12:])[:,2]>=self.height+.02).all())
  if held and self.first_held is None:self.first_held=n;self.held_reference=relative.copy()
  fingers=int((np.linalg.norm(force,axis=-1)>.1).sum(-1).min())
  travel=float(np.linalg.norm(pose[:2]-self.origin));root=float(np.linalg.norm(pose[:3]-goal[:3]))
  angle=float(2*np.arccos(np.clip(abs(np.dot(pose[3:],goal[3:])),0,1)))
  drift=None if self.held_reference is None else float(np.linalg.norm(relative-self.held_reference))
  # Weakening retains a current contact and elevation; never deliberately waits for a drop.
  recoverable=self.first_held is not None and fingers>=1 and pose[2]>=self.height+.01
  p=self.configuration
  criteria=dict(transport=held and travel>=p["transport_xy_m"],near=held and root<=p["near_root_m"],
   alignment=held and distance<=p["alignment_keypoint_m"],
   weakening=recoverable and root<=p["weak_root_m"] and (fingers<2 or drift>=p["weak_relative_drift_m"]),
   post_subgoal=recoverable and root<=p["weak_root_m"]
    and int(goal_index_before)>=p.get('minimum_completed_subgoals',1)
    and int(goal_index_after)>=p.get('minimum_completed_subgoals',1))
  self.last=dict(completed_tick=n,held12=held,fingers=fingers,height_above_settled_m=float(pose[2]-self.height),
   travel_xy_m=travel,root_goal_distance_m=root,keypoint_max_distance_m=distance,orientation_error_rad=angle,
   palm_relative_drift_m=drift,goal_before=int(goal_index_before),goal_after=int(goal_index_after))
  self.confirm=self.confirm+1 if criteria[self.configuration['stage']] else 0
  if self.confirm>=self.configuration['confirm_ticks']:
   self.scheduled_tick=n;self.trigger_completed_tick=n;self.reason='completed_student_'+self.configuration['stage']
   self.at_trigger=dict(self.last);return n
  return None
 def receipt(self):
  return dict(configuration=self.configuration,fallback_tick=self.fallback_tick,actual_scheduled_takeover_tick=self.scheduled_tick,
   reason=self.reason,completed_student_transitions_observed=len(self.poses),settled_height_m=self.height,
   first_held_completed_tick=self.first_held,trigger_completed_tick=self.trigger_completed_tick,
   at_trigger=self.at_trigger,last=self.last,diagnostic_only=True,student_inputs_changed=False,
   native_success_changed=False,no_expert_fallback=True,phase_alignment='next action after causal confirmation; no modulo12 rounding')
