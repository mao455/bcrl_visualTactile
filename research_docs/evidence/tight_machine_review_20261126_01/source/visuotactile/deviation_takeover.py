"""Privileged causal task-regression screens for collection, not policy inputs.

Screens identify observable regressions, not proven irreversible errors. Normal
regrasp/support transitions may also satisfy them; live expert recovery is a gate.
"""
import numpy as np
VERSION='task_regression_takeover_v2'
STAGES=('grasp','transport','near','any')
def profile(stage='any'):
 if stage not in STAGES:raise ValueError(stage)
 return dict(version=VERSION,stage=stage,confirm_ticks=2,window=12,height_loss_m=.01,relative_slip_m=.012,goal_regression_m=.02,near_root_m=.08,stall_ticks=72,stall_progress_m=.002,alignment_floor_m=.015)
def validate(c,f):
 if c!=profile(c.get('stage')) or f!=10**9:raise ValueError('Exact causal regression profile/never fallback required')
 return dict(c)
class DeviationTrigger:
 def __init__(self,configuration,fallback):
  self.configuration=validate(configuration,fallback);self.scheduled_tick=fallback;self.fallback_tick=fallback
  self.records=[];self.height=None;self.first_held=None;self.confirm=0;self.last_reason=None;self.onset=None;self.at_trigger=None;self.reason='no_observed_task_regression';self.last=None
 def observe(self,tick,pose,palm,force,*,keypoint_max_distance_m,goal_index_before,goal_index_after,goal_pose_before,goal_pose_after):
  if tick!=len(self.records) or tick>=self.scheduled_tick:raise ValueError('Each real transition once, in order')
  pose,palm,force,goal=map(np.asarray,(pose,palm,force,goal_pose_before))
  if [v.shape for v in (pose,palm,force,goal)]!=[(7,),(7,),(2,5,3),(7,)]:raise ValueError('Shapes')
  if not all(np.isfinite(v).all() for v in (pose,palm,force,goal)) or not np.isfinite(keypoint_max_distance_m):raise ValueError('Finite truth required')
  inv=palm[3:].copy();inv[1:]*=-1;v=pose[:3]-palm[:3];cross=2*np.cross(inv[1:],v);rel=v+inv[0]*cross+np.cross(inv[1:],cross)
  row=dict(z=float(pose[2]),xy=pose[:2].copy(),relative=rel,fingers=int((np.linalg.norm(force,axis=-1)>.1).sum(-1).min()),root=float(np.linalg.norm(pose[:3]-goal[:3])),key=float(keypoint_max_distance_m),goal=int(goal_index_before),goal_changed=int(goal_index_before)!=int(goal_index_after) or not np.array_equal(goal_pose_before,goal_pose_after))
  self.records.append(row);n=len(self.records);p=self.configuration
  if n==48:self.height=float(np.median([r['z'] for r in self.records[24:48]]))
  if self.height is None:return None
  recent=self.records[-p['window']:];held=all(r['fingers']>=2 and r['z']>=self.height+.02 for r in recent)
  if held and self.first_held is None:self.first_held=n
  same=all(r['goal']==row['goal'] and not r['goal_changed'] for r in recent)
  loss=max(r['z'] for r in recent)-row['z'];slip=float(np.linalg.norm(row['relative']-recent[0]['relative']));reg=row['root']-min(r['root'] for r in recent)
  previous_contact=max(r['fingers'] for r in recent[:-1]);near=row['root']<=p['near_root_m'];elevated=row['z']>=self.height+.005
  # A relative motion alone can be intended manipulation: require contact/height loss.
  unstable=previous_contact>=2 and row['fingers']<2 and (loss>=p['height_loss_m'] or (all(r['fingers']==0 for r in recent[-4:]) and slip>=p['relative_slip_m'])) and elevated
  stalled=False
  w=self.records[-p['stall_ticks']:]
  if len(w)==p['stall_ticks'] and all(r['goal']==row['goal'] and not r['goal_changed'] for r in w):
   stalled=near and held and row['key']>p['alignment_floor_m'] and min(r['key'] for r in w[:12])-min(r['key'] for r in w[-12:])<p['stall_progress_m']
  reason=None;stage=None
  if same and unstable:stage='near' if near else ('transport' if self.first_held is not None else 'grasp');reason='contact_loss_with_height_or_palm_slip'
  elif same and self.first_held is not None and row['fingers']>=1 and elevated and reg>=p['goal_regression_m']:stage='near' if near else 'transport';reason='held_goal_distance_regression'
  elif stalled:stage='near';reason='near_alignment_progress_stall'
  self.last=dict(completed_tick=n,stage=stage,screen=reason,fingers=row['fingers'],height_m=row['z']-self.height,height_loss12_m=loss,palm_slip12_m=slip,goal_regression12_m=reg,root_m=row['root'],keypoint_m=row['key'],goal_index=row['goal'],held12=held)
  qualifies=reason is not None and row['goal']==0 and row['key']>p['alignment_floor_m'] and p['stage'] in ('any',stage)
  if qualifies:
   if self.last_reason!=(stage,reason):self.confirm=0;self.onset=tick
   self.confirm+=1;self.last_reason=(stage,reason)
  else:self.confirm=0;self.last_reason=None;self.onset=None
  if self.confirm>=p['confirm_ticks']:
   self.scheduled_tick=n;self.reason=reason;self.at_trigger=dict(self.last);return n
  return None
 def receipt(self):
  return dict(configuration=self.configuration,actual_scheduled_takeover_tick=self.scheduled_tick,trigger_completed_tick=None if self.at_trigger is None else self.scheduled_tick,first_regression_transition_tick=self.onset,reason=self.reason,at_trigger=self.at_trigger,last=self.last,first_held_completed_tick=self.first_held,diagnostic_only=True,student_inputs_changed=False,no_expert_fallback=True,phase_alignment='next action after causal confirmation, no planner rounding',limitation='Observable regression screen, not a proven harmful causal error or guaranteed recoverability')
