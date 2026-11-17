"""Causal trigger/negative admission checks without simulator or optimizer."""
import copy
import numpy as np
from visuotactile.phase_takeover import profile,PhaseTrigger
from visuotactile.takeover_trigger import validate_trigger

def main():
 for stage in ('transport','near','alignment','weakening','post_subgoal'):
  cfg=profile(stage);trigger=PhaseTrigger(cfg,10**9)
  bad=dict(cfg,confirm_ticks=0)
  try:validate_trigger(bad,10**9)
  except ValueError:pass
  else:raise AssertionError('Changed phase profile admitted')
  try:validate_trigger(cfg,80)
  except ValueError:pass
  else:raise AssertionError('Fixed fallback admitted')
  for tick in range(48):
   p=np.array([0.,0.,.5,1.,0.,0.,0.]);hand=p.copy();hand[2]+=.05
   goal=p.copy();goal[0]=.08;goal[2]=.53
   assert trigger.observe(tick,p,hand,np.zeros((2,5,3)),keypoint_max_distance_m=.2,goal_index_before=0,goal_index_after=0,goal_pose_before=goal,goal_pose_after=goal) is None
  for tick in range(48,80):
   p=np.array([.08,0.,.53,1.,0.,0.,0.]);hand=p.copy();hand[2]+=.05
   if tick>=61:hand[0]+=.02
   force=np.zeros((2,5,3));force[:,:2,2]=1
   goals=int(stage=='post_subgoal' and tick>=65)
   scheduled=trigger.observe(tick,p,hand,force,keypoint_max_distance_m=.02,goal_index_before=goals,goal_index_after=goals,goal_pose_before=goal,goal_pose_after=goal)
   if stage=='post_subgoal' and not goals:assert scheduled is None
   if scheduled is not None:
    assert scheduled==tick+1 and scheduled%12!=0
    break
  else:raise AssertionError('Causal eligible stage never triggered')
  assert trigger.receipt()['student_inputs_changed'] is False
 # A completed subgoal alone cannot justify rescuing an unsupported object.
 trigger=PhaseTrigger(profile('post_subgoal'),10**9)
 for tick in range(100):
  p=np.array([0.,0.,.5,1.,0.,0.,0.]);hand=p.copy();hand[2]+=.05
  assert trigger.observe(tick,p,hand,np.zeros((2,5,3)),keypoint_max_distance_m=.01,
   goal_index_before=1,goal_index_after=1,goal_pose_before=p,goal_pose_after=p) is None
 print('Five causal stages: no pre-held switch, explicit profiles, no fixed fallback, non-planning-boundary switches; post-subgoal requires actual prior success and remaining support')
if __name__=='__main__':main()
