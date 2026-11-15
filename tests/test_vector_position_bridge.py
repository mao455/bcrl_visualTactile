import torch
from test_batched_position_bridge import environment,native_pipeline
from visuotactile.batched_position_bridge import BatchedExpertPositionBridge
from visuotactile.vector_position_bridge import VectorExpertPositionBridge
from visuotactile.position_control import ExpertPositionBridge

def test_vector_native_targets_labels_and_independent_lane_replacement():
 env=environment(3);speed=ExpertPositionBridge.speed_scheme;take=[3,5,7]
 old=BatchedExpertPositionBridge(env,take,speed,pipeline=native_pipeline());old.dynamic=True
 new=VectorExpertPositionBridge(env,take,speed,pipeline=native_pipeline())
 g=torch.Generator().manual_seed(91)
 for tick in range(20):
  if tick==11:
   before=old.bridges[1].queue.clone();old.replace_lane(0,2);new.replace_lane(0,2)
   assert torch.equal(old.bridges[1].queue,before)
  active=[True,tick<17,True]
  raw=torch.randn(3,29,generator=g);clipped=raw.clamp(-1,1);student=torch.randn(3,29,generator=g)*.1
  draw=torch.randint(0,4,(3,),generator=g)
  old.submit(raw,clipped,tick,active);new.submit(raw,clipped,tick,active)
  expected=old.select(env,student,draw);actual=new.select(env,student,draw)
  assert torch.equal(expected,actual)
  for key in old.trace:
   assert torch.equal(old.trace[key][active],new.trace[key][active]),key
  env._prev_targets=actual[:,env._perm_canon_to_lab].clone();env._cur_targets=env._prev_targets.clone()
