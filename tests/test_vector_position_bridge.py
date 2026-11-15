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

def test_expert_lstm_reset_and_active_selection_leave_other_lanes_untouched():
 from types import SimpleNamespace
 from visuotactile.queued_rollout import reset_expert_lanes,active_expert_action
 states=[torch.arange(24.).reshape(1,3,8),torch.ones(1,3,8)]
 player=SimpleNamespace(is_rnn=True,states=states)
 reset_expert_lanes(player,[1],3)
 assert not player.states[0][:,1].any() and torch.equal(player.states[0][:,2],torch.arange(16.,24.).reshape(1,8))
 def act(actor,is_deterministic):
  assert len(actor)==2
  player.states=[s+1 for s in player.states]
  return actor[:,:1].expand(-1,29)
 player.get_action=act
 before=[s.clone() for s in states]
 action=active_expert_action(player,torch.arange(3.)[:,None],[0,2])
 assert action[:,0].tolist()==[0,1]
 for after,prior in zip(player.states,before):
  assert torch.equal(after[:,1],prior[:,1]);assert torch.equal(after[:,[0,2]],prior[:,[0,2]]+1)


def test_indexed_refill_preserves_peer_control_observation():
    from visuotactile.queued_rollout import replace_reset_observations
    old={'image':torch.arange(18).reshape(3,2,3),'nested':{'step':torch.tensor([21,32,43])}}
    new={'image':torch.full((3,2,3),-1),'nested':{'step':torch.tensor([0,32,43])}}
    merged=replace_reset_observations(old,new,[0],3)
    assert torch.equal(merged['image'][1:],old['image'][1:])
    assert torch.equal(merged['image'][0],new['image'][0])
    assert merged['nested']['step'].tolist()==[0,32,43]
    new['image'].zero_();assert (merged['image'][0]==-1).all()
