"""Explicit auxiliary identity, strict checkpoint loading and no deployment path."""
import copy
import pytest
import torch
from visuotactile.models.native_dp import NativePolicy,load_policy
from visuotactile.position_control import continuous_image_contract,visual_auxiliary_image_contract
from visuotactile.visual_pose_aux import VisualPoseHead


@pytest.fixture(scope='module')
def payload():
    torch.manual_seed(7)
    model=NativePolicy('I-HistoryDP',24,modality='G00')
    model.visual_pose_head=VisualPoseHead(seed=51)
    return dict(group=model.group,updates=1,
        contract=visual_auxiliary_image_contract('q_reached','shared_vision'),
        ema=model.state_dict())


def test_new_checkpoint_requires_exact_head_and_contract(payload):
    loaded=load_policy(payload)
    assert not loaded.training and hasattr(loaded,'visual_pose_head')
    for key,value in loaded.state_dict().items():assert torch.equal(value,payload['ema'][key])
    missing=dict(payload,ema={k:v for k,v in payload['ema'].items() if not k.startswith('visual_pose_head.')})
    with pytest.raises(RuntimeError,match='Missing key'):load_policy(missing)
    incorrect=copy.deepcopy(payload['contract']);incorrect['perception_auxiliary']['deploy_predictions']=True
    with pytest.raises(ValueError):load_policy(dict(payload,contract=incorrect))
    # New head tensors cannot silently appear in a historical v6 checkpoint.
    with pytest.raises(RuntimeError,match='Unexpected key'):
        load_policy(dict(payload,contract=continuous_image_contract('q_reached')))


def test_head_is_not_in_observation_or_planning_condition(payload):
    model=load_policy(payload)
    def forbidden(*args,**kwargs):raise AssertionError('Training-only head called during execution')
    model.visual_pose_head.forward=forbidden
    robot=torch.zeros(178);robot[176]=1
    frame=dict(patches=torch.zeros(3,384,18,32),depth=torch.ones(3,2,18,32),
        valid=torch.ones(3,dtype=torch.bool),time=torch.zeros(3,dtype=torch.float64))
    with torch.no_grad():
        memory=model.observe(robot,torch.tensor(0.,dtype=torch.float64),0,frame)
        condition=model.condition(memory,torch.tensor(0.,dtype=torch.float64))
    assert condition.shape==(1,512) and torch.isfinite(condition).all()
