"""One command affine for matched residuals, independent Q nominal affine."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch

from visuotactile.compact import sha256
from visuotactile.models.native_dp import LegacyTargetResidual,TargetResidual,load_residual
from visuotactile import native_controller,anchor_continue,state_resources


def statistics():
    return dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),
        action_scale=torch.linspace(.5,2.,29),action_offset=torch.linspace(-.2,.2,29))


@pytest.fixture
def controller_payload(tmp_path,monkeypatch):
    command=statistics();reached=copy.deepcopy(command)
    reached['action_scale']*=.7;reached['action_offset']+=.1
    class FrozenBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for key,value in reached.items():self.register_buffer(key,value.clone())
    monkeypatch.setattr(native_controller,'NativeController',lambda *a,**kw:
        SimpleNamespace(model=FrozenBase(),device=torch.device('cpu')))
    stats=tmp_path/'statistics.pt';torch.save(command,stats)
    base=tmp_path/'base.pt';torch.save({'contract':{'supervision':'q_reached'}},base)
    model=TargetResidual(command,[.1]*29)
    payload=dict(residual_version=model.version,updates=1,model=model.state_dict(),
        contract={'supervision':'q_reached'},configuration=dict(base_checkpoint=str(base),base_sha256=sha256(base),
        residual_statistics_path=str(stats),residual_statistics_sha256=sha256(stats),
        residual_normalization_identity='u_expert_train166_affine_v1'))
    return payload,command,reached


def test_new_controller_preserves_separate_frontend_and_command_affines(controller_payload):
    payload,command,reached=controller_payload
    controller=native_controller.ResidualController(payload,{},device='cpu')
    assert torch.equal(controller.model.action_scale,reached['action_scale'])
    assert torch.equal(controller.residual.action_scale,command['action_scale'])
    assert not torch.equal(controller.model.action_scale,controller.residual.action_scale)
    # A matched command-front and Q-front backend must use identical physical
    # correction and objective for the same raw output and physical operands.
    raw=torch.ones(1,29);base=torch.zeros(1,29);limits=torch.tensor([[-3.,3.]]).expand(29,-1)
    reference=TargetResidual(command,[.1]*29)
    action,delta=controller.residual.compose(base,raw,torch.tensor([False]),limits)
    expected,expected_delta=reference.compose(base,raw,torch.tensor([False]),limits)
    assert torch.equal(action,expected) and torch.equal(delta,expected_delta)
    assert torch.equal(controller.residual.objective(action,base,delta,torch.tensor([True])),
        reference.objective(expected,base,expected_delta,torch.tensor([True])))


@pytest.mark.parametrize('corruption',['identity','artifact','buffer','robot'])
def test_new_controller_rejects_wrong_common_statistics(controller_payload,corruption):
    payload,_,_=controller_payload
    if corruption=='identity':payload['configuration']['residual_normalization_identity']='q_reached_train166_affine_v1'
    elif corruption=='artifact':payload['configuration']['residual_statistics_sha256']='wrong'
    elif corruption=='buffer':payload['model']['action_scale']=payload['model']['action_scale']*.7
    else:payload['model']['robot_mean']=payload['model']['robot_mean']+1
    with pytest.raises(ValueError,match='affine|statistics|buffer'):
        native_controller.ResidualController(payload,{},device='cpu')


def test_legacy_controller_needs_no_new_artifact_and_keeps_old_affine(controller_payload):
    payload,_,reached=controller_payload
    legacy=LegacyTargetResidual(reached,[.1]*29)
    payload.update(residual_version=legacy.version,model=legacy.state_dict())
    payload['configuration']={k:v for k,v in payload['configuration'].items() if not k.startswith('residual_')}
    controller=native_controller.ResidualController(payload,{},device='cpu')
    assert type(controller.residual) is LegacyTargetResidual
    assert torch.equal(controller.model.action_scale,controller.residual.action_scale)
    payload['model']['action_scale']=payload['model']['action_scale']*2
    with pytest.raises(ValueError,match='normalizer'):
        native_controller.ResidualController(payload,{},device='cpu')


def test_selected_real_historical_checkpoint_retains_exact_legacy_tensors():
    path=Path('outputs/visuotactile/run/tight_residual_20261110_01/G11/training/best.pt')
    if not path.exists():pytest.skip('Historical weights stay local')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=load_residual(saved,'cpu')
    assert type(model) is LegacyTargetResidual
    assert saved['updates']==88771
    assert all(torch.equal(tensor,model.state_dict()[name]) for name,tensor in saved['model'].items())


def test_training_setup_uses_common_command_artifact_and_binds_both_identities(tmp_path,monkeypatch):
    root=tmp_path/'Q01';(root/'state_data').mkdir(parents=True);(root/'nominal').mkdir()
    command=statistics();reached=copy.deepcopy(command);reached['action_scale']*=.7
    common=tmp_path/'command/state_data';common.mkdir(parents=True)
    torch.save(command,common/'statistics.pt');torch.save(reached,root/'state_data/statistics.pt')
    data=dict(splits={'train':['t'],'validation':['v']},normalization_identity='q_reached_train166_affine_v1')
    def write(path,value):path.write_text(json.dumps(value))
    write(root/'state_data/data.json',data)
    write(common/'data.json',dict(data,normalization_identity='u_expert_train166_affine_v1'))
    base=tmp_path/'base.pt';base.write_bytes(b'frozen base identity')
    normalization=dict(residual_statistics_sha256=sha256(common/'statistics.pt'),
        residual_normalization_identity='u_expert_train166_affine_v1')
    episodes={}
    for identity in ('t','v'):
        path=root/'nominal'/(identity+'.pt');torch.save({'identity':identity},path)
        episodes[identity]={'sha256':sha256(path)}
    index=dict(normalization,base_sha256=sha256(base),data_sha256=sha256(root/'state_data/data.json'),episodes=episodes)
    write(root/'nominal/index.json',index)
    write(root/'nominal/scales.json',dict(normalization,scales=[.1]*29))
    write(root/'configuration.json',{'residual_pair':dict(version=TargetResidual.version,initialization_seed=1,
        residual_statistics_path=str(common/'statistics.pt'),bases={'Q01':{'checkpoint':str(base),'sha256':sha256(base)}})})
    write(root/'session.json',dict(dino_sha256='dino',resource_root=str(root),training_deadline=1))
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext))
    monkeypatch.setattr(TargetResidual,'cuda',lambda self:self)
    _,_,_,_,model,configuration,_=anchor_continue.residual_setup(root)
    assert torch.equal(model.action_scale,command['action_scale'])
    assert configuration['statistics_sha256']==sha256(root/'state_data/statistics.pt')
    assert configuration['residual_statistics_sha256']==sha256(common/'statistics.pt')
    assert configuration['statistics_sha256']!=configuration['residual_statistics_sha256']
    index['residual_statistics_sha256']=configuration['statistics_sha256']
    write(root/'nominal/index.json',index)
    with pytest.raises(ValueError,match='normalization identity'):anchor_continue.residual_setup(root)
