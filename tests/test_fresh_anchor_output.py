"""Only output coordinates change; real prior targets and shared learning remain causal."""
import copy
from contextlib import nullcontext
import json
import time
from types import SimpleNamespace

import pytest
import torch

from tools.tight_fresh_backend import prepare
from visuotactile import anchor_continue
from visuotactile.compact import sha256
from visuotactile.execution_learning import SPAN_PROJECTION_FY_LOSS
from visuotactile.models.native_dp import (TargetResidual, load_residual,
    CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION)
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION, project_processed_position_targets
from visuotactile.residual_output import (REACHABLE_CENTERED_OUTPUT, REACHABLE_PREVIOUS_TARGET_OUTPUT,
    MATCHED_FINAL_HEAD_RESET, centered_proposal)
from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT
from visuotactile.state_history import trained_continuation_updates


def configuration(output=REACHABLE_CENTERED_OUTPUT):
    return dict(execution_adapter=REACHABLE_POSITION_EXECUTION, loss_contract=SPAN_PROJECTION_FY_LOSS,
        history_contract=dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION), output_contract=output,
        residual_statistics_sha256='statistics')


def model(output=REACHABLE_CENTERED_OUTPUT):
    statistics=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),
                    action_scale=torch.linspace(.5,2.,29),action_offset=torch.zeros(29))
    config=configuration(output)
    return TargetResidual(statistics,[1.]*29,**{k:v for k,v in config.items() if k!='residual_statistics_sha256'})


def payload(instance, *, updates=17, extra=None):
    config=configuration(instance.output_contract)
    config.update(extra or {})
    return dict(residual_version=instance.version, updates=updates, new_optimizer_updates=updates,
        residual_output_version=instance.output_contract['version'],
        residual_loss_version=instance.loss_contract['version'],
        residual_history_version=instance.history_contract['version'],
        model=copy.deepcopy(instance.state_dict()), configuration=config)


def initialized_pair():
    torch.manual_seed(171)
    source=model()
    with torch.no_grad():
        source.policy[-1].weight.normal_(0.,.03)
        source.policy[-1].bias.fill_(.1)
    saved=payload(source)
    candidates=[]
    for output in (REACHABLE_CENTERED_OUTPUT,REACHABLE_PREVIOUS_TARGET_OUTPUT):
        candidate=model(output)
        audit=anchor_continue.load_residual_continuation(candidate,saved,'statistics',head_reset=MATCHED_FINAL_HEAD_RESET)
        candidates.append((candidate,audit))
    return source,saved,candidates


def record(length=5, offset=0):
    generator=torch.Generator().manual_seed(321+offset)
    rand=lambda *shape:torch.randn(*shape,generator=generator)*.03
    robot=rand(length,178)
    robot[:,:29]=-.3                 # Measured q is deliberately different.
    robot[:,58:87]=.2                # Actual pre-action actuator target.
    robot[:,116:145]=.6              # Strict-past issued request is also distinct.
    robot[:,176]=1
    cursor=torch.arange(length).remainder(12)
    mask=torch.arange(length)>=min(2,length-1)
    return dict(robot=robot,visual=rand(length,512),plans=rand(length,24,29),
        plan_valid=cursor[:,None]+torch.arange(24)<24,plan_cursor=cursor,
        previous_nominal=rand(length,29),previous_nominal_valid=torch.ones(length,dtype=torch.bool),
        tactile=dict(force_w_N=rand(length*2,5,3),force_local_N=rand(length*2,5,3),
            position_base_m=rand(length*2,5,3),
            rotation_base=torch.eye(3).expand(length*2,5,3,3).clone(),time_s=torch.arange(length*2)/120),
        tactile_valid=torch.ones(length*2,dtype=torch.bool),base=torch.full((length,29),-.4),
        target=torch.full((length,29),.2003),gate=torch.zeros(length,dtype=torch.bool),
        limits=torch.tensor([-1.,1.]).expand(29,2).clone(),supervision_mask=mask,
        control_segment=mask.long(),time=torch.arange(length)/60)


def test_coordinate_maps_are_distinct_but_share_reachability_and_derivative():
    bounds=torch.tensor([[-.1,.1],[.2,.6]],dtype=torch.double)
    base=torch.tensor([-.5,1.],dtype=torch.double)
    previous=torch.tensor([0.,.4],dtype=torch.double)
    raw=torch.zeros_like(base,requires_grad=True)
    nominal=centered_proposal(base,raw,bounds,REACHABLE_CENTERED_OUTPUT)
    candidate=centered_proposal(base,raw,bounds,REACHABLE_PREVIOUS_TARGET_OUTPUT,previous=previous)
    assert not torch.equal(nominal,candidate)
    assert torch.equal(candidate,previous)
    assert torch.equal(torch.autograd.grad(nominal.sum(),raw)[0],torch.autograd.grad(candidate.sum(),raw)[0])
    target=torch.tensor([.09,.25],dtype=torch.double)
    for output,anchor in ((REACHABLE_CENTERED_OUTPUT,nominal.detach()),(REACHABLE_PREVIOUS_TARGET_OUTPUT,previous)):
        inverse=(target-anchor)/(bounds[:,1]-bounds[:,0])
        torch.testing.assert_close(centered_proposal(base,inverse,bounds,output,previous=previous),target)
    with pytest.raises(ValueError,match='actual pre-action target'):
        centered_proposal(base,raw,bounds,REACHABLE_PREVIOUS_TARGET_OUTPUT)


def test_common_reset_changes_only_final_layer_and_never_mutates_source():
    source,saved,candidates=initialized_pair()
    for candidate,audit in candidates:
        for name,value in candidate.state_dict().items():
            if name in ('policy.2.weight','policy.2.bias'):
                assert torch.count_nonzero(value)==0
                assert torch.count_nonzero(saved['model'][name])>0
            else:
                assert torch.equal(value,saved['model'][name]),name
        assert audit['requires_training_before_deployment'] and not audit['optimizer_state_restored']
        assert audit['new_optimizer_updates']==0 and audit['continuation_head_reset']==MATCHED_FINAL_HEAD_RESET
    assert all(torch.equal(v,saved['model'][k]) for k,v in source.state_dict().items())
    assert all(torch.equal(v,candidates[1][0].state_dict()[k]) for k,v in candidates[0][0].state_dict().items())


def test_new_coordinates_require_explicit_reset_instead_of_silent_legacy_load():
    source,saved,_=initialized_pair()
    with pytest.raises(ValueError,match='explicit common head reset'):
        anchor_continue.load_residual_continuation(model(REACHABLE_PREVIOUS_TARGET_OUTPUT),saved,'statistics')
    with pytest.raises(ValueError,match='head-reset contract'):
        anchor_continue.load_residual_continuation(model(),saved,'statistics',head_reset=dict(MATCHED_FINAL_HEAD_RESET,module='policy'))
    with pytest.raises(ValueError,match='same nominal source'):
        candidate=model(REACHABLE_PREVIOUS_TARGET_OUTPUT)
        candidate.loss_contract=dict(candidate.loss_contract,regularizer_weight=.01)
        anchor_continue.load_residual_continuation(candidate,saved,'statistics',head_reset=MATCHED_FINAL_HEAD_RESET)
    inherited=model()
    anchor_continue.load_residual_continuation(inherited,saved,'statistics')
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in inherited.state_dict().items())


def test_trained_previous_anchor_cannot_silently_revert_but_can_continue_unchanged():
    source=model(REACHABLE_PREVIOUS_TARGET_OUTPUT)
    with torch.no_grad():source.policy[-1].bias.fill_(.2)
    saved=payload(source)
    with pytest.raises(ValueError,match='explicit common head reset'):
        anchor_continue.load_residual_continuation(model(),saved,'statistics')
    continued=model(REACHABLE_PREVIOUS_TARGET_OUTPUT)
    audit=anchor_continue.load_residual_continuation(continued,saved,'statistics')
    assert not audit['output_contract_changed'] and not audit['requires_training_before_deployment']
    assert all(torch.equal(value,source.state_dict()[name]) for name,value in continued.state_dict().items())


def test_reset_update_zero_cannot_borrow_parent_training_to_deploy(tmp_path):
    _,saved,candidates=initialized_pair()
    parent=tmp_path/'parent.pt';torch.save(saved,parent)
    for candidate,audit in candidates:
        config=dict(continuation_from=dict(path=str(parent),sha256=sha256(parent)),
            continuation_head_reset=MATCHED_FINAL_HEAD_RESET,continuation_audit=audit)
        zero=payload(candidate,updates=0,extra=config)
        assert trained_continuation_updates(zero)==0
        with pytest.raises(ValueError,match='requires new training'):
            load_residual(zero)
        trained=dict(zero,updates=1,new_optimizer_updates=1)
        loaded=load_residual(trained)
        assert loaded.output_contract==candidate.output_contract
        broken=copy.deepcopy(trained);broken['configuration']['continuation_audit'].pop('continuation_head_reset')
        with pytest.raises(ValueError,match='initialization audit'):
            load_residual(broken)
    # The established no-reset, identical-parameter update0 convention remains.
    unchanged=dict(saved,updates=0,new_optimizer_updates=0,configuration=dict(saved['configuration'],
        continuation_from=dict(path=str(parent),sha256=sha256(parent)),
        continuation_audit=dict(source_updates=17,source_checkpoint_updates=17)))
    assert trained_continuation_updates(unchanged)==17
    load_residual(unchanged)


def test_real_prediction_anchors_to_actual_target_not_q_or_issued_request():
    candidate=model(REACHABLE_PREVIOUS_TARGET_OUTPUT).eval()
    actual=record()
    prediction=anchor_continue.residual_prediction(candidate,actual,SimpleNamespace(block=nullcontext))
    expected=project_processed_position_targets(actual['robot'][:,58:87],actual['robot'][:,58:87],
        actual['limits'],REACHABLE_POSITION_EXECUTION)
    torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
    assert not torch.allclose(prediction,actual['robot'][:,:29])
    assert not torch.allclose(prediction,actual['robot'][:,116:145])


def test_output_contract_does_not_remove_complete_dp_plans_or_visual_history():
    from visuotactile.native_data import residual_batch
    source,_,candidates=initialized_pair()
    control,candidate=candidates[0][0],candidates[1][0]
    for instance in (control,candidate):
        instance.policy[-1].load_state_dict(source.policy[-1].state_dict())
        instance.eval()
    inputs=residual_batch([record()],0,5,'cpu')
    with torch.no_grad():
        before,state=control(inputs)
        after,other=candidate(inputs)
        changed=copy.deepcopy(inputs);changed['plans']+=.5
        different,_=candidate(changed)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    for key in ('visual','motion','memory','history'):
        torch.testing.assert_close(state[key],other[key],rtol=0,atol=0)
    assert (different-after).abs().max()>1e-6


def test_both_anchors_have_identical_actual_optimizer_exposure(tmp_path,monkeypatch):
    _,_,candidates=initialized_pair()
    ids=['one','two','three'];records={name:record(n,i) for i,(name,n) in enumerate(zip(ids,[7,4,6]))}
    records['validation']=record(5,9)
    data=dict(splits=dict(train=ids,validation=['validation']))
    lease=SimpleNamespace(block=nullcontext,wait_s=0.,work_s=0.)
    monkeypatch.setattr(anchor_continue.signal,'signal',lambda *a:None)
    monkeypatch.setattr(anchor_continue,'residual_curves',lambda *a:None)
    logs=[];checkpoints=[]
    for name,(candidate,audit) in zip(('nominal','previous'),candidates):
        root=tmp_path/name;root.mkdir()
        session=dict(training_deadline=time.time()+3600,base_contracts={name:{}})
        config=dict(maximum_control_tick_epochs=2,minimum_control_tick_epochs=2,
            validation_epoch_interval=1,microbatch=2,initialization_seed=19,learning_rate=1e-4,
            tick_window_contract=TICK_WINDOW_CONTRACT,continuation_head_reset=MATCHED_FINAL_HEAD_RESET,
            continuation_audit=audit,**configuration(candidate.output_contract))
        monkeypatch.setattr(anchor_continue,'residual_setup',lambda *a,**kw:
            (session,config,data,records,candidate,config,lease))
        anchor_continue.residual_train(root)
        logs.append([json.loads(row) for row in (root/'training/loss.jsonl').read_text().splitlines()])
        checkpoints.append(torch.load(root/'training/last.pt',map_location='cpu',weights_only=False))
    fields=('update','epoch','supervised_ticks','contact_ticks','history_ticks','window_chunks','fixed_denominator','group_tail')
    assert [{key:row[key] for key in fields} for row in logs[0]]==[{key:row[key] for key in fields} for row in logs[1]]
    a,b=checkpoints
    assert a['episode_counts']==b['episode_counts']==dict.fromkeys(ids,2)
    assert a['supervised_ticks']==b['supervised_ticks']==22
    assert a['control_ticks']==b['control_ticks']==34
    assert torch.equal(a['schedule_rng'],b['schedule_rng'])
    for checkpoint in checkpoints:
        assert all(int(state['step'])==checkpoint['updates'] for state in checkpoint['optimizer']['state'].values())


def test_prepare_requires_explicit_continuation_before_creating_candidate(tmp_path):
    with pytest.raises(ValueError,match='head reset'):
        prepare(tmp_path,'A',15,continue_rear=True,output_contract=REACHABLE_PREVIOUS_TARGET_OUTPUT)
    with pytest.raises(ValueError,match='backend continuation'):
        prepare(tmp_path,'A',15,continuation_head_reset=MATCHED_FINAL_HEAD_RESET)
    assert not list(tmp_path.iterdir())
