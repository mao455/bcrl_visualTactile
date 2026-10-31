import copy

import pytest
import torch

from visuotactile.action_history import ExecutionHistory
from visuotactile.models.transformer_residual import TransformerResidual, TransformerResidualConfig, combine, residual_loss
from visuotactile.private_history import PrivateProposalHistory, private_action_rows
from visuotactile.transformer_data import remaining_plans, residual_inputs, residual_labels, tactile_features


def record_fixture(length=18):
    torch.manual_seed(471)
    tactile = torch.randn(length,2,5,20)
    tactile[...,-2] = 0.
    tactile[...,-1] = 1.
    return dict(time=torch.arange(length,dtype=torch.float64)/60.,proprio=torch.randn(length,58),
        action=torch.randn(length,61),target=torch.randn(length,29),tracking=torch.randn(length,29),
        tactile=tactile,base=torch.randn(length,29).clamp(-1,1),expert=torch.randn(length,29).clamp(-1,1),
        gate=torch.arange(length)%2==0,plans=torch.randn((length+11)//12,64,29),
        visual=torch.randn((length+11)//12,384))


def test_native_memory_matches_streaming_and_future_is_not_input():
    record = record_fixture()
    model = TransformerResidual().eval()
    torch.nn.init.normal_(model.output[-1].weight,std=.01)
    with torch.no_grad():
        batched,state = model(residual_inputs(record,0,18))
        memory = None
        streamed = []
        for step in range(18):
            anchors = torch.tensor([step]) if step%6==0 else torch.empty(0,dtype=torch.long)
            prediction,memory = model(residual_inputs(record,step,step+1,anchors),memory)
            if len(anchors):
                streamed.append(prediction)
        assert torch.allclose(batched,torch.cat(streamed),atol=2e-6,rtol=1e-5)
        assert torch.allclose(state,memory,atol=2e-6,rtol=1e-5)
        changed = copy.deepcopy(record)
        for name in ('proprio','action','target','tracking','tactile','base','expert'):
            changed[name][7:] += 100
        after,_ = model(residual_inputs(changed,0,18))
        assert torch.equal(batched[:2],after[:2])
        changed = copy.deepcopy(record)
        changed['base'][0] = -changed['base'][0]
        changed_prediction,_ = model(residual_inputs(changed,0,18))
        assert not torch.equal(batched[0],changed_prediction[0])


def test_force_mask_before_memory_and_no_contact_exact_bypass():
    record = record_fixture()
    model = TransformerResidual(TransformerResidualConfig(force_masked=True)).eval()
    torch.nn.init.normal_(model.output[-1].weight,std=.01)
    changed = copy.deepcopy(record)
    changed['tactile'][...,:6] *= 1000
    with torch.no_grad():
        before,state = model(residual_inputs(record,0,18))
        after,other_state = model(residual_inputs(changed,0,18))
    assert torch.equal(before,after)
    assert torch.equal(state,other_state)
    base = torch.tensor([[1.,-1.,.4]])
    assert torch.equal(combine(base,torch.ones_like(base)*20,torch.tensor([False])),base)


def test_raw_delta_loss_retains_gradient_at_execution_clip():
    prediction = torch.ones(1,6,29,requires_grad=True)
    base = torch.ones_like(prediction)
    target = -torch.ones_like(prediction)*2
    valid = torch.ones(1,6,dtype=torch.bool)
    executed = combine(base,prediction,valid)
    assert torch.equal(executed,base)
    residual_loss(prediction,target,valid)['loss'].backward()
    assert (prediction.grad>0).all()
    prediction.grad.zero_()
    valid[:,3:] = False
    residual_loss(prediction,target,valid)['loss'].backward()
    assert not prediction.grad[:,3:].any()


def test_plan_and_target_alignment_at_six_and_twelve_ticks():
    record = record_fixture(14)
    anchors = torch.tensor([0,6,12])
    plans,valid = remaining_plans(record['plans'],anchors)
    assert torch.equal(plans[1,:58],record['plans'][0,6:])
    assert not valid[1,58:].any()
    assert torch.equal(plans[2],record['plans'][1])
    labels = residual_labels(record,anchors)
    assert torch.equal(labels['delta'][1],record['expert'][6:12]-record['base'][6:12])
    assert labels['valid'][2].tolist()==[True,True,False,False,False,False]
    inputs = residual_inputs(record,0,14)
    assert inputs['recent_tactile'][0,:24].eq(0).all()


def test_private_proposals_equal_actual_without_correction_and_ignore_mixed():
    private,actual = PrivateProposalHistory(),ExecutionHistory(1)
    times = torch.arange(8,dtype=torch.float64)/60
    commands = torch.randn(8,29).clamp(-1,1)
    selected = times[(torch.arange(8)-2).clamp_min(0)]
    offline = private_action_rows(commands,times,selected,0,8)
    for step in range(7):
        packet = dict(episode_id=torch.tensor([4]),step_id=torch.tensor([step]),time_s=times[step:step+1])
        context = private.context(packet)
        assert torch.equal(context['previous_issued_command'][0],offline[step,:29])
        transition = dict(episode_id=packet['episode_id'],step_id=packet['step_id'],time_before_s=packet['time_s'],
            time_after_s=times[step+1:step+2],command_clipped_pre_delay=commands[step:step+1],
            control_trace=dict(valid=torch.tensor([True]),selected_command_time_s=selected[step:step+1],
                delayed_command=commands[max(0,step-2):max(0,step-2)+1]))
        actual.observe(transition,commands[step:step+1],packet['episode_id'],packet['step_id'])
        mixed = copy.deepcopy(transition)
        mixed['command_clipped_pre_delay'] = -mixed['command_clipped_pre_delay']
        mixed['control_trace']['delayed_command'] *= -1
        private.observe_proposal(mixed,commands[step:step+1],packet)
        next_packet = dict(episode_id=packet['episode_id'],step_id=torch.tensor([step+1]),time_s=times[step+1:step+2])
        for name,value in actual.context(next_packet).items():
            assert torch.equal(value,private.context(next_packet)[name])
    with pytest.raises(ValueError):
        private.observe(transition,commands[:1],packet['episode_id'],packet['step_id'])


def test_tactile_rejects_future_sensor_sample():
    force = torch.zeros(2,5,3)
    position = force.clone()
    rotation = torch.eye(3).expand(2,5,3,3)
    with pytest.raises(ValueError):
        tactile_features(force,force,position,rotation,torch.tensor([1.,1.1]),torch.tensor(1.),torch.tensor(True))
