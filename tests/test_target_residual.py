"""Causal Transformer residual acceptance without simulation or trained weights."""
import copy

import pytest
import torch

from visuotactile.models.native_dp import LegacyTargetResidual, TargetResidual, load_residual
from visuotactile.models.native_dp import PRECLIP_RESIDUAL_LOSS_VERSION


def statistics():
    return dict(robot_mean=torch.zeros(178), robot_std=torch.ones(178),
                action_scale=torch.linspace(.5, 2., 29), action_offset=torch.linspace(-.2, .2, 29))


def observations(steps=37, batch=2):
    generator = torch.Generator().manual_seed(712)
    def rand(*shape):
        return torch.randn(*shape, generator=generator)*.03
    cursor = torch.arange(steps).remainder(12).expand(batch, -1).clone()
    robot = rand(batch, steps, 178)
    robot[..., 176] = 1
    robot[:, 0, 116:177] = 0
    time = (torch.arange(2*steps, dtype=torch.float64)/120).expand(batch, -1).clone()
    inputs = dict(visual=rand(batch, steps, 512), robot=robot,
        tactile=dict(force_w_N=rand(batch, 2*steps, 5, 3),
            force_local_N=rand(batch, 2*steps, 5, 3),
            position_base_m=rand(batch, 2*steps, 5, 3),
            rotation_base=torch.eye(3).expand(batch, 2*steps, 5, 3, 3).clone(), time_s=time),
        tactile_valid=torch.ones(batch, 2*steps, dtype=torch.bool),
        plans=rand(batch, steps, 24, 29), plan_valid=torch.arange(24)[None, None]+cursor[..., None] < 24,
        plan_cursor=cursor, previous_nominal=rand(batch, steps, 29),
        previous_nominal_valid=torch.ones(batch, steps, dtype=torch.bool),
        valid=torch.ones(batch, steps, dtype=torch.bool))
    inputs['previous_nominal_valid'][:, 0] = False
    return inputs


def segment(inputs, start, stop):
    return {k: ({field: value[:, 2*start:2*stop] for field, value in v.items()} if k == 'tactile'
                else v[:, 2*start:2*stop] if k == 'tactile_valid' else v[:, start:stop])
            for k, v in inputs.items()}


@pytest.fixture
def active_model():
    torch.manual_seed(981)
    model = TargetResidual(statistics(), [.1]*29).eval()
    # Initial output is exactly zero; exercise the fusion after output learning.
    with torch.no_grad(): model.policy[-1].weight.normal_(0., .02)
    return model


def test_full_prefix_chunked_and_native_tick_match(active_model):
    inputs = observations()
    with torch.no_grad():
        full, full_state = active_model(inputs)
        chunks, state = [], None
        for start, stop in ((0, 13), (13, 32), (32, 37)):
            raw, state = active_model(segment(inputs, start, stop), state)
            chunks.append(raw)
        ticks, tick_state = [], None
        for tick in range(37):
            raw, tick_state = active_model(segment(inputs, tick, tick+1), tick_state)
            ticks.append(raw)
    torch.testing.assert_close(full, torch.cat(chunks, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(full, torch.cat(ticks, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(full_state['history'], tick_state['history'], atol=2e-6, rtol=2e-5)
    assert full_state['history_valid'].all()


def test_future_feedback_cannot_change_prefix_and_masked_plan_has_no_effect(active_model):
    inputs = observations(steps=9)
    changed = copy.deepcopy(inputs)
    for key in ('visual', 'robot', 'plans', 'previous_nominal'):
        changed[key][:, 5:] += 2
    for key in ('force_w_N', 'force_local_N', 'position_base_m'):
        changed['tactile'][key][:, 10:] += 4
    masked = copy.deepcopy(inputs)
    masked['plans'][~inputs['plan_valid']] = 100
    with torch.no_grad():
        original, _ = active_model(inputs)
        future, _ = active_model(changed)
        no_padding, _ = active_model(masked)
    torch.testing.assert_close(original[:, :5], future[:, :5], atol=1e-7, rtol=1e-6)
    assert (original[:, 5:]-future[:, 5:]).abs().max() > 1e-5
    torch.testing.assert_close(original, no_padding, atol=1e-7, rtol=1e-6)


def test_previous_nominal_and_actual_execution_are_separate_effective_inputs(active_model):
    inputs = observations(steps=4)
    previous = copy.deepcopy(inputs)
    executed = copy.deepcopy(inputs)
    previous['previous_nominal'][:, 2] += .3
    executed['robot'][:, 2, 116:145] += .3
    with torch.no_grad():
        raw, _ = active_model(inputs)
        nominal_change, _ = active_model(previous)
        execution_change, _ = active_model(executed)
    assert (raw[:, 2]-nominal_change[:, 2]).norm() > 1e-5
    assert (raw[:, 2]-execution_change[:, 2]).norm() > 1e-5
    assert (nominal_change[:, 2]-execution_change[:, 2]).norm() > 1e-5
    # A missing previous nominal is masked independently of executed feedback.
    previous = copy.deepcopy(inputs)
    previous['previous_nominal'][:, 0] = 1000
    with torch.no_grad(): ignored, _ = active_model(previous)
    torch.testing.assert_close(raw, ignored)


def test_all_fusion_branches_have_real_gradients(active_model):
    inputs = observations(steps=5)
    raw, _ = active_model(inputs)
    raw.square().mean().backward()
    for name in ('visual', 'tactile', 'motion', 'feedback', 'memory', 'intent', 'transformer'):
        grad = sum(float(p.grad.abs().sum()) for p in getattr(active_model, name).parameters() if p.grad is not None)
        assert grad > 0, name
    with torch.no_grad():
        masked, _ = active_model(inputs, mask_force=True)
    assert (raw.detach()-masked).abs().max() > 1e-6


def test_padding_does_not_advance_recurrence(active_model):
    inputs = observations(steps=6, batch=1)
    inputs['valid'][:, 3:] = False
    inputs['plan_valid'][:, 3:] = False
    with torch.no_grad():
        padded, state = active_model(inputs)
        actual, expected = active_model(segment(inputs, 0, 3))
    torch.testing.assert_close(padded[:, :3], actual)
    assert torch.count_nonzero(padded[:, 3:]) == 0
    for key in ('visual', 'motion', 'memory', 'history', 'history_valid'):
        torch.testing.assert_close(state[key], expected[key])
    torch.testing.assert_close(state['tactile']['hidden'], expected['tactile']['hidden'])


def test_noncontact_correction_and_full_tick_objective_have_gradients():
    model = TargetResidual(statistics(), [.1]*29)
    base = torch.zeros(1, 3, 29)
    raw = torch.full_like(base, .5, requires_grad=True)
    limits = torch.tensor([[-1., 1.]]).repeat(29, 1)
    action, delta = model.compose(base, raw, torch.zeros(1, 3, dtype=torch.bool), limits)
    assert action.abs().sum() > 0
    valid = torch.tensor([[True, True, False]])
    model.objective(action, base, delta, valid).backward()
    assert raw.grad[:, :2].abs().sum() > 0
    assert raw.grad[:, 2:].abs().sum() == 0
    zero, _ = model.compose(base, raw.detach()*0, torch.ones_like(valid), limits)
    assert torch.equal(zero, base)


@pytest.mark.parametrize('frontend_supervision', ['q_reached', 'u_expert'])
@pytest.mark.parametrize('location', ['inside', 'reachable_outside', 'joint_outside', 'boundary_near'])
def test_preclip_loss_preserves_execution_and_recovers_only_main_fit_gradient(frontend_supervision, location):
    from visuotactile.position_control import (REACHABLE_POSITION_EXECUTION,
        project_processed_position_targets, reachable_position_bounds)
    adapter = REACHABLE_POSITION_EXECUTION
    model = TargetResidual(statistics(), [.2]*7+[.9]*22, execution_adapter=adapter,
        loss_contract=dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=1.))
    previous = torch.zeros(1, 1, 29)
    limits = torch.tensor([[-1., 1.]]).expand(29, 2)
    bounds = reachable_position_bounds(previous, limits, adapter)
    sign = 1 if frontend_supervision == 'q_reached' else -1
    # Both frontend meanings end in the same *expert command* supervision.
    # A's nominal configuration is deliberately not substituted for the label.
    target = previous.clone()
    u_pre = (dict(inside=bounds[..., 1]*.25, reachable_outside=bounds[..., 1]+.1,
        joint_outside=torch.ones_like(previous)*2., boundary_near=bounds[..., 1]+2e-6)[location])*sign
    raw = torch.full_like(previous, .13, requires_grad=True)
    base = u_pre-model.scales*raw.detach().tanh()/model.action_scale
    chain = model.compose_chain(base, raw, torch.ones(1, 1, dtype=torch.bool), limits, previous=previous)
    # Reconstruct the pre-change execution formula verbatim, independently of
    # compose_chain. New supervision must not relax either hard constraint.
    old_delta = model.scales*raw.tanh()
    old_joint = torch.maximum(torch.minimum(base+old_delta/model.action_scale, limits[..., 1]), limits[..., 0])
    old_action = project_processed_position_targets(old_joint, previous, limits, adapter)
    assert torch.equal(chain['u_exec'], old_action)
    assert torch.equal(chain['u_joint'], old_joint)
    assert torch.equal(chain['delta'], old_delta)
    composed, delta = model.compose(base, raw, torch.ones(1, 1, dtype=torch.bool), limits, previous=previous)
    assert torch.equal(composed, old_action) and torch.equal(delta, old_delta)
    terms = model.objective_components(chain['u_exec'], target, chain['delta'], torch.ones(1, 1),
        u_pre=chain['u_pre'])
    exec_gradient = torch.autograd.grad(terms['exec_fit'], chain['u_pre'], retain_graph=True)[0]
    pre_gradient = torch.autograd.grad(terms['pre_fit'], chain['u_pre'], retain_graph=True)[0]
    if location == 'inside':
        assert torch.count_nonzero(exec_gradient) == 29
    else:
        assert torch.count_nonzero(exec_gradient) == 0
        assert torch.count_nonzero((chain['u_exec']-target).abs()>1e-6) == 29
    assert torch.all(pre_gradient*(chain['u_pre']-target)>0)
    # Main supervision is distinguished from the small residual regularizer.
    main = terms['exec_fit']+terms['pre_fit']
    main_gradient = torch.autograd.grad(main, raw, retain_graph=True)[0]
    assert torch.all(main_gradient*(chain['u_pre']-target)>0)
    torch.testing.assert_close(terms['total'], main+terms['regularizer'])


def test_preclip_loss_uses_same_mask_recovery_weights_and_command_affine_for_both_terms():
    model = TargetResidual(statistics(), [.2]*7+[.9]*22,
        loss_contract=dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=1.))
    u_pre = torch.tensor([1., 2., 10.])[None, :, None].expand(1, 3, 29).clone().requires_grad_()
    action = (u_pre*.5)
    target = torch.zeros_like(u_pre)
    weights = torch.tensor([[3., 1., 0.]])  # Early recovery, later expert, student prefix.
    terms = model.objective_components(action, target, torch.zeros_like(u_pre), weights, u_pre=u_pre)
    joint_factor = .5*model.action_scale[:7].square().mean()+.5*model.action_scale[7:].square().mean()
    expected = (3*1.+1*4.)/4*joint_factor
    torch.testing.assert_close(terms['pre_fit'], expected)
    torch.testing.assert_close(terms['exec_fit'], expected*.25)
    terms['total'].backward(retain_graph=True)
    assert torch.count_nonzero(u_pre.grad[:, 2]) == 0
    assert torch.all(u_pre.grad[:, :2]>0)
    none = model.objective(action, target, torch.zeros_like(u_pre), torch.zeros_like(weights), u_pre=u_pre)
    assert none.item() == 0.
    assert torch.count_nonzero(torch.autograd.grad(none, u_pre)[0]) == 0


def test_missing_loss_contract_keeps_legacy_objective_bitwise():
    model = TargetResidual(statistics(), [.1]*29)
    generator = torch.Generator().manual_seed(381)
    action, target, delta = [torch.randn(2, 4, 29, generator=generator) for _ in range(3)]
    weights = torch.tensor([[3., 1., 0., 1.], [1., 1., 0., 0.]])
    error = ((action-target)*model.action_scale).square()
    fit = .5*error[..., :7].mean(-1)+.5*error[..., 7:].mean(-1)
    penalty = .5*delta[..., :7].square().mean(-1)+.5*delta[..., 7:].square().mean(-1)
    expected = ((fit+1e-4*penalty)*weights).sum()/weights.sum().clamp_min(1)
    assert torch.equal(model.objective(action, target, delta, weights), expected)
    assert model.loss_contract is None


def test_preclip_loss_checkpoint_identity_and_legacy_load_are_explicit():
    model = TargetResidual(statistics(), [.1]*29)
    old = dict(updates=1, residual_version=model.version, model=model.state_dict(), configuration={})
    loaded = load_residual(old)
    assert loaded.loss_contract is None
    loss_contract = dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=1.)
    new = dict(old, configuration=dict(loss_contract=loss_contract), residual_loss_version=loss_contract['version'])
    loaded_new = load_residual(new)
    assert loaded_new.loss_contract == loss_contract
    inputs = observations(steps=2, batch=1)
    with torch.no_grad():
        raw_old, _ = loaded(inputs)
        raw_new, _ = loaded_new(inputs)
    assert torch.equal(raw_old, raw_new)
    with pytest.raises(ValueError, match='loss identity'):
        load_residual(dict(new, residual_loss_version='incorrect'))
    with pytest.raises(ValueError, match='loss identity'):
        load_residual(dict(old, configuration=dict(loss_contract=loss_contract)))
    with pytest.raises(ValueError, match='unclipped absolute'):
        loaded_new.objective(torch.zeros(1, 29), torch.zeros(1, 29), torch.zeros(1, 29), torch.ones(1))


@pytest.mark.parametrize('weight', [0, -1, float('inf'), float('nan'), True])
def test_preclip_loss_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match='positive weight'):
        TargetResidual(statistics(), [.1]*29,
            loss_contract=dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=weight))


def test_current_weight_prefix_state_only_matches_full_forward_after_weight_update(active_model):
    from visuotactile.anchor_continue import residual_prefix_state
    inputs = observations(steps=65, batch=2)
    inputs['valid'][1, 50:] = False
    chunks = [segment(inputs, start, stop) for start, stop in ((0, 32), (32, 64), (64, 65))]
    def tensors(value):
        return [tensor for item in value.values() for tensor in tensors(item)] if isinstance(value, dict) else [value]
    old_state = residual_prefix_state(active_model, chunks)
    with torch.no_grad():
        active_model.motion.weight_ih.add_(.03)
        active_model.feedback[0].weight.mul_(1.02)
    refreshed = residual_prefix_state(active_model, chunks)
    full = residual_prefix_state(active_model, chunks, state_only=False)
    assert any(not torch.equal(a,b) for a,b in zip(tensors(old_state),tensors(refreshed)))
    assert all(torch.equal(a,b) for a,b in zip(tensors(full),tensors(refreshed)))
    assert all(not tensor.requires_grad for tensor in tensors(refreshed))
    assert residual_prefix_state(active_model, []) is None
    with torch.no_grad():
        raw, state = active_model(inputs, state_only=True)
        _, expected = active_model(inputs)
    assert raw is None
    assert all(torch.equal(a,b) for a,b in zip(tensors(state), tensors(expected)))


def test_history_checkpoint_identity_is_explicit_and_does_not_change_deployment(active_model):
    from visuotactile.models.native_dp import CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION
    payload = dict(updates=1, residual_version=active_model.version, model=active_model.state_dict(),
        configuration=dict(history_contract=dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION)),
        residual_history_version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION)
    loaded = load_residual(payload)
    assert loaded.history_contract == dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION)
    inputs = observations(steps=4, batch=1)
    with torch.no_grad():
        old, _ = active_model(inputs)
        new, _ = loaded(inputs)
    assert torch.equal(old, new)
    del payload['residual_history_version']
    with pytest.raises(ValueError, match='history identity'):
        load_residual(payload)


def test_controller_keeps_correction_and_memory_across_contact_loss(active_model, monkeypatch):
    """Exercise deployment composition, including known-zero vs missing touch."""
    from types import SimpleNamespace
    from visuotactile.native_controller import ResidualController, InferenceTimings
    from visuotactile import native_data
    data=observations(steps=6,batch=1)
    for key in ('force_w_N','force_local_N'):
        data['tactile'][key].zero_();data['tactile'][key][:,2:4,0,0]=1.
    data['tactile_valid'][:,6:8]=False  # Tick3 unknown, versus known zero at2/4.
    base=SimpleNamespace(limits=torch.tensor([[-2.,2.]]).expand(29,-1),memory=None)
    base_calls=[]
    def nominal(packet,diagnostics):
        tick=int(packet['step_id'][0]);base_calls.append(tick)
        base.last_robot=data['robot'][0,tick];base.plan=torch.zeros(1,24,29)
        return torch.zeros(1,29)
    base.act=nominal
    monkeypatch.setattr(native_data,'residual_visual',lambda *args:data['visual'][:,0])
    controller=ResidualController.__new__(ResidualController)
    controller.base=base;controller.model=SimpleNamespace(contract={});controller.residual=active_model
    controller.device=torch.device('cpu');controller.state=None;controller.episode=None
    controller.previous_nominal=None;controller.timings=InferenceTimings('cpu')
    controller.action_scale_cpu=active_model.action_scale;controller.residual_scales_cpu=active_model.scales
    original=active_model.forward;incoming=[];shapes=[]
    def legal_nonzero(inputs,state):
        incoming.append(state is not None);raw,next_state=original(inputs,state)
        shapes.append((tuple(inputs['plans'].shape),tuple(raw.shape)))
        # Inject only the output, not parameters: trained models need not emit
        # nonzero corrections at every no-contact observation.
        return torch.full_like(raw,.25),next_state
    monkeypatch.setattr(active_model,'forward',legal_nonzero)
    states=[];gates=[]
    for tick in range(6):
        touch={k:v[:,2*tick:2*tick+2] for k,v in data['tactile'].items()}
        packet=dict(episode_id=torch.tensor([5]),step_id=torch.tensor([tick]),
            time_s=torch.tensor([(tick+1)/60],dtype=torch.float64),contact_valid=data['tactile_valid'][:,2*tick],
            contact_time_s=touch['time_s'],contact_w_N=touch['force_w_N'][:,-1],
            contact_substeps_w_N=touch['force_w_N'],contact_substeps_local_N=touch['force_local_N'],
            finger_position_base_m=touch['position_base_m'],finger_rotation_base=touch['rotation_base'])
        command=controller.act(packet)
        expected=active_model.scales*torch.tensor(.25).tanh()/active_model.action_scale
        torch.testing.assert_close(command[0],expected)
        assert int(controller.state['history_valid'].sum())==tick+1
        assert bool(controller.residual_diagnostics['contact_valid'])==(tick!=3)
        states.append(copy.deepcopy(controller.state));gates.append(controller.contact_gate)
    assert base_calls==list(range(6)) and incoming==[False,True,True,True,True,True]
    assert gates==[False,True,False,False,False,False]
    assert shapes==[((1,1,24,29),(1,1,29))]*6
    assert torch.equal(states[3]['tactile']['hidden'],states[2]['tactile']['hidden'])
    assert not torch.equal(states[4]['tactile']['hidden'],states[3]['tactile']['hidden'])
    assert not torch.equal(states[3]['memory'],states[2]['memory'])


def test_architecture_identity_loads_legacy_explicitly_and_rejects_reinterpretation(active_model):
    old = LegacyTargetResidual(statistics(), [.1]*29)
    historical = load_residual(dict(residual_version=old.version, model=old.state_dict()))
    assert type(historical) is LegacyTargetResidual
    restored = load_residual(dict(residual_version=active_model.version, model=active_model.state_dict()))
    assert type(restored) is TargetResidual
    with pytest.raises(RuntimeError):
        load_residual(dict(residual_version=active_model.version, model=old.state_dict()))
    with pytest.raises(ValueError, match='architecture'):
        load_residual(dict(residual_version='unspecified', model=old.state_dict()))
    info = active_model.architecture()
    assert info['layers'] == 2 and info['parameters'] == sum(p.numel() for p in active_model.parameters())


def test_rejects_missing_plan_cursor_and_unknown_privileged_input(active_model):
    inputs = observations(steps=1)
    del inputs['plan_cursor']
    with pytest.raises(ValueError, match='allowlist'): active_model(inputs)
    inputs = observations(steps=1)
    inputs['object_pose'] = torch.zeros(1, 7)
    with pytest.raises(ValueError, match='allowlist'): active_model(inputs)


def test_shared_batch_builder_preserves_nominal_history_and_rejects_mixed_contracts(active_model):
    from visuotactile.native_data import residual_batch
    inputs = observations(steps=4, batch=1)
    record = {k: {field: value[0] for field, value in v.items()} if k == 'tactile' else v[0]
              for k, v in inputs.items() if k != 'valid'}
    padded = residual_batch([record], 0, 6, 'cpu')
    assert padded['valid'].tolist() == [[True, True, True, True, False, False]]
    assert padded['previous_nominal_valid'].tolist() == [[False, True, True, True, False, False]]
    assert padded['plan_cursor'].tolist() == [[0, 1, 2, 3, 0, 0]]
    with torch.no_grad():
        unpadded, _ = active_model(inputs)
        batched, _ = active_model(padded)
    torch.testing.assert_close(unpadded, batched[:, :4])
    legacy = {k: v for k, v in record.items() if k not in ('plan_cursor', 'previous_nominal', 'previous_nominal_valid')}
    assert 'plan_cursor' not in residual_batch([legacy], 0, 4, 'cpu')
    for records in ([legacy, record], [record, legacy]):
        with pytest.raises(ValueError, match='contract'): residual_batch(records, 0, 4, 'cpu')


def test_controller_rejects_future_tactile_before_advancing_base():
    from visuotactile.native_controller import ResidualController
    controller = ResidualController.__new__(ResidualController)
    controller.episode = None
    packet = dict(episode_id=torch.tensor([1]), step_id=torch.tensor([0]),
        time_s=torch.tensor([1.]), contact_time_s=torch.tensor([[.99, 1.01]]),
        contact_valid=torch.tensor([True]))
    # No base is installed: validation must reject before any nominal update.
    with pytest.raises(ValueError, match='Future tactile'): controller.act(packet)


def test_inference_timing_cpu_records_wall_without_cuda(monkeypatch):
    from visuotactile.native_controller import InferenceTimings
    clock = iter([5., 5.012])
    monkeypatch.setattr('visuotactile.native_controller.time.perf_counter', lambda: next(clock))
    def forbidden(*args, **kwargs): raise AssertionError('CPU timer touched CUDA')
    monkeypatch.setattr(torch.cuda, 'Event', forbidden)
    timer = InferenceTimings('cpu')
    timer.finish(timer.begin('residual_forward_compose', 3))
    report = timer.report()
    assert report['pending_cuda_events'] == 0
    assert report['summary']['residual_forward_compose']['wall_s']['mean'] == pytest.approx(.012)
    assert report['summary']['residual_forward_compose']['cuda_stream_s']['count'] == 0


def test_cuda_timing_queries_only_and_reports_unfinished_events(monkeypatch):
    from visuotactile.native_controller import InferenceTimings
    ready = [False]
    class Event:
        def __init__(self, **kwargs): assert kwargs == dict(enable_timing=True)
        def record(self, stream): assert stream == 'existing-stream'
        def query(self): return ready[0]
        def elapsed_time(self, end):
            assert ready[0]
            return 7.5
    def forbidden(*args, **kwargs): raise AssertionError('Diagnostic added a GPU synchronization')
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: 'existing-stream')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    timer = InferenceTimings('cuda:0')
    timer.finish(timer.begin('ddim_plan', 0))
    pending = timer.report()
    assert pending['pending_cuda_events'] == 1
    assert pending['summary']['ddim_plan']['cuda_stream_s']['count'] == 0
    ready[0] = True
    complete = timer.report()
    assert complete['pending_cuda_events'] == 0
    assert complete['summary']['ddim_plan']['cuda_stream_s']['mean'] == pytest.approx(.0075)


def test_residual_diagnostics_separate_recontact_unknown_and_open_loss():
    import numpy as np
    from visuotactile.state_benchmark import residual_episode_metrics
    arrays = dict(residual_contact_gate=np.array([0,1,0,0,1,0,0,1,0], dtype=bool),
        residual_contact_valid=np.array([0,1,1,1,1,1,0,1,1], dtype=bool),
        residual_applied_rad=np.full((9,29), .01),
        residual_proposed_rad=np.full((9,29), .02),
        residual_clip_mask=np.zeros((9,29), dtype=bool),
        residual_tanh_fraction=np.full((9,29), .2))
    arrays['residual_clip_mask'][2,8] = True
    arrays['residual_tanh_fraction'][3,7] = .99
    result = residual_episode_metrics(arrays)
    gaps = result['observed_contact_loss_regain']
    assert gaps['completed_count'] == 1
    assert gaps['completed'][0]['loss_tick'] == 2 and gaps['completed'][0]['regain_tick'] == 4
    assert gaps['completed'][0]['noncontact_seconds'] == pytest.approx(2/60)
    assert gaps['completed'][0]['correction_active_ticks'] == 2
    assert gaps['interrupted_by_invalid_observation'] == [dict(loss_tick=5,invalid_tick=6)]
    assert gaps['open_loss_tick'] == 8 and gaps['open_noncontact_ticks'] == 1
    assert result['unknown_contact_ticks'] == 2
    assert result['joint_limit_clip']['joint_components'] == 1
    assert result['tanh_saturation']['joint_components'] == 1
    assert result['all_ticks']['hand']['applied_rms_rad'] == pytest.approx(.01)
    # Labels/future outcomes do not enter even these read-only event statistics.
    arrays['success'] = np.ones(9, dtype=bool)
    assert residual_episode_metrics(arrays) == result
    from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT, REACHABLE_TAIL_OUTPUT
    arrays['residual_tanh_fraction'][:] = 100.
    for contract in (REACHABLE_CENTERED_OUTPUT, REACHABLE_TAIL_OUTPUT):
        revised = residual_episode_metrics(arrays, output_contract=contract)
        assert revised['tanh_saturation']['applicable'] is False
        assert revised['legacy_normalized_amplitude']['maximum_fraction'] == 100.
        assert revised['joint_limit_clip'] == result['joint_limit_clip']
        assert revised['observed_contact_loss_regain'] == result['observed_contact_loss_regain']
