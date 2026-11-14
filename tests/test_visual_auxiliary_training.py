import copy
import json

import pytest
import torch
from torch import nn

from visuotactile import visual_aux_training as training
from visuotactile.anchor_sampling import planning_anchors, labels as action_labels
from visuotactile.checkpointing import training_state, restore_training
from visuotactile.compact import sha256
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.position_control import continuous_image_contract
from visuotactile.visual_pose_aux import (VisualPoseLabels, VisualPoseHead, ROUTES,
                                        clip_auxiliary_gradients)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(2, 512)
        self.action = nn.Linear(512, 2)
        self.contract = continuous_image_contract('q_reached')


def spec(route='shared_vision'):
    return dict(route=route, head_seed=71, loss_weight=.01)


def context(n=13):
    return VisualPoseLabels(torch.ones(n, 3)*.1,
        torch.tensor([[1., 0., 0., 0., 1., 0.]]).repeat(n, 1),
        torch.ones(n, dtype=torch.bool), torch.arange(n, dtype=torch.float64)/60)


def optimizer(model):
    return torch.optim.AdamW(training.optimizer_groups(model), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)


def test_control_detach_keeps_original_policy_adam_update_exact():
    torch.manual_seed(97)
    original = TinyPolicy()
    control = copy.deepcopy(original)
    rng = torch.get_rng_state().clone()
    training.attach_auxiliary_head(control, spec('detached_readout'))
    assert torch.equal(rng, torch.get_rng_state())
    data = torch.randn(13, 2)
    original_opt = torch.optim.AdamW(original.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    control_opt = optimizer(control)
    baseline = original.action(original.vision(data)).square().mean()
    baseline.backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), 1.)
    visual = control.vision(data)
    pose, _ = training.auxiliary_episode_loss(control, dict(time=context().time_s),
        dict(readout=visual, ticks=torch.arange(13)), context(), spec('detached_readout'))
    total = control.action(visual).square().mean()+pose
    total.backward()
    clip_auxiliary_gradients(control, control.visual_pose_head)
    for name, parameter in original.named_parameters():
        assert torch.equal(parameter.grad, dict(control.named_parameters())[name].grad)
    original_opt.step()
    control_opt.step()
    for name, parameter in original.named_parameters():
        assert torch.equal(parameter, dict(control.named_parameters())[name])


def test_candidate_pose_gradient_reaches_only_vision_and_head():
    model = TinyPolicy()
    training.attach_auxiliary_head(model, spec())
    visual = model.vision(torch.randn(13, 2))
    weighted, parts = training.auxiliary_episode_loss(model, dict(time=context().time_s),
        dict(readout=visual, ticks=torch.arange(13)), context(), spec())
    weighted.backward()
    assert model.vision.weight.grad.abs().sum() > 0
    assert model.visual_pose_head.linear.weight.grad.abs().sum() > 0
    assert model.action.weight.grad is None
    assert parts['valid_ticks'] == 13


def test_single_encoding_keeps_action_masks_and_all_real_prefix_frames(monkeypatch):
    from visuotactile import continuous_vision
    model = TinyPolicy()
    training.attach_auxiliary_head(model, spec())
    n = 25
    record = dict(time=torch.arange(n, dtype=torch.float64)/60, command=torch.ones(n, 29),
        supervision_mask=torch.arange(n)>=13, control_segment=(torch.arange(n)>=13).long())
    original_mask = record['supervision_mask'].clone()
    expected_anchors = planning_anchors(record)
    expected_labels = action_labels(record, expected_anchors, 24)
    calls = []
    def encode(m, rec, *, return_visual):
        calls.append(return_visual)
        return torch.zeros(3, 512), torch.tensor([0, 12, 24]), dict(ticks=torch.arange(n), readout=torch.zeros(n, 512))
    monkeypatch.setattr(continuous_vision, 'encode_dense', encode)
    _, anchors, visual = training.supervised_visual_conditions(model, record)
    assert calls == [True] and torch.equal(anchors, expected_anchors) and len(visual['ticks']) == n
    actual_labels = action_labels(record, anchors, 24)
    assert all(torch.equal(a, b) for a,b in zip(expected_labels, actual_labels))
    assert torch.equal(record['supervision_mask'], original_mask)
    record['supervision_mask'].zero_()
    with pytest.raises(ValueError, match='No legal'):
        training.supervised_visual_conditions(model, record)
    assert not record['supervision_mask'].any()


def test_aux_checkpoint_restores_policy_head_optimizer_ema_rng_and_contract():
    model = TinyPolicy()
    training.attach_auxiliary_head(model, spec())
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMAModel(averaged, power=.75)
    opt = optimizer(model)
    visual = model.vision(torch.randn(13, 2))
    loss, _ = training.auxiliary_episode_loss(model, dict(time=context().time_s),
        dict(readout=visual, ticks=torch.arange(13)), context(), spec())
    (loss+model.action(visual).square().mean()).backward()
    clip_auxiliary_gradients(model, model.visual_pose_head)
    opt.step();ema.step(model)
    schedule = torch.Generator().manual_seed(8)
    noise = torch.Generator().manual_seed(9)
    saved = training_state(model, averaged, opt, ema, schedule, noise, auxiliary_exposure={'test':1})
    restored = TinyPolicy()
    training.attach_auxiliary_head(restored, spec())
    restored_avg = copy.deepcopy(restored).eval().requires_grad_(False)
    restored_ema = EMAModel(restored_avg, power=.75)
    restored_opt = optimizer(restored)
    restored_schedule, restored_noise = torch.Generator(), torch.Generator()
    restore_training(saved, restored, restored_avg, restored_opt, restored_ema, restored_schedule, restored_noise)
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v, restored_avg.state_dict()[k]) for k,v in averaged.state_dict().items())
    assert restored_ema.optimization_step == ema.optimization_step
    assert torch.equal(restored_schedule.get_state(), schedule.get_state())
    assert len(restored_opt.param_groups) == 2 and len(restored_opt.state) == len(opt.state)
    bad = dict(saved, contract=continuous_image_contract('q_reached'))
    with pytest.raises(ValueError, match='contract'):
        restore_training(bad, restored, restored_avg, restored_opt, restored_ema, restored_schedule, restored_noise)
    missing = copy.deepcopy(saved)
    del missing['model']['visual_pose_head.linear.weight']
    with pytest.raises(RuntimeError):
        restore_training(missing, restored, restored_avg, restored_opt, restored_ema, restored_schedule, restored_noise)


def configured_root(tmp_path):
    root = tmp_path
    directory = root/'A'
    (directory/'state_data').mkdir(parents=True)
    original = [f'original{i:02d}' for i in range(30)]
    validation = [f'val{i:02d}' for i in range(6)]
    data = dict(normalization_train_ids=original, splits=dict(train=original+['correction'], validation=validation, test=[]),
        entries={i:dict(family=i) for i in original+validation})
    data['entries']['correction'] = dict(family=original[0])
    (directory/'state_data/data.json').write_text(json.dumps(data))
    (directory/'state_data/statistics.pt').write_bytes(b'explicit fake statistics fixture')
    source = root/'source.pt'
    source.write_bytes(b'explicit fake source fixture')
    continuation = dict(path=str(source), sha256=sha256(source))
    receipt = dict(version=training.CALIBRATION_VERSION, passed=True, roles=['A','B'], routes=list(ROUTES),
        shared_weight=True, head_seed=71, gradient_ratio_target=.1, estimator=training.ESTIMATOR, loss_weight=.01,
        data_sha256={'A':sha256(directory/'state_data/data.json'), 'B':'b'*64}, frontend_sha256={'A':sha256(source), 'B':'c'*64},
        statistics_sha256={'A':sha256(directory/'state_data/statistics.pt'), 'B':'d'*64},
        target_window_contract={'A':None, 'B':None}, subset_ids=training.calibration_ids(data))
    calibration = root/'calibration.json'
    calibration.write_text(json.dumps(receipt))
    selection = dict(spec(), version=training.SPEC_VERSION, calibration=dict(path='calibration.json',sha256=sha256(calibration)))
    (directory/'visual_auxiliary.json').write_text(json.dumps(selection))
    return root, continuation, selection, receipt


def test_spec_requires_matching_real_receipt_and_missing_is_exact_legacy(tmp_path):
    assert training.load_auxiliary_spec(tmp_path, 'A', None) is None
    root, continuation, selection, _ = configured_root(tmp_path)
    admitted = training.load_auxiliary_spec(root, 'A', continuation)
    assert admitted['loss_weight'] == selection['loss_weight']
    assert admitted['spec_sha256'] == sha256(root/'A/visual_auxiliary.json')
    (root/'calibration.json').unlink()
    with pytest.raises(FileNotFoundError): training.load_auxiliary_spec(root, 'A', continuation)


@pytest.mark.parametrize('change', ['unknown_spec', 'no_version', 'bad_route', 'seed', 'weight', 'bad_receipt_hash',
    'receipt_passed', 'receipt_weight', 'receipt_source', 'receipt_stats', 'receipt_subset', 'receipt_seed',
    'receipt_estimator', 'missing_B', 'receipt_window'])
def test_spec_and_calibration_mismatch_cannot_fall_back(tmp_path, change):
    root, continuation, selected, receipt = configured_root(tmp_path)
    if change == 'unknown_spec': selected['automatic'] = True
    elif change == 'no_version': del selected['version']
    elif change == 'bad_route': selected['route'] = 'none'
    elif change == 'seed': selected['head_seed'] = True
    elif change == 'weight': selected['loss_weight'] = float('nan')
    elif change == 'bad_receipt_hash': selected['calibration']['sha256'] = 'a'*64
    else:
        if change == 'receipt_passed': receipt['passed'] = False
        elif change == 'receipt_weight': receipt['loss_weight'] = .1
        elif change == 'receipt_source': receipt['frontend_sha256']['A'] = 'a'*64
        elif change == 'receipt_stats': receipt['statistics_sha256']['A'] = 'a'*64
        elif change == 'receipt_subset': receipt['subset_ids'] = ['val00']
        elif change == 'receipt_seed': receipt['head_seed'] = 1
        elif change == 'missing_B': del receipt['frontend_sha256']['B']
        elif change == 'receipt_window': receipt['target_window_contract']['A'] = 'recovery_onset_commit12_v1'
        else: receipt['estimator'] = 'guessed'
        (root/'calibration.json').write_text(json.dumps(receipt))
        selected['calibration']['sha256'] = sha256(root/'calibration.json')
    (root/'A/visual_auxiliary.json').write_text(json.dumps(selected))
    with pytest.raises(ValueError): training.load_auxiliary_spec(root, 'A', continuation)


def test_exposure_tracks_real_phase_population_and_update_counts():
    ledger = training.empty_exposure()
    report = dict(ticks=5, valid_perception_ticks=4, perception_exposure={
        'student_prefix':dict(observed_ticks=2, perception_ticks=1, invalid_perception_ticks=1),
        'transition':dict(observed_ticks=1, perception_ticks=1, invalid_perception_ticks=0),
        'legal_expert':dict(observed_ticks=2, perception_ticks=2, invalid_perception_ticks=0)})
    for _ in range(3): training.add_exposure(ledger, 'correction', report, ['original'])
    assert ledger['episode_updates'] == 3 and ledger['per_episode'] == {'correction':3}
    assert ledger['populations']['correction'] == dict(episode_updates=3, observed_ticks=15, perception_ticks=12)
    assert ledger['phases']['legal_expert']['perception_ticks'] == 6
    training.validate_exposure(ledger, {'correction':report}, ['correction'], ['original'], completed_epochs=3, updates=3)
    with pytest.raises(ValueError):
        training.validate_exposure(ledger, {'correction':report}, ['correction'], ['original'], completed_epochs=2, updates=3)
    ledger['phases']['legal_expert']['perception_ticks'] += 1
    with pytest.raises(ValueError):
        training.validate_exposure(ledger, {'correction':report}, ['correction'], ['original'], completed_epochs=3, updates=3)


def test_head_cannot_be_attached_twice_or_to_reinterpreted_checkpoint():
    model = TinyPolicy()
    training.attach_auxiliary_head(model, spec())
    with pytest.raises(ValueError): training.attach_auxiliary_head(model, spec())
    model = TinyPolicy();model.contract['version'] = 5
    with pytest.raises(ValueError): training.attach_auxiliary_head(model, spec())


def test_real_aux_context_admission_binds_action_masks_to_raw_authority(monkeypatch):
    from visuotactile.visual_pose_aux import _tensor_sha
    pose = context(4)
    record = dict(time=pose.time_s, supervision_mask=torch.tensor([False,False,True,True]),
        control_segment=torch.tensor([0,1,2,2]))
    data = dict(splits=dict(train=['c'],validation=[],test=[]), entries=dict(c=dict(family='f')))
    report = dict(authority_tensor_sha256=dict(expert_label_valid=_tensor_sha(record['supervision_mask']),
        segment_id=_tensor_sha(record['control_segment'])))
    monkeypatch.setattr(training,'load_pose_labels',lambda *args,**kwargs:(pose,report))
    contexts, _ = training.prepare_pose_contexts(data,{'c':record})
    assert contexts['c'] is pose
    record['supervision_mask'] = torch.ones(4,dtype=torch.bool)
    with pytest.raises(ValueError,match='recorded expert authority'):
        training.prepare_pose_contexts(data,{'c':record})
