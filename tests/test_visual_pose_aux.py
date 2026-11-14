import copy
from dataclasses import replace
import hashlib
import json
import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from visuotactile import visual_pose_aux as aux


def tensors(n=4):
    stamp = torch.arange(n, dtype=torch.float64)/60
    object_pose = torch.tensor([[.1, 0., 0., 1., 0., 0., 0.]], dtype=torch.float64).repeat(n, 1)
    palm_pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], dtype=torch.float64).repeat(n, 1)
    return dict(observation_time_s=stamp, time_before_s=stamp.clone(), camera_time_s=stamp[:, None].repeat(1, 3),
                camera_valid=torch.ones(n, 3, dtype=torch.bool), object_pose_w=object_pose, palm_pose_w=palm_pose)


def exact_prediction(labels):
    return torch.cat((labels.position_m, labels.rotation6), -1).float()


def test_current_center_and_rotation_columns_have_explicit_convention():
    values = tensors()
    qz = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    values['object_pose_w'][:] = torch.tensor([2., 2., 3., *qz])
    values['palm_pose_w'][:] = torch.tensor([1., 2., 3., *qz])
    labels = aux.current_pose_labels(**values)
    assert torch.allclose(labels.position_m, torch.tensor([[0., -.975, 0.]]*4, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(labels.rotation6, torch.tensor([[1., 0., 0., 0., 1., 0.]]*4, dtype=torch.float64), atol=1e-12)
    values['object_pose_w'][:, 3:] = torch.tensor([1., 0., 0., 0.])
    labels = aux.current_pose_labels(**values)
    assert torch.allclose(labels.rotation6, torch.tensor([[0., -1., 0., 1., 0., 0.]]*4, dtype=torch.float64), atol=1e-12)
    assert labels.valid.all()


def test_quaternion_sign_and_labels_never_receive_gradients():
    values = tensors()
    values['object_pose_w'].requires_grad_()
    values['palm_pose_w'].requires_grad_()
    baseline = aux.current_pose_labels(**values)
    changed = dict(values, object_pose_w=values['object_pose_w'].detach().clone(),
                   palm_pose_w=values['palm_pose_w'].detach().clone())
    changed['object_pose_w'][:, 3:] *= -1
    changed['palm_pose_w'][:, 3:] *= -1
    repeated = aux.current_pose_labels(**changed)
    assert torch.equal(baseline.position_m, repeated.position_m)
    assert torch.equal(baseline.rotation6, repeated.rotation6)
    assert not baseline.position_m.requires_grad and not baseline.rotation6.requires_grad
    prediction = exact_prediction(baseline).add(.1).requires_grad_()
    aux.pose_loss(prediction, baseline, time_s=baseline.time_s)['loss'].backward()
    assert values['object_pose_w'].grad is None and values['palm_pose_w'].grad is None


@pytest.mark.parametrize('change', ['after_label', 'future_camera', 'stale_camera', 'non60', 'missing_camera_shape', 'bad_camera_dtype'])
def test_clock_and_camera_contract_fails_closed(change):
    values = tensors()
    if change == 'after_label': values['time_before_s'] += 1/60
    elif change == 'future_camera': values['camera_time_s'][2, 0] += 1/60
    elif change == 'stale_camera': values['camera_time_s'][2, 0] -= 1/60
    elif change == 'non60':
        values['observation_time_s'][-1] += .001
        values['time_before_s'][-1] += .001
    elif change == 'missing_camera_shape': values['camera_valid'] = values['camera_valid'][:, :2]
    else: values['camera_valid'] = values['camera_valid'].long()
    with pytest.raises(ValueError): aux.current_pose_labels(**values)


def test_bad_pose_and_missing_camera_mask_only_perception_with_finite_fillers():
    values = tensors(5)
    values['object_pose_w'][1, 0] = float('nan')
    values['palm_pose_w'][2, 3:] = 0
    values['camera_valid'][3, 0] = False
    values['camera_time_s'][3, 0] = float('nan')
    action_mask = torch.tensor([False, False, False, True, True])
    saved = action_mask.clone()
    labels = aux.current_pose_labels(**values)
    assert labels.valid.tolist() == [True, False, False, False, True]
    assert torch.isfinite(labels.position_m).all() and torch.isfinite(labels.rotation6).all()
    assert torch.equal(action_mask, saved)
    assert labels.valid[0] and not action_mask[0]  # Actual student observation, no expert action.


def test_loss_has_fixed_physical_scale_and_counts_perception_not_actions():
    labels = aux.current_pose_labels(**tensors())
    prediction = exact_prediction(labels)
    prediction[:, 0] += .1
    prediction[:, 3:] += 1
    parts = aux.pose_loss(prediction, labels, time_s=labels.time_s)
    assert float(parts['position']) == pytest.approx(1.)
    assert float(parts['orientation6']) == pytest.approx(1.)
    assert float(parts['loss']) == pytest.approx(2.)
    assert parts['valid_ticks'] == 4 and parts['invalid_ticks'] == 0


def test_invalid_and_empty_phases_do_not_leak_nan_or_create_action_labels():
    labels = aux.current_pose_labels(**tensors())
    labels = replace(labels, valid=torch.tensor([True, False, True, False]))
    prediction = exact_prediction(labels)
    prediction[~labels.valid] = float('nan')
    prediction.requires_grad_()
    parts = aux.pose_loss(prediction, labels, time_s=labels.time_s)
    assert parts['valid_ticks'] == 2 and float(parts['loss']) == 0
    parts['loss'].backward()
    assert torch.isfinite(prediction.grad).all() and not prediction.grad.any()
    labels = replace(labels, valid=torch.zeros(4, dtype=torch.bool))
    prediction = torch.full((4, 9), float('nan'), requires_grad=True)
    parts = aux.pose_loss(prediction, labels, time_s=labels.time_s)
    parts['loss'].backward()
    assert float(parts['loss']) == 0 and not prediction.grad.any()
    metrics = aux.pose_metrics(prediction, labels, time_s=labels.time_s)
    assert metrics['position_vector_rmse_mm'] is None and metrics['valid_ticks'] == 0


@pytest.mark.parametrize('change', ['prediction_nan', 'wrong_time', 'label_grad', 'nonorthogonal_target', 'mask_dtype'])
def test_loss_rejects_bad_supervision_context(change):
    labels = aux.current_pose_labels(**tensors())
    prediction = exact_prediction(labels)
    time = labels.time_s
    if change == 'prediction_nan': prediction[0, 0] = float('nan')
    elif change == 'wrong_time': time = time+1/60
    elif change == 'label_grad': labels = replace(labels, position_m=labels.position_m.requires_grad_())
    elif change == 'nonorthogonal_target': labels = replace(labels, rotation6=torch.ones_like(labels.rotation6))
    else: labels = replace(labels, valid=labels.valid.long())
    with pytest.raises(ValueError): aux.pose_loss(prediction, labels, time_s=time)


def test_rotation_diagnostics_report_degenerate_predictions_without_identity_fallback():
    columns = torch.tensor([[1., 0., 0., 0., 1., 0.], [0.]*6, [1., 0., 0., 2., 0., 0.], [float('nan')]*6])
    rotation, valid = aux.rotation6d_to_matrix(columns)
    assert valid.tolist() == [True, False, False, False]
    assert torch.equal(rotation[0], torch.eye(3)) and not rotation[1:].any()
    labels = aux.current_pose_labels(**tensors())
    prediction = exact_prediction(labels)
    prediction[1:3, 3:] = columns[1:3]
    # A known +90deg z orientation produces 90deg diagnostic error.
    prediction[3, 3:] = torch.tensor([0., 1., 0., -1., 0., 0.])
    metrics = aux.pose_metrics(prediction, labels, time_s=labels.time_s)
    assert metrics['valid_orientation_predictions'] == 2 and metrics['degenerate_orientation_predictions'] == 2
    assert metrics['orientation_mean_degrees'] == pytest.approx(45.)
    assert metrics['orientation_rms_degrees'] == pytest.approx(90/math.sqrt(2))


def test_head_initialization_preserves_rng_and_only_accepts_explicit_routes():
    state = torch.random.get_rng_state().clone()
    head = aux.VisualPoseHead(seed=19)
    assert torch.equal(state, torch.random.get_rng_state())
    same = aux.VisualPoseHead(seed=19)
    assert all(torch.equal(a, b) for a,b in zip(head.parameters(), same.parameters()))
    assert sum(p.numel() for p in head.parameters()) == 4617
    with pytest.raises(ValueError): head(torch.zeros(2, 512), route='automatic')
    with pytest.raises(ValueError): head(torch.zeros(2, 690), route='shared_vision')
    with pytest.raises(ValueError): aux.VisualPoseHead(seed=True)
    assert not torch.cuda.is_initialized()


def test_detached_control_and_shared_candidate_have_identical_head_gradients():
    labels = aux.current_pose_labels(**tensors())
    readout = torch.randn(4, 512, requires_grad=True)
    shared = aux.VisualPoseHead(seed=3)
    control = aux.VisualPoseHead(seed=3)
    shared_prediction = shared(readout, route='shared_vision')
    control_prediction = control(readout, route='detached_readout')
    assert torch.equal(shared_prediction, control_prediction)
    control_loss = aux.pose_loss(control_prediction, labels, time_s=labels.time_s)['loss']
    control_grad = torch.autograd.grad(control_loss, (readout, *control.parameters()), allow_unused=True)
    shared_loss = aux.pose_loss(shared_prediction, labels, time_s=labels.time_s)['loss']
    shared_grad = torch.autograd.grad(shared_loss, (readout, *shared.parameters()))
    assert control_grad[0] is None and shared_grad[0].abs().sum() > 0
    assert all(torch.equal(a, b) for a,b in zip(control_grad[1:], shared_grad[1:]))


@pytest.mark.parametrize('registered', [False, True])
def test_separate_clipping_preserves_control_policy_gradient(registered):
    policy = nn.Linear(2, 2)
    reference = copy.deepcopy(policy)
    head = aux.VisualPoseHead(seed=20)
    if registered: policy.add_module('aux_head', head)
    for p,q in zip(policy.linear.parameters() if hasattr(policy, 'linear') else (policy.weight, policy.bias), reference.parameters()):
        p.grad = torch.ones_like(p)*3
        q.grad = torch.ones_like(q)*3
    for p in head.parameters(): p.grad = torch.ones_like(p)*1e6
    original, auxiliary = aux.parameter_partitions(policy, head)
    assert len(original) == 2 and len(auxiliary) == 2
    assert not {id(p) for p in original} & {id(p) for p in auxiliary}
    norm = aux.clip_auxiliary_gradients(policy, head)
    nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    assert float(norm['head_norm']) > 1e6
    assert torch.equal(policy.weight.grad, reference.weight.grad)
    assert torch.equal(policy.bias.grad, reference.bias.grad)


def test_nonfinite_aux_gradient_does_not_partially_clip_policy():
    policy = nn.Linear(2, 2)
    head = aux.VisualPoseHead(seed=20)
    for p in policy.parameters(): p.grad = torch.ones_like(p)*100
    head.linear.weight.grad = torch.full_like(head.linear.weight, float('nan'))
    before = policy.weight.grad.clone()
    with pytest.raises(FloatingPointError): aux.clip_auxiliary_gradients(policy, head)
    assert torch.equal(policy.weight.grad, before)


def test_actual_visual_path_batch_parity_causality_and_all_tick_gradient():
    from visuotactile.models.native_dp import HistoryVision
    from visuotactile.continuous_vision import frame_features, read_batch
    torch.manual_seed(173)
    n = 4
    vision = HistoryVision().train()
    patches = (torch.randn(n, 3, 384, 18, 32)*.01).requires_grad_()
    depth = torch.ones(n, 3, 2, 18, 32)
    values = tensors(n)
    time, ct, valid = values['observation_time_s'], values['camera_time_s'], values['camera_valid']
    tokens, _ = frame_features(SimpleNamespace(vision=vision), patches, depth, time[:, None]-ct, valid)
    indices = torch.arange(n)[:, None]+torch.arange(-2, 1)[None]
    safe = indices.clamp_min(0)
    readout = read_batch(vision, tokens[safe], time[:, None, None]-ct[safe], valid[safe]&(indices>=0)[..., None])
    serial = torch.cat([vision.read(tokens[max(0, t-2):t+1], time[t]-ct[max(0, t-2):t+1], valid[max(0, t-2):t+1]) for t in range(n)])
    assert float((readout-serial).abs().max()) < 2e-5
    head = aux.VisualPoseHead(seed=21)
    labels = aux.current_pose_labels(**values)
    prediction = head(readout, route='shared_vision')
    loss = aux.pose_loss(prediction, labels, time_s=time)['loss']
    gradients = torch.autograd.grad(loss, (patches, *vision.parameters()), retain_graph=True)
    assert (gradients[0].flatten(1).abs().sum(-1) > 0).all()
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients[1:])
    last = torch.autograd.grad(prediction[-1].square().mean(), patches, retain_graph=True)[0]
    assert not last[0].any() and (last[1:].flatten(1).abs().sum(-1) > 0).all()
    first = torch.autograd.grad(prediction[0].square().mean(), patches)[0]
    assert first[0].abs().sum() > 0 and not first[1:].any()
    assert not torch.cuda.is_initialized()


def recorded_source(tmp_path):
    urdf = tmp_path/'object.urdf'
    urdf.write_text('<robot><link name="object"><collision><origin xyz="0 .025 0"/>'
        '<geometry><box size=".02 .05 .02"/></geometry></collision></link></robot>')
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    path = tmp_path/'source.h5'
    values = tensors()
    with h5py.File(path, 'w') as stream:
        stream.attrs['metadata_json'] = json.dumps(dict(initial_family='family', dataset_split='train',
            config=dict(assets=dict(object_urdf=str(urdf)))))
        group = stream.create_group('episodes/case')
        group.attrs['complete'] = True
        paths = dict(observation_time_s='obs/time_s', time_before_s='transition/time_before_s',
            camera_time_s='obs/camera_time_s', camera_valid='obs/camera_valid',
            object_pose_w='transition/diagnostics_before/object_pose_w',
            palm_pose_w='transition/diagnostics_before/palm_body_pose_w')
        for name,value in values.items(): group.create_dataset(paths[name], data=value.numpy())
        group.create_dataset('obs/step_id', data=np.arange(4))
        group.create_dataset('terminated', data=[False, False, False, True])
        group.create_dataset('truncated', data=[False]*4)
        # There are deliberately no action/expert fields: perception never invents them.
    entry = dict(path=str(path), episode='case', family='family', length=4, source_sha256='a'*64)
    return entry, digest


def test_source_loader_reads_current_pose_without_any_action_fields(tmp_path):
    entry, digest = recorded_source(tmp_path)
    context, report = aux.load_pose_labels(entry, split='train', allowed_families={'family'}, expected_urdf_sha256=digest)
    assert context.valid.all() and report['valid_perception_ticks'] == 4
    assert report['source_manifest_sha256'] == 'a'*64 and report['object_urdf_sha256'] == digest
    assert set(report['input_tensor_sha256']) == set(tensors())
    assert report['perception_exposure'] == {'original':dict(observed_ticks=4, perception_ticks=4, invalid_perception_ticks=0)}


def test_recorded_takeover_exposure_keeps_prefix_transition_and_legal_masks_separate(tmp_path):
    entry, digest = recorded_source(tmp_path)
    entry['trajectory_kind'] = 'expert_takeover_v1'
    authority = dict(control_source=np.array([0, 0, 1, 1], dtype=np.int64),
        expert_label_valid=np.array([False, False, False, True]),
        switch_clean=np.array([False, False, True, True]), segment_id=np.array([0, 0, 1, 2], dtype=np.int64))
    with h5py.File(entry['path'], 'r+') as stream:
        for key,value in authority.items(): stream['episodes/case'].create_dataset('transition/dagger/'+key, data=value)
    context, report = aux.load_pose_labels(entry, split='train', allowed_families={'family'}, expected_urdf_sha256=digest)
    assert context.valid.all()
    assert {key:value['perception_ticks'] for key,value in report['perception_exposure'].items()} == dict(student_prefix=2, transition=1, legal_expert=1)
    with h5py.File(entry['path']) as stream:
        assert np.array_equal(stream['episodes/case/transition/dagger/expert_label_valid'][:], authority['expert_label_valid'])


def test_zero_action_labels_can_describe_perception_without_becoming_action_supervision():
    labels = aux.current_pose_labels(**tensors())
    fields = dict(control_source=torch.zeros(4, dtype=torch.long), expert_label_valid=torch.zeros(4, dtype=torch.bool),
        switch_clean=torch.zeros(4, dtype=torch.bool), segment_id=torch.zeros(4, dtype=torch.long))
    report = aux.perception_exposure(labels, **fields)
    assert report['student_prefix']['perception_ticks'] == 4 and report['legal_expert']['perception_ticks'] == 0
    assert not fields['expert_label_valid'].any()


@pytest.mark.parametrize('change', ['student_label', 'unclean_label', 'segment', 'switchback', 'missing'])
def test_exposure_rejects_illegal_actual_authority(change):
    labels = aux.current_pose_labels(**tensors())
    fields = dict(control_source=torch.tensor([0, 0, 1, 1]), expert_label_valid=torch.tensor([False, False, False, True]),
        switch_clean=torch.tensor([False, False, True, True]), segment_id=torch.tensor([0, 0, 1, 2]))
    if change == 'student_label': fields['expert_label_valid'][1] = True
    elif change == 'unclean_label': fields['switch_clean'][-1] = False
    elif change == 'segment': fields['segment_id'][-1] = 1
    elif change == 'switchback': fields['control_source'][-1] = 0
    else: fields['switch_clean'] = None
    with pytest.raises(ValueError): aux.perception_exposure(labels, **fields)


@pytest.mark.parametrize('change', ['test', 'unknown_family', 'entry_split'])
def test_population_admission_precedes_payload_access(tmp_path, monkeypatch, change):
    entry = dict(path='must_not_open.h5', episode='case', family='family', length=4)
    split = 'train'
    families = {'family'}
    if change == 'test': split = 'test'
    elif change == 'unknown_family': families = {'other'}
    else: entry['split'] = 'validation'
    monkeypatch.setattr(h5py, 'File', lambda *a, **k:pytest.fail('Opened inadmissible source'))
    with pytest.raises(ValueError): aux.load_pose_labels(entry, split=split, allowed_families=families)


@pytest.mark.parametrize('change', ['asset', 'metadata_family', 'metadata_split', 'incomplete', 'early_terminal', 'skipped_tick', 'future_label'])
def test_source_loader_rejects_bad_provenance_and_boundaries(tmp_path, change):
    entry, digest = recorded_source(tmp_path)
    with h5py.File(entry['path'], 'r+') as stream:
        group = stream['episodes/case']
        if change == 'asset': digest = 'b'*64
        elif change in ('metadata_family', 'metadata_split'):
            metadata = json.loads(stream.attrs['metadata_json'])
            metadata['initial_family' if change == 'metadata_family' else 'dataset_split'] = 'wrong'
            stream.attrs['metadata_json'] = json.dumps(metadata)
        elif change == 'incomplete': group.attrs['complete'] = False
        elif change == 'early_terminal': group['terminated'][1] = True
        elif change == 'skipped_tick': group['obs/step_id'][2] = 4
        else: group['transition/time_before_s'][:] += 1/60
    with pytest.raises(ValueError):
        aux.load_pose_labels(entry, split='train', allowed_families={'family'}, expected_urdf_sha256=digest)
