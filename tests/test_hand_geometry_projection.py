"""Contact-geometry acceptance on immutable recorded train states, CPU only."""
import json
from pathlib import Path

import h5py
import pytest
import torch

from visuotactile.state_image_loss import (GeometryObjective, HandGeometryProjection, HandGeometrySoftLoss,
    geometry_loss_context, geometry_soft_acceptance, calibrate_geometry_update, DIRECT_HAND_GEOMETRY_ID, PALM_LOSS_ID)


@pytest.fixture(scope='module')
def recorded():
    index_path = Path('outputs/visuotactile/run/tight_palm_20261108_01/state_data/data.json')
    if not index_path.exists():
        pytest.skip('Recorded train-only acceptance data is local')
    index = json.loads(index_path.read_text())
    entry = index['entries'][sorted(index['splits']['train'])[0]]
    with h5py.File(entry['path']) as f:
        metadata = json.loads(f.attrs['metadata_json'])
        q = torch.from_numpy(f['episodes'][entry['episode']]['transition/diagnostics_after/joint_position_rad'][:]).double()
    inventory = json.loads((Path(entry['path']).parent.parent/'initial_collision.json').read_text())
    objective = HandGeometryProjection(metadata, inventory)
    return metadata, inventory, objective, q, geometry_loss_context(entry)


def test_recorded_alignment_and_contact_stages(recorded):
    _, _, _, q, context = recorded
    assert len(q) == len(context['object_pose_base'])
    assert set(context['stage'].tolist()) == {0, 1, 2}
    assert not context['contact_mask'][context['stage'] != 1].any()
    assert not context['terminal'][:-1].any()


def test_exact_collision_witness_gradient_matches_finite_difference(recorded):
    _, _, objective, q, context = recorded
    q, pose = q[150:151].clone().requires_grad_(), context['object_pose_base'][150:151].double()
    distance, jacobian = objective.geometry.distances(q, pose, jacobian=True)
    automatic = torch.autograd.grad(distance.sum(), q)[0]
    assert automatic[:, :7].abs().sum() == 0
    assert torch.allclose(automatic[:, 7:], jacobian.sum(1), atol=1e-10)
    for joint in (7, 9, 12, 14, 17, 21, 25):
        perturbation = torch.zeros_like(q); perturbation[:, joint] = 1e-5
        finite = (objective.geometry.distances(q+perturbation, pose)-
                  objective.geometry.distances(q-perturbation, pose))/2e-5
        assert torch.allclose(finite, jacobian[:, :, joint-7], atol=1e-6, rtol=1e-4)


def test_projection_repairs_penetration_preserves_contacts_and_arm(recorded):
    _, _, objective, q, context = recorded
    indices = torch.tensor([0, 150, 250])
    predicted = q[indices].clone(); predicted[:, 7:] += .08
    predicted.requires_grad_()
    loss, report = objective.loss(predicted, **{k: context[k][indices] for k in ('object_pose_base', 'contact_mask', 'stage')})
    assert report['feasible'] == 3
    assert report['rows'][1]['before_min_distance_m'] < -.004
    assert report['rows'][2]['before_min_distance_m'] < -.004
    for row in report['rows']:
        assert row['after_min_distance_m'] >= -.0010001
        assert row['maximum_hand_change_rad'] <= .1500001
        for required, distance in zip(row['required_contact_mask'], row['after_fingertip_distance_m']):
            assert not required or distance <= .0020001
    loss.backward()
    assert predicted.grad[:, :7].abs().sum() == 0
    assert torch.isfinite(predicted.grad).all() and predicted.grad[:, 7:].norm() > 0


def test_release_and_approach_do_not_attract_unrequired_fingers(recorded):
    _, _, objective, q, context = recorded
    predicted = q[:1].clamp(objective.limits[:, 0], objective.limits[:, 1])
    pose = context['object_pose_base'][:1].double().clone(); pose[:, :3] += 2.
    no_contacts = torch.zeros(1, 5, dtype=torch.bool)
    for phase in (0, 2):
        target, report = objective.project(predicted, pose, no_contacts, torch.tensor([phase]))
        assert report['feasible'] == 1
        assert torch.equal(predicted, target)


def test_infeasible_projection_is_reason_coded_and_never_fake_target(recorded):
    _, _, objective, q, context = recorded
    predicted = q[:1].clone(); predicted[:, 7] = 100.; predicted.requires_grad_()
    target, report = objective.project(predicted, **{k: context[k][:1] for k in ('object_pose_base', 'contact_mask', 'stage')})
    assert report['feasible'] == 0
    assert report['rows'][0]['reason'] == 'joint_limits_incompatible_with_locality'
    assert torch.equal(target, predicted)
    loss, _ = objective.loss(predicted, **{k: context[k][:1] for k in ('object_pose_base', 'contact_mask', 'stage')})
    loss.backward()
    assert predicted.grad.abs().sum() == 0


def test_geometry_training_requires_live_identity_and_assets(recorded):
    metadata, inventory, _, _, _ = recorded
    objective = GeometryObjective(metadata)
    with pytest.raises(ValueError, match='Live collision parity'):
        objective.enable_hand_geometry(inventory, dict(live_collision_parity=False))
    with pytest.raises(ValueError, match='identity'):
        objective.enable_hand_geometry(inventory, dict(live_collision_parity=True, geometry_identity='fake'))


def test_contact_stage_cannot_be_silently_forced_during_release(recorded):
    _, _, objective, q, context = recorded
    with pytest.raises(ValueError, match='only contact'):
        objective.project(q[:1], context['object_pose_base'][:1], torch.ones(1, 5, dtype=torch.bool), torch.tensor([2]))


def test_live_auxiliary_exception_is_blocked_and_returns_for_caller_cleanup(monkeypatch):
    from types import SimpleNamespace
    from visuotactile import state_image_loss
    env = SimpleNamespace(_sim_step_counter=11, sim=SimpleNamespace(current_time=1.25))
    def fail(_):
        raise KeyError('config')
    monkeypatch.setattr(state_image_loss, '_live_geometry_probe', fail)
    result = state_image_loss.live_geometry_probe(env)
    assert result['status'] == 'BLOCKED' and not result['live_collision_parity']
    assert not result['complete'] and not result['force']['accepted']
    assert result['physics_steps'] == 0 and result['physics_elapsed_s'] == 0
    assert result['exception_class'] == 'KeyError' and 'config' in result['traceback']
    # The caller remains able to write this report and execute normal cleanup.
    json.dumps(result)


def test_projection_does_not_depend_on_outer_autograd_mode(recorded):
    _, _, objective, q, context = recorded
    predicted = q[150:151].clone(); predicted[:, 7:] += .08
    arguments = {k: context[k][150:151] for k in ('object_pose_base', 'contact_mask', 'stage')}
    ordinary, ordinary_report = objective.project(predicted, **arguments)
    with torch.no_grad():
        validation, validation_report = objective.project(predicted, **arguments)
    assert ordinary_report['feasible'] == validation_report['feasible'] == 1
    assert torch.equal(ordinary, validation)


@pytest.fixture(scope='module')
def direct_geometry(recorded):
    metadata, _, _, q, context = recorded
    root = Path('outputs/visuotactile/run/tight_transformer_20261110_01')
    report_path = root/'physics_gate_meshfix/geometry_live.json'
    if not report_path.exists():
        pytest.skip('Accepted same-SDK collision representation is local')
    live = json.loads(report_path.read_text())
    acceptance = geometry_soft_acceptance(root/'configuration/state_data/data.json', live)
    assert acceptance['passed']
    objective = GeometryObjective(metadata, mode=PALM_LOSS_ID)
    objective.enable_hand_geometry(live['inventory'], acceptance, loss_kind=DIRECT_HAND_GEOMETRY_ID)
    objective.weights['hand_geometry'] = acceptance['hand_geometry_weight']
    return objective, q, context, acceptance


def test_direct_geometry_new_scale_contact_effect_and_arm_detach(direct_geometry):
    objective, q, context, acceptance = direct_geometry
    assert acceptance['loss_identity'] == DIRECT_HAND_GEOMETRY_ID
    assert acceptance['effective_geometry_rows'] == 4
    assert acceptance['effective_contact_attraction_rows'] == 2
    assert acceptance['solver_calls'] == 0 and acceptance['dropped'] == 0
    assert 0 < acceptance['hand_geometry_weight'] < .001  # New units, not projection's old rad² weight.
    assert all(row['weighted_gradient_fraction'] <= .050000001 for row in acceptance['rows'])
    assert all(row['arm_geometry_gradient_norm'] == 0 for row in acceptance['rows'])
    contact = [row for row in acceptance['rows'] if row['contact_loss']>0]
    assert all(row['hand_geometry_gradient_norm']>0 and row['descent_contact_loss']<row['contact_loss'] for row in contact)
    predicted = q[183:184].clone(); predicted[:, 7:] -= .08; predicted.requires_grad_()
    arguments = {k:context[k][183:184] for k in ('object_pose_base', 'contact_mask', 'stage')}
    loss, report = objective.hand_geometry.loss(predicted, **arguments)
    gradient = torch.autograd.grad(loss, predicted)[0]
    assert gradient[:, :7].abs().sum()==0
    assert report['rows'][0]['penetration_loss']==0
    assert report['rows'][0]['contact_hand_gradient_norm_per_rad']==pytest.approx(float(gradient[:, 7:].norm()))
    joint = 7+int(gradient[:, 7:].abs().argmax())
    delta = torch.zeros_like(predicted); delta[:, joint] = 1e-5
    high, _ = objective.hand_geometry.loss(predicted+delta, **arguments)
    low, _ = objective.hand_geometry.loss(predicted-delta, **arguments)
    assert torch.allclose((high-low)/2e-5, gradient[0, joint], atol=1e-5, rtol=1e-4)


def test_direct_loss_keeps_difficult_contact_without_local_solver(direct_geometry, monkeypatch):
    import scipy.optimize
    objective, q, context, _ = direct_geometry
    def forbidden(*args, **kwargs):
        raise AssertionError('SLSQP must never run in the direct loss')
    monkeypatch.setattr(scipy.optimize, 'minimize', forbidden)
    predicted = q[183:184].clone(); predicted[:, 7:] -= .5; predicted.requires_grad_()
    loss, report = objective.hand_geometry.loss(predicted,
        **{k:context[k][183:184] for k in ('object_pose_base', 'contact_mask', 'stage')})
    assert report['included']==report['attempted']==1 and report['dropped']==report['solver_calls']==0
    assert report['rows'][0]['contact_loss']>0
    gradient = torch.autograd.grad(loss, predicted)[0]
    assert torch.isfinite(gradient).all() and gradient[:, 7:].norm()>0 and gradient[:, :7].norm()==0


def test_direct_approach_release_no_attraction(direct_geometry):
    objective, q, context, _ = direct_geometry
    pose = context['object_pose_base'][:1].clone(); pose[:, :3] += 2.
    for phase in (0, 2):
        predicted = q[:1].clone().requires_grad_()
        loss, report = objective.hand_geometry.loss(predicted, pose, torch.zeros(1, 5, dtype=torch.bool), torch.tensor([phase]))
        assert loss==0 and report['rows'][0]['contact_loss']==0
        assert torch.autograd.grad(loss, predicted)[0].abs().sum()==0


def test_direct_complete_update_microbatch_sum_and_padding_equivalent(direct_geometry):
    from visuotactile.anchor_sampling import weighted_parts
    objective, q, source, _ = direct_geometry
    command = q[183].expand(3, 24, 29).clone().float()
    prediction = command.clone(); prediction[..., 7:] -= .08; prediction.requires_grad_()
    valid = torch.arange(24)[None]<torch.tensor([24, 5, 17])[:, None]
    context = {k:source[k][183].expand(3, 24, *source[k].shape[1:]).clone() for k in ('object_pose_base', 'contact_mask', 'stage')}
    context['selected_mask'] = torch.zeros(3, 24, dtype=torch.bool)
    context['selected_mask'][0, 3] = True; context['selected_mask'][2, 8] = True
    context['stage'][2] = 0; context['contact_mask'][2] = False
    context['update_selected_count'] = 2
    scale, offset = torch.ones(29), torch.zeros(29)
    full = objective.components(prediction, command, valid, scale, offset, context)['hand_geometry']
    expected = torch.autograd.grad(full, prediction, retain_graph=True)[0]
    accumulated = 0.
    for sl in (slice(0, 1), slice(1, 3)):
        subcontext = {k:v[sl] if torch.is_tensor(v) else v for k,v in context.items()}
        parts = objective.components(prediction[sl], command[sl], valid[sl], scale, offset, subcontext)
        totals = (int(valid.sum()), int((valid[:, 1:]&valid[:, :-1]).sum()), int(valid[:, :12].sum()))
        _, scaled = weighted_parts(parts, valid[sl], totals, {'hand_geometry':1.}, global_hand_geometry=True)
        accumulated += scaled['hand_geometry']
    actual = torch.autograd.grad(accumulated, prediction)[0]
    assert torch.allclose(full, accumulated, atol=1e-7)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
    assert actual[~context['selected_mask']].abs().sum()==0 and actual[..., :7].abs().sum()==0


def test_direct_warmup_and_unselected_microbatch_skip_distance_work(direct_geometry, monkeypatch):
    from visuotactile.anchor_continue import projection_batch
    objective, q, _, _ = direct_geometry
    def forbidden(*args, **kwargs):
        raise AssertionError('Inactive geometry must not evaluate collision distance')
    monkeypatch.setattr(objective.hand_geometry.geometry, 'distances', forbidden)
    assert projection_batch(objective, [('unused', 0)], 'cpu', active=False)=={'active':False}
    command = q[:1, None].expand(1, 24, 29).float()
    prediction = command.clone().requires_grad_()
    valid = torch.ones(1, 24, dtype=torch.bool)
    for context in ({'active':False}, {'selected_mask':torch.zeros_like(valid), 'update_selected_count':2}):
        value = objective.components(prediction, command, valid, torch.ones(29), torch.zeros(29), context)['hand_geometry']
        assert value==0 and objective.last_geometry_metrics['attempted']==0
        assert objective.last_geometry_metrics['elapsed_s']==0
    monkeypatch.setattr(objective, 'gain', 0.)
    value = objective.components(prediction, command, valid, torch.ones(29), torch.zeros(29), {})['hand_geometry']
    assert value==0 and objective.last_geometry_metrics['attempted']==0


def test_complete_update_sampling_stratifies_without_advancing_model_rng(direct_geometry):
    from types import SimpleNamespace
    from visuotactile.anchor_continue import geometry_selection
    _, q, context, _ = direct_geometry
    objective = SimpleNamespace(loss_context={'recorded':context})
    selected = [('recorded',0), ('recorded',len(q)//2), ('recorded',len(q)-1)]
    state = torch.get_rng_state().clone()
    mask, report = geometry_selection(objective, selected, {'recorded':len(q)}, 417)
    repeated, _ = geometry_selection(objective, selected, {'recorded':len(q)}, 417)
    assert torch.equal(torch.get_rng_state(), state) and torch.equal(mask, repeated)
    assert mask.sum()==2 and report['selected_count']==2 and report['selected_contact_count']==1
    assert not mask[:, 12:].any() and not mask[-1, 1:].any()


@pytest.fixture
def complete_update_calibration_progress():
    root = Path('outputs/visuotactile/run/tight_transformer_20261110_02/configuration/geometry_update_calibration')
    if not (root/'training.jsonl').exists():
        pytest.skip('One actual train-only complete-update calibration receipt is local')
    configuration = json.loads((root/'configuration.json').read_text())
    assert configuration['disposable'] is True
    rows = [json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
    assert len(rows)==1
    return dict(rows[0], disposable=configuration['disposable'])


def test_complete_update_scale_corrects_normalization_without_mutating_evidence(direct_geometry, complete_update_calibration_progress):
    _, _, _, initial = direct_geometry
    before = json.dumps(initial, sort_keys=True)
    updated = calibrate_geometry_update(initial, complete_update_calibration_progress)
    calibration = updated['complete_update_calibration']
    assert json.dumps(initial, sort_keys=True)==before
    assert calibration['initial_weighted_gradient_ratio']==pytest.approx(6.12211195, rel=1e-7)
    assert calibration['final_weighted_gradient_ratio']==pytest.approx(.2)
    assert 0 < updated['hand_geometry_weight'] < updated['initial_hand_geometry_weight']
    assert all(row['weighted_gradient_fraction']<=.050000001 for row in updated['rows'])
    assert calibration['final_weighted_loss_fraction']<.001
    with pytest.raises(ValueError, match='once-only'):
        calibrate_geometry_update(updated, complete_update_calibration_progress)


@pytest.mark.parametrize('field,value,match', [
    ('base_full_update_output_hand_gradient_norm', None, 'Finite'),
    ('geometry_full_update_output_hand_gradient_norm', 0., 'Positive'),
    ('disposable', False, 'disposable'),
    ('formal_updates', 1, 'formal updates'),
    ('geometry_solver_calls', 1, 'solver calls')])
def test_complete_update_scale_rejects_missing_or_invalid_measurement(direct_geometry, complete_update_calibration_progress, field, value, match):
    _, _, _, initial = direct_geometry
    complete_update_calibration_progress[field] = value
    with pytest.raises(ValueError, match=match):
        calibrate_geometry_update(initial, complete_update_calibration_progress)


def test_complete_update_scale_rejects_extra_arm_or_incomplete_gradient_sum(direct_geometry, complete_update_calibration_progress):
    _, _, _, initial = direct_geometry
    complete_update_calibration_progress['geometry_probe_gradients'][0]['arm_norm'] = 1e-12
    with pytest.raises(ValueError, match='arm gradient'):
        calibrate_geometry_update(initial, complete_update_calibration_progress)
    complete_update_calibration_progress['geometry_probe_gradients'][0]['arm_norm'] = 0.
    complete_update_calibration_progress['geometry_full_update_output_hand_gradient_norm'] *= 2.
    with pytest.raises(ValueError, match='reconstruct'):
        calibrate_geometry_update(initial, complete_update_calibration_progress)


@pytest.mark.parametrize('status,count,callback,expected', [
    (0, 1, True, 'accepted'), (2, 0, True, 'sdk_nonvalid_result'),
    (0, 0, True, 'unexpected_convex_count'), (0, 2, True, 'unexpected_convex_count'),
    (0, 1, False, 'synchronous_callback_missing')])
def test_cooked_request_preserves_distinct_sdk_outcomes(status, count, callback, expected):
    from types import SimpleNamespace
    from visuotactile.state_image_loss import _request_cooked_convex
    points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    class API:
        def request_convex_collision_representation(self, **arguments):
            assert arguments['stage_id'] == 123 and arguments['collision_prim_id'] == 456
            assert arguments['run_asynchronously'] is False
            if callback:
                arguments['on_result'](status, [SimpleNamespace(vertices=points)]*count)
    report, vertices = _request_cooked_convex(API(), 123, 456, 0)
    assert report['reason'] == expected
    assert report['callback_count'] == int(callback)
    if callback:
        assert report['callbacks'][0]['status_int'] == status
        assert report['callbacks'][0]['hull_count'] == count
    assert (vertices is not None) == (expected == 'accepted')
    json.dumps(report)


def test_cooked_request_retains_callback_and_request_exceptions():
    from types import SimpleNamespace
    from visuotactile.state_image_loss import _request_cooked_convex
    class BrokenPayload:
        def request_convex_collision_representation(self, **arguments):
            arguments['on_result'](0, [SimpleNamespace(vertices=[[float('nan'), 0, 0]]*4)])
    report, vertices = _request_cooked_convex(BrokenPayload(), 1, 2, 0)
    assert vertices is None and report['reason'] == 'callback_payload_exception'
    assert report['callbacks'][0]['exception_class'] == 'ValueError'
    class BrokenRequest:
        def request_convex_collision_representation(self, **_):
            raise RuntimeError('SDK diagnostic failure')
    report, vertices = _request_cooked_convex(BrokenRequest(), 1, 2, 0)
    assert vertices is None and report['reason'] == 'request_exception'
    assert report['exception_message'] == 'SDK diagnostic failure'


def test_imported_collision_xform_resolves_unique_mesh_and_preserves_adapter_input():
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    from visuotactile.state_image_loss import _resolve_collision_mesh, _cooking_mesh_stage
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    body = UsdGeom.Xform.Define(stage, '/Body').GetPrim()
    owner = UsdGeom.Xform.Define(stage, '/Body/Collider')
    owner.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3))
    collision = owner.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collision).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision).CreateApproximationAttr('convexHull')
    collision.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.002)
    collision.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    mesh = UsdGeom.Mesh.Define(stage, '/Body/Collider/mesh')
    mesh.AddTranslateOp().Set(Gf.Vec3d(.01, 0., 0.))
    points = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .02, 0.], [0., 0., .03]], dtype=np.float32)
    mesh.CreatePointsAttr(points.tolist()); mesh.CreateFaceVertexCountsAttr([3]*4)
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    source = stage.GetRootLayer().ExportToString()
    transformed = points.astype(float)+[.11, .2, .3]
    record = dict(path=str(collision.GetPath()), body=str(body.GetPath()),
                  center=(transformed.min(0)+transformed.max(0))/2, half=np.ptp(transformed, axis=0)/2)
    resolved, child, transform, evidence = _resolve_collision_mesh(stage, record)
    assert child == mesh.GetPrim() and not evidence['mesh_has_direct_collision_api']
    assert evidence['inventory_bounds_max_error_m'] < 1e-12
    assert np.allclose(transform[3, :3], [.11, .2, .3])
    copy_stage, proxy, audit = _cooking_mesh_stage(stage, resolved, child)
    assert proxy.IsA(UsdGeom.Mesh) and proxy.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(proxy).GetApproximationAttr().Get() == 'convexHull'
    assert proxy.GetAttribute('physxCollision:contactOffset').Get() == collision.GetAttribute('physxCollision:contactOffset').Get()
    assert audit['source_points_sha256'] == audit['copied_points_sha256']
    assert audit['runtime_cooked_memory_read'] is False and not audit['live_stage_modified']
    assert stage.GetRootLayer().ExportToString() == source
    # Ambiguity blocks before any SDK request rather than choosing one mesh.
    UsdGeom.Mesh.Define(stage, '/Body/Collider/extra')
    with pytest.raises(ValueError, match='exactly one'):
        _resolve_collision_mesh(stage, record)
