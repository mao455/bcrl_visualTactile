import copy
import json
from types import SimpleNamespace

import pytest
import torch

from tools.tight_visual_aux_acceptance import (TOLERANCES, calibration_admission,
    check_head_rng, compare_maps, compare_tensors, first_original, register,
    run, same_draws, step_snapshot)
from tools.tight_fresh_train import fresh_objective
from visuotactile.visual_aux_training import CALIBRATION_VERSION, ESTIMATOR, calibration_ids
from visuotactile.visual_pose_aux import ROUTES, VisualPoseHead


def test_fixed_tolerances_and_parameter_identity_rejections():
    assert TOLERANCES == dict(forward=dict(atol=2e-5, rtol=0.),
        gradient=dict(atol=2e-6, rtol=2e-4),
        updated_parameter=dict(atol=2e-6, rtol=2e-6),
        update_delta=dict(atol=2e-7, rtol=2e-4))
    assert compare_tensors(torch.zeros(2), torch.ones(2)*1e-5, 'forward')['max_abs'] > 0
    with pytest.raises(ValueError, match='parity failed'):
        compare_tensors(torch.zeros(2), torch.ones(2)*3e-5, 'forward')
    with pytest.raises(ValueError, match='finite'):
        compare_tensors(torch.ones(1), torch.tensor([float('nan')]), 'forward')
    with pytest.raises(ValueError, match='identities'):
        compare_maps(dict(a=torch.ones(1)), dict(b=torch.ones(1)), 'gradient')
    with pytest.raises(ValueError, match='connectivity'):
        compare_maps(dict(a=torch.ones(1)), dict(a=None), 'gradient')
    with pytest.raises(ValueError, match='shape/dtype'):
        compare_tensors(torch.zeros(1), torch.zeros(2), 'forward')


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = torch.nn.Linear(3, 512)
        self.action = torch.nn.Linear(512, 2)


def test_actual_separate_clip_and_adam_update_match_detached_control():
    torch.manual_seed(91)
    base = TinyPolicy(); control = copy.deepcopy(base)
    control.visual_pose_head = VisualPoseHead(seed=3)
    x = torch.randn(4, 3)
    visual = base.vision(x)
    b = step_snapshot(base, base.action(visual).square().mean(), auxiliary=False, learning_rate=1e-4)
    visual = control.vision(x)
    # A very large auxiliary head gradient would alter joint clipping.
    loss = control.action(visual).square().mean()+1000*control.visual_pose_head(
        visual, route='detached_readout').square().mean()
    c = step_snapshot(control, loss, auxiliary=True, learning_rate=1e-4)
    assert c['norms']['head_norm'] > 1
    assert b['norms']['policy_norm'] == c['norms']['policy_norm']
    for field, kind in [('gradients','gradient'), ('updated','updated_parameter'), ('delta','update_delta')]:
        assert compare_maps(b[field], c[field], kind)['max_abs'] == 0
    assert any(value.abs().sum() > 0 for value in b['delta'].values())
    bad = copy.deepcopy(c['delta']); bad['action.bias'] += .001
    with pytest.raises(ValueError, match='update_delta parity'):
        compare_maps(b['delta'], bad, 'update_delta')


def test_rng_probe_and_diffusion_draw_order_are_exact_cpu():
    torch.manual_seed(83); before = torch.get_rng_state().clone()
    report = check_head_rng(7, torch.device('cpu'))
    assert torch.equal(before, torch.get_rng_state())
    assert report['cpu_rng_unchanged'] is True and report['cuda_devices_checked'] == 0
    model = SimpleNamespace(action_scale=torch.ones(29), action_offset=torch.zeros(29))
    truth = torch.zeros(2,24,29); noise, ticks = same_draws(truth, model)
    generator = torch.Generator().manual_seed(2026111403)
    assert torch.equal(noise, torch.randn(truth.shape, generator=generator))
    assert torch.equal(ticks, torch.randint(100, (2,), generator=generator))
    assert torch.equal(before, torch.get_rng_state())


def test_registration_preserves_existing_attempt_and_failure_before_cuda(tmp_path):
    output = tmp_path/'acceptance.json'
    started = register(output, dict(version='example'))
    assert json.loads(started.read_text())['version'] == 'example'
    with pytest.raises(FileExistsError, match='Preserve'):
        register(output, dict(version='overwrite'))
    root = tmp_path/'root'; root.mkdir()
    (root/'session.json').write_text(json.dumps(dict(deadline=0)))
    failed = tmp_path/'failed.json'
    with pytest.raises(TimeoutError, match='expired'):
        run(root, tmp_path/'missing_calibration.json', failed)
    receipt = json.loads(failed.read_text())
    assert receipt['passed'] is False and receipt['optimizer_updates'] == 0
    assert receipt['formal_training_updates'] == 0 and receipt['device'] == 'cuda:0'
    original = failed.read_bytes()
    with pytest.raises(FileExistsError):
        run(root, tmp_path/'missing_calibration.json', failed)
    assert failed.read_bytes() == original


def admission_fixture():
    ids = [f'e{i}' for i in range(30)]
    data = dict(normalization_train_ids=ids, splits=dict(train=['c']+ids, validation=['v'], test=[]),
        entries={identity:dict(family=f'f{j:02d}') for j, identity in enumerate(ids)})
    data['entries'].update(c=dict(family='f00', trajectory_kind='expert_takeover_v1'), v=dict(family='heldout'))
    bindings = {r:dict(files=dict(data=r+'data', frontend=r+'frontend', statistics=r+'stats', numeric=r+'numeric'),
        target_window_contract='window') for r in ('A','B')}
    source_sha = {'tools/example.py':'source'}
    receipt = dict(version=CALIBRATION_VERSION, passed=True, roles=['A','B'], routes=list(ROUTES),
        shared_weight=True, gradient_ratio_target=.1, estimator=ESTIMATOR,
        subset_ids=calibration_ids(data), head_seed=7, loss_weight=.5,
        source_sha256=source_sha, optimizer_updates=0, source_binding=copy.deepcopy(bindings),
        target_window_contract={r:'window' for r in ('A','B')},
        **{field:{r:bindings[r]['files'][name] for r in ('A','B')}
            for field,name in [('data_sha256','data'), ('frontend_sha256','frontend'), ('statistics_sha256','statistics')]})
    return receipt, dict(A=data,B=copy.deepcopy(data)), bindings, source_sha


def test_first_original_uses_actual_order_and_calibration_is_complete():
    receipt, data, bindings, source = admission_fixture()
    assert first_original(data['A']) == 'e0'
    calibration_admission(receipt,data,bindings,source)
    changed = copy.deepcopy(receipt); changed['statistics_sha256'].pop('B')
    with pytest.raises(ValueError, match='Complete'):
        calibration_admission(changed,data,bindings,source)
    changed = copy.deepcopy(receipt); changed['source_binding']['A']['files']['numeric']='changed'
    with pytest.raises(ValueError, match='numeric/cache'):
        calibration_admission(changed,data,bindings,source)
    with pytest.raises(ValueError, match='frozen-code'):
        calibration_admission(receipt,data,bindings,{'other.py':'other'})


def test_shared_fresh_objective_retains_role_b_weights(tmp_path, monkeypatch):
    import visuotactile.state_image_loss as module
    (tmp_path/'metadata.json').write_text('{"source":"metadata"}')
    class Objective:
        def __init__(self, meta, mode):
            self.meta, self.mode = meta, mode
    monkeypatch.setattr(module,'GeometryObjective',Objective)
    objective, geometry, proof = fresh_objective(tmp_path,'B',{}, {})
    assert objective.meta == dict(source='metadata')
    assert objective.mode == module.PALM_LOSS_ID
    assert objective.weights == dict(position=.3342396121814294,rotation=.029946394754032265)
    assert geometry is None and proof is None
