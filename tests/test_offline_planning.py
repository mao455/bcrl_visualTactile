import copy
import json
from pathlib import Path
from types import SimpleNamespace, MethodType

import pytest
import torch

from tools.tight_fresh_train import predict_on_native_clock
from tools import tight_offline_planning_gate as gate
from visuotactile.offline_planning import profile, predict_contiguous, contiguous_blocks
from test_selected_frontend_fit import artifacts, write


class NoiseWitness:
    horizon = 24
    training = False

    def __init__(self):
        self.draws = []
        self.rows = []
        self.states = []

    def predict(self, condition, generator, *, inference_steps=16, clip_sample=False, scheduler_index_mode='legacy'):
        assert inference_steps == 16 and clip_sample is False and scheduler_index_mode == 'host'
        assert not torch.is_grad_enabled()
        if isinstance(generator, list):
            assert all(g is generator[0] for g in generator)
            noise = torch.cat([torch.randn((1,24,29), generator=g) for g in generator])
            g = generator[0]
        else:
            assert len(condition) == 1
            noise = torch.randn((1,24,29), generator=generator)
            g = generator
        self.draws.extend(noise.unbind())
        self.rows.extend(condition.detach().unbind())
        self.states.append(g.get_state().clone())
        return noise + condition[:, :1, None]


@pytest.mark.parametrize('batch', [1, 2, 8, 32, 128])
@pytest.mark.parametrize('ticks', [[0], [0,12,24,36,48], [24,36,48,72,84,132]])
def test_exact_serial_noise_and_condition_rows_across_prefix_and_gaps(batch, ticks):
    anchors = torch.tensor(ticks)
    condition = torch.arange(len(ticks)*3).reshape(-1,3).float().requires_grad_()
    serial, candidate = NoiseWitness(), NoiseWitness()
    global_before = torch.get_rng_state().clone()
    reference = predict_on_native_clock(serial, condition, anchors)
    trace = {}
    actual = predict_contiguous(candidate, condition, anchors, configuration=profile(batch), receipt=trace)
    assert torch.equal(reference, actual)
    assert torch.equal(torch.stack(serial.draws), torch.stack(candidate.draws))
    assert torch.equal(serial.states[-1], candidate.states[-1])
    assert torch.equal(torch.stack(candidate.rows), condition.detach())
    assert torch.equal(global_before, torch.get_rng_state())
    assert trace['total_noise_draws'] == ticks[-1]//12+1
    assert trace['skipped_noise_draws'] == ticks[-1]//12+1-len(ticks)
    assert trace['predict_calls'] == len(contiguous_blocks(ticks,batch))
    assert not actual.requires_grad


def test_blocks_do_not_cross_gaps_even_when_under_batch_budget():
    assert contiguous_blocks([24,36,48,72,84,132],2) == [(0,2),(2,3),(3,5),(5,6)]
    assert contiguous_blocks([24,36,48,72,84,132],32) == [(0,3),(3,5),(5,6)]


def test_actual_native_predict_ddim_loop_preserves_noise_with_cpu_independent_denoiser():
    from diffusers import DDPMScheduler
    from visuotactile.models.native_dp import NativePolicy
    def make_model():
        model = SimpleNamespace(horizon=24, training=False,
            noise=DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                clip_sample=False, prediction_type='sample'),
            action_offset=torch.linspace(-.2,.2,29), action_scale=torch.linspace(.5,2,29),
            denoiser=lambda sample,timestep,global_cond: sample*.5+global_cond[:,:1,None])
        model.predict = MethodType(NativePolicy.predict, model)
        return model
    anchors = torch.tensor([24,36,48,84,96,120])
    conditions = torch.arange(18).reshape(6,3).float()/10
    expected = predict_on_native_clock(make_model(),conditions,anchors)
    for batch in (1,8,32):
        actual = predict_contiguous(make_model(),conditions,anchors,configuration=profile(batch))
        assert torch.equal(expected,actual)


@pytest.mark.parametrize('value', [None, True, 0, -1, 129, 1.5, '8'])
def test_explicit_finite_profile_required(value):
    with pytest.raises(ValueError): profile(value)


@pytest.mark.parametrize('ticks', [[], [12,0], [0,0], [-12], [1], [12.0], [True]])
def test_invalid_clock_never_predicts(ticks):
    with pytest.raises(ValueError): contiguous_blocks(ticks,8)


def test_profile_and_training_mode_cannot_silently_change():
    config = profile(8); config['seed'] = 42
    with pytest.raises(ValueError,match='registered'):
        predict_contiguous(NoiseWitness(), torch.zeros(1,3), torch.tensor([0]), configuration=config)
    model = NoiseWitness(); model.training = True
    with pytest.raises(ValueError,match='Eval-only'):
        predict_contiguous(model, torch.zeros(1,3), torch.tensor([0]), configuration=profile(8))


def test_deadline_check_stops_before_next_denoiser_call():
    model = NoiseWitness()
    def stop(): raise TimeoutError('owner deadline')
    with pytest.raises(TimeoutError):
        predict_contiguous(model, torch.zeros(2,3), torch.tensor([24,36]), configuration=profile(8), check=stop)
    assert not model.draws


def test_gate_admits_only_train_and_completed_actual_best(artifacts):
    root, frozen, _ = artifacts
    finished = root/'B/training/finished.json'
    with pytest.raises(ValueError,match='complete'):gate.admit(root,'B',frozen,['new1'])
    write(finished,dict(complete=True,checkpoint_sha256=gate.sha256(root/'B/training/best.pt')))
    receipt = gate.admit(root,'B',frozen,['new1'])
    assert receipt['paths']['checkpoint'].endswith('/best.pt')
    assert receipt['identities'] == ['new1']
    for identity in ['val1','test1']:
        with pytest.raises(ValueError,match='training episodes'):gate.admit(root,'B',frozen,[identity])
    (root/'B/training/best.pt').unlink()
    torch.save({},root/'B/training/last.pt')
    with pytest.raises(FileNotFoundError):gate.admit(root,'B',frozen,['new1'])


def test_gate_fixed_target_tolerance_and_legal_commit_metric_only():
    reference = torch.zeros(2,24,29)
    truth = torch.ones_like(reference)
    valid = torch.zeros(2,24,dtype=torch.bool);valid[0,4:12] = True;valid[1,:3] = True
    scale = torch.linspace(.5,2,29)
    candidate = reference + 1e-5
    row = gate.compare_plans(reference,candidate,reference,candidate,truth,valid,scale)
    assert row['passed'] and row['reference_metrics']['legal_commit_slots'] == 11
    candidate[0,0] = 1e-4  # Invalid target still fails the complete command parity gate.
    row = gate.compare_plans(reference,candidate,reference,candidate,truth,valid,scale)
    assert not row['passed']
    assert row['candidate_metrics'] == gate.error_metrics(reference+1e-5,truth,valid,scale)
    candidate[0,0] = float('nan')
    with pytest.raises(ValueError,match='Finite'):gate.compare_plans(reference,candidate,reference,candidate,truth,valid,scale)


def test_projection_never_touches_heldout_tensor_payload(tmp_path):
    class Poison:
        def values(self):pytest.fail('Held-out payload touched')
    data = dict(splits=dict(train=['x'],validation=['v'],test=['t']),
        entries={i:dict(family=i,split=('train' if i=='x' else 'test')) for i in ['x','v','t']})
    visual = tmp_path/'shard';visual.write_bytes(b'frozen')
    numeric = dict(data_sha256='bound',records=dict(x=dict(robot=torch.arange(6)),v=Poison(),t=Poison()))
    receipt = dict(identities=['x'],sha256=dict(data='bound'),selected_visual_cache={'x':dict(path=str(visual))})
    projected = gate.train_projection(tmp_path,data,numeric,receipt)
    saved = torch.load(Path(projected['root'])/'numeric.pt',weights_only=False)
    assert set(saved['records']) == {'x'}
    assert torch.equal(saved['records']['x']['robot'],numeric['records']['x']['robot'])
    assert (Path(projected['root'])/'perception/x.pt').resolve() == visual


def test_check_only_preserves_error_evidence_and_never_imports_frozen_model(artifacts,tmp_path):
    root, frozen, _ = artifacts
    rr = root/'B'
    write(rr/'training/finished.json',dict(complete=True,checkpoint_sha256=gate.sha256(rr/'training/best.pt')))
    torch.save(dict(data_sha256=gate.sha256(rr/'state_data/data.json'),
        records=dict(new1=dict(robot=torch.arange(6)))),rr/'state_data/numeric.pt')
    out = tmp_path/'gate'
    result = gate.run(root,'B',frozen,out,identities=['new1'],check_only=True)
    assert result['admission_only'] and result['passed'] is None and result['epoch'] == 3
    started = json.loads((out/'started.json').read_text())
    assert started['requested_identities'] == ['new1'] and started['root'] == str(root.resolve())
    assert not torch.cuda.is_initialized()
    with pytest.raises(FileExistsError):gate.run(root,'B',frozen,out,identities=['new1'],check_only=True)
    bad = tmp_path/'failure'
    with pytest.raises(ValueError,match='training episodes'):
        gate.run(root,'B',frozen,bad,identities=['test1'],check_only=True)
    assert json.loads((bad/'error.json').read_text())['automatic_retry'] is False
