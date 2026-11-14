from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from tools.tight_fresh_forensics_gate import compare, freeze, run_sequence
from visuotactile.first_plan_forensics import FirstPlanRecorder
from visuotactile.native_controller import NativeController
from test_first_plan_forensics import MockDino, MockPolicy, packet


class CpuLease:
    @contextmanager
    def block(self):
        yield


class MockCompleteController:
    """Real NativeController transport plus a deterministic recurrent mock backend."""
    def __init__(self, recorder=None, *, corrupt_tick=None):
        self.base = NativeController({'updates': 1}, {'joint_limits_rad': [[[-2., 2.]]*29]},
            dino=MockDino(), device='cpu', first_plan_recorder=recorder, scheduler_index_mode='host')
        self.state = None; self.corrupt_tick = corrupt_tick

    def act(self, p, d):
        nominal = self.base.act(p, d)
        if int(p['step_id'][0]) == 0:
            self.state = dict(hidden=torch.zeros(1, 29), ticks=0)
        self.state['hidden'] = self.state['hidden']*.9+nominal*.01
        self.state['ticks'] += 1
        if int(p['step_id'][0]) == self.corrupt_tick:
            self.state['hidden'][0, 0] += .001
        self.visual = self.base.memory.visual.clone()
        self.delta = self.state['hidden']*.01
        self.nominal = nominal; self.previous_nominal = nominal.clone()
        self.command = nominal+self.delta; self.plan = self.base.plan
        self.contact_gate = False; self.residual_diagnostics = {'proposal': self.delta.clone()}
        self.base.command = self.command
        return self.command

    def observe_execution(self, value):
        self.base.observe_execution(value)


def samples(episode=0):
    diagnostic = dict(joint_position_rad=torch.linspace(-.05, .05, 29)[None],
        joint_velocity_rad_s=torch.zeros(1, 29))
    for tick in range(25):
        yield packet(episode, tick), diagnostic, torch.full((1, 29), tick*.0001)


def test_exact_gate_catches_signed_zero_and_dtype_changes():
    positive = freeze(dict(a=torch.tensor([0.]), nested=SimpleNamespace(value=torch.ones(2))))
    negative = freeze(dict(a=torch.tensor([-0.]), nested=SimpleNamespace(value=torch.ones(2))))
    result = compare('signed_zero', positive, negative)
    assert result['bitwise_equal'] is False
    assert result['mismatches'][0]['max_abs_difference'] == 0.
    assert compare('same', positive, positive)['bitwise_equal'] is True
    assert compare('dtype', freeze(torch.ones(2)), freeze(torch.ones(2).double()))['bitwise_equal'] is False
    assert compare('shape', freeze(torch.ones(2)), freeze(torch.ones(1, 2)))['bitwise_equal'] is False


@pytest.mark.parametrize('enabled_first', [False, True])
def test_gate_checks_25_actual_ticks_three_plans_and_first_record_only(tmp_path, monkeypatch, enabled_first):
    import visuotactile.native_controller as module
    monkeypatch.setattr(module, 'load_policy', lambda saved, device: MockPolicy())
    recorder = FirstPlanRecorder(tmp_path/'forensics')
    baseline, recorded = MockCompleteController(), MockCompleteController(recorder)
    before = torch.random.get_rng_state().clone()
    result = run_sequence(baseline, recorded, samples(), lease=CpuLease(), cuda=False,
        enabled_first=enabled_first)
    assert result['passed'] and result['evaluated_ticks'] == 25 and result['plans_compared'] == 3
    assert all(row['bitwise_equal'] for row in result['rows'])
    assert len(baseline.base.model.observed) == len(recorded.base.model.observed) == 25
    assert torch.equal(before, torch.random.get_rng_state())
    assert recorder.summary()['episodes'] == 1
    assert len(list((tmp_path/'forensics').glob('*.json'))) == 1
    assert len(list((tmp_path/'forensics').glob('*.npz'))) == 1
    assert baseline.base.position_previous[0].equal(torch.full((29,), .0024))
    assert baseline.base.position_previous[1].equal(torch.full((29,), .0024))
    receipt = json.loads(next((tmp_path/'forensics').glob('*.json')).read_text())
    assert receipt['step_id'] == 0 and receipt['tensors']['plan']['shape'] == [1, 24, 29]
    # Identical original tensors must also survive an episode reset, without a
    # synthesized time or episode ID replacing the supplied observation.
    second = run_sequence(baseline, recorded, samples(episode=1), lease=CpuLease(), cuda=False)
    assert second['passed'] and recorder.summary()['episodes'] == 2
    assert baseline.base.memory is not recorded.base.memory


def test_difference_fails_admission_and_preserves_first_divergence(tmp_path, monkeypatch):
    import visuotactile.native_controller as module
    monkeypatch.setattr(module, 'load_policy', lambda saved, device: MockPolicy())
    recorder = FirstPlanRecorder(tmp_path/'forensics')
    result = run_sequence(MockCompleteController(), MockCompleteController(recorder, corrupt_tick=5),
        samples(), lease=CpuLease(), cuda=False)
    assert result['passed'] is False and result['evaluated_ticks'] == 6
    last = result['rows'][-1]
    assert last['step_id'] == 5 and not last['bitwise_equal']
    difference = next(c for c in last['checks'] if c['name'] == 'complete_history_plan_and_rng_after_act')
    assert any('backend_state' in x['path'] for x in difference['mismatches'])
    assert recorder.summary()['episodes'] == 1
