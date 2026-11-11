from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visuotactile.state_image_offline import deployment_sample, diagnostic_rows, summarize_diagnostics


@pytest.mark.parametrize('full_horizon',[False,True])
@pytest.mark.parametrize('takeover',[False,True])
def test_deployment_sampler_visits_once_uses_one_plan_rng_and_real_tail(monkeypatch,full_horizon,takeover):
    import visuotactile.state_image_offline as module
    seen, draws = [], []
    class Bank:
        def observe(self, lane, episode, tick, model, actor, robot, geo, stamp):
            seen.append(tick)
            return torch.zeros(1, 512) if tick % 12 == 0 else None
    monkeypatch.setattr(module, 'HistoryBank', Bank)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1));self.horizon=24
        def predict(self, condition, generator, **kwargs):
            assert kwargs == dict(inference_steps=16, clip_sample=False)
            value = torch.randn(1, 24, 29, generator=generator)
            draws.append(value); return value
    def fk(q):
        return torch.zeros(len(q), 6, 3), torch.eye(3).expand(len(q), 6, 3, 3)
    n = 25
    rec = dict(time=torch.arange(n)/60, actor=torch.zeros(n, 141), robot=torch.zeros(n, 178),
        geometry=torch.zeros(n, 85), command=torch.zeros(n, 29), joint_limits=torch.tensor([[-2., 2.]]*29))
    rec['robot'][:, 58:87] = torch.arange(n)[:, None]/100
    if takeover:
        rec['supervision_mask']=torch.arange(n)>=7
        rec['control_segment']=(torch.arange(n)>=7).long()
    sample = deployment_sample(Model(), rec, SimpleNamespace(fk=fk), SimpleNamespace(block=nullcontext), seed=42,full_horizon=full_horizon)
    expected_rng = torch.Generator().manual_seed(42)
    assert seen == list(range(n))
    assert len(draws) == 3 and sample['prediction'].shape == (n, 29)
    for draw in draws: assert torch.equal(draw, torch.randn(1, 24, 29, generator=expected_rng))
    assert torch.equal(sample['prediction'][-1], draws[-1][0, 0].clamp(-2, 2))
    assert torch.equal(sample['hold'][:, 0], torch.tensor([0.]*12+[.12]*12+[.24]))
    # The plan issued by student tick0 does not become an expert label at tick7.
    # History nevertheless consumed every prefix tick exactly once above.
    assert sample['valid'].tolist()==([False]*12+[True]*13 if takeover else [True]*n)
    if full_horizon:
        full=sample['full_horizon']
        assert len(full['prediction'])==24+13+1
        assert full['target_ticks']==list(range(24))+list(range(12,25))+[24]
        assert full['anchor_ticks']==[0]*24+[12]*13+[24]
        assert torch.equal(full['prediction'][:24],draws[0][0].clamp(-2,2))
        assert full['valid'].tolist()==([False]*24+[True]*14 if takeover else [True]*38)


def test_stage_error_uses_real_ticks_and_empty_is_unmeasured():
    n = 3
    sample = dict(truth=torch.zeros(n, 29), prediction=torch.ones(n, 29)*.2,
        hold=torch.tensor([[0.]*29, [.1]*29, [.1]*29]), ee_mm=torch.ones(n)*2,
        finger_mm=torch.ones(n, 5)*3, rotation_deg=torch.ones(n))
    phases = np.array(['approach_no_object_contact', 'contact_low_object', 'contact_low_object'])
    rows = diagnostic_rows(sample, phases, identity='one', split='train')
    summary = summarize_diagnostics(rows)
    select = lambda phase, motion: next(r for r in summary if r['phase'] == phase and r['motion'] == motion and r['group'] == 'arm')
    assert select('all', 'all')['ticks'] == 3
    assert select('all', 'near_static')['ticks'] == 1
    assert select('contact_low_object', 'moving')['rmse_rad'] == pytest.approx(.2)
    assert select('contact_low_object', 'moving')['hold_rmse_rad'] == pytest.approx(.1)
    assert select('second_insertion', 'all')['rmse_rad'] is None
    assert select('second_insertion', 'all')['ticks'] == 0


def test_diagnostic_nan_and_misaligned_events_rejected():
    sample = dict(truth=torch.zeros(1, 29), prediction=torch.full((1, 29), float('nan')),
        hold=torch.zeros(1, 29), ee_mm=torch.zeros(1), finger_mm=torch.zeros(1, 5), rotation_deg=torch.zeros(1))
    with pytest.raises(ValueError, match='lengths'):
        diagnostic_rows(sample, [], identity='one', split='train')
    with pytest.raises(FloatingPointError):
        diagnostic_rows(sample, ['approach_no_object_contact'], identity='one', split='train')
