import torch
import pytest

from visuotactile.study_metrics import masked_objectives, joint_groups
from visuotactile.study_resources import violation_reasons


def test_group_weight_gradient_and_tail_mask():
    prediction = torch.ones(2,64,29,requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.zeros(2,64,dtype=torch.bool)
    valid[0,:12] = True
    valid[1,:3] = True
    objectives = masked_objectives(prediction,target,valid)
    gradient_a = torch.autograd.grad(objectives['A'],prediction,retain_graph=True)[0]
    gradient_b = torch.autograd.grad(objectives['B'],prediction)[0]
    assert torch.allclose(gradient_a[..., :7].sum(),torch.tensor(14/29))
    assert torch.allclose(gradient_b[..., :7].sum(),torch.tensor(1.))
    assert torch.allclose(gradient_b[..., 7:].sum(),torch.tensor(1.))
    assert not gradient_b[~valid].any()
    changed = prediction.detach().clone()
    changed[~valid] = 1000
    assert masked_objectives(changed,target,valid)['B']==objectives['B']


def test_per_finger_mapping_respects_unequal_counts():
    names = [f'iiwa14_joint_{index}' for index in range(1,8)]
    for finger,count in (('thumb',5),('index',4),('middle',4),('ring',4),('pinky',5)):
        names += [f'left_{finger}_{index}' for index in range(count)]
    groups = joint_groups(names)
    assert [len(groups[name]) for name in ('thumb','index','middle','ring','pinky')]==[5,4,4,4,5]
    assert groups['pinky']==list(range(24,29))
    with pytest.raises(ValueError):
        joint_groups(names[:-1])


def test_resource_violation_uses_whole_card_and_memory():
    row = dict(gpu_util_percent=100,gpu_free_mib=18000,ram_available_bytes=20*1024**3,rolling_60s_percent=39.)
    summary = dict(gpu_time_weighted_mean_percent=37.)
    assert violation_reasons(row,summary)==[]
    row['rolling_60s_percent'] = 50.1
    row['ram_available_bytes'] = 9*1024**3
    assert set(violation_reasons(row,summary))=={'complete60_above_50','ram_below_10gib'}
    row['gpu_util_percent'] = None
    assert 'missing_gpu_telemetry' in violation_reasons(row,summary)
