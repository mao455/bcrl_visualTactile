import numpy as np
import pytest
import torch

from tools.grasp_diagnosis_worker import OverlapPredictor
from tools.grasp_diagnosis_analysis import window_summary,event_windows
from tools.grasp_diagnosis_precision import current_position


def test_overlap_changes_only_aligned_first_four_and_consumes_same_rng():
    generator=torch.Generator().manual_seed(7);reference=torch.Generator().manual_seed(7)
    def predict():return torch.randn((1,24,29),generator=generator)
    wrapped=OverlapPredictor(predict,4)
    first=wrapped();expected_first=torch.randn((1,24,29),generator=reference)
    assert torch.equal(first,expected_first)
    second=wrapped();fresh=torch.randn((1,24,29),generator=reference)
    weights=torch.tensor([.25,.5,.75,1.])[None,:,None]
    assert torch.equal(second[:,:4],first[:,12:16]*(1-weights)+fresh[:,:4]*weights)
    assert torch.equal(second[:,4:],fresh[:,4:])
    assert torch.equal(wrapped.raw,fresh) and torch.equal(generator.get_state(),reference.get_state())
    with pytest.raises(ValueError):OverlapPredictor(predict,3)


def test_zero_overlap_is_exact_and_common_prefix_excludes_timeout_tail():
    tensor=torch.randn(1,24,29);wrapped=OverlapPredictor(lambda:tensor,0)
    assert wrapped() is tensor and wrapped() is tensor
    values=np.ones(25);values[12]=2;values[13:]=100
    row=window_summary(dict(arm_jump=values,hand_jump=values,arm_tracking=values,hand_tracking=values),13)
    assert row['hand_jump_boundary_rms_rad']==2
    assert row['hand_jump_internal_rms_rad']==1


def test_contact_loss_requires_sustained_contact_and_reports_timing_without_causality():
    n=20;force=np.zeros(n);force[8:12]=1.;force[16]=1.
    data=dict(contact=force,hand_jump=np.zeros(n),hand_tracking=np.ones(n),clipping=np.full(n,np.nan))
    data['hand_jump'][12]=.2
    rows=event_windows(data)
    assert len(rows)==1 and rows[0]['tick']==12
    assert rows[0]['replan_in_window'] and rows[0]['max_hand_jump_rad']==.2
    assert rows[0]['max_clipping_rad'] is None


def test_precision_reads_only_current_robot_feedback_and_preserves_inputs():
    target=torch.linspace(-1,1,29)[None];tracking=torch.ones_like(target)*.125
    packet={'actuator_target_rad':target,'tracking_error_rad':tracking}
    result=current_position(packet)
    assert torch.equal(result,target[0]-tracking[0])
    assert torch.equal(target,torch.linspace(-1,1,29)[None])
    with pytest.raises(KeyError):current_position({'actuator_target_rad':target})
