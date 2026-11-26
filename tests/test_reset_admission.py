from types import SimpleNamespace
import json
import pytest
import torch
from visuotactile.paired_execution import verify_registered_layout


def sample():
    expected=dict(object_xy_m=[-.12,.02],hole_xy_m=[.04,-.04],unshifted_object_xy_m=[-.03,-.01],displacement_xy_m=[-.09,.03],tolerance_m=1e-5)
    case=dict(name='case',spec=dict(seed=1,reset_admission=expected))
    diagnostics=dict(object_pose_w=torch.tensor([[3.88,4.02,0,1,0,0,0]]),hole_pose_w=torch.tensor([[4.04,3.96,0,1,0,0,0]]))
    env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[4.,4.,0.]])),_diagnostics=lambda:diagnostics)
    return env,case,diagnostics


def test_actual_asset_xy_and_signed_displacement(tmp_path):
    env,case,_=sample();verify_registered_layout(env,[case],tmp_path)
    receipt=json.loads((tmp_path/'actual_layout_admission.json').read_text())
    assert receipt['passed']
    assert receipt['rows'][0]['actual_minus_unshifted_xy_m']==pytest.approx([-.09,.03],abs=1e-6)


@pytest.mark.parametrize('asset,axis', [('object_pose_w',0),('hole_pose_w',1)])
def test_metadata_cannot_hide_wrong_physical_reset(tmp_path,asset,axis):
    env,case,diagnostics=sample();diagnostics[asset][0,axis]+=.001
    with pytest.raises(ValueError,match='no student action admitted'):verify_registered_layout(env,[case],tmp_path)
    assert not json.loads((tmp_path/'actual_layout_admission.json').read_text())['passed']
