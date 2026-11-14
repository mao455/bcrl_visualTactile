import math

import pytest
import torch

from tools.tight_fresh_orientation_probe import relative_columns, decode_rotation, errors, summarize


def pose(quaternion):
    return torch.tensor([[0.,0.,0.,*quaternion]],dtype=torch.float64)


def test_relative_rotation_columns_and_quaternion_sign_invariance():
    identity=pose([1,0,0,0]);quarter=pose([math.sqrt(.5),0,0,math.sqrt(.5)])
    expected=torch.tensor([[0.,1.,0.,-1.,0.,0.]],dtype=torch.float64)
    assert torch.allclose(relative_columns(identity,quarter),expected,atol=1e-12)
    negative=quarter.clone();negative[:,3:]*=-1
    assert torch.equal(relative_columns(identity,quarter),relative_columns(identity,negative))
    own=relative_columns(quarter,quarter)
    assert torch.allclose(own,torch.tensor([[1.,0.,0.,0.,1.,0.]],dtype=torch.float64),atol=1e-12)
    inverse=relative_columns(quarter,identity)
    assert torch.allclose(inverse,torch.tensor([[0.,-1.,0.,1.,0.,0.]],dtype=torch.float64),atol=1e-12)


def test_geodesic_uses_column_order_and_retains_invalid_denominator():
    truth=torch.tensor([[1.,0,0,0,1,0]]*3,dtype=torch.float64)
    prediction=torch.tensor([[0.,1,0,-1,0,0],[0,0,0,0,0,0],[1,0,0,2,0,0]],dtype=torch.float64)
    value=errors(prediction,truth)
    assert value['valid'].tolist()==[True,False,False]
    assert value['angle_rad'][0].item()==pytest.approx(math.pi/2)
    report=summarize({'x':value},{'x':{'all':torch.ones(3,dtype=torch.bool)}},['x'],['all'])['all']
    assert report['frames']==3 and report['valid_rotation_frames']==1 and report['invalid_rotation_frames']==2
    assert report['valid_geodesic_mean_rad']==pytest.approx(math.pi/2)
    assert report['column_mse']==pytest.approx(float((prediction-truth).square().mean()))


def test_decode_recovers_proper_rotation_with_scaled_sheared_columns():
    matrix,valid=decode_rotation(torch.tensor([[0.,2,0,-3,1,0]],dtype=torch.float64))
    assert valid.all()
    assert torch.allclose(matrix.transpose(1,2)@matrix,torch.eye(3,dtype=torch.float64)[None])
    assert torch.det(matrix).item()==pytest.approx(1.)


def test_nonunit_or_nonfinite_label_is_rejected():
    with pytest.raises(ValueError):relative_columns(pose([1,0,0,0]),pose([2,0,0,0]))
    with pytest.raises(ValueError):decode_rotation(torch.full((1,6),float('nan')))
