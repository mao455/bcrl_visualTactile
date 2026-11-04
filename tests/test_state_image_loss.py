import json
from pathlib import Path
import h5py
import pytest
import torch

from visuotactile.state_image_loss import GeometryObjective, calibrated_weights, rotation_chord


@pytest.fixture
def objective():
    index = json.loads(Path('outputs/visuotactile/native_dp_3way/native_20261101_02/data.json').read_text())
    row = index['entries'][index['splits']['train'][0]]
    with h5py.File(row['path'], 'r') as f: metadata=json.loads(f.attrs['metadata_json'])
    return GeometryObjective(metadata)


def test_all_geometry_components_mask_padding_and_have_finite_joint_gradients(objective):
    torch.manual_seed(7)
    command=torch.randn(2, 4, 29, dtype=torch.double)*.1
    scale=torch.linspace(.5, 2, 29, dtype=torch.double);offset=torch.linspace(-1,1,29,dtype=torch.double)
    x=(command*scale+offset+.03*torch.arange(1,5,dtype=torch.double)[None,:,None]).requires_grad_()
    valid=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
    values=objective.components(x,command,valid,scale,offset)
    for name,loss in values.items():
        grad=torch.autograd.grad(loss,x,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad[valid].norm()>0, name
        assert grad[~valid].abs().sum()==0, name
    changed=command.clone();changed[~valid]=100
    again=objective.components(x,changed,valid,scale,offset)
    assert all(torch.equal(v,again[k]) for k,v in values.items())


def test_rotation_chord_zero_pi_and_double_gradcheck():
    eye=torch.eye(3,dtype=torch.double)
    angle=torch.tensor(.4,dtype=torch.double,requires_grad=True)
    def loss(a):
        z=a*0;o=z+1;c=a.cos();s=a.sin()
        r=torch.stack((c,-s,z,s,c,z,z,z,o)).reshape(3,3)
        return rotation_chord(r,eye)
    assert loss(torch.tensor(0.))==0
    assert loss(torch.tensor(torch.pi))==pytest.approx(1.)
    assert torch.autograd.gradcheck(loss,(angle,))


def test_finger_fk_and_rotation_gradient_and_equal_target_zero(objective):
    q=torch.full((1,2,29),.05,dtype=torch.double,requires_grad=True)
    y=torch.zeros_like(q);valid=torch.ones(1,2,dtype=torch.bool);scale=torch.ones(29,dtype=torch.double)
    def fn(x):
        parts=objective.components(x,y,valid,scale,scale*0)
        return parts['position']+parts['rotation']+parts['fingertip']
    assert torch.autograd.gradcheck(fn,(q,),fast_mode=True,atol=1e-5)
    grad=torch.autograd.grad(fn(q),q)[0]
    assert grad[...,:7].norm()>0 and grad[...,7:].norm()>0
    parts=objective.components(y,y,valid,scale,scale*0)
    assert all(v==0 for v in parts.values())


def test_calibration_bounds_every_training_batch():
    rows=[dict(loss=dict(base=.4,position=12.,rotation=.7,fingertip=3.,temporal=2.),
        gradient=dict(base=.1,position=30.,rotation=.2,fingertip=2.,temporal=1.))]
    weights=calibrated_weights(rows)
    for r in rows:
        for k,w in weights.items():
            assert 0<w<=1
            assert w*r['loss'][k]<=.05*r['loss']['base']+1e-12
            assert w*r['gradient'][k]<=.05*r['gradient']['base']+1e-12
