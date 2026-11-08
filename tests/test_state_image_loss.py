import json
from pathlib import Path
import h5py
import pytest
import torch

from visuotactile.state_image_loss import GeometryObjective, calibrated_weights, rotation_chord, PALM_LOSS_ID, calibrated_palm_weights


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


@pytest.fixture
def palm_objective():
    data=json.loads(Path('outputs/visuotactile/run/tight_state_20261107_01/composition_06/state_data/data.json').read_text())
    with h5py.File(data['entries'][data['splits']['train'][0]]['path']) as f:meta=json.loads(f.attrs['metadata_json'])
    return GeometryObjective(meta,mode=PALM_LOSS_ID)


def test_palm_commit_mask_full_diffusion_hand_and_empty_batch(palm_objective):
    q=torch.full((2,24,29),.05,requires_grad=True);target=torch.zeros_like(q)
    valid=torch.ones(2,24,dtype=torch.bool);valid[1,5:]=False
    parts=palm_objective.components(q,target,valid,torch.ones(29),torch.zeros(29))
    for name in ('position','rotation'):
        grad=torch.autograd.grad(parts[name],q,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad[:,:12,:7].norm()>0
        assert grad[:,12:].abs().sum()==0 and grad[...,7:].abs().sum()==0
        assert grad[~valid].abs().sum()==0
    base=torch.autograd.grad(parts['base'],q,retain_graph=True)[0]
    assert base[0,12:,7:].norm()>0  # Hand29D and full H24 remain supervised.
    empty=palm_objective.components(q,target,torch.zeros_like(valid),torch.ones(29),torch.zeros(29))
    loss=sum(empty.values());assert loss==0 and torch.isfinite(torch.autograd.grad(loss,q)[0]).all()


def test_palm_microbatch_padding_matches_global_gradient(palm_objective):
    from visuotactile.anchor_sampling import weighted_parts,mask_totals
    torch.manual_seed(42);q=(torch.randn(3,24,29)*.03).requires_grad_();y=torch.zeros_like(q)
    valid=torch.arange(24)[None]<torch.tensor([24,5,17])[:,None]
    totals=mask_totals(dict(a=24,b=5,c=17),[('a',0),('b',0),('c',0)],12)
    weights=dict(position=.1,rotation=.02)
    full=palm_objective.components(q,y,valid,torch.ones(29),torch.zeros(29))
    loss=full['base']+sum(weights[k]*full[k] for k in weights)
    expected=torch.autograd.grad(loss,q,retain_graph=True)[0]
    split=0.
    for sl in (slice(0,2),slice(2,3)):
        part=palm_objective.components(q[sl],y[sl],valid[sl],torch.ones(29),torch.zeros(29))
        split+=weighted_parts(part,valid[sl],totals,weights,12)[0]
    actual=torch.autograd.grad(split,q)[0]
    assert torch.allclose(loss,split,atol=1e-8) and torch.allclose(expected,actual,atol=2e-8,rtol=2e-5)


def test_palm_weights_frozen_from_train_scales():
    rows=[dict(loss=dict(base=.4,position=.012,rotation=.7),gradient=dict(base=.1,position=.3,rotation=.2))]
    weights=calibrated_palm_weights(rows)
    assert set(weights)=={'position','rotation'}
    for key,w in weights.items():
        assert w>0 and w*rows[0]['loss'][key]<=.05*.4+1e-12
        assert w*rows[0]['gradient'][key]<=.05*.1+1e-12
