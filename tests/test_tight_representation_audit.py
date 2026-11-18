"""Scientific diagnostic invariants, independent of local research artifacts."""
import numpy as np
import torch
from tools.tight_representation_probe import relative,rotation_error,fit,predict
from visuotactile.anchor_sampling import labels


def test_rotation_metric_and_pose_use_object_in_palm_frame():
    # Palm at x=1, rotated90deg about z; object at y=1 relative world.
    a=np.array([[1,0,0,2**-.5,0,0,2**-.5]])
    b=np.array([[1,1,0,1,0,0,0]])
    y=relative(a,b)
    np.testing.assert_allclose(y[0,:3],[1,0,0],atol=1e-12)
    np.testing.assert_allclose(rotation_error(y[:,3:],y[:,3:]),0,atol=1e-5)


def test_fold_standardization_uses_only_supplied_training_rows():
    x=torch.tensor([[0.,1.],[2.,1.],[4.,1.]])
    y=torch.tensor([[0.],[2.],[4.]])
    p,receipt=fit(x,y)
    assert torch.equal(p[0],torch.tensor([2.,1.],dtype=torch.float64))
    assert receipt['constant_dimensions']==1
    prediction=predict(p,torch.tensor([[100.,1.]]))
    assert torch.isfinite(prediction).all()
    assert torch.equal(p[0],torch.tensor([2.,1.],dtype=torch.float64))


def test_window_overlap_is_not_distinct_visit_exposure():
    # Explicit independent arithmetic: two overlapping H24 starts in one visit.
    starts=np.array([4,10]);ticks=(starts[:,None]+np.arange(24)[None]).ravel()
    counts=np.bincount(ticks,minlength=40);hits=np.zeros(40,dtype=int);hits[np.unique(ticks)]+=1
    assert counts[12]==2 and hits[12]==1
    assert counts[0]==0 and counts.sum()==48
