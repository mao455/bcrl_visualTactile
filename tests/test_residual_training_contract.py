"""Exercise actual trainer scheduling/selection with tiny synthetic CPU tensors."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from visuotactile import anchor_continue
from visuotactile.models.native_dp import TargetResidual


class TinyResidual(torch.nn.Module):
    version = TargetResidual.version

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(29))
        self.register_buffer('action_scale', torch.ones(29))
        self.seen = []

    def forward(self, inputs, state):
        self.seen.extend(inputs['robot'][..., 0][inputs['valid']].tolist())
        raw = self.weight.expand(*inputs['valid'].shape, 29)
        return raw, {'last': raw[:, -1]}

    def compose(self, base, raw, contact, limits):
        return base+raw, raw

    objective = TargetResidual.objective


def record(identity, length):
    return dict(robot=torch.full((length, 1), float(identity)), visual=torch.zeros(length, 1),
        plans=torch.zeros(length, 1, 29), plan_valid=torch.ones(length, 1, dtype=torch.bool),
        tactile={'dummy': torch.zeros(length*2, 1)}, tactile_valid=torch.ones(length*2, dtype=torch.bool),
        base=torch.zeros(length, 29), target=torch.ones(length, 29),
        # Every synthetic tick is noncontact: counting only contact would fail.
        gate=torch.zeros(length, dtype=torch.bool), limits=torch.tensor([[-2., 2.]]).expand(29, -1))


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    root=tmp_path/'R01';root.mkdir()
    train=[f't{i}' for i in range(6)];validation=[f'v{i}' for i in range(4)]
    records={identity:record(i, i+1) for i,identity in enumerate(train)}
    records.update({identity:record(99, 1) for identity in validation})
    data={'splits':dict(train=train,validation=validation)}
    config=dict(maximum_control_tick_epochs=5,minimum_control_tick_epochs=1,
        microbatch=4,validation_epoch_interval=1,initialization_seed=3)
    session=dict(training_deadline=float('inf'),base_contracts={'R01':{}})
    lease=SimpleNamespace(block=nullcontext,wait_s=0.,work_s=0.)
    model=TinyResidual()
    monkeypatch.setattr(anchor_continue,'residual_setup',lambda *a,**kw:
        (session,config,data,records,model,{},lease))
    monkeypatch.setattr(anchor_continue.signal,'signal',lambda *a:None)
    monkeypatch.setattr(anchor_continue,'residual_curves',lambda *a:None)
    # Constant validation exercises the registered four-validation plateau.
    calls=[]
    def metrics(model,records,ids,lease):
        calls.append(list(ids))
        return {'all':{'normalized_equal_mse':1.}}, []
    monkeypatch.setattr(anchor_continue,'residual_metrics',metrics)
    return root,session,model,calls


def test_probe_is_four_complete_train_episodes_and_disposable(trainer):
    root,_,model,calls=trainer
    anchor_continue.residual_train(root,probe=True)
    finished=json.loads((root/'training_probe/finished.json').read_text())
    saved=torch.load(root/'training_probe/best.pt',map_location='cpu',weights_only=False)
    assert sorted(model.seen)==[0.]+[1.]*2+[2.]*3+[3.]*4
    assert finished['supervised_ticks']==finished['required_supervised_ticks']==10
    assert finished['target_reached'] and finished['control_tick_epochs']==1
    assert finished['effective_batch']==finished['train_episode_count']==4
    assert finished['updates']==saved['updates']==1
    assert finished['formal_updates']==saved['formal_updates']==0
    assert finished['disposable'] and saved['disposable']
    assert saved['train_ids']==['t0','t1','t2','t3']
    assert calls[0]==['v0','v1','v2','v3']
    assert finished['wall_s']>=finished['training_wall_s']>0
    assert finished['supervised_ticks_per_wall_s']>0
    assert not (root/'training').exists()
    with pytest.raises(ValueError,match='second residual'):anchor_continue.residual_train(root,probe=True)


def test_flat_validation_stops_after_registered_minimum_and_four_checks(trainer):
    root,_,_,_=trainer
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    assert finished['control_tick_epochs']==4
    assert finished['supervised_ticks']==84 and finished['required_supervised_ticks']==105
    assert not finished['target_reached'] and not finished['disposable']
    assert finished['minimum_exposure_reached'] and finished['stop_reason']=='validation_plateau_after_minimum_exposure'
    assert finished['formal_updates']==finished['updates']==8
    assert finished['sampled_contact_anchors']==0


def test_plateau_cannot_bypass_minimum_and_common_exposure_is_saved(trainer):
    root,_,_,_=trainer
    _,config,*_=anchor_continue.residual_setup(root)
    config.update(minimum_control_tick_epochs=5,comparison_control_tick_epochs=[2])
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    assert finished['control_tick_epochs']==5 and finished['target_reached']
    common=torch.load(root/'training/comparison_epoch_2.pt',weights_only=False)
    assert common['epoch']==2 and common['supervised_ticks']==42 and common['updates']==4


def test_budget_stop_does_not_claim_full_exposure(trainer):
    root,session,_,_=trainer
    session['training_deadline']=0.
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    assert finished['stop_reason']=='fixed_wall_budget'
    assert not finished['target_reached']
    assert finished['supervised_ticks']<finished['required_supervised_ticks']


def test_selection_metric_counts_noncontact_and_equal_arm_hand(monkeypatch):
    model=TinyResidual()
    records={'a':dict(target=torch.zeros(2,29),gate=torch.tensor([True,False]))}
    prediction=torch.zeros(2,29)
    prediction[1,:7]=2.;prediction[1,7:]=4.
    monkeypatch.setattr(anchor_continue,'residual_prediction',lambda *a,**kw:prediction)
    metrics,_=anchor_continue.residual_metrics(model,records,['a'],None)
    # Tick0 = zero error; tick1 = .5*2^2 + .5*4^2 = 10.
    assert metrics['all']['normalized_equal_mse']==5.
    assert metrics['contact']['normalized_equal_mse']==0.
    assert metrics['noncontact']['normalized_equal_mse']==10.
