import copy
import json

import pytest
import torch

from visuotactile import anchor_continue
from visuotactile.anchor_sampling import residual_recovery_weights
from visuotactile.fresh_contracts import FRESH_AGGREGATE_CONTRACT
from visuotactile.fresh_recovery import (FRESH_RECOVERY_WEIGHTING_VERSION,
    fresh_recovery_weighting_config, fresh_recovery_plan, fresh_recovery_exposure)
from visuotactile.residual_tick_window import ResidualTickWindow, TICK_WINDOW_CONTRACT
from test_residual_training_contract import trainer, record, TinyResidual


def contract(weight=3., ticks=12):
    return dict(version=FRESH_RECOVERY_WEIGHTING_VERSION, ticks=ticks, backend_weight=weight)


def cohort():
    original = [f't{i}' for i in range(30)]
    data = dict(visual_stride=1, supervision='u_expert',
        normalization_identity='u_expert_fresh_train_only_affine_v1',
        normalization_scope='unchanged original training affine; no aggregate refit',
        normalization_train_ids=original,
        aggregation=dict(version='expert_takeover_aggregate_v1', student_prefix_supervised=False),
        splits=dict(train=original+['correction'], validation=[f'v{i}' for i in range(6)], test=[]),
        entries={'correction':dict(trajectory_kind='expert_takeover_v1')})
    records = {i: record(j, 1) for j, i in enumerate(original+data['splits']['validation'])}
    correction = record(80, 90)
    correction.update(time=torch.arange(90)/60, supervision_mask=torch.arange(90)>=34,
        control_segment=(torch.arange(90)>=34).long())
    records['correction'] = correction
    configuration = dict(fresh_data_contract=FRESH_AGGREGATE_CONTRACT,
        tick_window_contract=dict(TICK_WINDOW_CONTRACT), fresh_recovery_weighting=contract())
    return data, configuration, records


def test_missing_opt_in_leaves_legacy_untouched():
    assert fresh_recovery_weighting_config(None) is None
    assert fresh_recovery_plan({}, {}, {}) == (None, {}, None)


@pytest.mark.parametrize('bad', [dict(version='unknown',ticks=12,backend_weight=3.),
    dict(ticks=12,backend_weight=3.), dict(contract(),extra=True),
    *[dict(contract(),ticks=x) for x in (True,0,-1,1.5)],
    *[dict(contract(),backend_weight=x) for x in (True,0.9,float('inf'),float('nan'))]])
def test_rejects_unknown_or_invalid_contract(bad):
    with pytest.raises(ValueError): fresh_recovery_weighting_config(bad)


@pytest.mark.parametrize('mutation', ['legacy','strata','empty_strata','old_weighting','window',
    'refit','validation','kind','missing_mask','missing_segment','clock','empty','length'])
def test_requires_actual_fresh_aggregate_and_unchanged_schedule(mutation):
    data, configuration, records = cohort()
    if mutation == 'legacy': configuration['fresh_data_contract']='old'
    elif mutation == 'strata': data['sampling_strata']={'core':data['normalization_train_ids']}
    elif mutation == 'empty_strata': configuration['sampling_strata']={}
    elif mutation == 'old_weighting': configuration['recovery_prefix']={}
    elif mutation == 'window': configuration.pop('tick_window_contract')
    elif mutation == 'refit': data['normalization_scope']='all episodes'
    elif mutation == 'validation': data['splits']['validation'].pop()
    elif mutation == 'kind': data['entries']['correction']['trajectory_kind']='student'
    elif mutation == 'missing_mask': records['correction'].pop('supervision_mask')
    elif mutation == 'missing_segment': records['correction'].pop('control_segment')
    elif mutation == 'clock': records['correction'].pop('time')
    elif mutation == 'empty': records['correction']['supervision_mask'].zero_()
    elif mutation == 'length': records['correction']['robot']=records['correction']['robot'][:-1]
    with pytest.raises(ValueError): fresh_recovery_plan(data, configuration, records)


@pytest.mark.parametrize('boundary', ['hole','segment','requested_ticks'])
def test_only_first_contiguous_legal_expert_ticks_are_weighted(boundary):
    data, configuration, records = cohort()
    correction = records['correction']
    end = 46
    if boundary == 'hole':
        correction['supervision_mask'][39] = False
        end = 39
    elif boundary == 'segment':
        correction['control_segment'][39:] = 2
        end = 39
    before = copy.deepcopy(records)
    spec, masks, population = fresh_recovery_plan(data, configuration, records)
    assert masks['correction'].nonzero().flatten().tolist() == list(range(34,end))
    assert all(not masks[i].any() for i in data['normalization_train_ids'])
    # The first two expert queue transitions and all earlier student history
    # remain ineligible; the later legal suffix keeps ordinary supervision.
    assert not masks['correction'][:34].any()
    weights, initial = residual_recovery_weights([masks['correction']],32,64,
        correction['supervision_mask'][None,32:64],spec['backend_weight'])
    assert not weights[0,:2].any()
    assert (weights[initial] == 3.).all()
    assert (weights[correction['supervision_mask'][None,32:64]&~initial] == 1.).all()
    assert population['initial_recovery_supervised_ticks'] == end-34
    assert population['loss_weight_mass'] == population['supervised_ticks']+2*(end-34)
    for key in ('robot','time','supervision_mask','control_segment'):
        assert torch.equal(correction[key],before['correction'][key])


def test_weight_one_has_identical_loss_gradient_and_physical_exposure():
    data, configuration, records = cohort()
    configuration['fresh_recovery_weighting']=contract(1.)
    spec, masks, population = fresh_recovery_plan(data, configuration, records)
    valid = records['correction']['supervision_mask'][None]
    weights, _ = residual_recovery_weights([masks['correction']],0,90,valid,spec['backend_weight'])
    values = torch.linspace(-1,1,90,requires_grad=True)[None]
    old = (values.square()*valid).sum()/valid.sum()
    new = (values.square()*weights).sum()/weights.sum()
    assert torch.equal(old,new)
    assert torch.equal(torch.autograd.grad(old,values,retain_graph=True)[0],torch.autograd.grad(new,values)[0])
    assert population['loss_weight_mass'] == population['supervised_ticks'] == 86
    exposure = fresh_recovery_exposure(records,masks,{i:2 for i in data['splits']['train']},spec)
    assert exposure['supervised_ticks'] == exposure['loss_weight_mass'] == 172
    assert exposure['initial_recovery_supervised_ticks'] == 24


def test_weighted_onset_uses_fixed128_denominator():
    data, configuration, records = cohort()
    spec, masks, _ = fresh_recovery_plan(data, configuration, records)
    valid = records['correction']['supervision_mask'][None]
    model = torch.nn.Linear(1,1,bias=False).double()
    model.weight.data.fill_(.1)
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    window = ResidualTickWindow(TICK_WINDOW_CONTRACT)
    for start,stop in ((0,32),(32,64),(64,90)):
        legal = valid[:,start:stop]
        if not legal.any(): continue
        weights, initial = residual_recovery_weights([masks['correction']],start,stop,legal,spec['backend_weight'])
        error = model(torch.ones(stop-start,1,dtype=torch.double)).square().flatten()[None]
        loss = (error*weights).sum()/weights.sum()
        window.backward(loss,optimizer,weight_mass=float(weights.sum()),supervised_ticks=int(legal.sum()),
            contact_ticks=0,history_ticks=stop-start,weight_report=dict(
                initial_recovery_supervised_ticks=int(initial.sum()),remaining_supervised_ticks=int((legal&~initial).sum())))
    assert model.weight.grad.item() == pytest.approx(2*.1*(56+2*12)/128)
    report = window.step(model,optimizer)
    assert report['fixed_denominator'] == 128 and report['supervised_ticks'] == 56
    assert report['loss_weight_mass'] == 80 and report['initial_recovery_supervised_ticks'] == 12
    assert report['remaining_supervised_ticks'] == 44 and report['group_tail']


def configure_trainer(trainer, monkeypatch, *, weight):
    root, session, _, calls = trainer
    _, base, _, _, _, _, lease = anchor_continue.residual_setup(root)
    data, config, records = cohort()
    config.update(base,maximum_control_tick_epochs=1,minimum_control_tick_epochs=1)
    if weight is None:
        config.pop('fresh_recovery_weighting')
        receipt = {}
    else:
        config['fresh_recovery_weighting']=contract(weight)
        spec, _, population = fresh_recovery_plan(data,config,records)
        receipt = dict(fresh_recovery_weighting=spec,fresh_recovery_population=population)
    model = TinyResidual()
    setup = (session,config,data,records,model,receipt,lease)
    monkeypatch.setattr(anchor_continue,'residual_setup',lambda *args,**kwargs:setup)
    return setup


def test_actual_trainer_weight_one_matches_unweighted_and_receipts(trainer,monkeypatch):
    root, session, _, calls = trainer
    setup = configure_trainer(trainer,monkeypatch,weight=None)
    anchor_continue.residual_train(root)
    old = torch.load(root/'training/last.pt',weights_only=False)
    old_rows = [json.loads(line) for line in (root/'training/loss.jsonl').read_text().splitlines()]
    setup = configure_trainer(trainer,monkeypatch,weight=1.)
    other = root.parent/'control';other.mkdir();session['base_contracts']['control']={}
    anchor_continue.residual_train(other)
    new = torch.load(other/'training/last.pt',weights_only=False)
    rows = [json.loads(line) for line in (other/'training/loss.jsonl').read_text().splitlines()]
    assert torch.equal(old['model']['weight'],new['model']['weight'])
    assert old['episode_order'] == new['episode_order'] and old['episode_counts'] == new['episode_counts']
    assert old['updates'] == new['updates'] and old['supervised_ticks'] == new['supervised_ticks'] == 86
    assert all(a['loss']==b['loss'] and a['gradient_norm']==b['gradient_norm'] for a,b in zip(old_rows,rows))
    assert all(row['fresh_recovery_weighting']==contract(1.) for row in rows)
    assert sum(row['initial_recovery_supervised_ticks'] for row in rows) == 12
    finished = json.loads((other/'training/finished.json').read_text())
    assert finished['fresh_recovery_exposure']['loss_weight_mass'] == finished['supervised_ticks'] == 86
    assert new['validation_ids'] == [f'v{i}' for i in range(6)]


def test_trainer_rechecks_setup_contract_before_writing(trainer,monkeypatch):
    root, _, _, _ = trainer
    setup = configure_trainer(trainer,monkeypatch,weight=3.)
    setup[5]['fresh_recovery_population']['initial_recovery_supervised_ticks'] += 1
    with pytest.raises(ValueError,match='setup/training'): anchor_continue.residual_train(root)
    assert not (root/'training').exists()
