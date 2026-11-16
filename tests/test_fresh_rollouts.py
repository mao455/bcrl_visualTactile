import json
from pathlib import Path
import pytest
import torch
from tools.tight_fresh_rollouts import completed_pair
from visuotactile.compact import sha256


def fixture(tmp_path, monkeypatch):
    base=tmp_path/'A/backend/A'
    (base/'training').mkdir(parents=True)
    (base/'offline').mkdir()
    torch.save({'contract': {'visual_stride': 1}},base/'training/best.pt')
    (base/'training/finished.json').write_text(json.dumps({'target_reached': True,'disposable': False}))
    (base/'offline/result.json').write_text(json.dumps({'complete': True,'test_used': False,
        'checkpoint_sha256':sha256(base/'training/best.pt')}))
    import visuotactile.state_benchmark as benchmark
    monkeypatch.setattr(benchmark,'dagger_policy_identity',lambda saved,path: {'backend':path})
    return base


def test_complete_pair_with_bound_offline_is_admitted(tmp_path,monkeypatch):
    base=fixture(tmp_path,monkeypatch)
    assert completed_pair(tmp_path,'A')['backend']==str(base/'training/best.pt')


@pytest.mark.parametrize('kind',['model_drift','incomplete_training','test_evidence','sparse_vision'])
def test_inadmissible_pair_rejected_before_simulator(tmp_path,monkeypatch,kind):
    base=fixture(tmp_path,monkeypatch)
    if kind=='model_drift':
        with (base/'training/best.pt').open('ab') as f:f.write(b'changed')
    elif kind=='incomplete_training':
        (base/'training/finished.json').write_text(json.dumps({'target_reached':False}))
    elif kind=='test_evidence':
        path=base/'offline/result.json';value=json.loads(path.read_text());value['test_used']=True
        path.write_text(json.dumps(value))
    else:
        torch.save({'contract':{'visual_stride':12}},base/'training/best.pt')
        path=base/'offline/result.json';value=json.loads(path.read_text())
        value['checkpoint_sha256']=sha256(base/'training/best.pt');path.write_text(json.dumps(value))
    with pytest.raises(ValueError):completed_pair(tmp_path,'A')


@pytest.mark.parametrize('fault',[None,'profile','minimum','stopping','disposable'])
def test_phase_plateau_requires_valid_complete_receipt(tmp_path,monkeypatch,fault):
    from visuotactile.bounded_extension import phase_dagger_profile
    base=fixture(tmp_path,monkeypatch)
    receipt=dict(target_reached=False,disposable=False,
        bounded_extension=phase_dagger_profile('backend'),bounded_training_complete=True,
        minimum_exposure_reached=True,stop_reason='validation_plateau_after_lr_reduction',
        plateau_state={'stop':True})
    if fault=='profile':receipt['bounded_extension']['minimum_epochs']=1
    elif fault=='minimum':receipt['minimum_exposure_reached']=False
    elif fault=='stopping':receipt['plateau_state']['stop']=False
    elif fault=='disposable':receipt['disposable']=True
    (base/'training/finished.json').write_text(json.dumps(receipt))
    if fault is None:assert completed_pair(tmp_path,'A')['backend']==str(base/'training/best.pt')
    else:
        with pytest.raises(ValueError):completed_pair(tmp_path,'A')


@pytest.mark.parametrize('count',[8,20])
def test_registered_final_wave_has_exact_pair_and_never_uses_development(tmp_path,count):
    from scripts.tight_layout_20261116.final_test import validate
    seeds=list(range(28823000,28823000+count));pairs={'A':{'hash':'a'},'B':{'hash':'b'}}
    reg=tmp_path/'registration.json';reg.write_text(json.dumps(dict(seeds=seeds,pairs=pairs)))
    cases=[dict(role=role,spec=dict(seed=seed,split='test',narrow_reset={'same':seed})) for seed in seeds[:4] for role in 'AB']
    plan=dict(kind='autonomous',test_used=True,holdout_contract='layout_final_test_v1',
      final_test_registration=str(reg),final_test_registration_sha256=sha256(reg),pairs=pairs,cases=cases)
    validate(plan)
    cases[-1]['role']='A'
    with pytest.raises(AssertionError):validate(plan)
    cases[-1]['role']='B';cases[-1]['spec']['split']='development'
    with pytest.raises(AssertionError):validate(plan)


@pytest.mark.parametrize('fault',[None,'selected_earlier','wrong_boundary','incomplete'])
def test_matched_budget_requires_exact_final_snapshot(tmp_path,monkeypatch,fault):
    from visuotactile.bounded_extension import matched_update_profile
    base=fixture(tmp_path,monkeypatch);cap=dict(updates=2000,boundary=dict(epoch=20,next_group=24))
    receipt=dict(target_reached=False,disposable=False,bounded_extension=matched_update_profile('backend',64),
        matched_update_stop=cap,matched_update_complete=True,updates=2000,matched_boundary=cap['boundary'],
        stop_reason='registered_matched_update_boundary',required_control_tick_epochs=64)
    saved=dict(contract=dict(visual_stride=1),updates=2000,configuration=dict(matched_update_stop=cap))
    if fault=='selected_earlier':saved['updates']=1990
    elif fault=='wrong_boundary':receipt['matched_boundary']=dict(epoch=20,next_group=20)
    elif fault=='incomplete':receipt['matched_update_complete']=False
    torch.save(saved,base/'training/best.pt')
    (base/'training/finished.json').write_text(json.dumps(receipt))
    (base/'offline/result.json').write_text(json.dumps(dict(complete=True,test_used=False,checkpoint_sha256=sha256(base/'training/best.pt'))))
    if fault is None:assert completed_pair(tmp_path,'A')
    else:
        with pytest.raises(ValueError):completed_pair(tmp_path,'A')
