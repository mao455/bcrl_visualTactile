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
