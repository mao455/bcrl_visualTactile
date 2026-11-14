import pytest
import torch

from tools import tight_trajectory_training_gate as gate
from test_trajectory_batch_gate import complete_fixture as old_fixture, artifacts


def complete_fixture(artifacts):
    import json
    root,frozen=old_fixture(artifacts)
    rr=root/'B';config=gate.read(rr/'training/configuration.json')
    config.update(learning_rate=3e-5,weight_decay=1e-6)
    (rr/'training/configuration.json').write_text(json.dumps(config))
    path=rr/'training/best.pt';saved=torch.load(path,weights_only=False)
    saved['configuration']=config;torch.save(saved,path)
    (rr/'training/finished.json').write_text(json.dumps(dict(complete=True,checkpoint_sha256=gate.sha256(path))))
    return root,frozen


def test_train_only_cpu_receipt_keeps_repeated_visit_and_tail(artifacts,tmp_path):
    root,frozen=complete_fixture(artifacts)
    output=tmp_path/'prepare'
    result=gate.run(root,'B',frozen,output,root,check_only=True)
    assert result['complete'] and result['passed'] is None and not result['cuda_initialized']
    assert all(value==0 for value in result['optimizer_steps_confirmed'].values())
    receipt=gate.read(output/'admission.json')
    assert receipt['registered_deck']==['old1','new1','old1']
    assert receipt['registered_batches']==[['old1','new1'],['old1']]
    assert set(receipt['explicit_candidate_helpers'])=={'trajectory_batch','trajectory_training'}
    projected=torch.load(output/'train_only/numeric.pt',weights_only=False)
    assert set(projected['records'])=={'old1','new1'}
    assert not (output/'diagnostic_boundary.pt').exists()
    with pytest.raises(FileExistsError):gate.run(root,'B',frozen,output,root,check_only=True)


def test_bad_train_identity_cannot_reach_update(artifacts,tmp_path):
    root,frozen=complete_fixture(artifacts);output=tmp_path/'bad'
    with pytest.raises(ValueError,match='training episodes'):
        gate.run(root,'B',frozen,output,root,identities=['old1','val1'])
    failed=gate.read(output/'failed.json')
    assert all(value==0 for value in failed['optimizer_steps_confirmed'].values())
    assert failed['active_update_may_be_partial'] is None


def test_stop_before_cuda_retains_failed_attempt(artifacts,tmp_path):
    root,frozen=complete_fixture(artifacts);(root/'STOP').touch()
    output=tmp_path/'stop'
    with pytest.raises(InterruptedError):gate.run(root,'B',frozen,output,root)
    assert not torch.cuda.is_initialized() and (output/'failed.json').exists()
    assert not (output/'result.json').exists()


def test_complete_state_tree_checks_optimizer_rng_and_tensor_structure():
    reference={'model':{'weight':torch.tensor([1.,0.])},
        'optimizer':{'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.zeros(2)}}},
        'noise_rng':torch.tensor([7,9],dtype=torch.uint8),'updates':2,'visits':3}
    candidate={'model':{'weight':torch.tensor([1.+1e-6,0.])},
        'optimizer':{'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.zeros(2)}}},
        'noise_rng':torch.tensor([7,9],dtype=torch.uint8),'updates':2,'visits':3}
    assert all(v['passed'] for v in gate.compare_state(reference,candidate).values())
    assert not all(v['passed'] for v in gate.compare_state(reference,candidate,exact=True).values())
    candidate['noise_rng'][0]=8
    assert not gate.compare_state(reference,candidate)['state/noise_rng']['passed']
    candidate['optimizer']['state'][0]['exp_avg']=torch.zeros(3)
    with pytest.raises(ValueError,match='structure'):gate.compare_state(reference,candidate)


def test_state_tree_rejects_missing_ema_and_nan():
    with pytest.raises(ValueError,match='structure'):
        gate.compare_state({'ema':torch.zeros(1)},{})
    result=gate.compare_state(torch.zeros(1),torch.tensor([float('nan')]))
    assert not result['state']['passed']


def test_diagnostic_optimizer_uses_frozen_formal_learning_rate():
    spec=gate.optimizer_spec(dict(learning_rate=3e-5,weight_decay=1e-6))
    assert spec==dict(lr=3e-5,weight_decay=1e-6,betas=(.95,.999))
    with pytest.raises(ValueError):gate.optimizer_spec(dict(learning_rate=float('nan'),weight_decay=1e-6))
    with pytest.raises(ValueError):gate.optimizer_spec(dict(learning_rate=True,weight_decay=1e-6))
