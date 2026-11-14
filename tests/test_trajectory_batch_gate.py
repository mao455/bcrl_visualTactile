import json
import time

import pytest
import torch

from tools import tight_trajectory_batch_gate as gate
from test_selected_frontend_fit import artifacts, write


def complete_fixture(artifacts):
    root, frozen, saved = artifacts
    rr = root/'B'
    cache = gate.read(rr/'cache_index.json')
    visual = rr/'state_data/perception/old1.pt'
    write(visual, {'fixture': True})
    cache['entries']['old1'] = dict(path=str(visual), sha256=gate.sha256(visual),
        source_sha256='source-old1', frames=3)
    write(rr/'cache_index.json', cache)
    config = gate.read(rr/'training/configuration.json')
    config['cache_index_sha256'] = gate.sha256(rr/'cache_index.json')
    write(rr/'training/configuration.json', config)
    saved['configuration'] = config
    torch.save(saved, rr/'training/best.pt')
    write(rr/'training/finished.json', dict(complete=True, checkpoint_sha256=gate.sha256(rr/'training/best.pt')))
    torch.save(dict(data_sha256=config['data_sha256'], records={identity:dict(time=torch.arange(3))
        for identity in ('old1','new1','val1','test1')}), rr/'state_data/numeric.pt')
    write(root/'session.json', dict(deadline=time.time()+900))
    return root, frozen


def test_cpu_admission_projects_only_two_train_records_and_never_imports_trainer(artifacts, tmp_path):
    root, frozen = complete_fixture(artifacts)
    output = tmp_path/'check'
    result = gate.run(root, 'B', frozen, output, root, check_only=True)
    assert result['passed'] is None and result['complete'] and not result['cuda_initialized']
    numeric = torch.load(output/'train_only/numeric.pt', weights_only=False)
    assert set(numeric['records']) == {'old1','new1'}
    receipt = gate.read(output/'admission.json')
    assert receipt['candidate_helper'].endswith('/trajectory_batch.py')
    assert receipt['candidate_helper_sha256'] == gate.sha256(receipt['candidate_helper'])
    with pytest.raises(FileExistsError): gate.run(root, 'B', frozen, output, root, check_only=True)


def test_heldout_rejected_with_failure_evidence(artifacts, tmp_path):
    root, frozen = complete_fixture(artifacts)
    output = tmp_path/'wrong'
    with pytest.raises(ValueError, match='training episodes'):
        gate.run(root, 'B', frozen, output, root, identities=['old1','val1'], check_only=True)
    assert gate.read(output/'failed.json')['optimizer_updates'] == 0
    assert not (output/'result.json').exists()


def test_stop_prevents_cuda_even_after_cpu_admission(artifacts, tmp_path):
    root, frozen = complete_fixture(artifacts)
    (root/'STOP').touch()
    with pytest.raises(InterruptedError):
        gate.run(root, 'B', frozen, tmp_path/'stop', root)
    assert not torch.cuda.is_initialized()


def test_fixed_tolerance_rejects_nonfinite_and_over_limit():
    reference = torch.tensor([0., 1.])
    assert gate.difference(reference, reference+1e-6)['passed']
    assert not gate.difference(reference, reference+1e-3)['passed']
    assert not gate.difference(reference, torch.tensor([float('nan'), 1.]))['passed']
