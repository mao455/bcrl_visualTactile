"""A fitting continuation must reuse the exact current DP and actual source masks."""
import json
from pathlib import Path
import pytest
from tools import tight_fresh_rearfit as experiment
from visuotactile.compact import sha256


def source(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    write = lambda path, value: path.write_text(json.dumps(value))
    for name, value in [('session.json', {'deadline': 123456}), ('gate.json', {'passed': True}),
                        ('metadata.json', {}), ('shared_resources.json', {'monitor': 1})]:
        write(root / name, value)
    pairs = {}
    for role in ('A', 'B'):
        directory = root / role
        (directory / 'state_data').mkdir(parents=True)
        write(directory / 'state_data/data.json', dict(aggregation={'audited': True}, entries={'recovery': {'round_index': 1}}))
        (directory / 'training').mkdir()
        (directory / 'training/best.pt').write_bytes((role + ' actual DP').encode())
        write(directory / 'cache_index.json', {'frozen_actual_frames': True})
        rear = directory / 'backend'
        (rear / role / 'nominal').mkdir(parents=True)
        (rear / role / 'training').mkdir()
        (rear / role / 'training/best.pt').write_bytes((role + ' actual backend').encode())
        pairs[role] = dict(frontend=str(directory / 'training/best.pt'),
            frontend_sha256=sha256(directory / 'training/best.pt'), backend=str(rear / role / 'training/best.pt'),
            backend_sha256=sha256(rear / role / 'training/best.pt'))
        write(rear / 'configuration.json', {'residual_pair': dict(residual_statistics_sha256='same command affine',
            residual_normalization_identity='same original30', maximum_control_tick_epochs=5)})
        write(rear / 'session.json', {'deadline': 123456, 'base_contracts': {role: {'visual_stride': 1}}})
        write(rear / role / 'nominal/index.json', dict(base_sha256=pairs[role]['frontend_sha256'],
            supervision_mask_sha256='actual legal mask', data_sha256=sha256(directory / 'state_data/data.json'),
            residual_statistics_sha256='same command affine'))
    monkeypatch.setattr(experiment, 'completed_pair', lambda root, role: pairs[role])
    monkeypatch.setattr(experiment.subprocess, 'check_output', lambda *args, **kw: 'frozen source')
    return root, pairs, write


def test_preserves_current_dp_cache_deadline_and_live_monitor(tmp_path, monkeypatch):
    root, pairs, write = source(tmp_path, monkeypatch)
    output = tmp_path / 'new'
    experiment.prepare(root, output, 20)
    assert json.loads((output / 'session.json').read_text())['deadline'] == 123456
    write(root / 'shared_resources.json', {'monitor': 2})
    assert json.loads((output / 'shared_resources.json').read_text())['monitor'] == 2
    for role in ('A', 'B'):
        new = output / role / 'backend' / role
        v = json.loads((new / 'configuration.json').read_text())['residual_pair']
        assert v['continuation_from']['sha256'] == pairs[role]['backend_sha256']
        assert v['maximum_control_tick_epochs'] == 20
        assert sha256(output / role / 'training/best.pt') == pairs[role]['frontend_sha256']
        assert (new / 'nominal').resolve() == root / role / 'backend' / role / 'nominal'
        assert not list((new / 'training').iterdir())
    with pytest.raises(FileExistsError):
        experiment.prepare(root, output, 20)


@pytest.mark.parametrize('key,value', [('base_sha256', 'stale DP'), ('supervision_mask_sha256', None),
    ('data_sha256', 'wrong trajectories'), ('residual_statistics_sha256', 'different affine')])
def test_rejects_relabelled_or_stale_cache(tmp_path, monkeypatch, key, value):
    root, _, write = source(tmp_path, monkeypatch)
    path = root / 'A/backend/A/nominal/index.json'
    index = json.loads(path.read_text()); index[key] = value; write(path, index)
    with pytest.raises(ValueError, match='cache required'):
        experiment.prepare(root, tmp_path / 'rejected', 20)
