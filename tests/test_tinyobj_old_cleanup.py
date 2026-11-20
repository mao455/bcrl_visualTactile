"""Exercise the authorized old-data boundary entirely within temporary outputs."""
import gzip
import importlib.util
import json
import os
from pathlib import Path

import pytest


def fixture(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'research_docs/evidence/tight_adaptive_dagger_20261119_01/cleanup_tinyobj_old_data.py'
    spec = importlib.util.spec_from_file_location('tiny_cleanup_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / 'outputs'
    base.mkdir()
    monkeypatch.setattr(module, 'BASE', base)
    monkeypatch.setattr(module, 'LEDGER', tmp_path / 'ledger')
    monkeypatch.setattr(module, 'process_guard', lambda: (set(), set(), {}))

    def add(relative, *, old=True):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'generated test data')
        stamp = module.CUTOFF - 1 if old else module.CUTOFF
        os.utime(path, (stamp, stamp))
        return path

    clean = add('production/datasets/cleandataset/clean.pkl')
    return module, base, add, clean


def manifest(module):
    with gzip.open(module.LEDGER / 'manifest.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def test_plan_and_apply_preserve_clean_recent_code_config_and_links(tmp_path, monkeypatch):
    module, base, add, clean = fixture(tmp_path, monkeypatch)
    old = add('production/datasets/2026-10-16/work/obsolete.pkl')
    retained = [clean, add('old/clean_dataset/nested/clean.pkl'),
                add('old/process.py'), add('old/config/settings.json'),
                add('recent/20261120/old_vendor.npz'), add('old/new.pkl', old=False),
                add('rax_dexbin_tinyobj/older/data.h5')]
    linked = add('old/hard.pkl')
    os.link(linked, base / 'old/alias.pkl')
    retained.extend([linked, base / 'old/alias.pkl'])
    symbolic = base / 'old/symbol.pkl'
    symbolic.symlink_to(old)
    module.plan()
    rows = manifest(module)
    assert [row['path'] for row in rows] == [str(old)]
    clean_before = module.helper.identity(clean)
    plan = json.loads((module.LEDGER / 'plan.json').read_text())
    module.apply(plan['manifest_sha256'])
    assert not old.exists()
    assert all(p.exists() for p in retained) and symbolic.is_symlink()
    assert module.helper.identity(clean) == clean_before
    result = json.loads((module.LEDGER / 'result.json').read_text())
    assert result['files_deleted'] == 1 and result['clean_stat_identities_unchanged']


def test_open_inode_and_active_directory_are_excluded(tmp_path, monkeypatch):
    module, base, add, clean = fixture(tmp_path, monkeypatch)
    opened = add('old/open.pkl')
    active = add('old/queued/future.pkl')
    eligible = add('old/closed.pkl')
    info = opened.stat()
    monkeypatch.setattr(module, 'process_guard', lambda: ({(info.st_dev, info.st_ino)}, {active.parent}, {}))
    module.plan()
    assert [r['path'] for r in manifest(module)] == [str(eligible)]


def test_changed_candidate_is_not_unlinked(tmp_path, monkeypatch):
    module, base, add, clean = fixture(tmp_path, monkeypatch)
    candidate = add('old/data.pkl')
    module.plan()
    candidate.write_bytes(b'changed after inventory')
    plan = json.loads((module.LEDGER / 'plan.json').read_text())
    with pytest.raises(AssertionError):
        module.apply(plan['manifest_sha256'])
    assert candidate.exists() and clean.exists()


def test_clean_change_aborts_before_any_delete(tmp_path, monkeypatch):
    module, base, add, clean = fixture(tmp_path, monkeypatch)
    candidate = add('old/data.pkl')
    module.plan()
    clean.write_bytes(b'changed by another writer')
    plan = json.loads((module.LEDGER / 'plan.json').read_text())
    with pytest.raises(AssertionError):
        module.apply(plan['manifest_sha256'])
    assert candidate.exists() and not (module.LEDGER / 'operations.jsonl').exists()
