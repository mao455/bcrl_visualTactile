import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import project_maintenance as maintenance


def test_only_exited_zombie_is_exempt_from_unreadable_open_file_inventory():
    import os
    import subprocess
    import sys
    import time
    assert maintenance.restricted_system_process(os.getpid()) is None
    child=subprocess.Popen([sys.executable,'-c','import os; os._exit(0)'])
    try:
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            state=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[0]
            if state=='Z':break
            time.sleep(.01)
        assert state=='Z'
        assert 'zombie' in maintenance.restricted_system_process(child.pid)['reason']
    finally:child.wait(timeout=2)


def make_plan(tmp_path):
    root=tmp_path/'artifacts';root.mkdir()
    keep=root/'current.pt';keep.write_bytes(b'protected')
    old=root/'old.pt';old.write_bytes(b'obsolete')
    alias=root/'old-link.pt';alias.hardlink_to(old)
    request=tmp_path/'request.json';ledger=tmp_path/'ledger.json'
    request.write_text(json.dumps(dict(allowed_root=str(root),protected={str(keep):'current checkpoint'},
        candidates=[dict(path=str(root/'old.pt'),reason='superseded',hash=True),
            dict(path=str(alias),reason='duplicate'),dict(path=str(keep),reason='must be protected') ])))
    with patch.object(maintenance,'CODE',tmp_path),patch.object(maintenance.subprocess,'check_output',return_value='commit'):
        maintenance.plan(request,ledger)
    return root,ledger


def test_prune_protects_current_and_counts_hardlink_storage_once(tmp_path):
    root,ledger=make_plan(tmp_path)
    blocks=(root/'old.pt').stat().st_blocks*512
    with patch.object(maintenance,'open_inodes',return_value=(set(),[])):
        result=maintenance.apply(ledger)
    assert result['deleted_checkpoints']==2 and result['released_blocks_bytes']==blocks
    assert (root/'current.pt').read_bytes()==b'protected'
    assert not (root/'old.pt').exists() and not (root/'old-link.pt').exists()


def test_mutated_candidate_aborts_before_any_removal(tmp_path):
    root,ledger=make_plan(tmp_path)
    (root/'old.pt').write_bytes(b'changed after review')
    with patch.object(maintenance,'open_inodes',return_value=(set(),[])),pytest.raises(AssertionError,match='Candidate changed'):
        maintenance.apply(ledger)
    assert (root/'old-link.pt').exists() and (root/'current.pt').exists()


def test_open_candidate_is_not_deleted(tmp_path):
    root,ledger=make_plan(tmp_path);s=(root/'old.pt').stat()
    with patch.object(maintenance,'open_inodes',return_value=({(s.st_dev,s.st_ino)},[])),pytest.raises(AssertionError,match='Candidate opened'):
        maintenance.apply(ledger)
    assert (root/'old.pt').exists()


def test_source_migration_records_retirement_without_changing_data(tmp_path,monkeypatch):
    monkeypatch.setattr(maintenance,'CODE',tmp_path)
    root=tmp_path/'run';root.mkdir();source=tmp_path/'current.py';data=tmp_path/'data.pt'
    source.write_text('before');data.write_bytes(b'protected data')
    before=maintenance.digest(source);data_sha=maintenance.digest(data)
    (root/'offline_freeze.json').write_text(json.dumps({str(source):before,str(data):data_sha}))
    (root/'runtime_freeze.json').write_text(json.dumps({'current.py':before,'retired.py':'old_digest'}))
    source.write_text('after')
    changes=maintenance.migrate_freezes(root,{'current.py':'M','retired.py':'D'},root/'migration')
    assert json.loads((root/'offline_freeze.json').read_text())[str(data)]==data_sha
    assert changes['runtime_freeze.json'][-1]['after'] is None
    assert json.loads((root/'migration/before_runtime_freeze.json').read_text())['current.py']==before


def test_source_migration_rejects_unreviewed_data_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(maintenance,'CODE',tmp_path)
    root=tmp_path/'run';root.mkdir();data=tmp_path/'data.pt';data.write_bytes(b'original')
    (root/'offline_freeze.json').write_text(json.dumps({str(data):maintenance.digest(data)}))
    data.write_bytes(b'changed')
    with pytest.raises(AssertionError,match='Unreviewed frozen input drift'):
        maintenance.migrate_freezes(root,{},root/'migration')


def test_failed_assessment_never_stops_or_changes_parent(tmp_path,monkeypatch):
    import time
    root=tmp_path/'assessment';root.mkdir();(root/'maintenance').mkdir()
    (root/'session.json').write_text(json.dumps(dict(parent_root=str(tmp_path/'parent'))))
    request=tmp_path/'request.json';ledger=root/'maintenance/finalization.json'
    request.write_text(json.dumps(dict(assessment_root=str(root),deadline=time.time()+100,analysis_service='owned-analysis.service')))
    monkeypatch.setattr(maintenance.subprocess,'check_output',lambda *a,**k:'ActiveState=failed\nExecMainStatus=1\n')
    with patch.object(maintenance.subprocess,'Popen') as launch,patch.object(maintenance.os,'killpg') as kill:
        with pytest.raises(AssertionError,match='analysis service failed'):maintenance.finalize(request,ledger)
        launch.assert_not_called();kill.assert_not_called()
    result=json.loads(ledger.read_text());assert result['state']=='blocked'
