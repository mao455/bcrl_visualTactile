import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import project_maintenance as maintenance


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
