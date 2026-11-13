"""An incomplete or failed finite stage cannot silently advance the research cycle."""
import json
import time
from types import SimpleNamespace
import pytest
from tools import tight_fresh_cycle as cycle


@pytest.mark.parametrize('fault',['exit','missing','incomplete'])
def test_failed_stage_retains_evidence_and_never_advances(tmp_path,monkeypatch,fault):
    root=tmp_path/'initial';root.mkdir();collection=tmp_path/'collection';collection.mkdir()
    output=tmp_path/'round';ledger=tmp_path/'round_stages';calls=[]
    (root/'session.json').write_text(json.dumps({'deadline':time.time()+100}))
    (collection/'result.json').write_text(json.dumps({'software_complete':True}))
    (collection/'correction_entries.json').write_text(json.dumps({'real':{}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cycle.subprocess,'check_output',lambda *a,**k:'source commit')
    def execute(command,**kwargs):
        calls.append(command)
        if fault!='missing':
            output.mkdir()
            (output/'prepared.json').write_text(json.dumps({'complete':fault!='incomplete'}))
        return SimpleNamespace(returncode=1 if fault=='exit' else 0)
    monkeypatch.setattr(cycle.subprocess,'run',execute)
    with pytest.raises(RuntimeError):cycle.run(root,collection,output,10,5,4,123)
    assert len(calls)==1
    assert (ledger/'failure.json').exists() and (ledger/'aggregate.json').exists()
    assert not (ledger/'finished.json').exists()
    failure=json.loads((ledger/'failure.json').read_text())
    assert len(failure['stages'])==1
    with pytest.raises(FileExistsError):cycle.run(root,collection,output,10,5,4,123)
