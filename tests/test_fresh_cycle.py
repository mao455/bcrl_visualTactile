"""An incomplete or failed finite stage cannot silently advance the research cycle."""
import json
import os
import time
from types import SimpleNamespace
import pytest
from tools import tight_fresh_cycle as cycle


@pytest.mark.parametrize('fault',['exit','missing','incomplete','deadline'])
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
        code=1 if fault=='exit' else 0
        return SimpleNamespace(pid=os.getpid(),returncode=code,poll=lambda:code)
    monkeypatch.setattr(cycle.subprocess,'Popen',execute)
    if fault=='deadline':monkeypatch.setattr(cycle,'wait_owned',lambda *a,**kw:'original_session_deadline')
    with pytest.raises(RuntimeError):cycle.run(root,collection,output,10,5,4,123)
    assert len(calls)==1
    assert (ledger/'failure.json').exists() and (ledger/'aggregate.json').exists()
    assert not (ledger/'finished.json').exists()
    assert json.loads((ledger/'status.json').read_text())['stage']=='failed'
    failure=json.loads((ledger/'failure.json').read_text())
    assert len(failure['stages'])==1
    with pytest.raises(FileExistsError):cycle.run(root,collection,output,10,5,4,123)


@pytest.mark.parametrize('exact_tail',[False,True])
def test_exact_tail_profile_reaches_registered_evaluation_only(tmp_path,monkeypatch,exact_tail):
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    root=tmp_path/'source';root.mkdir();collection=tmp_path/'collection';collection.mkdir()
    output=tmp_path/'next';ledger=tmp_path/'next_stages';calls=[]
    (root/'session.json').write_text(json.dumps(dict(deadline=time.time()+1000)))
    (collection/'result.json').write_text(json.dumps(dict(software_complete=True)))
    (collection/'correction_entries.json').write_text(json.dumps({'real':{}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cycle.subprocess,'check_output',lambda *args,**kwargs:'commit')
    def emit(path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(complete=True)))
    def launch(command,**kwargs):
        calls.append(command);module=command[2]
        if module=='tools.tight_fresh_dagger':
            emit(output/('prepared.json' if command[3]=='aggregate' else 'B/cache_index.json'))
        elif module=='tools.tight_fresh_train':
            role=command[command.index('--role')+1];emit(output/role/'training/finished.json')
        elif module=='tools.tight_fresh_backend':
            role=command[command.index('--role')+1];emit(output/role/'backend/prepared.json')
        elif module=='tools.tight_finish':
            rear=Path(command[command.index('--root')+1]);emit(rear/'cache_complete.json')
        elif module=='visuotactile.anchor_continue':
            destination=Path(command[command.index('--output')+1])
            emit(destination/('finished.json' if command[3]=='residual-train' else 'result.json'))
        else:
            assert module=='tools.tight_fresh_batch_run' and command[3]=='prepare'
            assert ('--vision-exact-tail' in command)==exact_tail
            raise RuntimeError('CPU evaluation prepare boundary')
        return SimpleNamespace(pid=os.getpid(),returncode=0,poll=lambda:0)
    from pathlib import Path
    monkeypatch.setattr(cycle.subprocess,'Popen',launch)
    options=dict(vision_tail_profile=EXACT_TAIL_PROFILE) if exact_tail else {}
    with pytest.raises(RuntimeError,match='CPU evaluation prepare boundary'):
        cycle.run(root,collection,output,2,1,4,1,evaluation_lanes=8,vision_microbatch=8,**options)
    registration=json.loads((ledger/'registration.json').read_text())
    if exact_tail:assert registration['vision_tail_profile']==EXACT_TAIL_PROFILE
    else:assert 'vision_tail_profile' not in registration
    assert all('--vision-exact-tail' not in command for command in calls[:-1])


def test_exact_tail_cannot_silently_apply_to_serial_cycle(tmp_path):
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    with pytest.raises(ValueError,match='batch evaluation'):
        cycle.run(tmp_path,tmp_path,tmp_path/'next',2,1,4,1,vision_tail_profile=EXACT_TAIL_PROFILE)
    assert not (tmp_path/'next_stages').exists()
