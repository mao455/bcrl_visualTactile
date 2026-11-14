"""Real takeover registration across an expired source and active resource session."""
import json
import sys
import time
from types import SimpleNamespace

import h5py
import pytest

from tools import tight_fresh_batch_run as batch
from tools import tight_fresh_stage_takeover as stage
from visuotactile.compact import sha256


@pytest.fixture
def setup(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    active=tmp_path/'active';active.mkdir()
    expired=dict(deadline=time.time()-10,expert_sha256='expert')
    current=dict(deadline=time.time()+1000,expert_sha256='expert')
    (source/'session.json').write_text(json.dumps(expired))
    (active/'session.json').write_text(json.dumps(current))
    original=tmp_path/'expert.h5'
    with h5py.File(original,'w') as stream:
        stream.attrs['result_json']=json.dumps(dict(success=True,seed=91,narrow_reset={'x':1}))
    identities=[f'original{i}' for i in range(30)]
    data=dict(normalization_train_ids=identities,splits=dict(train=identities,validation=[],test=[]),
        entries={identity:dict(family=identity,path=str(original),source_sha256=sha256(original))
            for identity in identities})
    for role in ('A','B'):
        directory=source/role/'state_data';directory.mkdir(parents=True)
        (directory/'data.json').write_text(json.dumps(data))
    pair=dict(frontend='source_front.pt',frontend_sha256='front',backend='source_back.pt',backend_sha256='back')
    monkeypatch.setattr(stage,'completed_pair',lambda root,role:dict(pair,role=role))
    options=dict(lanes=8,identities=identities[:8],delays=[12,0,12,0,12,0,0,0],
        round_index=7,seed=28140700,vision_microbatch=8,writer_bytes=512<<20,
        verification_workers=2,verification_max_pending=4)
    return SimpleNamespace(source=source,active=active,output=tmp_path/'collection',
        options=options,current=current,expired=expired)


def test_active_session_registers_expired_source_without_changing_either(setup):
    x=setup
    before={path:path.read_bytes() for path in (x.source/'session.json',x.active/'session.json')}
    with pytest.raises(ValueError,match='deadline'):
        batch.prepare(x.source,x.output,**x.options)
    assert not x.output.exists()
    plan=batch.prepare(x.source,x.output,**x.options,session_root=x.active)
    batch.validate_takeover_plan(plan)
    assert len(plan['cases'])==16 and {case['role'] for case in plan['cases']}=={'A','B'}
    assert plan['source_root']==str(x.source) and plan['resource_root']==str(x.active)
    assert (x.output/'session.json').read_bytes()==before[x.active/'session.json']
    assert sha256(x.output/'session.json')==sha256(x.active/'session.json')==plan['source_session_sha256']
    assert plan['batch']['vision_microbatch']==8 and plan['batch']['writer_bytes']==512<<20
    assert plan['batch']['verification_workers']==2 and plan['batch']['verification_max_pending']==4
    assert all(path.read_bytes()==value for path,value in before.items())
    assert not (x.output/'run_started.json').exists()


@pytest.mark.parametrize('changed',['copied','resource'])
def test_changed_registered_session_is_rejected_before_simulator(setup,monkeypatch,changed):
    x=setup
    batch.prepare(x.source,x.output,**x.options,session_root=x.active)
    path=(x.output if changed=='copied' else x.active)/'session.json'
    path.write_text(json.dumps(dict(x.current,deadline=x.current['deadline']+3600)))
    import tools.tight_state as state
    def forbidden(*args,**kwargs):
        pytest.fail('Session drift reached the simulator')
    monkeypatch.setattr(state,'environment',forbidden)
    with pytest.raises(ValueError,match='Registered session bytes changed'):
        batch.run(x.output)
    assert not (x.output/'run_started.json').exists()


def test_autonomous_cannot_silently_ignore_independent_session(setup):
    x=setup
    with pytest.raises(ValueError,match='training takeovers only'):
        batch.prepare(x.source,x.output,lanes=8,session_root=x.active)
    assert not x.output.exists()


def test_cli_forwards_explicit_session_and_rejects_runtime_override(setup,monkeypatch):
    x=setup;calls=[]
    monkeypatch.setattr(batch,'prepare',lambda *args,**kwargs:calls.append((args,kwargs)))
    monkeypatch.setattr(sys,'argv',['batch','prepare','--root',str(x.source),'--output',str(x.output),
        '--session-root',str(x.active),'--identities','original0','--delays','0','--lanes','8',
        '--verification-workers','2','--verification-max-pending','4'])
    batch.main()
    assert calls[0][1]['session_root']==x.active
    assert calls[0][1]['verification_workers']==2 and calls[0][1]['verification_max_pending']==4
    monkeypatch.setattr(sys,'argv',['batch','run','--output',str(x.output),'--session-root',str(x.active)])
    with pytest.raises(SystemExit) as error:batch.main()
    assert error.value.code==2


@pytest.mark.parametrize('workers,pending',[(True,4),(-1,4),(2,1),(0,0),(0,False)])
def test_invalid_verifier_budgets_fail_before_registration(setup,workers,pending):
    x=setup
    options=dict(x.options,verification_workers=workers,verification_max_pending=pending)
    with pytest.raises(ValueError):batch.prepare(x.source,x.output,**options,session_root=x.active)
    assert not x.output.exists()
