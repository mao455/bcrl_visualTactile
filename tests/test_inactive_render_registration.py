"""Explicit rendering/memo registration only: no Kit or model inference."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from visuotactile.inactive_render_products import PROFILE
from test_training_expansion import setup_registry,write


@pytest.mark.parametrize('backend',[dict(physics_device='cuda:0'),dict(camera_backend='tiled')])
def test_incompatible_backend_rejected_before_any_output(tmp_path,backend):
    from tools.tight_fresh_batch_run import prepare,validate_inactive_render_request
    assert validate_inactive_render_request(None,backend) is None
    with pytest.raises(ValueError,match='CPU standard'):
        prepare(tmp_path,tmp_path/'bad',lanes=16,execution_backend=backend,inactive_render_profile=PROFILE)
    assert not (tmp_path/'bad').exists()


def test_unknown_profile_rejected_before_any_source_read(tmp_path):
    from tools import tight_fresh_batch_run as formal,tight_fresh_expansion as expansion
    from tools import tight_fresh_cycle as cycle
    calls=[lambda:formal.prepare(tmp_path,tmp_path/'bad',lanes=16,inactive_render_profile={}),
        lambda:expansion.prepare(tmp_path,tmp_path/'bad',session_root=tmp_path,registry_id='bad',seeds=[1],
            partition_manifest=tmp_path/'absent',frozen_families=tmp_path/'absent',schedules=['fixed40'],
            lanes=16,event_seed=2,round_index=9,inactive_render_profile={}),
        lambda:cycle.run(tmp_path,tmp_path,tmp_path/'bad',1,1,1,1,evaluation_lanes=16,inactive_render_profile={})]
    for call in calls:
        with pytest.raises(ValueError):call()
        assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('enabled',[False,True])
def test_formal_prepare_profile_is_absent_by_default(tmp_path,monkeypatch,enabled):
    from tools import tight_fresh_batch_run as formal,tight_fresh_rollouts as serial
    def prepare(root,output,count,seed):
        output.mkdir();write(output/'plan.json',dict(kind='autonomous'))
        write(output/'session.json',dict(expert_sha256='expert'))
    monkeypatch.setattr(serial,'prepare',prepare)
    plan=formal.prepare(tmp_path,tmp_path/'out',lanes=16,
        **(dict(inactive_render_profile=PROFILE) if enabled else {}))
    assert ('inactive_render_profile' in plan['batch'])==enabled
    if enabled:assert plan['batch']['inactive_render_profile']==PROFILE
    assert 'inactive_render_profile' not in plan['execution_backend']
    assert json.loads((tmp_path/'out/plan.json').read_text())==plan


@pytest.mark.parametrize('enabled',[False,True])
def test_expansion_registers_profile_with_existing_family_contract(setup_registry,monkeypatch,enabled):
    from tools import tight_fresh_expansion as expansion
    from visuotactile.training_expansion import validate_collection_plan
    import h5py
    setup=setup_registry
    write(setup['source']/'session.json',dict(deadline=time.time()+1000,domain=setup['domain'],expert_sha256='expert'))
    monkeypatch.setattr(expansion,'completed_pair',lambda root,role:dict(role=role))
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('Registration must not read trajectories'))
    plan=expansion.prepare(setup['source'],setup['root']/'prepared',session_root=setup['source'],
        registry_id='camera_fixture',seeds=[6000,6001],partition_manifest=setup['partition'],
        frozen_families=setup['families'],schedules=['fixed40','contact0'],lanes=16,event_seed=8000,round_index=9,
        **(dict(inactive_render_profile=PROFILE) if enabled else {}))
    assert ('inactive_render_profile' in plan['batch'])==enabled
    if enabled:assert plan['batch']['inactive_render_profile']==PROFILE
    validate_collection_plan(plan)


@pytest.mark.parametrize('enabled',[False,True])
def test_cycle_registers_only_optin_and_dispatches_to_appropriate_stage(tmp_path,monkeypatch,enabled):
    from tools import tight_fresh_cycle as cycle
    root=tmp_path/'source';collection=tmp_path/'collection';output=tmp_path/'next';calls=[]
    write(root/'session.json',dict(deadline=time.time()+1000))
    write(collection/'result.json',dict(software_complete=True));write(collection/'correction_entries.json',{'real':{}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cycle.subprocess,'check_output',lambda *a,**k:'commit')
    def emit(path):write(path,dict(complete=True))
    def launch(command,**kwargs):
        calls.append(command);module=command[2]
        if module=='tools.tight_fresh_dagger':emit(output/('prepared.json' if command[3]=='aggregate' else 'B/cache_index.json'))
        elif module=='tools.tight_fresh_train':emit(output/command[command.index('--role')+1]/'training/finished.json')
        elif module=='tools.tight_fresh_backend':emit(output/command[command.index('--role')+1]/'backend/prepared.json')
        elif module=='tools.tight_finish':emit(Path(command[command.index('--root')+1])/'cache_complete.json')
        elif module=='visuotactile.anchor_continue':
            emit(Path(command[command.index('--output')+1])/('finished.json' if command[3]=='residual-train' else 'result.json'))
        else:
            assert module=='tools.tight_fresh_batch_run' and command[3]=='prepare'
            assert ('--inactive-render-products' in command)==enabled
            raise RuntimeError('CPU prepare boundary')
        return SimpleNamespace(pid=os.getpid(),returncode=0,poll=lambda:0)
    monkeypatch.setattr(cycle.subprocess,'Popen',launch)
    options=dict(inactive_render_profile=PROFILE,payload_audit_memo=True) if enabled else {}
    with pytest.raises(RuntimeError,match='CPU prepare boundary'):
        cycle.run(root,collection,output,2,1,4,1,evaluation_lanes=16,vision_microbatch=8,**options)
    registration=json.loads((tmp_path/'next_stages/registration.json').read_text())
    assert ('inactive_render_profile' in registration)==enabled
    assert ('payload_audit_memo' in registration)==enabled
    for command in calls:
        is_cache=command[2:4]==['tools.tight_fresh_dagger','cache']
        assert ('--payload-audit-memo' in command)==(enabled and is_cache)
    assert all('--inactive-render-products' not in command for command in calls[:-1])


def test_cycle_serial_rejects_profile_and_memo_requires_boolean(tmp_path):
    from tools.tight_fresh_cycle import run
    with pytest.raises(ValueError,match='batch evaluation'):
        run(tmp_path,tmp_path,tmp_path/'out',1,1,1,1,inactive_render_profile=PROFILE)
    with pytest.raises(ValueError,match='boolean'):
        run(tmp_path,tmp_path,tmp_path/'out',1,1,1,1,payload_audit_memo=1)
    assert not (tmp_path/'out_stages').exists()
