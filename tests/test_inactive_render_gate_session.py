"""Actual gate environment session prerequisite, without Kit or CUDA."""
import json
from pathlib import Path
import time

import pytest

from tools import tight_inactive_render_gate as gate


@pytest.fixture
def registered(tmp_path):
    resource=tmp_path/'resource';resource.mkdir()
    output=tmp_path/'gate';output.mkdir()
    directory=output/'ordinary';directory.mkdir()
    deadline=time.time()+500
    # Deliberate whitespace/order: parsing then atomic_json would change SHA.
    raw=(' { "expert_checkpoint" : "existing_expert.pt", "expert_sha256":"bound",\n'
        '"precision_policy":"unified_fp32", "deadline" : '+repr(deadline)+' }\n\n').encode()
    (resource/'session.json').write_bytes(raw)
    plan=dict(resource_root=str(resource),session_sha256=gate.sha256(resource/'session.json'),
        deadline=deadline-100,work_deadline=deadline-130,execution_backend={})
    (output/'plan.json').write_text(json.dumps(plan))
    return resource,output,directory,plan,raw


def test_session_copy_is_byte_exact_preserves_deadlines_and_cannot_overwrite(registered):
    resource,output,directory,plan,raw=registered
    original_plan=(output/'plan.json').read_bytes()
    receipt=gate.prepare_worker_session(plan,directory)
    assert (directory/'session.json').read_bytes()==raw==(resource/'session.json').read_bytes()
    assert (output/'plan.json').read_bytes()==original_plan
    assert receipt['session_deadline']==json.loads(raw)['deadline']
    assert receipt['gate_deadline']==plan['deadline'] and receipt['work_deadline']==plan['work_deadline']
    assert json.loads((directory/'session_binding.json').read_text())==receipt
    with pytest.raises(FileExistsError):gate.prepare_worker_session(plan,directory)
    assert (directory/'session.json').read_bytes()==raw


@pytest.mark.parametrize('location',['source','local','symlink','deadline'])
def test_resource_local_drift_and_deadline_change_fail_closed(registered,location):
    resource,_,directory,plan,raw=registered
    gate.prepare_worker_session(plan,directory)
    if location=='source':(resource/'session.json').write_bytes(raw+b' ')
    elif location=='local':(directory/'session.json').write_bytes(raw+b' ')
    elif location=='symlink':
        (directory/'session.json').unlink()
        (directory/'session.json').symlink_to(resource/'session.json')
    else:plan['work_deadline']+=1
    with pytest.raises(ValueError):gate.verify_worker_session(plan,directory)


def test_changed_source_is_rejected_before_any_local_copy(registered):
    resource,_,directory,plan,raw=registered
    (resource/'session.json').write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='before worker copy'):gate.prepare_worker_session(plan,directory)
    assert not (directory/'session.json').exists() and not (directory/'session_binding.json').exists()


def test_worker_copies_after_plan_validation_and_before_environment(registered,monkeypatch):
    resource,output,directory,plan,raw=registered
    from tools import tight_state
    from visuotactile import state_resources
    import torch
    calls=[]
    def verify(actual,tree):
        assert actual==plan
        assert not (directory/'session.json').exists()
        calls.append('plan')
    monkeypatch.setattr(gate,'verify_plan',verify)
    def lease(root,*,deadline):
        assert root==resource and deadline==plan['work_deadline']
        assert (directory/'session.json').read_bytes()==raw
        calls.append('lease')
        return 'original-resource-lease'
    monkeypatch.setattr(state_resources,'lease_for',lease)
    def environment(root,**kwargs):
        assert root==directory and kwargs['lease']=='original-resource-lease'
        assert (directory/'session.json').read_bytes()==raw
        assert (directory/'session_binding.json').exists() and (directory/'run_started.json').exists()
        assert kwargs['evaluation'] is True and kwargs['num_envs']==16
        calls.append('environment')
        raise RuntimeError('CPU stop before Kit')
    monkeypatch.setattr(tight_state,'environment',environment)
    with pytest.raises(RuntimeError,match='CPU stop before Kit'):gate.worker(resource,output,'ordinary')
    assert calls==['plan','lease','environment'] and not torch.cuda.is_initialized()
    assert not (directory/'initialization.json').exists()


def test_invalid_plan_never_creates_local_session_or_run_marker(registered,monkeypatch):
    resource,output,directory,_,_=registered
    def reject(*args):raise ValueError('Frozen source mismatch')
    monkeypatch.setattr(gate,'verify_plan',reject)
    with pytest.raises(ValueError,match='Frozen source'):gate.worker(resource,output,'ordinary')
    assert not (directory/'session.json').exists() and not (directory/'run_started.json').exists()
