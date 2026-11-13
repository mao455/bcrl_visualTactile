import json
from pathlib import Path
import pytest
from tools import tight_research_supervisor as watch


def test_process_identity_rejects_pid_reuse(monkeypatch):
    monkeypatch.setattr(watch,'process',lambda pid:dict(pid=pid,start_ticks=20,state='R'))
    assert watch.same_process(dict(pid=123,start_ticks=10)) is None
    assert watch.running(dict(pid=123,start_ticks=20))
    monkeypatch.setattr(watch,'process',lambda pid:dict(pid=pid,start_ticks=20,state='Z'))
    assert not watch.running(dict(pid=123,start_ticks=20))


def test_telemetry_is_not_research_progress(tmp_path):
    before=watch.progress_signature(tmp_path)
    (tmp_path/'shared_resources.json').write_text('{}')
    assert watch.progress_signature(tmp_path)==before
    p=tmp_path/'formal/A/training';p.mkdir(parents=True);(p/'loss.jsonl').write_text('{}\n')
    assert watch.progress_signature(tmp_path)!=before


def test_no_progress_and_completed_event_are_deduplicated(tmp_path):
    plan=dict(id='stage1',root=str(tmp_path),completion_files=['result.json'],stall_seconds=20)
    assert watch.choose_event(plan,tmp_path,[],19,0) is None
    event=watch.choose_event(plan,tmp_path,[],21,0)
    assert event['kind']=='suspected_no_progress'
    assert watch.choose_event(plan,tmp_path,[event['key']],30,0) is None
    (tmp_path/'result.json').write_text('{"complete":true}')
    done=watch.choose_event(plan,tmp_path,[],21,0)
    assert done['kind']=='stage_result'


def test_watch_cannot_escape_authorized_run(tmp_path):
    plan=dict(id='a',root=str(tmp_path/'run'),completion_files=['result.json'])
    assert watch.validate_watch(plan,tmp_path)==tmp_path/'run'
    with pytest.raises(ValueError):watch.validate_watch(dict(plan,root='/etc'),tmp_path)
    with pytest.raises(ValueError):watch.validate_watch(dict(plan,completion_files=['../foreign.json']),tmp_path)
    with pytest.raises(ValueError):watch.validate_watch(dict(plan,status_file='../foreign.json'),tmp_path)
    with pytest.raises(ValueError):watch.validate_watch(dict(plan,owner_identity=dict(pid=1,start_ticks=1)),tmp_path)


def test_pinned_owner_exit_detected_without_launch_file(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,'process',lambda pid:None)
    plan=dict(id='newstage',root=str(tmp_path),owner_identity=dict(pid=123,start_ticks=456))
    assert watch.choose_event(plan,tmp_path,[],1,0)['kind']=='unexpected_owner_exit'


def test_final_eval_parks_coordinator_and_waits_for_child_exit(tmp_path,monkeypatch):
    root=tmp_path/'run';supervisor=tmp_path/'supervisor';root.mkdir();supervisor.mkdir()
    owner=dict(pid=10,start_ticks=1,state='R',ppid=1);child=dict(pid=20,start_ticks=2,state='R',ppid=10)
    processes={10:owner,20:child};signals=[]
    monkeypatch.setattr(watch,'process',lambda pid:processes.get(pid))
    monkeypatch.setattr(watch.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    watch.write(supervisor/'registration.json',dict(initial_owner=owner))
    watch.write(root/'follow_launch.json',dict(pid=10))
    watch.write(root/'active_stage.json',dict(name='G01_formal_eval',process=child,deadline=watch.time.time()+100))
    watch.write(root/'formal/QG01/backend/evaluation_complete.json',dict(complete=True))
    watch.write(root/'formal/G01/backend/G01/offline/result.json',dict(complete=True))
    assert watch.park_initial_owner(supervisor,root,watch.time.time()+100) is None
    assert signals==[(10,watch.signal.SIGSTOP)]
    child['state']='Z'
    assert watch.park_initial_owner(supervisor,root,watch.time.time()+100) is None
    assert signals[-2:]==[(10,watch.signal.SIGTERM),(10,watch.signal.SIGCONT)]
    del processes[10]
    assert watch.park_initial_owner(supervisor,root,watch.time.time()+100)['kind']=='full_policy_boundary'
    assert all(pid==10 for pid,sig in signals)
