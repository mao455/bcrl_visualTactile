"""The conditional handoff never mutates a live worker's frozen source."""
from pathlib import Path
import time

import pytest

from tools import tight_dagger as d


def registration(tmp_path, monkeypatch):
    root=tmp_path/'run';prepared=tmp_path/'prepared';deployment=tmp_path/'deployment'
    for p in (root,prepared,deployment):p.mkdir()
    request=dict(version='conditional_zero_success_v1',extension_seconds=21600,
        maximum_extra_rounds=1,prior_round=2,next_round=3,groups={'QG01':'A','G01':'B'},
        deployment_tree=str(deployment),prepared_source='reviewed',prepared_parent='previous',
        owner={'pid':123,'start_ticks':456},prior_deadline=time.time()+100,
        owner_service='only-owned-service')
    d.write(root/'conditional_extension_request.json',request)
    monkeypatch.setattr(d,'CODE',prepared)
    monkeypatch.setattr(d,'git',lambda *args:'reviewed' if args[0]=='rev-parse' else '')
    return root,deployment,request


def completed(root, request):
    for g in request['groups']:
        d.write(root/'round_02'/g/'backend/evaluation_complete.json',{'complete':True})


def test_boundary_waits_for_both_complete_receipts(tmp_path):
    request={'prior_round':2,'groups':['QG01','G01']}
    d.write(tmp_path/'round_02/QG01/backend/evaluation_complete.json',{'complete':True})
    assert not d.conditional_boundary_ready(tmp_path,request)
    d.write(tmp_path/'round_02/G01/backend/evaluation_complete.json',{'complete':False})
    assert not d.conditional_boundary_ready(tmp_path,request)
    d.write(tmp_path/'round_02/G01/backend/evaluation_complete.json',{'complete':True})
    assert d.conditional_boundary_ready(tmp_path,request)


def test_exited_incomplete_owner_is_not_zero_success_and_no_stop_is_written(tmp_path,monkeypatch):
    root,_,request=registration(tmp_path,monkeypatch)
    monkeypatch.setattr(d,'proc',lambda pid:None)
    with pytest.raises(ValueError,match='before both evaluations'):
        d.conditional_follow(root)
    assert not (root/'STOP').exists()
    assert d.read(root/'conditional_extension_status.json')['state']=='FAILED'


def test_source_drift_and_started_next_round_block_before_stop(tmp_path,monkeypatch):
    root,deployment,request=registration(tmp_path,monkeypatch);completed(root,request)
    monkeypatch.setattr(d,'proc',lambda pid:dict(pid=pid,start_ticks=456,state='S'))
    (deployment/'model.py').write_text('unreviewed edit')
    d.write(root/'source_freeze.json',{'model.py':'original hash'})
    with pytest.raises(ValueError,match='source drifted'):d.conditional_follow(root)
    assert not (root/'STOP').exists()
    d.write(root/'round_03/QG01/attempts.json',{'started':True})
    with pytest.raises(ValueError,match='Next round already started'):d.conditional_follow(root)
    assert not (root/'STOP').exists()


def test_boundary_stop_uses_registered_owner_then_requires_reaped_workers(tmp_path,monkeypatch):
    root,_,request=registration(tmp_path,monkeypatch);completed(root,request)
    d.write(root/'source_freeze.json',{})
    d.write(root/'status.json',{'owned_workers_exited':False})
    observations=iter([dict(pid=123,start_ticks=456,state='S'),None])
    monkeypatch.setattr(d,'proc',lambda pid:next(observations))
    with pytest.raises(ValueError,match='cleanup is incomplete'):d.conditional_follow(root)
    assert (root/'STOP').exists()
    receipt=d.read(root/'conditional_extension_boundary.json')
    assert receipt['owner']==request['owner'] and receipt['evaluation_complete'] is True
    assert not (root/'conditional_extension_deployment.json').exists()


def test_reused_pid_is_not_signalled_and_completed_handoff_is_one_shot(tmp_path,monkeypatch):
    root,_,request=registration(tmp_path,monkeypatch);completed(root,request)
    d.write(root/'source_freeze.json',{});d.write(root/'status.json',{'owned_workers_exited':False})
    monkeypatch.setattr(d,'proc',lambda pid:dict(pid=pid,start_ticks=999,state='S'))
    with pytest.raises(ValueError,match='cleanup is incomplete'):d.conditional_follow(root)
    assert not (root/'STOP').exists()
    d.write(root/'conditional_extension_follow_finished.json',{'state':'NOT_TRIGGERED'})
    with pytest.raises(ValueError,match='already resolved'):d.conditional_follow(root)


@pytest.mark.parametrize('triggered',[False,True])
def test_completed_handoff_deploys_before_launch_and_exports_decision_receipt(tmp_path,monkeypatch,triggered):
    from tools import tight_finish
    root,deployment,request=registration(tmp_path,monkeypatch);completed(root,request)
    d.write(root/'source_freeze.json',{});d.write(root/'status.json',{'owned_workers_exited':True})
    d.write(root/'worker_processes.json',[]);d.write(root/'public_git.json',{'head':'previous'})
    d.write(root/'session.json',{'compute_deadline':time.time()+100})
    monkeypatch.setattr(d,'proc',lambda pid:None)
    monkeypatch.setattr(tight_finish,'matrix_worker_absence',lambda workers:(True,[]))
    calls=[];head=['previous']
    def git_command(args,**kwargs):
        assert Path(kwargs['cwd'])==deployment
        calls.append(tuple(args))
        command=args[1:]
        if command[:2]==['rev-parse','HEAD']:return head[0]
        if command[:2]==['rev-parse','FETCH_HEAD']:return 'previous'
        if command[:2]==['remote','get-url']:return 'https://github.com/mao455/bcrl_visualTactile.git'
        if command[0]=='cherry-pick':head[0]='deployed'
        return ''
    def command(args,**kwargs):
        assert Path(kwargs['cwd'])==deployment
        calls.append(tuple(args))
        if 'dagger_conditional_extend' in args:
            assert head[0]=='deployed'
            assert any('push' in c for c in calls)
            d.write(root/'conditional_extension.json',{'triggered':triggered,'reason':'tested decision'})
        elif args[0]=='systemd-run':
            assert triggered and d.read(root/'conditional_extension.json')['triggered']
            d.write(root/'round_03/QG01/complete_policy_acceptance.json',{'passed':True})
        elif '-c' in args:
            finished=d.read(root/'conditional_extension_follow_finished.json')
            assert finished['triggered']==triggered
            assert finished['collection_startup_verified']==triggered
        else:raise AssertionError(args)
    monkeypatch.setattr(d.subprocess,'check_output',git_command)
    monkeypatch.setattr(d.subprocess,'run',command)
    d.conditional_follow(root)
    status=d.read(root/'conditional_extension_status.json')
    assert status['state']==('STARTUP_VERIFIED' if triggered else 'NOT_TRIGGERED')
    assert sum(c[0]=='systemd-run' for c in calls)==int(triggered)
    assert d.read(root/'public_git.json')['head']=='deployed'
    assert not (root/'STOP').exists()
