import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import tight_fresh_eval_recovery as recovery, tight_fresh_rollouts
from tools.tight_fresh_eval_recovery import run,stop_owned,wait_owned
from visuotactile.compact import sha256


def test_recovery_rejects_existing_control_initialization(tmp_path):
    previous=tmp_path/'previous';previous.mkdir();(previous/'initialization.json').write_text('{}')
    with pytest.raises(ValueError,match='pre-control'):
        run(tmp_path/'root',previous,tmp_path/'next',tmp_path,180)


def test_owned_startup_process_is_reaped():
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        stop_owned(process)
        assert process.returncode is not None
        stop_owned(process)
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_missing_startup_marker_stops_owned_group(tmp_path):
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    reason=wait_owned(process,time.time()+5,startup_marker=tmp_path/'init',startup_seconds=.1)
    assert reason=='pre_control_startup_timeout'
    assert process.returncode is not None


def test_initialized_worker_can_outlive_startup_window(tmp_path):
    marker=tmp_path/'init';marker.write_text('{}')
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.15)'],start_new_session=True)
    reason=wait_owned(process,time.time()+5,startup_marker=marker,startup_seconds=.05)
    assert reason is None
    assert process.returncode==0


def test_absolute_deadline_applies_after_initialization(tmp_path):
    marker=tmp_path/'init';marker.write_text('{}')
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    reason=wait_owned(process,time.time()+.1,startup_marker=marker,startup_seconds=5)
    assert reason=='original_session_deadline'
    assert process.returncode is not None


@pytest.fixture
def registered_plan(tmp_path,monkeypatch):
    root=tmp_path/'root';previous=tmp_path/'previous';source=tmp_path/'frozen';output=tmp_path/'recovery'
    for path in (root,previous,source/'tools',source/'visuotactile'):path.mkdir(parents=True)
    write=lambda path,value:path.write_text(json.dumps(value,indent=2)+'\n')
    session=dict(deadline=time.time()+1000,domain={'fixed':'original'},expert_checkpoint='expert.pt')
    write(root/'session.json',session);write(previous/'session.json',session)
    for filename in ('tools/worker.py','visuotactile/controller.py'):
        (source/filename).write_text('# frozen worker\n')
    pairs={role:dict(frontend=role+'_dp.pt',frontend_sha256=role+'_dp_hash',
        backend=role+'_rear.pt',backend_sha256=role+'_rear_hash') for role in ('A','B')}
    cases=[dict(name=role+'_train_'+str(index),role=role,spec=dict(identity='training_'+str(index),
        seed=100+27*index,event_seed=500+19*index,cohort='fresh',split='train',
        diagnostic_cohort='original_training_diagnostic',narrow_reset={'fixed':index}),video=index==0)
        for index in range(2) for role in (('A','B') if index==0 else ('B','A'))]
    old=dict(kind='autonomous',pairs=pairs,cases=cases,source_commit='frozen_commit',
        source_sha256={str(p.relative_to(source)):sha256(p) for p in source.rglob('*.py')},
        inference_scheduler_index_mode='host',cohort='training diagnostic; no exact replay',test_used=False)
    write(previous/'plan.json',old)
    write(previous/'run_started.json',dict(pid=999999999,plan_sha256=sha256(previous/'plan.json')))
    monkeypatch.setattr(tight_fresh_rollouts,'completed_pair',lambda path,role:copy.deepcopy(pairs[role]))
    monkeypatch.setattr(recovery.subprocess,'check_output',lambda *args,**kwargs:
        'frozen_commit\n' if kwargs['cwd']==source else 'owner_commit\n')
    calls=[]
    def child(command,**kwargs):
        calls.append(dict(command=command,**{k:v for k,v in kwargs.items() if k in ('cwd','env','start_new_session')}))
        write(output/'initialization.json',{'seconds':1})
        write(output/'result.json',{'software_complete':True,'cases':len(cases)})
        return SimpleNamespace(pid=123456,returncode=0,poll=lambda:0)
    monkeypatch.setattr(recovery.subprocess,'Popen',child)
    monkeypatch.setattr(recovery.subprocess,'run',lambda *a,**kw:pytest.fail('Registered plan must not regenerate development cases'))
    return SimpleNamespace(root=root,previous=previous,source=source,output=output,old=old,pairs=pairs,
        cases=cases,session=session,write=write,calls=calls)


def test_preserved_plan_reuses_exact_training_cases_pairs_source_and_deadline(registered_plan):
    x=registered_plan
    original_bytes=(x.previous/'plan.json').read_bytes()
    run(x.root,x.previous,x.output,x.source,180,preserve_registered_plan=True)
    assert (x.output/'plan.json').read_bytes()==original_bytes
    assert (x.output/'session.json').read_bytes()==(x.previous/'session.json').read_bytes()
    assert len(x.calls)==1 and x.calls[0]['cwd']==x.source
    assert x.calls[0]['command'][0]==str(x.source/'.venv_isaacsim/bin/python')
    assert x.calls[0]['start_new_session'] and x.calls[0]['command'][-1]==str(x.output)
    registration=json.loads((x.output/'recovery_registration.json').read_text())
    assert registration['preserve_registered_plan']
    assert registration['prior_plan_sha256']==registration['plan_sha256']==sha256(x.previous/'plan.json')
    assert registration['deadline']==x.session['deadline'] and registration['startup_seconds']==180
    assert registration['frozen_execution_source_commit']=='frozen_commit'
    assert registration['owner_source_commit']=='owner_commit'
    assert registration['owner_script_sha256']==sha256(recovery.__file__)
    assert registration['verified_source_files']==2
    assert json.loads((x.output/'recovery_finished.json').read_text())['software_complete']


@pytest.mark.parametrize('fault',['initialized','started_case','partial_case','completed_case','results',
    'live_worker','takeover','pair','source_bytes','source_extra','source_missing','source_commit',
    'plan_changed','session_changed','deadline_changed','startup_bound'])
def test_preserved_plan_rejects_changed_or_started_dependencies(registered_plan,monkeypatch,fault):
    x=registered_plan;old=copy.deepcopy(x.old);bound=180
    if fault=='initialized': x.write(x.previous/'initialization.json',{})
    elif fault in ('started_case','partial_case','completed_case'):
        directory=x.previous/x.cases[0]['name'];directory.mkdir()
        if fault!='partial_case':x.write(directory/('started.json' if fault=='started_case' else 'result.json'),{})
    elif fault=='results':x.write(x.previous/'results.json',[{'case':x.cases[0]['name']}])
    elif fault=='live_worker':x.write(x.previous/'run_started.json',dict(pid=os.getpid(),plan_sha256=sha256(x.previous/'plan.json')))
    elif fault=='takeover':old['kind']='takeover'
    elif fault=='pair':monkeypatch.setattr(tight_fresh_rollouts,'completed_pair',lambda *a:{'different':'pair'})
    elif fault=='source_bytes':(x.source/'tools/worker.py').write_text('# changed\n')
    elif fault=='source_extra':(x.source/'tools/extra.py').write_text('# extra\n')
    elif fault=='source_missing':(x.source/'tools/worker.py').unlink()
    elif fault=='source_commit':old['source_commit']='other_commit'
    elif fault=='plan_changed':old['cases'][0]['spec']['seed']+=1
    elif fault in ('session_changed','deadline_changed'):
        changed=copy.deepcopy(x.session)
        if fault=='deadline_changed':changed['deadline']+=100
        else:changed['domain']['fixed']='other'
        x.write(x.root/'session.json',changed)
    elif fault=='startup_bound':bound=181
    if old!=x.old:
        x.write(x.previous/'plan.json',old)
        if fault!='plan_changed':
            x.write(x.previous/'run_started.json',dict(pid=999999999,plan_sha256=sha256(x.previous/'plan.json')))
    with pytest.raises((ValueError,TimeoutError)):
        run(x.root,x.previous,x.output,x.source,bound,preserve_registered_plan=True)
    assert not x.output.exists() and not x.calls


def test_default_development_recovery_still_regenerates_plan(registered_plan,monkeypatch):
    x=registered_plan;prepared=[]
    def prepare(command,**kwargs):
        prepared.append(command);x.output.mkdir()
        x.write(x.output/'plan.json',x.old);x.write(x.output/'session.json',x.session)
    monkeypatch.setattr(recovery.subprocess,'run',prepare)
    run(x.root,x.previous,x.output,x.source,120)
    assert len(prepared)==1 and 'prepare' in prepared[0]
    assert prepared[0][prepared[0].index('--count')+1]=='2'
    assert prepared[0][prepared[0].index('--seed')+1]=='100'
    registration=json.loads((x.output/'recovery_registration.json').read_text())
    assert registration['startup_seconds']==120 and 'preserve_registered_plan' not in registration
