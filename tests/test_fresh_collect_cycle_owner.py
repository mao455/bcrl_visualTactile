"""Finite CPU subprocesses exercise the durable collection-to-cycle boundary."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools import tight_fresh_collect_cycle_owner as owner
from visuotactile.compact import sha256


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def registration(tmp_path,monkeypatch):
    session=tmp_path/'session';data=session/'pool';collection=session/'collection';output=session/'round';tree=tmp_path/'frozen';ledger=session/'owner'
    for p in (session,data,collection,tree,ledger):p.mkdir(parents=True,exist_ok=True)
    (tree/'.venv_isaacsim/bin').mkdir(parents=True);(tree/'.venv_isaacsim/bin/python').symlink_to(sys.executable)
    s=dict(deadline=time.time()+12000)
    for p in (session,data,collection):write(p/'session.json',s)
    for role in ('A','B'):write(data/role/'state_data/data.json',dict(train=['original'],role=role))
    pairs={role:dict(frontend_sha256='front'+role,backend_sha256='rear'+role) for role in ('A','B')}
    plan=dict(kind='takeover',test_used=False,batch=dict(version='whole_wave_complete_student_v1'),resource_root=str(session),
        source_data_sha256={role:sha256(data/role/'state_data/data.json') for role in ('A','B')},
        source_commit='a'*40,source_sha256={},pairs=pairs,
        cases=[dict(name='case'+str(i),role='AB'[i%2],spec=dict(identity='case'+str(i),family='family'+str(i))) for i in range(8)])
    write(collection/'plan.json',plan)
    r=dict(version=owner.VERSION,session_root=str(session),data_source=str(data),collection=str(collection),output=str(output),
        worker_tree=str(tree),worker_commit='a'*40,session_sha256=sha256(session/'session.json'),
        collection_plan_sha256=sha256(collection/'plan.json'),deadline=s['deadline'],collection_max_seconds=7200,
        minimum_cycle_remaining_seconds=3600,minimum_accepted=8,
        collection_command=[str(tree/'.venv_isaacsim/bin/python'),'-m','tools.tight_batch_owner','--output',str(collection),'--worker-tree',str(tree)],
        cycle_command=[str(tree/'.venv_isaacsim/bin/python'),'-m','tools.tight_fresh_cycle','--root',str(data),'--collection',str(collection),'--output',str(output)])
    path=ledger/'registration.json';write(path,r)
    monkeypatch.setattr(owner,'validate_plan',lambda plan:None)
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *args,**kwargs:'a'*40+'\n')
    return path


def finish_collection(context,accepted=8):
    p=context['paths']['collection'];plan=context['plan'];entries={};rows=[]
    for i,case in enumerate(plan['cases']):
        result=dict(case['spec'],software_complete=True,legal=True,success=i<accepted,accepted_for_training=i<accepted,
            dagger=dict(expert_valid_ticks=5),dataset_capture=dict(accepted_success=True,verified_source_fields=185,
                path=str(p/case['name']/'demonstration/data.h5'),sha256='capture'+str(i)))
        write(p/case['name']/'result.json',result);rows.append(dict(case=case['name'],role=case['role'],result=result))
        if i<accepted:entries[case['name']]=dict(episode=case['name'],split='train',family=case['spec']['family'],
            source_group=case['role'],path=result['dataset_capture']['path'],
            source_sha256=result['dataset_capture']['sha256'],**plan['pairs'][case['role']])
    write(p/'results.json',rows);write(p/'correction_entries.json',entries)
    write(p/'result.json',dict(software_complete=True,cases=8))
    write(p/'owner_finished.json',dict(software_complete=True,returncode=0,plan_sha256=sha256(p/'plan.json')))


def test_validate_is_read_only_and_rejects_reused_identity(registration):
    before={p:p.read_bytes() for p in registration.parent.rglob('*') if p.is_file()}
    c=owner.validate(registration)
    assert before=={p:p.read_bytes() for p in registration.parent.rglob('*') if p.is_file()}
    write(c['ledger']/'owner_started.json',{})
    with pytest.raises(ValueError,match='restart'):owner.validate(registration)


@pytest.mark.parametrize('fault',['deadline','budget','minimum','command','data','source','duplicate','existing'])
def test_registration_source_budget_and_command_guards(registration,fault):
    r=owner.read(registration)
    if fault=='deadline':r['deadline']+=100
    elif fault=='budget':r['collection_max_seconds']=7201
    elif fault=='minimum':r['minimum_cycle_remaining_seconds']=3599
    elif fault=='command':r['cycle_command'][2]='arbitrary.module'
    elif fault=='data':write(Path(r['data_source'])/'A/state_data/data.json',dict(changed=True))
    elif fault=='source':r['worker_commit']='b'*40
    elif fault=='duplicate':r['cycle_command']+=['--root',r['data_source']]
    elif fault=='existing':Path(r['output']).mkdir()
    write(registration,r)
    with pytest.raises(ValueError):owner.validate(registration)


@pytest.mark.parametrize('fault',['missing_case','unlisted_entry','capture_sha','wrong_pair','missing_expert','row_drift'])
def test_admission_requires_every_opportunity_and_verified_success_binding(registration,fault):
    c=owner.validate(registration);finish_collection(c);p=c['paths']['collection']
    if fault=='missing_case':(p/'case7/result.json').unlink()
    elif fault=='unlisted_entry':
        e=owner.read(p/'correction_entries.json');e['extra']=e['case0'];write(p/'correction_entries.json',e)
    elif fault=='capture_sha':
        e=owner.read(p/'correction_entries.json');e['case0']['source_sha256']='drift';write(p/'correction_entries.json',e)
    elif fault=='wrong_pair':
        e=owner.read(p/'correction_entries.json');e['case0']['frontend_sha256']='other';write(p/'correction_entries.json',e)
    elif fault=='missing_expert':
        rows=owner.read(p/'results.json');rows[0]['result']['dagger']['expert_valid_ticks']=0
        write(p/'case0/result.json',rows[0]['result']);write(p/'results.json',rows)
    else:
        rows=owner.read(p/'results.json');rows[0]['result']['success']=False;write(p/'results.json',rows)
    with pytest.raises((ValueError,FileNotFoundError)):owner.collection_admission(c,time.time())


def test_admission_keeps_minimum_count_and_work_time(registration):
    c=owner.validate(registration);finish_collection(c,7)
    assert owner.collection_admission(c,time.time())['reason']=='too_few_accepted'
    finish_collection(c,8)
    r=c['registration']
    assert owner.collection_admission(c,r['deadline']-owner.CLEANUP_SECONDS-3600)['admitted']
    assert owner.collection_admission(c,r['deadline']-owner.CLEANUP_SECONDS-3599)['reason']=='insufficient_cycle_time'


def test_durable_sequence_and_no_second_start(registration,monkeypatch):
    calls=[]
    def stage(name,command,c,deadline,stops):
        calls.append(name)
        if name=='collection':finish_collection(c)
        else:write(c['paths']['output'].with_name('round_stages')/'finished.json',dict(software_complete=True))
        return dict(stage=name,complete=True,returncode=0,deadline=deadline)
    monkeypatch.setattr(owner,'run_stage',stage)
    result=owner.run(registration)
    assert calls==['collection','cycle'] and result['software_complete'] and result['cycle_state']=='complete'
    assert result['stages'][1]['completion_sha256']
    assert owner.read(registration.parent/'owner_started.json')['identity']['start_ticks']>0
    assert len(owner.read(registration.parent/'partial_census.json'))==8
    with pytest.raises(ValueError,match='restart'):owner.run(registration)


@pytest.mark.parametrize('reason',['too_few','failed','stop','expired'])
def test_cycle_not_started_is_durable(registration,monkeypatch,reason):
    calls=[]
    if reason=='stop':Path(owner.read(registration)['session_root']).joinpath('STOP').touch()
    if reason=='expired':monkeypatch.setattr(owner.time,'time',lambda:owner.read(registration)['deadline']-3500)
    def stage(name,command,c,deadline,stops):
        calls.append(name);assert name=='collection'
        finish_collection(c,7 if reason=='too_few' else 8)
        return dict(stage=name,complete=reason!='failed',returncode=1 if reason=='failed' else 0)
    monkeypatch.setattr(owner,'run_stage',stage)
    result=owner.run(registration)
    assert result['cycle_state']=='not_started' and not result['software_complete']
    assert calls==([] if reason in ('stop','expired') else ['collection'])
    assert owner.read(registration.parent/'owner_finished.json')==result


def test_real_subprocess_exit_and_stop_cleanup_independent_session_descendant(registration):
    c=owner.validate(registration);child_file=c['ledger']/'child.json'
    code="""import json,os,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
open(sys.argv[1],'w').write(json.dumps({'pid':p.pid}))
time.sleep(30)
"""
    result=owner.run_stage('collection',[sys.executable,'-c',code,str(child_file)],c,time.time()+.5,())
    assert result['stop_reason']=='stage_work_deadline' and not result['complete']
    child=owner.read(child_file)['pid'];identity=owner.process_identity(child)
    assert identity is None or identity['state']=='Z'
    assert len(result['cleanup'])>=2 and all(row['live_descendants_exited'] for row in result['cleanup'])
    assert owner.read(c['ledger']/'collection_finished.json')['returncode'] is not None


def test_real_subprocess_stop_marker_and_failure_are_recorded(registration):
    c=owner.validate(registration);stop=c['ledger']/'STOP';stop.touch()
    result=owner.run_stage('collection',[sys.executable,'-c','import time;time.sleep(30)'],c,time.time()+20,(stop,))
    assert result['stop_reason']=='requested_stop' and result['returncode']<0
    result=owner.run_stage('cycle',[sys.executable,'-c','import time;time.sleep(.1);raise SystemExit(7)'],c,time.time()+20,())
    assert result['returncode']==7 and not result['complete']


def test_persisted_inner_group_rejects_reused_pid(registration):
    c=owner.validate(registration);binding=owner.process_identity(os.getpid())
    write(c['paths']['collection']/'owner_worker_started.json',dict(worker_group=dict(binding,
        start_ticks=binding['start_ticks']+1,pgid=binding['pid'],sid=binding['pid'])))
    groups=owner.discover_groups(binding,c)
    assert groups[binding['pid']]['start_ticks']==binding['start_ticks']
