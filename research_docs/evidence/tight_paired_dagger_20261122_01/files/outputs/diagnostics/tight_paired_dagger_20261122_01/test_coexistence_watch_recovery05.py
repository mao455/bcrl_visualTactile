"""Synthetic CPU-only identity fixtures: no watcher schedules or GPU work."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from outputs.diagnostics.tight_paired_dagger_20261122_01 import coexistence_watch_recovery05 as w


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bound(tmp_path,monkeypatch):
    root=tmp_path/'run';owner=root/'owners/collection_recovery05';out=root/'collection_recovery05'
    out.mkdir(parents=True);owner.mkdir(parents=True)
    adapter=tmp_path/'collection_recovery05.py';adapter.write_text('exact registered source\n')
    monkeypatch.setattr(w,'ADAPTER',adapter)
    spec=dict(name='collection_recovery05',owner=owner,module=w.MODULE,operation='dagger',job=root/'collection_recovery05_job.json')
    command=['python','-m',w.MODULE,'worker','--output',str(out)]
    job=dict(version='tight_project_job_v1',operation='dagger',resource_root=str(owner),stages=[dict(entry='paired_collection_recovery05',args=['--output',str(out)])])
    owner_proc=dict(pid=100,ppid=1,created=100.,command=['python'],cwd=str(root))
    worker=dict(pid=201,ppid=100,created=101.,command=command,cwd=str(root))
    procs={100:owner_proc,201:worker}
    reg=dict(spec=spec,job=job,receipt=dict(job=copy.deepcopy(job),commands=[command]),owner_process=copy.deepcopy(owner_proc))
    save(owner/'project_status.json',dict(stage=0,pid=201,utc=102.,command=command))
    save(out/'plan.json',dict(resource_root=str(root),cases=[]))
    save(out/'registration.json',dict(root=str(root),owner=str(owner),output=str(out),job=str(spec['job']),
        plan_sha256=w.base.sha((out/'plan.json').read_bytes()),recovery_adapter=str(adapter),recovery_adapter_sha256=w.base.sha(adapter.read_bytes())))
    return dict(root=root,owner=owner,out=out,reg=reg,procs=procs,adapter=adapter,worker=worker)


def result(b,pid=201):
    return w.registered_worker(pid,[b['reg']],b['root'],lookup=b['procs'].get)


def change_file(path,key,value):
    row=json.loads(path.read_text());row[key]=value;save(path,row)


def test_exact_registered_worker_excluded(bound):
    row=result(bound)
    assert row['pid']==201 and row['owner_pid']==100
    assert row['output']==str(bound['out']) and row['created']==101.


@pytest.mark.parametrize('field,value',[('pid',202),('stage',1),('stage',True),('utc',97.)])
def test_changed_worker_status_denied(bound,field,value):
    change_file(bound['owner']/'project_status.json',field,value)
    assert result(bound) is None


def test_external_worker_denied(bound):
    bound['procs'][999]=dict(bound['worker'],pid=999,ppid=1)
    assert result(bound,999) is None


@pytest.mark.parametrize('change',['actual_command','status_command','receipt_command','parent','parent_create','stage','module','receipt_job'])
def test_changed_command_parent_or_registration_denied(bound,change):
    wrong=['python','-m','external.test','worker','--output',str(bound['out'])]
    if change=='actual_command':bound['worker']['command']=wrong
    elif change=='status_command':change_file(bound['owner']/'project_status.json','command',wrong)
    elif change=='receipt_command':bound['reg']['receipt']['commands']=[wrong]
    elif change=='parent':bound['worker']['ppid']=1
    elif change=='parent_create':bound['procs'][100]['created']=150.
    elif change=='stage':bound['reg']['job']['stages'][0]['entry']='external_worker'
    elif change=='module':bound['reg']['spec']['module']='external.test'
    else:bound['reg']['receipt']['job']['operation']='evaluate'
    assert result(bound) is None


@pytest.mark.parametrize('field',['root','owner','output','job','plan_sha256','recovery_adapter_sha256'])
def test_changed_binding_denied(bound,field):
    change_file(bound['out']/'registration.json',field,'/other/path' if 'sha256' not in field else 'bad')
    assert result(bound) is None


def test_changed_source_bytes_denied(bound):
    bound['adapter'].write_text('changed source')
    assert result(bound) is None


def test_other_source_path_even_same_bytes_denied(bound):
    other=bound['adapter'].with_name('external.py');other.write_bytes(bound['adapter'].read_bytes())
    change_file(bound['out']/'registration.json','recovery_adapter',str(other))
    assert result(bound) is None


def test_changed_plan_bytes_denied(bound):
    change_file(bound['out']/'plan.json','cases',['changed'])
    assert result(bound) is None


def test_missing_status_denied(bound):
    (bound['owner']/'project_status.json').unlink()
    assert result(bound) is None


def test_specs_preserve_original_collection_main_and_eval(tmp_path):
    root=tmp_path/'run';config=tmp_path/'config'
    rows=w.specs(root,config);by_name={row['name']:row for row in rows}
    assert len(rows)==len(by_name)
    assert by_name['collection']['module']=='tools.tight_paired_collection'
    assert by_name['collection']['job']==root/'collection_job.json'
    assert by_name['main_train']['module']==w.parallel.proposal.MODULE
    assert by_name['main_train']['job']==w.parallel.proposal.JOB
    assert by_name['development']['module']==by_name['test']['module']=='tools.tight_paired_evaluate'
    assert by_name['collection_recovery05']==dict(name='collection_recovery05',owner=root/'owners/collection_recovery05',module=w.MODULE,operation='dagger',job=root/'collection_recovery05_job.json')
    assert w.base.PAUSES==(None,'A','B','G00','G10','G01','G11')
