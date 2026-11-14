"""The actual subprocess lifecycle is exercised with tiny CPU-only workers."""
import json
import subprocess
import sys
import time

import pytest

from tools import tight_fresh_stage_fit_owner as owner
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


WORKER="""
import hashlib,json,sys,time
from pathlib import Path
registration=Path(sys.argv[1]);arm=sys.argv[2];mode=sys.argv[3]
plan=json.loads(registration.read_text());digest=hashlib.sha256(registration.read_bytes()).hexdigest()
out=registration.parent/arm if arm!='compare' else registration.parent
if arm!='compare':out.mkdir()
if mode=='timeout':time.sleep(60)
if mode=='exit':sys.exit(7)
result=dict(registration_sha256=digest,optimizer_updates=0,rear_evaluated=False,
    validation_evaluated=False,test_evaluated=False,version=plan['version'])
if arm!='compare':result.update(complete=mode!='incomplete',arm=arm,checkpoint_sha256=plan[arm]['sha256'])
else:
    assert all((registration.parent/a/'result.json').exists() for a in ('source','selected'))
(out/('comparison.json' if arm=='compare' else 'result.json')).write_text(json.dumps(result))
"""


@pytest.fixture
def setup(tmp_path,monkeypatch):
    session=tmp_path/'session';session.mkdir();cycle=session/'round';cycle.mkdir()
    output=session/'cpu_audits/fit';output.mkdir(parents=True)
    tree=tmp_path/'frozen';(tree/'tools').mkdir(parents=True);(tree/'visuotactile').mkdir()
    evaluator=tree/'tools/tight_fresh_stage_fit.py';evaluator.write_text('# frozen evaluator fixture\n')
    deadline=time.time()+60
    for path in (session,cycle):atomic_json(path/'session.json',dict(deadline=deadline))
    for role in ('A','B'):
        directory=output/role;directory.mkdir()
        atomic_json(directory/'registration.json',dict(role=role,root=str(cycle),
            version='train_only_selected_frontend_stage_fit_v3',evaluator_sha256=sha256(evaluator),
            identities=['actual_training_case'],source=dict(sha256=role+'source'),selected=dict(sha256=role+'selected')))
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:'a'*40+'\n')
    real_popen=subprocess.Popen;calls=[];faults={}
    def popen(command,**kwargs):
        registration=command[command.index('--registration')+1]
        role=registration.split('/')[-2]
        arm=command[command.index('--arm')+1] if '--arm' in command else 'compare'
        calls.append(role+'_'+arm)
        assert kwargs['cwd']==tree and kwargs['start_new_session'] is True
        return real_popen([sys.executable,'-c',WORKER,registration,arm,faults.get(role+'_'+arm,'normal')],**kwargs)
    monkeypatch.setattr(owner.subprocess,'Popen',popen)
    return dict(output=output,session_root=session,worker_tree=tree,worker_commit='a'*7,
        calls=calls,faults=faults,deadline=deadline)


def run(setup,**kwargs):
    return owner.run(**{key:setup[key] for key in ('output','session_root','worker_tree','worker_commit')},**kwargs)


def test_four_fresh_processes_then_comparisons_preserve_all_bound_receipts(setup):
    result=run(setup)
    assert setup['calls']==['A_source','A_selected','B_source','B_selected','A_compare','B_compare']
    assert result['software_complete'] and not result['pending']
    assert result['deadline']==setup['deadline']==result['original_session_deadline']
    pids=[]
    for row in result['stages']:
        assert row['complete'] and row['cleanup']['live_descendants_exited']
        path=setup['output']/('owner_'+row['stage']+'_started.json')
        started=json.loads(path.read_text());pids.append(started['worker_group']['pid'])
        assert started['worker_group']['start_ticks']>0 and started['deadline']==setup['deadline']
        assert (setup['output']/('owner_'+row['stage']+'.log')).exists()
        assert row['completion_sha256']==sha256(row['completion'])
    assert len(set(pids))==6
    with pytest.raises(FileExistsError,match='previous diagnostic'):run(setup)
    assert len(setup['calls'])==6


@pytest.mark.parametrize('fault',['exit','incomplete','timeout'])
def test_failure_retains_successful_arms_and_never_starts_next_stage(setup,fault):
    # Keep the bounded wall fault first so the test requires only one short child.
    stage='A_source' if fault=='timeout' else 'B_source'
    setup['faults'][stage]=fault
    with pytest.raises(RuntimeError,match='owner incomplete'):
        run(setup,wall_seconds=.15 if fault=='timeout' else 1800)
    result=json.loads((setup['output']/'owner_finished.json').read_text())
    assert not result['software_complete']
    assert setup['calls']==(['A_source'] if fault=='timeout' else ['A_source','A_selected','B_source'])
    if fault!='timeout':assert (setup['output']/'A/selected/result.json').exists()
    assert result['stages'][-1]['cleanup']['live_descendants_exited']
    assert not (setup['output']/'B/selected').exists()
    if fault=='timeout':
        assert result['deadline']<setup['deadline']
        assert result['stages'][-1]['stop_reason']=='owner_wall_limit'
    elif fault=='exit':assert result['stages'][-1]['returncode']==7
    else:assert result['stages'][-1]['completion_sha256'] is not None


def test_stop_before_stage_has_durable_owner_failure_without_any_child(setup):
    (setup['session_root']/'STOP').touch()
    with pytest.raises(RuntimeError,match='STOP'):run(setup)
    result=json.loads((setup['output']/'owner_finished.json').read_text())
    assert not setup['calls'] and len(result['pending'])==6 and not result['stages']


def test_sigterm_exception_cleans_worker_and_records_failure(setup,monkeypatch):
    setup['faults']['A_source']='timeout'
    def terminate(*a,**k):raise SystemExit(143)
    monkeypatch.setattr(owner,'wait_backend_owned',terminate)
    with pytest.raises(RuntimeError,match='SystemExit'):run(setup)
    result=json.loads((setup['output']/'owner_finished.json').read_text())
    assert not result['software_complete'] and len(setup['calls'])==1
    assert result['stages'][0]['cleanup']['live_descendants_exited']
    assert result['stages'][0]['returncode'] is not None


@pytest.mark.parametrize('fault',['existing_arm','session','evaluator','population','role'])
def test_preflight_rejects_drift_and_existing_arm_before_owner_start(setup,fault):
    path=setup['output']/'B/registration.json';plan=json.loads(path.read_text())
    if fault=='existing_arm':(setup['output']/'A/source').mkdir()
    elif fault=='session':atomic_json(setup['session_root']/'round/session.json',dict(deadline=setup['deadline']+1))
    elif fault=='evaluator':plan['evaluator_sha256']='different'
    elif fault=='population':plan['identities']=['different_case']
    else:plan['role']='A'
    atomic_json(path,plan)
    with pytest.raises((ValueError,FileExistsError)):run(setup)
    assert not setup['calls'] and not (setup['output']/'owner_started.json').exists()


@pytest.mark.parametrize('bound',[0,1801,float('nan'),True])
def test_finite_wall_bound_cannot_silently_expand(setup,bound):
    with pytest.raises(ValueError,match='wall bound'):run(setup,wall_seconds=bound)
    assert not setup['calls']
