import json
import pytest
from tools.expert_speed_eval import ExpertSpeedEvaluation,report_body_hash,dependency_outcome
from tools.runtime import write,read


def setup_owner(tmp_path,state):
    parent=tmp_path/'parent';parent.mkdir();root=tmp_path/'new';root.mkdir()
    (parent/'status.json').write_text(json.dumps(state))
    (root/'session.json').write_text(json.dumps(dict(dependency_root=str(parent),deadline=10**12)))
    (root/'status.json').write_text('{}')
    return ExpertSpeedEvaluation(root),parent


def test_active_state_queue_remains_read_only_and_no_kit_is_started(tmp_path,monkeypatch):
    owner,parent=setup_owner(tmp_path,dict(state='evaluating',pid=1234))
    original=(parent/'status.json').read_bytes()
    owner.report=lambda:None
    class WaitReached(Exception):pass
    def wait(seconds):assert seconds==30;raise WaitReached()
    monkeypatch.setattr('tools.expert_speed_eval.time.sleep',wait)
    monkeypatch.setattr('tools.expert_speed_eval.subprocess.Popen',lambda *a,**kw:pytest.fail('GPU/worker launch during dependency wait'))
    with pytest.raises(WaitReached):owner.wait_dependency()
    assert (parent/'status.json').read_bytes()==original
    assert json.loads((owner.root/'status.json').read_text())['state']=='waiting_dependency'
    assert owner.attempts==[]


def test_incomplete_parent_exit_cannot_release_new_experiments(tmp_path):
    owner,parent=setup_owner(tmp_path,dict(state='blocked',pid=None))
    write(parent/'owner_exit.json',read(parent/'status.json'))
    write(parent/'summary.json',{})
    write(parent/'attempts.json',[])
    write(parent/'owner_error.json',dict(error='RuntimeError: Kit result missing'))
    with pytest.raises(AssertionError,match='different error'):owner.wait_dependency()
    assert owner.attempts==[]


def test_report_guard_ignores_only_the_generated_slow_result_section():
    a='Reviewed text\n<!-- SLOW_EXPERT_RESULTS_START -->\nwaiting\n<!-- SLOW_EXPERT_RESULTS_END -->\n'
    assert report_body_hash(a)==report_body_hash(a.replace('waiting','0/10'))
    assert report_body_hash(a)!=report_body_hash(a.replace('Reviewed text','Changed criteria'))


def exhausted_parent(parent):
    status=dict(state='blocked',pid=None,worker_pid=None)
    write(parent/'status.json',status);write(parent/'owner_exit.json',status)
    write(parent/'owner_error.json',dict(error='Traceback\nAssertionError: Preregistered legal-start reserve exhausted\n'))
    attempts=[];specs=[]
    for group,n in [('historical',100),('fresh',120)]:
        for i in range(n):
            spec=dict(identity=f'{group}_{i}',seed=12345+i,selection_index=i,reserve=i>=100,evaluation_group=group)
            specs.append(spec)
            rejected=group=='fresh' and i%2==0
            trial=dict(**spec,legal=not rejected,not_started=rejected)
            if rejected:trial['reason']='pre-action collision rejection; no replacement by outcome'
            record=dict(name=f'random_{group}_{i:03d}',valid=True,exit_code=0,resources_released=True,
                remaining_owned_pids=[],remaining_owned_gpu_pids=[],native_outcome_valid=not rejected,
                initialization_rejected=rejected,result=dict(trials=[trial]))
            attempts.append(record);write(parent/'stages'/record['name']/'committed.json',record)
    write(parent/'attempts.json',attempts)
    write(parent/'session.json',dict(specs=specs,cohort_targets=dict(historical=100,fresh=100)))
    write(parent/'summary.json',dict(valid=160,initialization_rejections=60,formal_attempts=220,software_missing=0,acceptance_failed=0))


def test_exhausted_legal_reserve_releases_only_after_resource_checks(tmp_path,monkeypatch):
    owner,parent=setup_owner(tmp_path,{})
    exhausted_parent(parent)
    write(parent/'monitor.json',dict(pid=999999999));(parent/'owner.lock').touch()
    original={p.relative_to(parent):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr('tools.expert_speed_eval.subprocess.Popen',lambda *a,**kw:pytest.fail('Wait audit launches no GPU worker'))
    owner.wait_dependency()
    result=read(owner.root/'dependency_release.json')
    assert result['reason']=='fresh_legal_reserve_exhausted' and result['parent_state']=='blocked'
    assert result['valid']==160 and result['missing_valid']==40
    assert original=={p.relative_to(parent):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    assert owner.attempts==[]


@pytest.mark.parametrize('damage,match',[
    ('software','software-failed'),('leak','resource release'),('unfinished','software-failed'),
    ('uncommitted','durably committed'),('wrong_reason',''),('wrong_seed',''),
    ('unresolved_reserve','Fresh reserve not fully resolved')])
def test_reserve_exception_does_not_mask_missing_evidence_or_faults(tmp_path,damage,match):
    parent=tmp_path/'parent';exhausted_parent(parent)
    attempts=read(parent/'attempts.json')
    if damage=='software':attempts[-1]['valid']=False
    elif damage=='leak':attempts[-1]['remaining_owned_pids']=[123]
    elif damage=='unfinished':del attempts[-1]['exit_code']
    elif damage=='uncommitted':write(parent/'stages'/attempts[-1]['name']/'committed.json',{})
    elif damage=='wrong_reason':attempts[100]['result']['trials'][0]['reason']='action interrupted by collision'
    elif damage=='wrong_seed':attempts[-1]['result']['trials'][0]['seed']+=1
    elif damage=='unresolved_reserve':
        attempts.pop();summary=read(parent/'summary.json');summary.update(valid=159,formal_attempts=219);write(parent/'summary.json',summary)
    write(parent/'attempts.json',attempts)
    with pytest.raises(AssertionError,match=match):dependency_outcome(parent)


def test_completed_200_dependency_still_requires_resource_release(tmp_path):
    parent=tmp_path/'parent';exhausted_parent(parent)
    status=dict(state='completed',pid=None,worker_pid=None)
    write(parent/'status.json',status);write(parent/'owner_exit.json',status)
    write(parent/'summary.json',dict(valid=200))
    attempts=read(parent/'attempts.json')[:200]
    for attempt in attempts:attempt.update(native_outcome_valid=True,initialization_rejected=False)
    write(parent/'attempts.json',attempts)
    assert dependency_outcome(parent)['reason']=='completed_200'
    attempts=read(parent/'attempts.json');attempts[0]['resources_released']=False;write(parent/'attempts.json',attempts)
    with pytest.raises(AssertionError,match='resource release'):dependency_outcome(parent)
