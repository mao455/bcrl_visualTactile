"""Synthetic fixtures only: never reads actual science, service or process state."""
from pathlib import Path
import base64
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent

def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

common=module('final13_test_common','final_recovery13_common.py')
prior=module('final13_inherited_fixture','test_final_evidence_recovery12.py')

def write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row,indent=2))

def repin(f):
    f.binding['small_file_sha256']={str(p.resolve()):common.sha(p) for p in f.pinned if p.is_file()}
    write(f.audit/common.BINDING_NAME,f.binding)
    write(f.audit/common.REVIEW_NAME,dict(status='cpu_review_passed_not_activated',
        binding_sha256=common.sha(f.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(f.audit/name) for name in common.FINAL_TOOLS}))

def inv(wave):
    rows={}
    for p in sorted(wave.rglob('*')):
        if p.is_dir():rows[str(p.relative_to(wave))]=dict(kind='directory',mtime_ns=p.stat().st_mtime_ns)
        else:rows[str(p.relative_to(wave))]=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=common.sha(p))
    return rows

@pytest.fixture
def closed(tmp_path):
    f=prior.closed.__wrapped__(tmp_path);r,a=f.root,f.audit
    def add(path,row):write(path,row);f.pinned.add(path);return path
    def remove(path):
        if path.exists():path.unlink()
        f.pinned.discard(path)
    for name in common.FINAL_TOOLS:(a/name).write_text('synthetic final13 '+name)
    for name in common.SOURCE_DEPENDENCIES_SHA256:
        (a/name).write_bytes((HERE/name).read_bytes());f.pinned.add(a/name)
    for name in common.RUNTIME13_SOURCES:
        if not (a/name).exists():(a/name).write_text('synthetic runtime13 '+name)
        f.pinned.add(a/name)
    test=r/'evaluation/test';owner=r/'owners/test_fragment13';driver=a/'driver_test_fragment13'
    testreg=common.read_small(test/'registration.json')
    jobpath=a/'original_test_job.json'
    originaljob=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(test),
        stages=[dict(entry='paired_evaluate',args=['worker','--output',str(test/f'wave{i:02d}')]) for i in range(64)])
    add(jobpath,originaljob)
    commands=[['python','-m','outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',*s['args']] for s in originaljob['stages']]
    add(test/'project_job_started.json',dict(pid=120,utc=232,job=originaljob,commands=commands))
    add(test/'project_status.json',dict(stage=1,pid=122,command=commands[1],utc=233))
    add(test/'monitor.json',dict(pid=121))
    add(test/'project_failure.json',dict(error='synthetic consumed initialization failure'))
    add(test/'project_exit.json',dict(utc=245,children_reaped=True))
    remove(test/'project_complete.json')
    remove(a/'driver_eval_recovery12/closure.json')
    add(a/'driver_eval_recovery12/status.json',dict(stage='stopped_with_evidence',utc=245))
    add(a/'driver_eval_recovery12/failure.json',dict(error='synthetic failed original12'))
    add(a/'driver_eval_recovery12/driver_exit.json',dict(pid=123,utc=246,returned_after_owned_helpers=True))
    add(r/'cycle_logs_recovery12/evaluate_test_exit.json',dict(returncode=1,utc=245))
    for name in ('summarize_test','audit_collection_cuda_noise_recovery05'):
        remove(r/'cycle_logs_recovery12'/(name+'_exit.json'))
        remove(a/'driver_eval_recovery12'/(name+'_capacity_admission.json'))
    remove(test/'wave01/results.json');remove(test/'wave01/process_exit_requested.json')
    sessions={};plans={}
    for i in range(64):
        w=test/f'wave{i:02d}';add(w/'plan.json',dict(index=i,strict_autonomous=True));add(w/'session.json',dict(deadline=f.binding['original_deadline']))
        sessions[str(w/'session.json')]=common.sha(w/'session.json');plans[str(w)]=common.sha(w/'plan.json')
    testreg.update(job=str(jobpath),job_sha256=common.sha(jobpath),wave_plan_sha256=plans)
    add(test/'registration.json',testreg)
    for i in range(2):(test/f'project_stage_{i:02d}.log').write_text('')
    add(test/'session.json',dict(deadline=f.binding['original_deadline'],resource_policy='full_gpu_v1'))
    add(owner/'session.json',dict(deadline=f.binding['original_deadline'],resource_root=str(owner)))
    census=dict(lifecycle_only=True,outcome_values_interpreted=False,registered_waves=64,registered_opportunities=256,
        consumed_waves=2,eligible_indices=list(range(2,64)),
        original_registration_sha256=common.sha(test/'registration.json'),original_job_sha256=common.sha(jobpath),
        original_started_sha256=common.sha(test/'project_job_started.json'),original_status_sha256=common.sha(test/'project_status.json'),
        rows=[dict(index=i,wave=str(test/f'wave{i:02d}'),consumed=i<2,eligible_for_dispatch=i>=2,
            terminal_receipts_present=i==0,inventory=inv(test/f'wave{i:02d}') if i<2 else
            {name:inv(test/f'wave{i:02d}')[name] for name in ('plan.json','session.json')}) for i in range(64)])
    watch13=a/'watch_shared_capacity13';backup=watch13/'resource_watch_v4_backups_1_113'
    records=[]
    for i in range(2):
        target=r/f'fixture_watch13_{i}'/'concurrency.json';raw=b'{"restore":13}' if i==0 else None
        if raw:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        row=dict(path=str(target),existed=raw is not None,sha256=hashlib.sha256(raw).hexdigest() if raw else None,
            base64=base64.b64encode(raw).decode() if raw else None)
        add(backup/(hashlib.sha256(str(target).encode()).hexdigest()+'.json'),row)
        records.append({k:row[k] for k in ('path','existed','sha256')})
    add(watch13/'resource_watch_v4_exit.json',dict(pid=113,modified_schedules=0,conflicts=[],exact_original_byte_restoration=True,restored=[x['path'] for x in records]))
    marker=r/'STOP_RESOURCE_WATCH13';marker.write_text('owned13 stop')
    add(a/'WATCHER13_ACTUAL_CLOSED.json',dict(pid=113,source_pid_reaped=True,conflicts=[],
        service=dict(MainPID='0',ActiveState='inactive',Result='success'),exit_sha256=common.sha(watch13/'resource_watch_v4_exit.json'),
        restored_verified=records,actual_restoration_identities=len(records),stop_marker_preserved=str(marker)))
    required=[test/name for name in ('project_failure.json','project_exit.json','monitor.json','project_status.json','project_job_started.json')]
    required += [a/'driver_eval_recovery12/failure.json',a/'driver_eval_recovery12/driver_exit.json',a/'WATCHER12_ACTUAL_CLOSED.json']
    add(a/'ROOT_TEST_FRAGMENT13_BOUNDARY.json',dict(status='actual_original12_boundary_closed_for_missing_only_fragment',
        deadline=f.binding['original_deadline'],all_original12_actors_reaped=True,no_external_signals=True,
        closed_identities=[dict(pid=p,created=1.) for p in (120,121,122,123,112)],artifact_sha256={str(p):common.sha(p) for p in required}))
    fragmentjob=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),
        stages=[dict(entry='paired_evaluate_fragment13',args=originaljob['stages'][i]['args']) for i in range(2,64)])
    add(a/'test_fragment13_job.json',fragmentjob)
    tails=[dict(name='summarize_test',args=['-m','tools.tight_paired_evaluate','summarize','--output',str(test),'--reconstruct-cuda-noise'],skip_closed=False),
        dict(name='audit_collection_cuda_noise_recovery05',args=[str(a/'audit_collection_recovery05.py'),'noise','--collection',str(r/'collection'),'--output',str(a/'collection_noise_recovery05_actual01'),'--reconstruct-cuda-noise'],skip_closed=False)]
    reg=dict(version='test_missing_whole_suffix_fragment13_v1',root=str(r),owner=str(owner),test=str(test),
        deadline=f.binding['original_deadline'],job=str(a/'test_fragment13_job.json'),job_sha256=common.sha(a/'test_fragment13_job.json'),
        boundary_sha256=common.sha(a/'ROOT_TEST_FRAGMENT13_BOUNDARY.json'),census=census,
        original_registration_sha256=common.sha(test/'registration.json'),original_owner_session_sha256=common.sha(test/'session.json'),
        fragment_owner_session_sha256=common.sha(owner/'session.json'),original_wave_session_sha256=sessions,
        source_sha256=f.binding['production']['after_sha256'],original_cohorts_sha256=common.sha(r/'cohorts.json'),
        frozen_pairs=testreg['pairs'],eligible_indices=list(range(2,64)),tail_stages=tails,startup_timeout_seconds=1800,
        orchestration_sha256={str(a/name):common.sha(a/name) for name in common.RUNTIME13_SOURCES},
        scientific_changes=0,optimizer_updates=0,collection_opportunities=0,external_process_presence_is_not_a_veto=True,original12_remains_failed=True)
    add(a/'TEST_FRAGMENT13_REGISTRATION.json',reg)
    add(a/'ROOT_TEST_FRAGMENT13_REVIEW.json',dict(status='source_and_actual_missing_suffix_review_passed_inactive',registration_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json')))
    add(driver/'registration.json',dict(reg,activated=True,activated_utc=247))
    add(driver/'status.json',dict(stage='audit_collection_cuda_noise_recovery05_complete',utc=298))
    add(driver/'closure.json',dict(utc=299,status='missing_original_suffix_and_unattempted_audits_closed',original12_remains_failed=True))
    add(driver/'driver_exit.json',dict(pid=130,utc=300,returned_after_owned_helpers=True))
    add(owner/'project_job_started.json',dict(utc=248,job=fragmentjob,commands=[['python','-m',
        'outputs.diagnostics.tight_paired_dagger_20261122_01.test_fragment13',*s['args']] for s in fragmentjob['stages']]))
    add(owner/'project_complete.json',dict(stages=62));add(owner/'project_exit.json',dict(utc=290,children_reaped=True))
    add(owner/'project_status.json',dict(stage=61,pid=261));add(owner/'monitor.json',dict(pid=131))
    for j,i in enumerate(range(2,64)):
        (owner/f'project_stage_{j:02d}.log').write_text('')
        add(test/f'wave{i:02d}'/'run_started.json',dict(utc=249+j*.1,pid=200+j,plan_sha256=plans[str(test/f'wave{i:02d}')]))
        add(test/f'wave{i:02d}'/'process_exit_requested.json',dict(utc=280,pid=200+j))
        add(owner/'dispatches'/f'wave{i:02d}.json',dict(utc=249+j*.1,pid=200+j,index=i,wave=str(test/f'wave{i:02d}'),
            dispatch_attempt=1,rerun=False,plan_sha256=plans[str(test/f'wave{i:02d}')],registration_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json')))
    add(driver/'evaluate_fragment13_capacity_admission.json',dict(utc=248,capacity=dict(ready=True)))
    add(r/'cycle_logs_fragment13/evaluate_fragment13_exit.json',dict(returncode=0,utc=290))
    stages={name:str(r/'cycle_logs_recovery12'/(name+'_exit.json')) for name in ('summarize_development','prepare_test','evaluate_test')}
    stages['evaluate_fragment13']=str(r/'cycle_logs_fragment13/evaluate_fragment13_exit.json')
    for name in ('summarize_test','audit_collection_cuda_noise_recovery05'):
        stages[name]=str(add(r/'cycle_logs_fragment13'/(name+'_exit.json'),dict(returncode=0,utc=298)))
        add(driver/(name+'_capacity_admission.json'),dict(utc=291,capacity=dict(ready=True)))
    f.binding.update(version='paired_final_recovery13_bindings_v1',stage_exit_paths=stages,watcher_exit_path=str(watch13/'resource_watch_v4_exit.json'))
    f.reg13=reg;repin(f);return f


def test_truthful_failed12_completed_fragment13_full64_denominator(closed):
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['failed12_preserved']
    e=result['evaluation_execution']['test']
    assert (e['registered_waves'],e['attempted_waves'],e['terminal_waves'],e['software_unfinished_waves'],e['unexecuted_waves'])==(64,64,63,1,0)
    assert e['original12_attempted_indices']==[0,1] and e['fragment13_attempted_indices']==list(range(2,64))
    assert result['watcher_closure']['current_restoration_audit']['current_schedule_bytes_checked'] is True
    assert result['watcher_closure']['historical_restoration_audits']['watch_shared_capacity12']['current_schedule_bytes_checked'] is False

@pytest.mark.parametrize('change',['failure13','missing_fence','unknown_fence','duplicate_fence','prefix_log_gap','unknown_original_log','unknown_wave','changed_prefix','changed_plan','changed_session','incomplete13','replayed_helper','unreaped12','changed_boundary','changed_source','bad_service','restored_bytes','expired13'])
def test_refuses_bad_actual_proof_even_root_repin(closed,change):
    f=closed;r,a=f.root,f.audit;test=r/'evaluation/test';owner=r/'owners/test_fragment13'
    if change=='failure13':write(a/'driver_test_fragment13/failure.json',dict(error='bad'))
    elif change=='missing_fence':(owner/'dispatches/wave02.json').unlink()
    elif change=='unknown_fence':write(owner/'dispatches/wave64.json',dict(index=64))
    elif change=='duplicate_fence':write(owner/'dispatches/duplicate_wave02.json',dict(index=2))
    elif change=='prefix_log_gap':(test/'project_stage_00.log').unlink()
    elif change=='unknown_original_log':(test/'project_stage_99.log').write_text('')
    elif change=='unknown_wave':(test/'wave64').mkdir()
    elif change=='changed_prefix':write(test/'wave01/unexplained.json',dict(extra=True))
    elif change=='changed_plan':write(test/'wave02/plan.json',dict(changed=True))
    elif change=='changed_session':write(test/'wave02/session.json',dict(changed=True))
    elif change=='incomplete13':(owner/'project_complete.json').unlink()
    elif change=='replayed_helper':write(r/'cycle_logs_recovery12/summarize_test_exit.json',dict(returncode=0))
    elif change=='unreaped12':write(test/'project_exit.json',dict(children_reaped=False,utc=245))
    elif change=='changed_boundary':write(a/'ROOT_TEST_FRAGMENT13_BOUNDARY.json',dict(status='fake'))
    elif change=='changed_source':(a/'test_fragment13.py').write_text('changed')
    elif change=='bad_service':
        p=a/'WATCHER13_ACTUAL_CLOSED.json';v=common.read_small(p);v['service']['MainPID']='113';write(p,v)
    elif change=='restored_bytes':(r/'fixture_watch13_0/concurrency.json').write_text('changed')
    elif change=='expired13':
        p=a/'driver_test_fragment13/driver_exit.json';v=common.read_small(p);v['utc']=f.binding['original_deadline']+1;write(p,v)
    repin(f)
    if change=='incomplete13':assert common.closure_gate(r,a)['status']=='pending'
    else:
        with pytest.raises((ValueError,KeyError,FileNotFoundError)):common.closure_gate(r,a)


def test_absent_native_started_still_consumed_by_empty_original_launch_log(closed):
    f=closed;wave=f.root/'evaluation/test/wave01';(wave/'run_started.json').unlink()
    p=f.audit/'TEST_FRAGMENT13_REGISTRATION.json';reg=common.read_small(p);reg['census']['rows'][1]['inventory']=inv(wave)
    # This isolated inventory test deliberately updates only the supplied synthetic census.
    repin(f);result=common.fragment_execution_inventory(f.root,f.audit,reg,{'small_file_sha256':f.binding['small_file_sha256']})
    assert result['software_unfinished_indices']==[1] and result['attempted_waves']==64


def test_root_binding_hash_changes_refuse(closed):
    p=closed.audit/common.BINDING_NAME;v=common.read_small(p);v['unreviewed']=True;write(p,v)
    with pytest.raises(ValueError,match='binding changed'):common.reviewed_binding(closed.root,closed.audit)


def test_builder_synthetic_roundtrip(closed):
    actual=common.build_binding(closed.root,closed.audit)
    assert actual['version']=='paired_final_recovery13_bindings_v1'
    assert not any(Path(path).suffix=='.pt' for path in actual['small_file_sha256'])
    assert actual['stage_exit_paths']['summarize_test'].endswith('cycle_logs_fragment13/summarize_test_exit.json')


def test_no_runtime_import_side_effects(monkeypatch):
    writes=[]
    def blocked(*args,**kwargs):writes.append((args,kwargs));raise AssertionError('side effect')
    monkeypatch.setattr(subprocess,'Popen',blocked)
    monkeypatch.setattr(Path,'write_text',blocked);monkeypatch.setattr(Path,'mkdir',blocked)
    for name in ('final_recovery13_common','extract_selected_ledgers_recovery13','final_compute_recovery13','final_report_assembler_recovery13','export_final_evidence_recovery13'):
        module('inactive_import_'+name,name+'.py')
    assert not writes


def test_exact_two_payload_directory_exceptions():
    export=module('final13_export_fixture','export_final_evidence_recovery13.py')
    assert len(export.CACHE_PHASE_METADATA_PATHS)==2
    assert all(p.endswith('/caches/pipeline_events.json') for p in export.CACHE_PHASE_METADATA_PATHS)

extractor=module('final13_test_extract','extract_selected_ledgers_recovery13.py')
assembler=module('final13_test_assembly','final_report_assembler_recovery13.py')
exporter=module('final13_test_export','export_final_evidence_recovery13.py')
compute=module('final13_test_compute','final_compute_recovery13.py')

def test_eight_ledger_payload_reads_and_immutable_reuse(tmp_path,monkeypatch):
    fixtures=module('fixture13_selected','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    (audit/'extract_selected_ledgers_recovery13.py').write_bytes((HERE/'extract_selected_ledgers_recovery13.py').read_bytes())
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    selected=audit/'selected13';assert extractor.extract(root,audit,selected,loader=loader)['status']=='passed'
    assert len(owner.loads)==8
    assembler.CURRENT_ROOT=root;assembler.CURRENT_AUDIT=audit;assembler.SELECTED_LEDGERS=selected
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.pt'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            for component in ('frontend','backend'):
                directory=root/scope/role/('training' if component=='frontend' else f'backend/{role}/training')
                (directory/'family_phase_curves.csv').write_text('synthetic')
                assert assembler.actual_component(directory,component,None,None)['additional_checkpoint_reads']==0
    assert len(owner.loads)==8

@pytest.mark.parametrize('changed',['sibling_json','payload','wrong_sha','not_list','symlink','over_cap'])
def test_two_cache_exceptions_do_not_expand_payload_or_directory_access(tmp_path,monkeypatch,changed):
    relative=exporter.CACHE_PHASE_METADATA_REL
    path=tmp_path/relative;path.parent.mkdir(parents=True)
    path.write_text('[{"pid":1}]')
    binding=dict(small_file_sha256={str(path.resolve()):common.sha(path)})
    monkeypatch.setattr(exporter.common,'reviewed_binding',lambda *_:(binding,{}))
    if changed=='sibling_json':
        path=path.with_name('not_reviewed.json');path.write_text('[]');relative=str(path.relative_to(tmp_path))
    elif changed=='payload':
        path=path.with_suffix('.pt');path.write_bytes(b'payload');relative=str(path.relative_to(tmp_path))
    elif changed=='wrong_sha':binding['small_file_sha256'][str(path.resolve())]='0'*64
    elif changed=='not_list':path.write_text('{}');binding['small_file_sha256'][str(path.resolve())]=common.sha(path)
    elif changed=='symlink':
        source=tmp_path/'external.json';source.write_text('[]');path.unlink();path.symlink_to(source)
    elif changed=='over_cap':
        with path.open('wb') as stream:stream.truncate(exporter.base.MAX_FILE+1)
    with pytest.raises(ValueError):exporter.source_path(tmp_path,relative)


def test_exact_current_cache_metadata_is_allowed_only_with_bound_sha(tmp_path,monkeypatch):
    for relative in sorted(exporter.CACHE_PHASE_METADATA_PATHS):
        path=tmp_path/relative;path.parent.mkdir(parents=True);path.write_text('[]')
    binding=dict(small_file_sha256={str((tmp_path/name).resolve()):common.sha(tmp_path/name)
        for name in exporter.CACHE_PHASE_METADATA_PATHS})
    monkeypatch.setattr(exporter.common,'reviewed_binding',lambda *_:(binding,{}))
    for relative in exporter.CACHE_PHASE_METADATA_PATHS:
        assert exporter.source_path(tmp_path,relative)==tmp_path/relative


def test_compute_preserves_failed12_original_and_fragment13_and_uncertified_collection(closed,monkeypatch):
    r,a=closed.root,closed.audit
    write(r/'owners/main_cache/project_job_started.json',dict(utc=5))
    write(r/'owners/main_cache/project_exit.json',dict(utc=15,children_reaped=True))
    write(r/'owners/collection/project_job_started.json',dict(utc=1))
    write(r/'collection/collection_summary.json',dict(opportunities=96))
    write(r/'evaluation/test/summary/primary_summary.json',dict(n=64,episodes=256))
    monkeypatch.setattr(compute,'time',SimpleNamespace(time=lambda:250))
    result=compute.inventory(r,a)
    assert result['committed_parent_updates']==59926 and result['committed_main_updates']==97200
    assert result['gpu_hours'] is None
    assert result['evaluation_execution']['development']['software_unfinished_episodes']==4
    failed=[row for row in result['owners'] if row.get('failure')]
    assert any(row['path'].endswith('/development') for row in failed)
    unknown=next(row for row in result['owners'] if row['path'].endswith('/collection'))
    assert unknown['wall_time_certified'] is False and unknown['end_utc'] is None
    assert {row['stage'] for row in result['standalone_audits']}==set(common.STANDALONE_STAGES)
    assert any(row['path'].endswith('/evaluation/test') for row in failed)
    assert any(row['path'].endswith('/owners/test_fragment13') for row in result['owners'])
    assert len([row for row in result['owners'] if row['path'].endswith('/owners/test_fragment13')])==1
    assert result['evaluation_execution']['test']['software_unfinished_waves']==1


def refresh_registration(f,reg):
    p=f.audit/'TEST_FRAGMENT13_REGISTRATION.json';write(p,reg)
    write(f.audit/'ROOT_TEST_FRAGMENT13_REVIEW.json',dict(status='source_and_actual_missing_suffix_review_passed_inactive',registration_sha256=common.sha(p)))
    write(f.audit/'driver_test_fragment13/registration.json',dict(reg,activated=True,activated_utc=247))
    for path in (f.root/'owners/test_fragment13/dispatches').glob('*.json'):
        row=common.read_small(path);row['registration_sha256']=common.sha(p);write(path,row)
    repin(f)


def test_true_inherited_rc0_helper_selected_once_and_dynamic_summary_path(closed):
    f=closed;r,a=f.root,f.audit;reg=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json')
    name='summarize_test';old=r/'cycle_logs_fragment13'/(name+'_exit.json');new=r/'cycle_logs_recovery12'/(name+'_exit.json')
    new.parent.mkdir(exist_ok=True);old.rename(new);f.pinned.discard(old);f.pinned.add(new)
    admission=a/'driver_test_fragment13'/(name+'_capacity_admission.json');oldadmission=a/'driver_eval_recovery12'/(name+'_capacity_admission.json')
    admission.rename(oldadmission);f.pinned.discard(admission);f.pinned.add(oldadmission)
    reg['tail_stages'][0].update(skip_closed=True,original_exit_sha256=common.sha(new))
    f.binding['stage_exit_paths'][name]=str(new);refresh_registration(f,reg)
    result=common.closure_gate(r,a)
    assert result['status']=='closed' and result['fragment13']['tail_stage_exit_paths'][name]==str(new)
    actual=common.build_binding(r,a)
    assert actual['stage_exit_paths'][name]==str(new) and str(old) not in actual['small_file_sha256']


@pytest.mark.parametrize('change',['original_job','helper_args','skipped_admission','fence_pid','native_pid','original_exit_time'])
def test_additional_integrity_probes(closed,change):
    f=closed;r,a=f.root,f.audit;test=r/'evaluation/test';owner=r/'owners/test_fragment13'
    if change=='original_job':write(a/'original_test_job.json',dict(tampered=True))
    elif change=='helper_args':
        reg=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json');reg['tail_stages'][0]['args']=['wrong'];refresh_registration(f,reg)
    elif change=='skipped_admission':
        p=a/'driver_test_fragment13/summarize_test_capacity_admission.json';write(p,dict(capacity=dict(ready=False)))
    elif change=='fence_pid':
        p=owner/'dispatches/wave02.json';row=common.read_small(p);row['pid']+=1;write(p,row)
    elif change=='native_pid':
        p=test/'wave02/run_started.json';row=common.read_small(p);row['pid']+=1;write(p,row)
    elif change=='original_exit_time':
        p=test/'project_exit.json';row=common.read_small(p);row['utc']=100;write(p,row)
    repin(f)
    with pytest.raises((ValueError,KeyError)):common.closure_gate(r,a)


def test_assembler_costs_keep_failed12_and_fragment13_and_helpers_once(closed):
    f=closed;assembler.CURRENT_AUDIT=f.audit;assembler.CURRENT_ROOT=f.root
    result=assembler.attempt_costs(f.root)
    assert len([row for row in result['owner_attempts'] if row['owner']=='test_fragment13'])==1
    assert len([row for row in result['owner_attempts'] if row['owner']=='evaluation/test'])==1
    assert {row['stage'] for row in result['standalone_audits']}==set(common.STANDALONE_STAGES)
    assert len(result['standalone_audits'])==3
    assert result['owner_and_standalone_wall_union_seconds']>=result['closed_owner_wall_union_seconds']
