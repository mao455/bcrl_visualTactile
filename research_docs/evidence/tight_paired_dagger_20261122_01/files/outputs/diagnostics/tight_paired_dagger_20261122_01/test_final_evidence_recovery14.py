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

common=module('final13_test_common','final_recovery14_common.py')
prior=module('final14_inherited_fixture','test_final_evidence_recovery13.py')

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
    # Build from the pinned13 synthetic fixture, then truthfully retain its driver
    # as pre-native failure while the same native owner executes once under14.
    f=prior.closed.__wrapped__(tmp_path);r,a=f.root,f.audit
    def add(path,row):write(path,row);f.pinned.add(path);return path
    def remove(path):
        if path.exists():path.unlink()
        f.pinned.discard(path)
    for name in common.FINAL_TOOLS:(a/name).write_text('synthetic final14 '+name)
    for name in common.SOURCE_DEPENDENCIES_SHA256:
        (a/name).write_bytes((HERE/name).read_bytes());f.pinned.add(a/name)
    for name in common.RUNTIME14_SOURCES:
        if not (a/name).exists():(a/name).write_text('synthetic runtime14 '+name)
        f.pinned.add(a/name)
    test=r/'evaluation/test';owner=r/'owners/test_fragment13';driver13=a/'driver_test_fragment13';driver=a/'driver_test_admission14'
    remove(driver13/'closure.json')
    add(driver13/'status.json',dict(stage='stopped_with_evidence',utc=247.5))
    add(driver13/'failure.json',dict(error="ValueError('Original suffix changed')",utc=247.5))
    add(driver13/'driver_exit.json',dict(pid=130,utc=247.6,returned_after_owned_helpers=True))
    add(r/'cycle_logs_fragment13/evaluate_fragment13_exit.json',dict(returncode=1,utc=247.5))
    add(driver13/'evaluate_fragment13_capacity_admission.json',dict(utc=247.1,capacity=dict(ready=True)))
    log=r/'cycle_logs_fragment13/evaluate_fragment13.log';log.write_text('synthetic Original suffix changed before native owner')
    f.binding['local_retained_log_sha256']={str(log):common.sha(log)}
    for name in ('summarize_test','audit_collection_cuda_noise_recovery05'):
        remove(r/'cycle_logs_fragment13'/(name+'_exit.json'))
        remove(driver13/(name+'_capacity_admission.json'))
    watcher=a/'watch_shared_capacity14';backup=watcher/'resource_watch_v4_backups_1_114';records=[]
    for target,raw in ((r/'fixture_watch14_0/concurrency.json',b'{"restore":14}'),(test/'wave02/concurrency.json',None)):
        if raw:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        row=dict(path=str(target),existed=raw is not None,sha256=hashlib.sha256(raw).hexdigest() if raw else None,
            base64=base64.b64encode(raw).decode() if raw else None)
        add(backup/(hashlib.sha256(str(target).encode()).hexdigest()+'.json'),row)
        records.append({k:row[k] for k in ('path','existed','sha256')})
    add(watcher/'resource_watch_v4_exit.json',dict(pid=114,modified_schedules=0,conflicts=[],exact_original_byte_restoration=True,restored=[x['path'] for x in records]))
    marker=r/'STOP_RESOURCE_WATCH14';marker.write_text('owned14 stop')
    add(a/'WATCHER14_ACTUAL_CLOSED.json',dict(pid=114,source_pid_reaped=True,conflicts=[],
        service=dict(MainPID='0',ActiveState='inactive',Result='success'),exit_sha256=common.sha(watcher/'resource_watch_v4_exit.json'),
        restored_verified=records,actual_restoration_identities=len(records),stop_marker_preserved=str(marker)))
    artifacts=[driver13/'failure.json',driver13/'driver_exit.json',log,
        r/'cycle_logs_fragment13/evaluate_fragment13_exit.json',a/'WATCHER13_ACTUAL_CLOSED.json']
    boundary=add(a/'ROOT_TEST_ADMISSION14_BOUNDARY.json',dict(status='actual_fragment13_failed_before_native_owner',
        deadline=f.binding['original_deadline'],original13_driver_reaped=True,no_external_signals=True,
        closed_identities=[dict(pid=p,created=1.) for p in (130,113)],artifact_sha256={str(p):common.sha(p) for p in artifacts}))
    original=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json')
    reg=dict(version='test_admission14_pre_native_only_v1',utc=247.7,deadline=f.binding['original_deadline'],
        owner=str(owner),job=str(a/'test_fragment13_job.json'),original13_registry_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json'),
        original13_job_sha256=common.sha(a/'test_fragment13_job.json'),boundary_sha256=common.sha(boundary),
        original_test_registration_sha256=common.sha(test/'registration.json'),original13_driver_artifacts=common.read_small(boundary)['artifact_sha256'],
        native_zero_attempt_census=dict(native_owner13_started=False,native_stage_logs=0,dispatch_fences=0,
            eligible_indices=original['eligible_indices'],native_attempts=0,lifecycle_only=True,outcome_values_interpreted=False),
        **{k:original[k] for k in ('eligible_indices','tail_stages','source_sha256','frozen_pairs','original_cohorts_sha256')},
        admission=common.ADMISSION14,orchestration_sha256={str(a/name):common.sha(a/name) for name in common.RUNTIME14_SOURCES},
        original13_driver_remains_failed=True,native_owner13_reused_once=True,scientific_changes=0,training_updates=0,no_outcome_retry=True,
        schedule_metadata_policy='Only live watcher14-owned exact paused schedule with exact original backup binding; no other runtime metadata')
    add(a/'TEST_ADMISSION14_REGISTRATION.json',reg)
    add(a/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(a/'TEST_ADMISSION14_REGISTRATION.json')))
    add(driver/'registration.json',dict(reg,activated=True,activated_utc=247.8))
    add(driver/'status.json',dict(stage='audit_collection_cuda_noise_recovery05_complete',utc=298))
    add(driver/'closure.json',dict(utc=299,status='original_missing_suffix_and_audits_complete',original13_driver_remains_failed=True,native_owner13_used_once=True,report_required=True))
    add(driver/'driver_exit.json',dict(pid=140,utc=300,returned_after_owned_helpers=True))
    add(driver/'native_owner_dispatch14.json',dict(utc=247.9,job_sha256=common.sha(a/'test_fragment13_job.json'),dispatch_attempt=1,rerun=False))
    begin=common.read_small(owner/'project_job_started.json')
    begin['commands']=[[command[0],'-m','outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_admission14',*command[3:]] for command in begin['commands']]
    add(owner/'project_job_started.json',begin)
    for path in (owner/'dispatches').glob('*.json'):
        row=common.read_small(path);row.update(admission14_registration_sha256=common.sha(a/'TEST_ADMISSION14_REGISTRATION.json'),cooperative_metadata=None);add(path,row)
    add(driver/'evaluate_fragment14_capacity_admission.json',dict(utc=248,capacity=dict(ready=True),admission=common.ADMISSION14))
    add(r/'cycle_logs_admission14/evaluate_fragment14_exit.json',dict(returncode=0,utc=290))
    stages={name:str(r/'cycle_logs_recovery12'/(name+'_exit.json')) for name in ('summarize_development','prepare_test','evaluate_test')}
    stages.update(evaluate_fragment13_failed=str(r/'cycle_logs_fragment13/evaluate_fragment13_exit.json'),evaluate_fragment14=str(r/'cycle_logs_admission14/evaluate_fragment14_exit.json'))
    for name in ('summarize_test','audit_collection_cuda_noise_recovery05'):
        stages[name]=str(add(r/'cycle_logs_admission14'/(name+'_exit.json'),dict(returncode=0,utc=298)))
        add(driver/(name+'_capacity_admission.json'),dict(utc=291,capacity=dict(ready=True),admission=common.ADMISSION14))
    f.binding.update(version='paired_final_recovery14_bindings_v1',stage_exit_paths=stages,watcher_exit_path=str(watcher/'resource_watch_v4_exit.json'))
    f.reg14=reg;repin(f);return f


def test_truthful_failed12_completed_fragment13_full64_denominator(closed):
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['failed12_preserved']
    e=result['evaluation_execution']['test']
    assert (e['registered_waves'],e['attempted_waves'],e['terminal_waves'],e['software_unfinished_waves'],e['unexecuted_waves'])==(64,64,63,1,0)
    assert e['original12_attempted_indices']==[0,1] and e['fragment14_attempted_indices']==list(range(2,64))
    assert result['watcher_closure']['current_restoration_audit']['current_schedule_bytes_checked'] is True
    assert result['watcher_closure']['historical_restoration_audits']['watch_shared_capacity12']['current_schedule_bytes_checked'] is False

@pytest.mark.parametrize('change',['failure13','missing_fence','unknown_fence','duplicate_fence','prefix_log_gap','unknown_original_log','unknown_wave','changed_prefix','changed_plan','changed_session','incomplete13','replayed_helper','unreaped12','changed_boundary','changed_source','bad_service','restored_bytes','expired13'])
def test_refuses_bad_actual_proof_even_root_repin(closed,change):
    f=closed;r,a=f.root,f.audit;test=r/'evaluation/test';owner=r/'owners/test_fragment13'
    if change=='failure13':write(a/'driver_test_admission14/failure.json',dict(error='bad'))
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
        p=a/'WATCHER14_ACTUAL_CLOSED.json';v=common.read_small(p);v['service']['MainPID']='114';write(p,v)
    elif change=='restored_bytes':(r/'fixture_watch14_0/concurrency.json').write_text('changed')
    elif change=='expired13':
        p=a/'driver_test_admission14/driver_exit.json';v=common.read_small(p);v['utc']=f.binding['original_deadline']+1;write(p,v)
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
    assert actual['version']=='paired_final_recovery14_bindings_v1'
    assert not any(Path(path).suffix=='.pt' for path in actual['small_file_sha256'])
    assert actual['stage_exit_paths']['summarize_test'].endswith('cycle_logs_admission14/summarize_test_exit.json')
    write(closed.audit/common.BINDING_NAME,actual)
    write(closed.audit/common.REVIEW_NAME,dict(status='cpu_review_passed_not_activated',
        binding_sha256=common.sha(closed.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(closed.audit/name) for name in common.FINAL_TOOLS}))
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['original13_driver_failure']['additional_wave_attempts']==0


def test_no_runtime_import_side_effects(monkeypatch):
    writes=[]
    def blocked(*args,**kwargs):writes.append((args,kwargs));raise AssertionError('side effect')
    monkeypatch.setattr(subprocess,'Popen',blocked)
    monkeypatch.setattr(Path,'write_text',blocked);monkeypatch.setattr(Path,'mkdir',blocked)
    for name in ('final_recovery14_common','extract_selected_ledgers_recovery14','final_compute_recovery14','final_report_assembler_recovery14','export_final_evidence_recovery14'):
        module('inactive_import_'+name,name+'.py')
    assert not writes


def test_exact_two_payload_directory_exceptions():
    export=module('final13_export_fixture','export_final_evidence_recovery14.py')
    assert len(export.CACHE_PHASE_METADATA_PATHS)==2
    assert all(p.endswith('/caches/pipeline_events.json') for p in export.CACHE_PHASE_METADATA_PATHS)

extractor=module('final13_test_extract','extract_selected_ledgers_recovery14.py')
assembler=module('final13_test_assembly','final_report_assembler_recovery14.py')
exporter=module('final13_test_export','export_final_evidence_recovery14.py')
compute=module('final13_test_compute','final_compute_recovery14.py')

def test_eight_ledger_payload_reads_and_immutable_reuse(tmp_path,monkeypatch):
    fixtures=module('fixture13_selected','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    (audit/'extract_selected_ledgers_recovery14.py').write_bytes((HERE/'extract_selected_ledgers_recovery14.py').read_bytes())
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
    monkeypatch.setattr(compute,'time',SimpleNamespace(time=lambda:350))
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
    assert len(result['failed_driver_attempts'])==len(result['completed_driver_attempts'])==1
    assert result['failed_driver_attempts'][0]['attempt']=='driver_test_fragment13'
    assert result['completed_driver_attempts'][0]['attempt']=='driver_test_admission14'
    assert result['failed_driver_attempts'][0]['start_utc']==247
    assert result['completed_driver_attempts'][0]['start_utc']==247.8


def refresh14(f):
    a=f.audit;r=f.root;path=a/'TEST_ADMISSION14_REGISTRATION.json';reg=common.read_small(path)
    original=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json')
    reg.update(original13_registry_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json'),
        boundary_sha256=common.sha(a/'ROOT_TEST_ADMISSION14_BOUNDARY.json'),
        original13_driver_artifacts=common.read_small(a/'ROOT_TEST_ADMISSION14_BOUNDARY.json')['artifact_sha256'],
        **{k:original[k] for k in ('eligible_indices','tail_stages','source_sha256','frozen_pairs','original_cohorts_sha256')})
    write(path,reg)
    write(a/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(path)))
    write(a/'driver_test_admission14/registration.json',dict(reg,activated=True,activated_utc=247.8))
    for fence in (r/'owners/test_fragment13/dispatches').glob('*.json'):
        row=common.read_small(fence);row['admission14_registration_sha256']=common.sha(path);write(fence,row)
    repin(f)


def refresh_registration(f,reg):
    p=f.audit/'TEST_FRAGMENT13_REGISTRATION.json';write(p,reg)
    write(f.audit/'ROOT_TEST_FRAGMENT13_REVIEW.json',dict(status='source_and_actual_missing_suffix_review_passed_inactive',registration_sha256=common.sha(p)))
    write(f.audit/'driver_test_fragment13/registration.json',dict(reg,activated=True,activated_utc=247))
    for path in (f.root/'owners/test_fragment13/dispatches').glob('*.json'):
        row=common.read_small(path);row['registration_sha256']=common.sha(p);write(path,row)
    refresh14(f)


def test_true_inherited_rc0_helper_selected_once_and_dynamic_summary_path(closed):
    f=closed;r,a=f.root,f.audit;reg=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json')
    name='summarize_test';old=r/'cycle_logs_admission14'/(name+'_exit.json');new=r/'cycle_logs_recovery12'/(name+'_exit.json')
    new.parent.mkdir(exist_ok=True);old.rename(new);f.pinned.discard(old);f.pinned.add(new)
    admission=a/'driver_test_admission14'/(name+'_capacity_admission.json');oldadmission=a/'driver_eval_recovery12'/(name+'_capacity_admission.json')
    admission.rename(oldadmission);f.pinned.discard(admission);f.pinned.add(oldadmission)
    reg['tail_stages'][0].update(skip_closed=True,original_exit_sha256=common.sha(new))
    f.binding['stage_exit_paths'][name]=str(new);refresh_registration(f,reg)
    result=common.closure_gate(r,a)
    assert result['status']=='closed' and result['fragment14']['tail_stage_exit_paths'][name]==str(new)
    actual=common.build_binding(r,a)
    assert actual['stage_exit_paths'][name]==str(new) and str(old) not in actual['small_file_sha256']


@pytest.mark.parametrize('change',['original_job','helper_args','skipped_admission','fence_pid','native_pid','original_exit_time'])
def test_additional_integrity_probes(closed,change):
    f=closed;r,a=f.root,f.audit;test=r/'evaluation/test';owner=r/'owners/test_fragment13'
    if change=='original_job':write(a/'original_test_job.json',dict(tampered=True))
    elif change=='helper_args':
        reg=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json');reg['tail_stages'][0]['args']=['wrong'];refresh_registration(f,reg)
    elif change=='skipped_admission':
        p=a/'driver_test_admission14/summarize_test_capacity_admission.json';write(p,dict(capacity=dict(ready=False)))
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
    assert len(result['failed_driver_attempts'])==len(result['completed_driver_attempts'])==1
    assert result['failed_driver_attempts'][0]['attempt']=='driver_test_fragment13'
    assert result['completed_driver_attempts'][0]['attempt']=='driver_test_admission14'
    assert result['owner_and_standalone_wall_union_seconds']>=result['closed_owner_wall_union_seconds']


@pytest.mark.parametrize('change',['prior_native_start','prior_native_log','prior_native_fence','ram_allowance','gpu_reserve','schedule_policy',
    'missing_watcher_identity','bad_birth','failure13_changed','nonzero13_changed','unbound_cooperative','wrong_cooperative_pid',
    'wrong_cooperative_hash','duplicate13helper','unknown14dispatch','wrong14admission','raw_log_changed','raw_log_export'])
def test_admission14_specific_fail_closed(closed,change):
    f=closed;r,a=f.root,f.audit;regpath=a/'TEST_ADMISSION14_REGISTRATION.json';reg=common.read_small(regpath)
    if change.startswith('prior_native_'):
        key={'prior_native_start':'native_owner13_started','prior_native_log':'native_stage_logs','prior_native_fence':'dispatch_fences'}[change]
        reg['native_zero_attempt_census'][key]=True if key=='native_owner13_started' else 1
    elif change=='ram_allowance':reg['admission']['ram_resident_bytes']=16*2**30
    elif change=='gpu_reserve':reg['admission']['gpu_reserve_mib']=0
    elif change=='schedule_policy':reg['schedule_metadata_policy']='arbitrary schedules'
    elif change in ('missing_watcher_identity','bad_birth'):
        p=a/'ROOT_TEST_ADMISSION14_BOUNDARY.json';row=common.read_small(p)
        if change=='missing_watcher_identity':row['closed_identities']=row['closed_identities'][:1]
        else:row['closed_identities'][1]['created']=-1
        write(p,row);reg['boundary_sha256']=common.sha(p)
    elif change=='failure13_changed':write(a/'driver_test_fragment13/failure.json',dict(error='different failure'))
    elif change=='nonzero13_changed':write(r/'cycle_logs_fragment13/evaluate_fragment13_exit.json',dict(returncode=0,utc=247.5))
    elif 'cooperative' in change:
        wave=r/'evaluation/test/wave02';p=r/'owners/test_fragment13/dispatches/wave02.json';row=common.read_small(p)
        backup=a/'watch_shared_capacity14/resource_watch_v4_backups_1_114'/(hashlib.sha256(str(wave/'concurrency.json').encode()).hexdigest()+'.json')
        paused=dict(pause_groups=[None,'A','B','G00','G10','G01','G11'],paired_resource_watch_v4_pid=114)
        row['cooperative_metadata']=dict(path=str(wave/'concurrency.json'),sha256=hashlib.sha256(json.dumps(paused,indent=2,allow_nan=False).encode()).hexdigest(),
            watcher_pid=114,backup=str(backup),backup_sha256=common.sha(backup))
        if change=='unbound_cooperative':row['cooperative_metadata']['backup']=str(a/'arbitrary.json')
        elif change=='wrong_cooperative_pid':row['cooperative_metadata']['watcher_pid']=113
        else:row['cooperative_metadata']['sha256']='0'*64
        write(p,row)
    elif change=='duplicate13helper':write(r/'cycle_logs_fragment13/summarize_test_exit.json',dict(returncode=0,utc=298))
    elif change=='unknown14dispatch':write(a/'driver_test_admission14/native_owner_dispatch14.json',dict(utc=247.9,job_sha256=common.sha(a/'test_fragment13_job.json'),dispatch_attempt=2,rerun=True))
    elif change=='wrong14admission':write(a/'driver_test_admission14/evaluate_fragment14_capacity_admission.json',dict(utc=248,capacity=dict(ready=True),admission={}))
    elif change=='raw_log_changed':(r/'cycle_logs_fragment13/evaluate_fragment13.log').write_text('changed')
    elif change=='raw_log_export':f.pinned.add(r/'cycle_logs_fragment13/evaluate_fragment13.log')
    write(regpath,reg)
    write(a/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(regpath)))
    write(a/'driver_test_admission14/registration.json',dict(reg,activated=True,activated_utc=247.8))
    for p in (r/'owners/test_fragment13/dispatches').glob('*.json'):
        row=common.read_small(p);row['admission14_registration_sha256']=common.sha(regpath);write(p,row)
    repin(f)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):common.closure_gate(r,a)


def test_exact_backed_up_cooperative_metadata_accepted(closed):
    f=closed;r,a=f.root,f.audit;wave=r/'evaluation/test/wave02'
    backup=a/'watch_shared_capacity14/resource_watch_v4_backups_1_114'/(hashlib.sha256(str(wave/'concurrency.json').encode()).hexdigest()+'.json')
    paused=dict(pause_groups=[None,'A','B','G00','G10','G01','G11'],paired_resource_watch_v4_pid=114)
    p=r/'owners/test_fragment13/dispatches/wave02.json';row=common.read_small(p)
    row['cooperative_metadata']=dict(path=str(wave/'concurrency.json'),sha256=hashlib.sha256(json.dumps(paused,indent=2,allow_nan=False).encode()).hexdigest(),
        watcher_pid=114,backup=str(backup),backup_sha256=common.sha(backup));write(p,row);repin(f)
    result=common.closure_gate(r,a)
    assert result['status']=='closed' and result['original13_driver_failure']['additional_wave_attempts']==0
    assert result['failed13_driver_preserved'] and result['fragment14']['original13_driver_failed']


def test_frozen13_dependency_sources_remain_exact():
    for name in common.SOURCE_DEPENDENCIES_SHA256:
        assert common.sha(HERE/name)==common.SOURCE_DEPENDENCIES_SHA256[name]
