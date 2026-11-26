"""Source-only recovery12 review using synthetic metadata and checkpoint fixtures."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent

def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

common=module('fixture12_common','final_recovery12_common.py')
extractor=module('fixture12_extract','extract_selected_ledgers_recovery12.py')
assembler=module('fixture12_assemble','final_report_assembler_recovery12.py')
compute=module('fixture12_compute','final_compute_recovery12.py')
exporter=module('fixture12_export','export_final_evidence_recovery12.py')
prior=module('fixture12_ancestry10','test_final_evidence_recovery10.py')

def write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row))

def repin(f):
    f.binding['small_file_sha256']={str(p.resolve()):common.sha(p) for p in f.pinned}
    write(f.audit/common.BINDING_NAME,f.binding)
    write(f.audit/common.REVIEW_NAME,dict(status='cpu_review_passed_not_activated',
        binding_sha256=common.sha(f.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(f.audit/name) for name in common.FINAL_TOOLS}))

@pytest.fixture
def closed(tmp_path):
    f=prior.closed.__wrapped__(tmp_path);r,a=f.root,f.audit
    def add(path,row):write(path,row);f.pinned.add(path);return path
    def remove(path):
        if path.exists():path.unlink()
        f.pinned.discard(path)
    for name in common.FINAL_TOOLS:(a/name).write_text('synthetic final12 '+name)
    for name in common.SOURCE_DEPENDENCIES_SHA256:
        (a/name).write_bytes((HERE/name).read_bytes());f.pinned.add(a/name)
    for name in common.RUNTIME12_SOURCES:
        if not (a/name).exists():(a/name).write_text('synthetic runtime12 '+name)
        f.pinned.add(a/name)
    remove(a/'driver_backend_recovery10/closure.json')
    add(a/'driver_backend_recovery10/status.json',dict(stage='stopped_with_evidence',utc=230))
    add(a/'driver_backend_recovery10/failure.json',dict(error='dev wave07 startup failed'))
    add(r/'cycle_logs_recovery10/evaluate_development_exit.json',dict(returncode=1,utc=230))
    for stage in common.REMAINING_STAGES:
        remove(r/'cycle_logs_recovery10'/(stage+'_exit.json'))
        remove(a/'driver_backend_recovery10'/(stage+'_shared_capacity06_admission.json'))
    dev=r/'evaluation/development'
    remove(dev/'project_complete.json')
    add(dev/'project_failure.json',dict(error='owned startup bound'))
    devreg=copy.deepcopy(common.read_small(r/'evaluation/test/registration.json'));devreg.update(cohort='development',n=8)
    add(dev/'registration.json',devreg)
    add(dev/'project_job_started.json',dict(utc=190,job=dict(resource_root=str(dev),
        stages=[dict(entry='synthetic') for _ in range(8)]),commands=[['fixture'] for _ in range(8)]))
    rows=[]
    for i in range(8):
        wave=dev/f'wave{i:02d}'
        add(wave/'run_started.json',dict(utc=190+i))
        if i<7:
            add(wave/'results.json',dict(opaque_synthetic=True))
            add(wave/'process_exit_requested.json',dict(utc=200+i))
        inventory={str(p.relative_to(wave)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
            sha256=common.sha(p)) for p in wave.rglob('*') if p.is_file()}
        rows.append(dict(index=i,wave=str(wave),consumed=True,eligible_for_rerun=False,
            terminal_receipts_present=i<7,inventory=inventory))
    add(a/'EVAL_STARTUP12_OWNED_TERMINATION.json',dict(version='owned_precontrol_startup_termination12_v1',
        student_actions_observed=False,completed_wave_result_files=7,test_prepared=False,external_processes_signaled=0))
    exit10=a/'watch_shared_capacity10/resource_watch_v4_exit.json'
    records=[]
    for path in sorted((a/'watch_shared_capacity10').glob('resource_watch_v4_backups_*/*.json')):
        row=common.read_small(path);records.append({k:row[k] for k in ('path','existed','sha256')})
    add(a/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json',dict(pid=100,source_pid_reaped=True,
        service='MainPID=0\nResult=success\nActiveState=inactive\n',exit_sha256=common.sha(exit10),
        conflicts=[],restored_verified=records))
    watcher=a/'watch_shared_capacity12';backup=watcher/'resource_watch_v4_backups_1_112'
    restored=[]
    for i in range(3):
        target=r/f'fixture_watch12_{i}'/'concurrency.json';raw=b'{"restore":12}' if i==0 else None
        if raw:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        row=dict(path=str(target),existed=raw is not None,
            sha256=hashlib.sha256(raw).hexdigest() if raw else None,base64=base64.b64encode(raw).decode() if raw else None)
        add(backup/(hashlib.sha256(str(target).encode()).hexdigest()+'.json'),row)
        restored.append(str(target))
    add(watcher/'resource_watch_v4_exit.json',dict(pid=112,modified_schedules=0,conflicts=[],
        exact_original_byte_restoration=True,restored=restored))
    marker=r/'STOP_RESOURCE_WATCH';marker.write_text('owned close')
    actualrows=[]
    for path in sorted(backup.glob('*.json')):
        row=common.read_small(path);actualrows.append({k:row[k] for k in ('path','existed','sha256')})
    add(a/'WATCHER12_ACTUAL_CLOSED.json',dict(pid=112,source_pid_reaped=True,conflicts=[],
        service='MainPID=0\nResult=success\nActiveState=inactive\n',
        exit_sha256=common.sha(watcher/'resource_watch_v4_exit.json'),restored_verified=actualrows,
        actual_restoration_identities=len(actualrows),stop_marker_preserved=str(marker)))
    add(r/'cohorts.json',dict(sealed64='synthetic original identity'))
    stages=[]
    for stage in common.REMAINING_STAGES:
        gpu=stage!='prepare_test'
        args=(['-m','outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',
            'project','evaluate','--job',str(f.project/'configs/experiments/paired_dagger_20261122/evaluate_test.json')]
            if stage=='evaluate_test' else [stage])
        stages.append(dict(name=stage,args=args,gpu=gpu))
        add(r/'cycle_logs_recovery12'/(stage+'_exit.json'),dict(returncode=0,utc=240))
        if gpu:add(a/'driver_eval_recovery12'/(stage+'_capacity_admission.json'),dict(utc=220,capacity=dict(ready=True)))
    census=dict(development_population=8,development_consumed=8,development_closed_wave_receipts=7,
        development_reruns=0,test_prepared=False,test_registered_families=64,test_launched=0,
        lifecycle_only=True,outcome_values_interpreted=False,rows=rows,
        original_owner_failure_sha256=common.sha(dev/'project_failure.json'),
        original_driver_failure_sha256=common.sha(a/'driver_backend_recovery10/failure.json'))
    reg12=dict(version='paired_unstarted_test_recovery12_v1',root=str(r),original_deadline=f.reg['original_deadline'],
        source_sha256=f.reg['source_sha256'],scientific_changes=0,collection_retries=0,training_updates=0,
        startup_timeout_seconds=1800,lifecycle=census,stages=stages,frozen_pairs=devreg['pairs'],
        orchestration_sha256={str(a/name):common.sha(a/name) for name in common.RUNTIME12_SOURCES})
    for key,path in [('original_registration_sha256',a/'BACKEND_RECOVERY10_REGISTRATION.json'),
        ('development_registration_sha256',dev/'registration.json'),
        ('restoration_sha256',a/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json'),
        ('original_startup_failure_sha256',a/'EVAL_STARTUP12_OWNED_TERMINATION.json'),('cohorts_sha256',r/'cohorts.json')]:
        reg12[key]=common.sha(path)
    add(a/'EVALUATION_RECOVERY12_REGISTRATION.json',reg12)
    add(a/'ROOT_EVALUATION_RECOVERY12_REVIEW.json',dict(status='source_and_actual_boundary_review_passed_inactive',
        registration_sha256=common.sha(a/'EVALUATION_RECOVERY12_REGISTRATION.json')))
    add(a/'driver_eval_recovery12/registration.json',dict(reg12,activated=True,activated_utc=231))
    add(a/'driver_eval_recovery12/status.json',dict(stage='remaining_evaluation_complete',utc=241,original10_remains_failed=True))
    add(a/'driver_eval_recovery12/closure.json',dict(utc=241,original10_remains_failed=True,
        status='original_sealed_test_and_remaining_summaries_complete'))
    add(a/'driver_eval_recovery12/driver_exit.json',dict(utc=242,returned_after_owned_helpers=True))
    testowner=r/'evaluation/test'
    add(testowner/'project_job_started.json',dict(utc=232,job=dict(resource_root=str(testowner),
        stages=[dict(entry='synthetic') for _ in range(64)]),commands=[['fixture'] for _ in range(64)]))
    add(testowner/'project_complete.json',dict(stages=64))
    add(testowner/'project_exit.json',dict(utc=239,children_reaped=True))
    for i in range(64):
        wave=r/'evaluation/test'/f'wave{i:02d}'
        add(wave/'run_started.json',dict(utc=232))
        add(wave/'results.json',dict(opaque_synthetic=True))
        add(wave/'process_exit_requested.json',dict(utc=239))
    f.binding.update(version='paired_final_recovery12_bindings_v1',
        stage_exit_paths={s:str(r/'cycle_logs_recovery12'/(s+'_exit.json')) for s in common.REMAINING_STAGES},
        watcher_exit_path=str(watcher/'resource_watch_v4_exit.json'))
    repin(f);return f


def test_failed10_closed12_metadata_only_and_separate_counts(closed,monkeypatch):
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.pt','Lifecycle may not read actual model payloads'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['failed10_preserved'] is True
    assert result['evaluation_execution']['development']['software_unfinished_episodes']==4
    assert result['evaluation_execution']['test']['attempted_episodes']==256
    assert result['watcher_closure']['current_restoration_audit']['backup_count']==3
    assert result['watcher_closure']['previous10_restoration_audit']['backup_count']==35

@pytest.mark.parametrize('change',['false10_success','false10_closure','failed10_rc0','dev_success','dev_unreaped',
    'dev_wave07_labels','changed_census','wrong_runtime12','wrong_registry12','extra_stage12','failed12',
    'unreaped12','late12','timeout12','test_missing_launch','test_missing_terminal','watch12_conflict',
    'watch12_changed_bytes','watch10_conflict','watch10_unreaped','watch12_not_reaped','watch12_wrong_count',
    'watch12_no_marker','test_early12','test_wrong_native_count','dev_wrong_native_count'])
def test_rejects_false_history_or_incomplete12_even_after_root_repin(closed,change):
    r,a=closed.root,closed.audit
    def update(path,**changes):
        row=common.read_small(path);row.update(changes);write(path,row)
    if change=='false10_success':update(a/'driver_backend_recovery10/status.json',stage='main_stages_complete')
    elif change=='false10_closure':write(a/'driver_backend_recovery10/closure.json',dict(status='success'))
    elif change=='failed10_rc0':update(r/'cycle_logs_recovery10/evaluate_development_exit.json',returncode=0)
    elif change=='dev_success':write(r/'evaluation/development/project_complete.json',dict(stages=8))
    elif change=='dev_unreaped':update(r/'evaluation/development/project_exit.json',children_reaped=False)
    elif change=='dev_wave07_labels':write(r/'evaluation/development/wave07/initialization.json',dict(advanced=True))
    elif change=='changed_census':update(r/'evaluation/development/wave00/run_started.json',utc=191)
    elif change=='wrong_runtime12':(a/'eval_startup12.py').write_text('unexpected changed source')
    elif change=='wrong_registry12':update(a/'EVALUATION_RECOVERY12_REGISTRATION.json',training_updates=1)
    elif change=='extra_stage12':update(a/'driver_eval_recovery12/registration.json',activated=False)
    elif change=='failed12':write(a/'driver_eval_recovery12/failure.json',dict(error='real failure'))
    elif change=='unreaped12':update(a/'driver_eval_recovery12/driver_exit.json',returned_after_owned_helpers=False)
    elif change=='late12':update(a/'driver_eval_recovery12/status.json',utc=1795684572)
    elif change=='timeout12':write(a/'driver_eval_recovery12/startup_timeout_wave00.json',dict(consumed=True))
    elif change=='test_missing_launch':
        path=r/'evaluation/test/wave63/run_started.json';path.unlink();closed.pinned.discard(path)
    elif change=='test_missing_terminal':
        path=r/'evaluation/test/wave63/results.json';path.unlink();closed.pinned.discard(path)
    elif change=='watch12_conflict':update(a/'watch_shared_capacity12/resource_watch_v4_exit.json',conflicts=['fixture'])
    elif change=='watch12_changed_bytes':(r/'fixture_watch12_0/concurrency.json').write_text('different')
    elif change=='watch10_conflict':update(a/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json',conflicts=['fixture'])
    elif change=='watch10_unreaped':update(a/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json',source_pid_reaped=False)
    elif change=='watch12_not_reaped':update(a/'WATCHER12_ACTUAL_CLOSED.json',source_pid_reaped=False)
    elif change=='watch12_wrong_count':update(a/'WATCHER12_ACTUAL_CLOSED.json',actual_restoration_identities=2)
    elif change=='watch12_no_marker':(r/'STOP_RESOURCE_WATCH').unlink()
    elif change=='test_early12':update(r/'evaluation/test/project_job_started.json',utc=230)
    elif change=='test_wrong_native_count':update(r/'evaluation/test/project_complete.json',stages=63)
    elif change=='dev_wrong_native_count':update(r/'evaluation/development/project_job_started.json',commands=[])
    repin(closed)
    with pytest.raises((ValueError,FileNotFoundError)):common.closure_gate(r,a)


def test_builder_pending_and_no_payload_or_sealed_outcome_open(closed,monkeypatch):
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.pt'
        assert 'results.json'!=path.name,'Pending binding must not open sealed outcomes'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    (closed.audit/'driver_eval_recovery12/closure.json').unlink()
    result=common.build_binding(closed.root,closed.audit)
    assert result['status']=='pending' and result['checkpoint_reads']==0


def test_builder_roundtrip_preserves_all_source_ancestry(closed):
    candidate=common.build_binding(closed.root,closed.audit)
    assert candidate.get('status')!='pending',candidate
    assert str((closed.root/'evaluation/test/wave63/run_started.json').resolve()) in candidate['small_file_sha256']
    assert str((closed.root/'evaluation/development/wave06/process_exit_requested.json').resolve()) in candidate['small_file_sha256']
    closed.binding=candidate;closed.pinned={Path(p) for p in candidate['small_file_sha256']};repin(closed)
    assert common.closure_gate(closed.root,closed.audit)['status']=='closed'


def test_denominator_is_separate_from_consumed_and_terminal(tmp_path):
    root=tmp_path/'run';wave=root/'evaluation/test/wave00'
    write(wave/'run_started.json',dict(utc=1))
    result=common.evaluation_execution_inventory(root,'test',dict(n=64))
    assert result['registered_episodes']==256 and result['attempted_episodes']==4
    assert result['software_unfinished_episodes']==4 and result['unexecuted_episodes']==252


def test_pending_assembler_and_export_remain_inactive(tmp_path,monkeypatch):
    monkeypatch.setattr(assembler.common,'closure_gate',lambda *_:dict(status='pending',missing=['closed12']))
    monkeypatch.setattr(assembler.base,'assemble',lambda *_:pytest.fail('No science before lifecycle closure'))
    assert assembler.assemble(tmp_path/'run',tmp_path/'out',tmp_path/'audit',tmp_path/'selected')['status']=='pending'
    result=exporter.export(tmp_path,tmp_path/'run',tmp_path/'audit',tmp_path/'assembly',tmp_path/'public',selected_ledgers=tmp_path/'selected')
    assert result['status']=='blocked' and not (tmp_path/'public').exists()


def test_exact_two_reviewed_cache_metadata_exceptions():
    assert exporter.CACHE_PHASE_METADATA_PATHS==frozenset((
        'outputs/diagnostics/tight_paired_dagger_20261122_01/parallel_rear_recovery10_runtime/caches/pipeline_events.json',
        'outputs/diagnostics/tight_paired_dagger_20261122_01/parallel_rear_recovery09_runtime/caches/pipeline_events.json'))


def test_eight_ledger_payload_reads_and_immutable_reuse(tmp_path,monkeypatch):
    fixtures=module('fixture12_selected','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    (audit/'extract_selected_ledgers_recovery12.py').write_bytes((HERE/'extract_selected_ledgers_recovery12.py').read_bytes())
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    selected=audit/'selected12';assert extractor.extract(root,audit,selected,loader=loader)['status']=='passed'
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


def test_compute_preserves_failed10_development_and_uncertified_collection(closed,monkeypatch):
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
