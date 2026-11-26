"""Synthetic15 fixture only: no actual lifecycle, science, process or service reads."""
from pathlib import Path
import base64,copy,hashlib,importlib.util,json,subprocess,sys
from types import SimpleNamespace
import pytest
HERE=Path(__file__).resolve().parent

def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

common=module('final15_test_common','final_recovery15_common.py')
prior=module('final15_fixture14','test_final_evidence_recovery14.py')
extractor=module('final15_extract','extract_selected_ledgers_recovery15.py')
assembler=module('final15_assembly','final_report_assembler_recovery15.py')
exporter=module('final15_export','export_final_evidence_recovery15.py')
compute=module('final15_compute','final_compute_recovery15.py')

def write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row,indent=2,allow_nan=False))

def repin(f):
    f.binding['small_file_sha256']={str(p.resolve()):common.sha(p) for p in f.pinned if p.is_file()}
    write(f.audit/common.BINDING_NAME,f.binding)
    write(f.audit/common.REVIEW_NAME,dict(status='cpu_review_passed_not_activated',binding_sha256=common.sha(f.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(f.audit/name) for name in common.FINAL_TOOLS}))

def inventory(wave):
    return {k:v for k,v in prior.inv(wave).items() if k!='concurrency.json'}

@pytest.fixture
def closed(tmp_path):
    f=prior.closed.__wrapped__(tmp_path);r,a=f.root,f.audit
    def add(path,row):write(path,row);f.pinned.add(path);return path
    def remove(path):
        if path.exists():path.unlink()
        f.pinned.discard(path)
    for name in common.FINAL_TOOLS:(a/name).write_text('synthetic15 '+name)
    for name in common.SOURCE_DEPENDENCIES_SHA256:
        (a/name).write_bytes((HERE/name).read_bytes());f.pinned.add(a/name)
    for name in common.RUNTIME15_SOURCES:
        if not (a/name).exists():(a/name).write_text('synthetic15runtime '+name)
        f.pinned.add(a/name)
    # Driver.run reports the failed helper; its retained child log has the cause.
    add(a/'driver_test_fragment13/failure.json',dict(utc=247.5,
        error="ValueError('evaluate_fragment13 failed; no retry or replacement is launched')"))
    raw13=r/'cycle_logs_fragment13/evaluate_fragment13.log'
    raw13.write_text('Traceback (most recent call last):\nValueError: Original suffix changed\n')
    f.binding['local_retained_log_sha256'][str(raw13)]=common.sha(raw13)
    test=r/'evaluation/test';old=r/'owners/test_fragment13';owner=r/'owners/test_continuation15'
    d14=a/'driver_test_admission14';driver=a/'driver_test_continuation15'
    # Move the synthetic original12 census from two to thirteen consumed waves.
    for path in test.glob('project_stage_*.log'):path.unlink()
    for i in range(13):(test/f'project_stage_{i:02d}.log').write_text('')
    status=common.read_small(test/'project_status.json');status['stage']=12;add(test/'project_status.json',status)
    for i in range(1,12):
        add(test/f'wave{i:02d}/results.json',[]);add(test/f'wave{i:02d}/process_exit_requested.json',dict(utc=244,pid=122))
    for name in ('results.json','process_exit_requested.json'):remove(test/'wave12'/name)
    # Failed14 used owner13 for exactly four original waves13..16.
    remove(old/'project_complete.json');add(old/'project_failure.json',dict(error='native -6 startup abort'))
    add(old/'project_exit.json',dict(utc=290,children_reaped=True))
    add(old/'project_status.json',dict(stage=3,pid=216));add(old/'monitor.json',dict(pid=131))
    for path in old.glob('project_stage_*.log'):path.unlink()
    for i in range(4):(old/f'project_stage_{i:02d}.log').write_text('')
    for path in (old/'dispatches').glob('*.json'):remove(path)
    job=common.read_small(a/'test_fragment13_job.json');job['stages']=[dict(entry='paired_evaluate_fragment13',args=['worker','--output',str(test/f'wave{i:02d}')]) for i in range(13,64)]
    add(a/'test_fragment13_job.json',job)
    add(old/'project_job_started.json',dict(utc=248,job=job,commands=[['python','-m','outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_admission14',*row['args']] for row in job['stages']]))
    for i in range(13,17):
        wave=test/f'wave{i:02d}';add(wave/'run_started.json',dict(utc=260+i*.1,pid=200+i,plan_sha256=common.sha(wave/'plan.json')))
        if i==16:
            remove(wave/'results.json');remove(wave/'process_exit_requested.json')
        else:add(wave/'process_exit_requested.json',dict(utc=280,pid=200+i))
    b13=a/'ROOT_TEST_FRAGMENT13_BOUNDARY.json';boundary=common.read_small(b13)
    boundary['artifact_sha256']={path:common.sha(path) for path in boundary['artifact_sha256']};add(b13,boundary)
    reg13=common.read_small(a/'TEST_FRAGMENT13_REGISTRATION.json')
    reg13.update(job_sha256=common.sha(a/'test_fragment13_job.json'),boundary_sha256=common.sha(b13),eligible_indices=list(range(13,64)))
    reg13['census'].update(consumed_waves=13,eligible_indices=list(range(13,64)),original_status_sha256=common.sha(test/'project_status.json'),
        rows=[dict(index=i,wave=str(test/f'wave{i:02d}'),consumed=i<13,eligible_for_dispatch=i>=13,terminal_receipts_present=i<12,
            inventory=inventory(test/f'wave{i:02d}') if i<13 else {n:inventory(test/f'wave{i:02d}')[n] for n in ('plan.json','session.json')}) for i in range(64)])
    add(a/'TEST_FRAGMENT13_REGISTRATION.json',reg13)
    add(a/'ROOT_TEST_FRAGMENT13_REVIEW.json',dict(status='source_and_actual_missing_suffix_review_passed_inactive',registration_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json')))
    add(a/'driver_test_fragment13/registration.json',dict(reg13,activated=True,activated_utc=247))
    reg14=common.read_small(a/'TEST_ADMISSION14_REGISTRATION.json')
    b14=a/'ROOT_TEST_ADMISSION14_BOUNDARY.json';boundary14=common.read_small(b14)
    boundary14['artifact_sha256']={path:common.sha(path) for path in boundary14['artifact_sha256']};add(b14,boundary14)
    reg14.update(original13_registry_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json'),original13_job_sha256=common.sha(a/'test_fragment13_job.json'),
        eligible_indices=reg13['eligible_indices'],tail_stages=reg13['tail_stages'],
        boundary_sha256=common.sha(b14),original13_driver_artifacts=boundary14['artifact_sha256'])
    reg14['native_zero_attempt_census']['eligible_indices']=reg13['eligible_indices']
    add(a/'TEST_ADMISSION14_REGISTRATION.json',reg14)
    add(a/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(a/'TEST_ADMISSION14_REGISTRATION.json')))
    add(d14/'registration.json',dict(reg14,activated=True,activated_utc=247.8))
    add(d14/'native_owner_dispatch14.json',dict(utc=247.9,job_sha256=common.sha(a/'test_fragment13_job.json'),dispatch_attempt=1,rerun=False))
    remove(d14/'closure.json');add(d14/'failure.json',dict(error='native owner software failure'))
    add(r/'cycle_logs_admission14/evaluate_fragment14_exit.json',dict(returncode=1,utc=290))
    for i in range(13,17):
        wave=test/f'wave{i:02d}'
        add(old/'dispatches'/f'wave{i:02d}.json',dict(utc=249,pid=200+i,index=i,wave=str(wave),plan_sha256=common.sha(wave/'plan.json'),
            registration_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json'),admission14_registration_sha256=common.sha(a/'TEST_ADMISSION14_REGISTRATION.json'),
            cooperative_metadata=None,dispatch_attempt=1,rerun=False))
    for name in ('summarize_test','audit_collection_cuda_noise_recovery05'):
        remove(r/'cycle_logs_admission14'/(name+'_exit.json'));remove(d14/(name+'_capacity_admission.json'))
    # Create a current15 dynamic restoration proof, historical14 stays immutable.
    watcher=a/'watch_shared_capacity15';backup=watcher/'resource_watch_v4_backups_1_115';records=[]
    for target,raw in ((r/'fixture_watch15_0/concurrency.json',b'{"restore":15}'),(test/'wave17/concurrency.json',None)):
        if raw:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        row=dict(path=str(target),existed=raw is not None,sha256=hashlib.sha256(raw).hexdigest() if raw else None,base64=base64.b64encode(raw).decode() if raw else None)
        add(backup/(hashlib.sha256(str(target).encode()).hexdigest()+'.json'),row);records.append({k:row[k] for k in ('path','existed','sha256')})
    add(watcher/'resource_watch_v4_exit.json',dict(pid=115,modified_schedules=0,conflicts=[],exact_original_byte_restoration=True,restored=[x['path'] for x in records]))
    marker=r/'STOP_RESOURCE_WATCH15';marker.write_text('owned15 stop')
    add(a/'WATCHER15_ACTUAL_CLOSED.json',dict(pid=115,source_pid_reaped=True,conflicts=[],service=dict(MainPID='0',ActiveState='inactive',Result='success'),
        exit_sha256=common.sha(watcher/'resource_watch_v4_exit.json'),restored_verified=records,actual_restoration_identities=len(records),stop_marker_preserved=str(marker)))
    rawlog=r/'cycle_logs_admission14/evaluate_fragment14.log';rawlog.write_text('synthetic native -6')
    artifacts=[d14/'failure.json',d14/'driver_exit.json',a/'WATCHER14_ACTUAL_CLOSED.json',old/'project_failure.json',old/'project_exit.json',rawlog]
    boundary15=add(a/'ROOT_TEST_CONTINUATION15_BOUNDARY.json',dict(status='actual_admission14_failed_closed_missing_only15_boundary',deadline=f.binding['original_deadline'],
        all_admission14_actors_reaped=True,no_external_signals=True,eligible_indices=common.ELIGIBLE15,all_consumed_indices=list(range(17)),software_unfinished_indices=[12,16],
        closed_identities=[dict(pid=p,created=1.) for p in (140,131,114)],artifact_sha256={str(p):common.sha(p) for p in artifacts}))
    f.binding['local_retained_log_sha256'][str(rawlog)]=common.sha(rawlog)
    # Native15 owns one launch for all47; wave20 fails before native dispatch.
    originaljob=common.read_small(a/'original_test_job.json')
    job15=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=[dict(entry='paired_evaluate_continuation15',args=originaljob['stages'][i]['args']) for i in common.ELIGIBLE15])
    add(a/'test_continuation15_job.json',job15)
    add(owner/'session.json',dict(deadline=f.binding['original_deadline'],resource_policy='full_gpu_v1',resource_root=str(owner),owned_worker_roots=[str(test/f'wave{i:02d}') for i in common.ELIGIBLE15]))
    (owner/'launches').mkdir();(owner/'dispatches').mkdir();(owner/'stage_exits').mkdir();(owner/'startup_stops').mkdir()
    census=dict(consumed_indices=list(range(17)),eligible_indices=common.ELIGIBLE15,original12_stage_logs=list(range(13)),original14_stage_logs=list(range(4)),
        original14_dispatch_fences=[f'wave{i:02d}.json' for i in range(13,17)],prefix_inventory={str(test/f'wave{i:02d}'):inventory(test/f'wave{i:02d}') for i in range(17)},lifecycle_only=True,outcome_values_interpreted=False)
    reg=dict(version='test_continuation15_missing47_v1',utc=308,deadline=f.binding['original_deadline'],owner=str(owner),job=str(a/'test_continuation15_job.json'),
        job_sha256=common.sha(a/'test_continuation15_job.json'),owner_session_sha256=common.sha(owner/'session.json'),original13_registry_sha256=common.sha(a/'TEST_FRAGMENT13_REGISTRATION.json'),
        original14_registry_sha256=common.sha(a/'TEST_ADMISSION14_REGISTRATION.json'),original_test_registration_sha256=common.sha(test/'registration.json'),boundary_sha256=common.sha(boundary15),
        original14_artifact_sha256=common.read_small(boundary15)['artifact_sha256'],census=census,eligible_indices=common.ELIGIBLE15,
        **{k:reg13[k] for k in ('original_wave_session_sha256','source_sha256','frozen_pairs','original_cohorts_sha256','tail_stages')},
        admission=common.ADMISSION14,software_error_policy=common.POLICY15,orchestration_sha256={str(a/name):common.sha(a/name) for name in common.RUNTIME15_SOURCES},
        scientific_changes=0,training_updates=0,no_outcome_retry=True,original12_13_14_failures_preserved=True,external_process_presence_is_not_a_veto=True)
    add(a/'TEST_CONTINUATION15_REGISTRATION.json',reg)
    add(a/'ROOT_TEST_CONTINUATION15_REVIEW.json',dict(status='source_and_actual_missing47_review_passed_inactive',registration_sha256=common.sha(a/'TEST_CONTINUATION15_REGISTRATION.json')))
    add(driver/'registration.json',dict(reg,activated=True,activated_utc=309.8))
    add(driver/'native_owner_dispatch15.json',dict(utc=309.9,job_sha256=common.sha(a/'test_continuation15_job.json'),dispatch_attempt=1,rerun=False))
    add(driver/'status.json',dict(stage='audit_collection_cuda_noise_recovery05_complete',utc=410))
    add(driver/'closure.json',dict(utc=411,status='original_missing47_queue_and_audits_complete',queue_consumed_not_all_software_completed=True,original12_13_14_failures_preserved=True,report_required=True))
    add(driver/'driver_exit.json',dict(pid=150,utc=412,returned_after_owned_helpers=True))
    commands=[['python','-m','outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_continuation15',*stage['args']] for stage in job15['stages']]
    add(owner/'project_job_started.json',dict(utc=310,pid=151,job=job15,commands=commands,software_error_policy=common.POLICY15))
    add(owner/'project_status.json',dict(stage=46,pid=363,command=commands[-1]));add(owner/'monitor.json',dict(pid=152))
    stage_rows=[]
    for stage,index in enumerate(common.ELIGIBLE15):
        wave=test/f'wave{index:02d}';stamp=311+stage*2;(owner/f'project_stage_{stage:02d}.log').write_text('')
        launch=add(owner/'launches'/f'wave{index:02d}.json',dict(utc=stamp,stage=stage,index=index,command=commands[stage],registration_sha256=common.sha(a/'TEST_CONTINUATION15_REGISTRATION.json'),dispatch_attempt=1,rerun=False))
        row=dict(utc=stamp+1.5,stage=stage,index=index,command=commands[stage],returncode=-6 if index==20 else 0,child_reaped=True,process_started=True,queue_aborted=False,
            startup_stopped=False,software_completed=index!=20,consumed=True,rerun=False,next_only_if_pristine=True,policy_sha256=hashlib.sha256(json.dumps(common.POLICY15,sort_keys=True).encode()).hexdigest())
        add(owner/'stage_exits'/f'wave{index:02d}.json',row);stage_rows.append(row)
        if index==20:
            for name in ('run_started.json','results.json','process_exit_requested.json'):remove(wave/name)
        else:
            add(owner/'dispatches'/f'wave{index:02d}.json',dict(utc=stamp+.1,pid=300+index,index=index,wave=str(wave),plan_sha256=common.sha(wave/'plan.json'),registration_sha256=common.sha(a/'TEST_CONTINUATION15_REGISTRATION.json'),
                original13_registration_sha256=reg['original13_registry_sha256'],launch_sha256=common.sha(launch),cooperative_metadata=None,dispatch_attempt=1,rerun=False))
            add(wave/'run_started.json',dict(utc=stamp+.2,pid=300+index,plan_sha256=common.sha(wave/'plan.json')))
            add(wave/'process_exit_requested.json',dict(utc=stamp+1,pid=300+index))
    add(owner/'project_nonzero_stages.json',dict(utc=318.5,policy=common.POLICY15,stages=[row for row in stage_rows if not row['software_completed']],continued_next_never_launched=True))
    add(owner/'project_complete.json',dict(utc=405,status='finite47_queue_consumed_all',stages=47,all_children_reaped=True,software_completed_stages=46,software_unfinished_indices=[20],all_waves_software_completed=False,outcomes_interpreted=False))
    add(owner/'project_exit.json',dict(utc=405,children_reaped=True,worker_reaped=True,monitor_reaped=True))
    add(driver/'evaluate_continuation15_capacity_admission.json',dict(utc=309.9,capacity=dict(ready=True),admission=common.ADMISSION14))
    add(r/'cycle_logs_continuation15/evaluate_continuation15_exit.json',dict(returncode=0,utc=406))
    stages={name:str(r/'cycle_logs_recovery12'/(name+'_exit.json')) for name in ('summarize_development','prepare_test','evaluate_test')}
    stages.update(evaluate_fragment13_failed=str(r/'cycle_logs_fragment13/evaluate_fragment13_exit.json'),evaluate_fragment14_failed=str(r/'cycle_logs_admission14/evaluate_fragment14_exit.json'),evaluate_continuation15=str(r/'cycle_logs_continuation15/evaluate_continuation15_exit.json'))
    for offset,name in enumerate(('summarize_test','audit_collection_cuda_noise_recovery05')):
        stages[name]=str(add(r/'cycle_logs_continuation15'/(name+'_exit.json'),dict(returncode=0,utc=408+2*offset)))
        add(driver/(name+'_capacity_admission.json'),dict(utc=407+2*offset,capacity=dict(ready=True),admission=common.ADMISSION14))
    add(test/'summary/primary_summary.json',dict(cohort='test',n=64,episodes=256,absolute_success=[dict(role=role,software_incomplete=3) for role in ('A_new','B_new','A_parent','B_parent')]))
    f.binding.update(version='paired_final_recovery15_bindings_v1',stage_exit_paths=stages,watcher_exit_path=str(watcher/'resource_watch_v4_exit.json'))
    f.reg15=reg;f.stage_rows15=stage_rows;repin(f);return f


def test_no_runtime_import_side_effects(monkeypatch):
    writes=[]
    def blocked(*args,**kwargs):writes.append((args,kwargs));raise AssertionError('side effect')
    monkeypatch.setattr(subprocess,'Popen',blocked)
    monkeypatch.setattr(Path,'write_text',blocked);monkeypatch.setattr(Path,'mkdir',blocked)
    for name in ('final_recovery15_common','extract_selected_ledgers_recovery15','final_compute_recovery15','final_report_assembler_recovery15','export_final_evidence_recovery15'):
        module('inactive_import_'+name,name+'.py')
    assert not writes


def test_exact_two_payload_directory_exceptions():
    export=module('final13_export_fixture','export_final_evidence_recovery15.py')
    assert len(export.CACHE_PHASE_METADATA_PATHS)==2
    assert all(p.endswith('/caches/pipeline_events.json') for p in export.CACHE_PHASE_METADATA_PATHS)

def test_eight_ledger_payload_reads_and_immutable_reuse(tmp_path,monkeypatch):
    fixtures=module('fixture13_selected','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    (audit/'extract_selected_ledgers_recovery15.py').write_bytes((HERE/'extract_selected_ledgers_recovery15.py').read_bytes())
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



def test_failed12_13_14_and_completed15_queue_preserve_full64(closed):
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['failed12_preserved'] and result['failed14_preserved']
    assert result['original13_driver_failure']['additional_wave_attempts']==0
    e=result['evaluation_execution']['test']
    assert (e['registered_waves'],e['attempted_waves'],e['terminal_waves'],e['software_unfinished_waves'],e['software_unfinished_episodes'],e['unexecuted_waves'])==(64,64,61,3,12,0)
    assert e['original12_attempted_indices']==list(range(13)) and e['admission14_attempted_indices']==list(range(13,17))
    assert e['continuation15_attempted_indices']==list(range(17,64)) and e['continuation15_software_error_indices']==[20]
    assert len(e['continuation15_native_started_indices'])==46
    assert result['watcher_closure']['current_restoration_audit']['current_schedule_bytes_checked']
    assert not result['watcher_closure']['historical_restoration_audits']['watch_shared_capacity14']['current_schedule_bytes_checked']


def test_real_binding_builder_roundtrip_and_retained_raw_logs(closed):
    f=closed
    binding=common.build_binding(f.root,f.audit)
    assert binding.get('status')!='pending',binding.get('missing')
    assert binding['watcher_exit_path']==str(f.audit/'watch_shared_capacity15/resource_watch_v4_exit.json')
    assert len(binding['local_retained_log_sha256'])==2
    assert not set(binding['local_retained_log_sha256'])&set(binding['small_file_sha256'])
    assert all(Path(path).suffix in common.SAFE_SUFFIXES for path in binding['small_file_sha256'])
    write(f.audit/common.BINDING_NAME,binding)
    write(f.audit/common.REVIEW_NAME,dict(status='cpu_review_passed_not_activated',
        binding_sha256=common.sha(f.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(f.audit/name) for name in common.FINAL_TOOLS}))
    assert common.closure_gate(f.root,f.audit)['status']=='closed'


def test_compute_and_assembler_inventory_each_actual_attempt_once(closed):
    f=closed;r,a=f.root,f.audit
    write(r/'owners/main_cache/project_job_started.json',dict(utc=5))
    write(r/'owners/main_cache/project_exit.json',dict(utc=15,children_reaped=True))
    write(r/'owners/collection/project_job_started.json',dict(utc=1))
    write(r/'collection/collection_summary.json',dict(opportunities=96))
    result=compute.inventory(r,a)
    for directory in (r/'evaluation/test',r/'owners/test_fragment13',r/'owners/test_continuation15'):
        assert len([row for row in result['owners'] if row['path']==str(directory)])==1
    assert len(result['standalone_audits'])==3
    assert [row['attempt'] for row in result['failed_driver_attempts']]==[
        'driver_test_fragment13','driver_test_admission14']
    assert [row['attempt'] for row in result['completed_driver_attempts']]==['driver_test_continuation15']
    failed13=result['failed_driver_attempts'][0]
    assert failed13['native_owner_started'] is False and failed13['additional_wave_attempts']==0
    collection=next(row for row in result['owners'] if row['path']==str(r/'owners/collection'))
    assert collection['end_utc'] is None and collection['wall_time_certified'] is False
    spans=[[row['start_utc'],row['end_utc']] for row in result['owners'] if row['wall_time_certified']]
    spans.extend([row['start_admission_utc'],row['end_utc']] for row in result['standalone_audits'])
    spans.extend([row['start_utc'],row['end_utc']] for row in result['failed_driver_attempts']+result['completed_driver_attempts'])
    assert result['owner_and_standalone_interval_union']==compute.union(spans)
    assert result['committed_parent_updates']==59926 and result['committed_main_updates']==97200
    assert result['gpu_hours'] is None and 'not GPU-hours' in result['gpu_hours_reason']
    assembler.CURRENT_AUDIT=a;assembler.CURRENT_ROOT=r
    costs=assembler.attempt_costs(r)
    for name in ('evaluation/test','test_fragment13','test_continuation15'):
        assert len([row for row in costs['owner_attempts'] if row['owner']==name])==1
    assert len(costs['failed_driver_attempts'])==2 and len(costs['completed_driver_attempts'])==1
    assert costs['owner_and_standalone_wall_union_seconds']==result['owner_and_standalone_interval_union']['seconds']
    assert costs['open_or_unreaped_owner_count']==1


@pytest.mark.parametrize('change',[
    'prefix_bytes','original_job','plan','wave_session','source295','policy','eligible','noise',
    'launch_gap','launch_unknown','launch_retry','exit_gap','exit_unknown','dispatch_unknown',
    'launch_before_reap','missing_nonzero','null_returncode','not_started','queue_aborted','unreaped_child',
    'unreaped_owner','unreaped_monitor','owner_failure','driver_failure','driver_unreaped','fake14_closure',
    'removed12_failure','changed13_failure','changed14_failure','software_aggregate','terminal_pid',
    'raw_log_changed','raw_log_export','helper_repeat13','helper_repeat14','helper_repeat15',
    'global_queue_stop','final_status','underreported_software'
])
def test_finite15_semantics_fail_closed_after_repin(closed,change):
    f=closed;r,a=f.root,f.audit;owner=r/'owners/test_continuation15';test=r/'evaluation/test'
    path=None;row=None
    if change=='prefix_bytes':(test/'wave01/results.json').write_text('[0]')
    elif change=='original_job':write(a/'original_test_job.json',dict(tampered=True))
    elif change=='plan':
        path=test/'wave17/plan.json';row=common.read_small(path);row['tampered']=True
    elif change=='wave_session':
        path=test/'wave17/session.json';row=common.read_small(path);row['deadline']-=1
    elif change=='source295':
        path=a.parents[2]/next(iter(f.binding['production']['after_sha256']));path.write_text('tampered')
    elif change in ('policy','eligible','noise'):
        path=a/'TEST_CONTINUATION15_REGISTRATION.json';row=common.read_small(path)
        if change=='policy':row['software_error_policy']['retry_consumed_wave']=True
        elif change=='eligible':row['eligible_indices']=list(range(16,64))
        else:row['frozen_pairs']['A_new']['sampler_seed']=1
    elif change in ('launch_gap','exit_gap'):
        (owner/('launches' if change=='launch_gap' else 'stage_exits')/'wave20.json').unlink()
    elif change in ('launch_unknown','exit_unknown','dispatch_unknown'):
        directory={'launch_unknown':'launches','exit_unknown':'stage_exits','dispatch_unknown':'dispatches'}[change]
        write(owner/directory/'wave64.json',dict(unknown=True))
    elif change in ('launch_retry','launch_before_reap'):
        path=owner/'launches/wave21.json';row=common.read_small(path)
        row['dispatch_attempt' if change=='launch_retry' else 'utc']=2 if change=='launch_retry' else 0
    elif change=='missing_nonzero':(owner/'project_nonzero_stages.json').unlink()
    elif change in ('null_returncode','not_started','queue_aborted','unreaped_child'):
        path=owner/'stage_exits/wave20.json';row=common.read_small(path)
        key={'null_returncode':'returncode','not_started':'process_started','queue_aborted':'queue_aborted','unreaped_child':'child_reaped'}[change]
        row[key]=None if change=='null_returncode' else change=='queue_aborted'
    elif change in ('unreaped_owner','unreaped_monitor'):
        path=owner/'project_exit.json';row=common.read_small(path);row['children_reaped' if change=='unreaped_owner' else 'monitor_reaped']=False
    elif change=='owner_failure':write(owner/'project_failure.json',dict(error='infrastructure abort'))
    elif change=='driver_failure':write(a/'driver_test_continuation15/failure.json',dict(error='failed'))
    elif change=='driver_unreaped':
        path=a/'driver_test_continuation15/driver_exit.json';row=common.read_small(path);row['returned_after_owned_helpers']=False
    elif change=='fake14_closure':write(a/'driver_test_admission14/closure.json',dict(closed=True))
    elif change=='removed12_failure':(test/'project_failure.json').unlink()
    elif change=='changed13_failure':write(a/'driver_test_fragment13/failure.json',dict(error='rewritten'))
    elif change=='changed14_failure':write(a/'driver_test_admission14/failure.json',dict(error='rewritten'))
    elif change=='software_aggregate':
        path=owner/'project_complete.json';row=common.read_small(path);row['software_unfinished_indices']=[]
    elif change=='terminal_pid':
        path=test/'wave17/process_exit_requested.json';row=common.read_small(path);row['pid']+=1
    elif change=='raw_log_changed':(r/'cycle_logs_admission14/evaluate_fragment14.log').write_text('rewritten')
    elif change=='raw_log_export':f.pinned.add(r/'cycle_logs_admission14/evaluate_fragment14.log')
    elif change=='global_queue_stop':(owner/'STOP').touch()
    elif change=='final_status':
        path=owner/'project_status.json';row=common.read_small(path);row['stage']=45
    elif change=='underreported_software':
        path=test/'summary/primary_summary.json';row=common.read_small(path);row['absolute_success'][0]['software_incomplete']=2
    elif change.startswith('helper_repeat'):
        generation=change.removeprefix('helper_repeat')
        directory={'13':'cycle_logs_fragment13','14':'cycle_logs_admission14','15':'cycle_logs_recovery12'}[generation]
        write(r/directory/'summarize_test_exit.json',dict(returncode=0,utc=300))
    if row is not None:write(path,row)
    repin(f)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):common.closure_gate(r,a)


def test_inherited36_dependency_sources_remain_exact():
    assert len(common.SOURCE_DEPENDENCIES_SHA256)==36
    for name,digest in common.SOURCE_DEPENDENCIES_SHA256.items():assert common.sha(HERE/name)==digest


def pinset(f):
    return dict(small_file_sha256=f.binding['small_file_sha256'],
        local_retained_log_sha256=f.binding['local_retained_log_sha256'])


def record15(f,path,row):
    write(path,row);f.pinned.add(path)


def refresh15(f,reg):
    r,a=f.root,f.audit;path=a/'TEST_CONTINUATION15_REGISTRATION.json';record15(f,path,reg)
    record15(f,a/'ROOT_TEST_CONTINUATION15_REVIEW.json',dict(status='source_and_actual_missing47_review_passed_inactive',registration_sha256=common.sha(path)))
    record15(f,a/'driver_test_continuation15/registration.json',dict(reg,activated=True,activated_utc=309.8))
    for directory in ('launches','dispatches'):
        for fence in (r/'owners/test_continuation15'/directory).glob('*.json'):
            row=common.read_small(fence);row['registration_sha256']=common.sha(path)
            if directory=='dispatches':
                row['original13_registration_sha256']=reg['original13_registry_sha256']
                row['launch_sha256']=common.sha(r/'owners/test_continuation15/launches'/fence.name)
            record15(f,fence,row)
    repin(f)


def update15_aggregate(f):
    owner=f.root/'owners/test_continuation15'
    rows=[common.read_small(owner/'stage_exits'/f'wave{i:02d}.json') for i in common.ELIGIBLE15]
    errors=[row for row in rows if not row['software_completed']]
    path=owner/'project_nonzero_stages.json'
    if errors:record15(f,path,dict(utc=max(row['utc'] for row in errors),policy=common.POLICY15,stages=errors,continued_next_never_launched=True))
    elif path.exists():path.unlink();f.pinned.discard(path)
    path=owner/'project_complete.json';row=common.read_small(path)
    row.update(software_completed_stages=47-len(errors),software_unfinished_indices=[r['index'] for r in errors],all_waves_software_completed=not errors)
    record15(f,path,row)


def test_all15_software_complete_is_queue_complete_with_original_failures_retained(closed):
    f=closed;r,a=f.root,f.audit;owner=r/'owners/test_continuation15';wave=r/'evaluation/test/wave20'
    path=owner/'stage_exits/wave20.json';row=common.read_small(path);row.update(returncode=0,software_completed=True);record15(f,path,row)
    launch=owner/'launches/wave20.json';stamp=common.read_small(launch)['utc']
    record15(f,owner/'dispatches/wave20.json',dict(utc=stamp+.1,pid=320,index=20,wave=str(wave),plan_sha256=common.sha(wave/'plan.json'),
        registration_sha256=common.sha(a/'TEST_CONTINUATION15_REGISTRATION.json'),original13_registration_sha256=f.reg15['original13_registry_sha256'],
        launch_sha256=common.sha(launch),cooperative_metadata=None,dispatch_attempt=1,rerun=False))
    record15(f,wave/'run_started.json',dict(utc=stamp+.2,pid=320,plan_sha256=common.sha(wave/'plan.json')))
    record15(f,wave/'process_exit_requested.json',dict(utc=stamp+1,pid=320));record15(f,wave/'results.json',[])
    update15_aggregate(f)
    path=r/'evaluation/test/summary/primary_summary.json';summary=common.read_small(path)
    for row in summary['absolute_success']:row['software_incomplete']=2
    record15(f,path,summary);repin(f)
    execution=common.closure_gate(r,a)['evaluation_execution']['test']
    assert execution['terminal_waves']==62 and execution['software_unfinished_episodes']==8
    assert execution['continuation15_software_error_indices']==[]


def test_closed_lanes_survive_later_worker_error_and_are_not_fourfold_inferred(closed):
    f=closed;r,a=f.root,f.audit;wave=r/'evaluation/test/wave20';owner=r/'owners/test_continuation15'
    launch=owner/'launches/wave20.json';stamp=common.read_small(launch)['utc']
    record15(f,owner/'dispatches/wave20.json',dict(utc=stamp+.1,pid=320,index=20,wave=str(wave),plan_sha256=common.sha(wave/'plan.json'),
        registration_sha256=common.sha(a/'TEST_CONTINUATION15_REGISTRATION.json'),original13_registration_sha256=f.reg15['original13_registry_sha256'],
        launch_sha256=common.sha(launch),cooperative_metadata=None,dispatch_attempt=1,rerun=False))
    record15(f,wave/'run_started.json',dict(utc=stamp+.2,pid=320,plan_sha256=common.sha(wave/'plan.json')))
    path=r/'evaluation/test/summary/primary_summary.json';summary=common.read_small(path)
    for i,row in enumerate(summary['absolute_success']):row['software_incomplete']=3 if i==0 else 2
    record15(f,path,summary);repin(f)
    execution=common.closure_gate(f.root,f.audit)['evaluation_execution']['test']
    assert execution['software_unfinished_waves']==3 and execution['software_unfinished_episodes']==9
    assert 'Closed lanes survive' in execution['software_unfinished_wave_scope']


def timeout_fixture(f,*,abandoned=False):
    r,a=f.root,f.audit;owner=r/'owners/test_continuation15';wave=r/'evaluation/test/wave17'
    # Move later synthetic events1800s, preserving native start and launch time.
    targets=list(owner.rglob('*.json'))+list((a/'driver_test_continuation15').glob('*.json'))+list((r/'cycle_logs_continuation15').glob('*.json'))
    for i in common.ELIGIBLE15:targets.extend((r/'evaluation/test'/f'wave{i:02d}').glob('*.json'))
    for path in targets:
        row=common.read_small(path)
        if isinstance(row,dict) and type(row.get('utc')) in (int,float) and row['utc']>=312:
            row['utc']+=1800;record15(f,path,row)
    for path in (owner/'dispatches').glob('*.json'):
        row=common.read_small(path);row['launch_sha256']=common.sha(owner/'launches'/path.name);record15(f,path,row)
    status=dict(stage=0,pid=317,command=common.read_small(owner/'project_job_started.json')['commands'][0])
    intent=owner/'startup_intent_stage00_wave17.json'
    record15(f,intent,dict(utc=2111.21,stage=0,index=17,status=status,timeout_seconds=1800,
        physical_steps_unknown=True,entire_wave_consumed=True,rerun=False))
    record15(f,owner/'startup_decision_stage00_wave17.json',dict(utc=2111.23,intent_sha256=common.sha(intent),
        decision='abandoned' if abandoned else 'scoped_stop_committed',stage_changed=False,
        initialized=abandoned,stop_written=not abandoned,global_stop_written=False))
    if not abandoned:
        record15(f,owner/'startup_stops/wave17.json',dict(utc=2111.22,index=17,intent_sha256=common.sha(intent)))
        path=owner/'stage_exits/wave17.json';row=common.read_small(path);row.update(returncode=-15,startup_stopped=True,software_completed=False);record15(f,path,row)
        for name in ('results.json','process_exit_requested.json'):
            (wave/name).unlink();f.pinned.discard(wave/name)
        path=r/'evaluation/test/summary/primary_summary.json';summary=common.read_small(path)
        for row in summary['absolute_success']:row['software_incomplete']=4
        record15(f,path,summary)
    update15_aggregate(f);repin(f)
    return owner


@pytest.mark.parametrize('abandoned',[False,True])
def test_scoped_startup_decision_closes_and_next_pristine_wave_is_once_only(closed,abandoned):
    owner=timeout_fixture(closed,abandoned=abandoned)
    result=common.closure_gate(closed.root,closed.audit)['evaluation_execution']['test']
    assert result['attempted_waves']==64 and result['unexecuted_waves']==0
    assert result['startup_decisions']==dict(intents=1,scoped_stops=0 if abandoned else 1,global_stop_written=False)
    assert (17 in result['continuation15_software_error_indices']) is not abandoned
    assert len(list((owner/'launches').glob('wave18.json')))==1


@pytest.mark.parametrize('change',['early','wrong_cause','unpaired','global','missing_stop','watchdog','before_complete','null_exit'])
def test_startup_and_complete_queue_cannot_hide_abort(closed,change):
    f=closed;owner=timeout_fixture(f);intent=owner/'startup_intent_stage00_wave17.json';decision=owner/'startup_decision_stage00_wave17.json'
    if change=='early':
        row=common.read_small(intent);row['utc']=400;write(intent,row)
    elif change in ('wrong_cause','global'):
        row=common.read_small(decision);row['initialized' if change=='wrong_cause' else 'global_stop_written']=True;write(decision,row)
    elif change=='unpaired':decision.unlink()
    elif change=='missing_stop':(owner/'startup_stops/wave17.json').unlink()
    elif change=='watchdog':write(owner/'startup_watchdog15_error.json',dict(error='infrastructure'))
    elif change=='before_complete':
        row=common.read_small(owner/'project_complete.json');row['utc']=400;write(owner/'project_complete.json',row)
    else:
        row=common.read_small(owner/'stage_exits/wave17.json');row['returncode']=None;write(owner/'stage_exits/wave17.json',row)
    repin(f)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):common.closure_gate(f.root,f.audit)


@pytest.mark.parametrize('change',['missing','wrong_cause','wrong_wrapper'])
def test_failed13_wrapper_and_child_cause_are_separately_required(closed,change):
    f=closed;r,a=f.root,f.audit;path=r/'cycle_logs_fragment13/evaluate_fragment13.log'
    if change=='missing':path.unlink()
    elif change=='wrong_cause':path.write_text('a different error')
    else:write(a/'driver_test_fragment13/failure.json',dict(error="ValueError('Original suffix changed')"))
    # Update all provenance SHAs to exercise semantic validation, not stale pins.
    boundary=a/'ROOT_TEST_ADMISSION14_BOUNDARY.json';row=common.read_small(boundary)
    row['artifact_sha256']={p:common.sha(p) for p in row['artifact_sha256'] if Path(p).exists()};write(boundary,row)
    regpath=a/'TEST_ADMISSION14_REGISTRATION.json';reg=common.read_small(regpath)
    reg.update(boundary_sha256=common.sha(boundary),original13_driver_artifacts=row['artifact_sha256']);write(regpath,reg)
    write(a/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(regpath)))
    if path.exists():f.binding['local_retained_log_sha256'][str(path)]=common.sha(path)
    else:f.binding['local_retained_log_sha256'].pop(str(path))
    repin(f)
    with pytest.raises((ValueError,FileNotFoundError)):
        common.admission14_gate(r,a,f.binding,pinset(f),dict(driver_exit_utc=0))


def test_builder_and_review_accept_exact_three_boundary_log_union_only(closed):
    f=closed;r,a=f.root,f.audit;log=r/'historical_extra_boundary.log';log.write_text('retained local history')
    path=a/'ROOT_TEST_FRAGMENT13_BOUNDARY.json';row=common.read_small(path);row['artifact_sha256'][str(log)]=common.sha(log);write(path,row)
    f.binding['local_retained_log_sha256'][str(log)]=common.sha(log);repin(f)
    assert len(common.reviewed_binding(r,a)[0]['local_retained_log_sha256'])==3
    built=common.build_binding(r,a)
    assert set(built['local_retained_log_sha256'])==set(f.binding['local_retained_log_sha256'])
    assert str(log) not in built['small_file_sha256']
    extra=r/'unbound.log';extra.write_text('unbound');f.binding['local_retained_log_sha256'][str(extra)]=common.sha(extra);repin(f)
    with pytest.raises(ValueError,match='exact root-boundary'):common.reviewed_binding(r,a)


def test_true_original12_helper_is_reused_once_through_all_failed_generations(closed):
    f=closed;r,a=f.root,f.audit;name='summarize_test'
    old=r/'cycle_logs_continuation15'/(name+'_exit.json');new=r/'cycle_logs_recovery12'/(name+'_exit.json')
    old.rename(new);f.pinned.discard(old);f.pinned.add(new)
    old=a/'driver_test_continuation15'/(name+'_capacity_admission.json');newadmission=a/'driver_eval_recovery12'/(name+'_capacity_admission.json')
    old.rename(newadmission);f.pinned.discard(old);f.pinned.add(newadmission)
    p13=a/'TEST_FRAGMENT13_REGISTRATION.json';reg13=common.read_small(p13)
    reg13['tail_stages'][0].update(skip_closed=True,original_exit_sha256=common.sha(new));record15(f,p13,reg13)
    record15(f,a/'ROOT_TEST_FRAGMENT13_REVIEW.json',dict(status='source_and_actual_missing_suffix_review_passed_inactive',registration_sha256=common.sha(p13)))
    record15(f,a/'driver_test_fragment13/registration.json',dict(reg13,activated=True,activated_utc=247))
    p14=a/'TEST_ADMISSION14_REGISTRATION.json';reg14=common.read_small(p14)
    reg14.update(original13_registry_sha256=common.sha(p13),tail_stages=reg13['tail_stages']);record15(f,p14,reg14)
    record15(f,a/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(p14)))
    record15(f,a/'driver_test_admission14/registration.json',dict(reg14,activated=True,activated_utc=247.8))
    for path in (r/'owners/test_fragment13/dispatches').glob('*.json'):
        row=common.read_small(path);row.update(registration_sha256=common.sha(p13),admission14_registration_sha256=common.sha(p14));record15(f,path,row)
    reg15=common.read_small(a/'TEST_CONTINUATION15_REGISTRATION.json')
    reg15.update(original13_registry_sha256=common.sha(p13),original14_registry_sha256=common.sha(p14),tail_stages=reg13['tail_stages'])
    f.binding['stage_exit_paths'][name]=str(new);refresh15(f,reg15)
    assert common.closure_gate(r,a)['continuation15']['tail_stage_exit_paths'][name]==str(new)
    built=common.build_binding(r,a)
    assert built.get('status')!='pending',built.get('missing')
    assert built['stage_exit_paths'][name]==str(new)
    assert str(r/'cycle_logs_continuation15'/(name+'_exit.json')) not in built['small_file_sha256']


def cooperative15(f):
    r,a=f.root,f.audit;wave=r/'evaluation/test/wave17';pid=115
    backup=a/'watch_shared_capacity15/resource_watch_v4_backups_1_115'/(hashlib.sha256(str(wave/'concurrency.json').encode()).hexdigest()+'.json')
    paused=dict(pause_groups=[None,'A','B','G00','G10','G01','G11'],paired_resource_watch_v4_pid=pid)
    return dict(path=str(wave/'concurrency.json'),sha256=hashlib.sha256(json.dumps(paused,indent=2,allow_nan=False).encode()).hexdigest(),
        watcher_pid=pid,backup=str(backup),backup_sha256=common.sha(backup))


def test_exact_current15_cooperative_pause_and_dynamic_restoration(closed):
    f=closed;path=f.root/'owners/test_continuation15/dispatches/wave17.json';row=common.read_small(path)
    row['cooperative_metadata']=cooperative15(f);record15(f,path,row);repin(f)
    result=common.closure_gate(f.root,f.audit)
    assert result['watcher_closure']['current_restoration_audit']['backup_count']==2
    assert result['status']=='closed'


@pytest.mark.parametrize('change',['pid','pause_hash','backup_sha','backup_path','current_bytes','service','unreaped','missing_backup'])
def test_current15_cooperation_and_restoration_are_exact(closed,change):
    f=closed;r,a=f.root,f.audit;metadata=cooperative15(f)
    if change=='pid':metadata['watcher_pid']=114
    elif change=='pause_hash':metadata['sha256']='0'*64
    elif change=='backup_sha':metadata['backup_sha256']='0'*64
    elif change=='backup_path':metadata['backup']=str(a/'unbound.json')
    elif change=='current_bytes':(r/'fixture_watch15_0/concurrency.json').write_text('changed')
    elif change=='missing_backup':Path(metadata['backup']).unlink()
    else:
        path=a/'WATCHER15_ACTUAL_CLOSED.json';row=common.read_small(path)
        if change=='service':row['service']['MainPID']='115'
        else:row['source_pid_reaped']=False
        record15(f,path,row)
    path=r/'owners/test_continuation15/dispatches/wave17.json';row=common.read_small(path);row['cooperative_metadata']=metadata;record15(f,path,row);repin(f)
    with pytest.raises((ValueError,FileNotFoundError)):common.closure_gate(r,a)


def test_export_blocked_before_destination_and_raw_logs_remain_local(closed,tmp_path):
    f=closed;destination=tmp_path/'public_export';assembly=tmp_path/'assembly';assembly.mkdir()
    result=exporter.export(f.audit.parents[2],f.root,f.audit,assembly,destination,apply=False)
    assert result['status']=='blocked' and not destination.exists()
    with pytest.raises(ValueError,match='before destination'):
        exporter.export(f.audit.parents[2],f.root,f.audit,assembly,destination,apply=True)
    assert not destination.exists()
    for path in f.binding['local_retained_log_sha256']:
        with pytest.raises(ValueError):exporter.source_path(f.audit.parents[2],str(Path(path).relative_to(f.audit.parents[2])))
