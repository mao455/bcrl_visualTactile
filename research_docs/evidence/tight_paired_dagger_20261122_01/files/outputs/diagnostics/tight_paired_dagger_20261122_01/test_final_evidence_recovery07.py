"""CPU/filesystem fixtures only: no real checkpoints, outcomes or GPU calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent


def module(name,filename):
    spec = importlib.util.spec_from_file_location(name,HERE/filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


common = module('fixture_common07','final_recovery07_common.py')
extractor = module('fixture_extract07','extract_selected_ledgers_recovery07.py')
assembler = module('fixture_assembler07','final_report_assembler_recovery07.py')
compute = module('fixture_compute07','final_compute_recovery07.py')
exporter = module('fixture_export07','export_final_evidence_recovery07.py')


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload))


def repin(fixture):
    binding = fixture.binding
    binding['small_file_sha256'] = {str(path):common.sha(path) for path in fixture.pinned}
    write(fixture.audit/common.BINDING_NAME,binding)
    write(fixture.audit/common.REVIEW_NAME,dict(status='cpu_review_passed',
        binding_sha256=common.sha(fixture.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(fixture.audit/name) for name in common.FINAL_TOOLS}))


@pytest.fixture
def closed(tmp_path):
    project = tmp_path/'project'; audit = project/'outputs/diagnostics/test07'
    root = project/'outputs/visuotactile/run/test07'; audit.mkdir(parents=True)
    pinned = set()
    def add(path,value):
        write(path,value);pinned.add(path);return path
    for name in common.FINAL_TOOLS:(audit/name).write_text('synthetic source '+name)
    write(root/'session.json',dict(started_utc=0,deadline=1795684571.2261844))
    for name in common.RUNTIME_SOURCES:
        (audit/name).write_text('synthetic runtime source');pinned.add(audit/name)
    old_receipts = {
        audit/'driver_main_recovery05/status.json':dict(stage='stopped_with_evidence'),
        audit/'driver_main_recovery05/failure.json':dict(error='preserved old failure'),
        audit/'wait_main_shared06/status.json':dict(stage='stopped_with_evidence'),
        audit/'wait_main_shared06/failure.json':dict(error='preserved waiter failure'),
        audit/'wait_main_shared06/reviewed_main_exit.json':dict(returncode=1,owned_helper_returned_after_cleanup=True),
        root/'owners/main_train/project_failure.json':dict(error='Stage2 returned1'),
        root/'owners/main_train/project_exit.json':dict(utc=20,children_reaped=True),
        root/'cycle_logs/train_four_components_parallel_rears_exit.json':dict(returncode=1),
    }
    for path,value in old_receipts.items():add(path,value)
    write(root/'owners/main_train/project_job_started.json',dict(utc=10))
    for stage in common.OLD_COMPLETED_STAGES:add(root/'cycle_logs'/(stage+'_exit.json'),dict(returncode=0))
    stage_paths = {}
    for stage in common.REMAINING_STAGES:
        path = root/'cycle_logs_recovery07'/(stage+'_exit.json')
        add(path,dict(returncode=0,utc=120));stage_paths[stage]=str(path)
    write(audit/'driver_backend_recovery07/status.json',dict(stage='main_stages_complete',utc=125))
    add(audit/'driver_backend_recovery07/closure.json',dict(stage='main_stages_complete',status='backend_and_remaining_main_stages_complete',utc=125))
    add(audit/'driver_backend_recovery07/driver_exit.json',dict(returned_after_owned_helpers=True,utc=126))
    for directory in (root/'owners/backend_recovery07',root/'evaluation/development',root/'evaluation/test'):
        job = dict(resource_root=str(directory),stages=[dict(entry='fixture')])
        write(directory/'project_job_started.json',dict(utc=30,job=job,commands=[['fixture']]))
        write(directory/'project_complete.json',dict(stages=1))
        write(directory/'project_exit.json',dict(utc=110,children_reaped=True))
        if directory.name == 'backend_recovery07':add(audit/'proposed_backend_recovery07.json',job)
    preserved = {}; pairs = {role:{} for role in ('A_new','B_new','A_parent','B_parent')}
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            write(root/scope/role/'state_data/data.json',{})
            for component,sub in [('frontend','training'),('backend',f'backend/{role}/training')]:
                directory=root/scope/role/sub;directory.mkdir(parents=True,exist_ok=True)
                path=directory/'best.pt';path.write_bytes((scope+role+component).encode())
                updates = (7200 if component=='frontend' else 22763) if scope=='common180_parents' else (
                    10200 if component=='frontend' else 38400)
                deployment=dict(epoch=60,update=updates,checkpoint_sha256=common.sha(path))
                finished=dict(epochs=60,updates=updates,complete=True) if component=='frontend' else dict(
                    updates=updates,target_reached=True,control_tick_epochs=60)
                add(directory/'deployment.json',deployment);add(directory/'finished.json',finished)
                pairs[role+('_new' if scope=='learning' else '_parent')][component+'_sha256']=deployment['checkpoint_sha256']
                if scope=='learning' and component=='frontend':
                    preserved[role]=dict(checkpoint_sha256=deployment['checkpoint_sha256'],
                        checkpoint_stat_identity=common.identity(path.stat()))
    before,after,snapshots={},{},{}
    for i in range(295):
        relative=sorted(common.AMENDED_PATHS)[i] if i<3 else f'tools/fixture_source_{i:03d}.py'
        path=project/relative;path.parent.mkdir(exist_ok=True)
        old=f'old fixture source{i}';new=f'new fixture source{i}' if i<3 else old
        path.write_text(new);pinned.add(path)
        before[relative]=hashlib.sha256(old.encode()).hexdigest();after[relative]=common.sha(path)
        if i<3:
            snapshot=audit/'frozen_source'/relative;snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_text(old);pinned.add(snapshot);snapshots[relative]=str(snapshot)
            amended=audit/'recovery07_source_snapshot'/relative;amended.parent.mkdir(parents=True,exist_ok=True)
            amended.write_text(new);pinned.add(amended)
    write(root/'learning/training_source_identity.json',dict(sources=before))
    write(root/'evaluation/test/registration.json',dict(cohort='test',n=64,pairs=pairs,source_sha256=after))
    write(audit/'watch07/exit.json',dict(modified_schedules=0,conflicts=[],exact_original_byte_restoration=True))
    write(audit/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json',dict(source_pid_reaped=True))
    add(audit/'driver_main_recovery05/registration.json',dict(source_sha256=before,deadline=1795684571.2261844))
    changes={path:dict(old_sha256=before[path],new_sha256=after[path]) for path in common.AMENDED_PATHS}
    add(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json',dict(changed_paths=changes,optimizer_updates=0,
        physical_steps=0,affine_or_label_changes=False,snapshot=str(audit/'recovery07_source_snapshot')))
    add(audit/'BACKEND_RECOVERY07_ACTIVATION_RECONCILIATION.json',dict(record_kind='synthetic_fixture'))
    add(audit/'watch_shared_capacity07/resource_watch_v4_exit.json',dict(modified_schedules=0,
        conflicts=[],exact_original_byte_restoration=True))
    add(audit/'driver_backend_recovery07/backend_source_identity.json',dict(sources=after))
    stages=[]
    for stage in common.REMAINING_STAGES:
        gpu=stage not in ('verify_final_actual_training','prepare_development','prepare_test')
        command=[str(audit/'presealed_guard_main_recovery07.json')] if stage=='verify_final_actual_training' else [stage]
        stages.append(dict(name=stage,command=command,gpu=gpu))
        if gpu:add(audit/'driver_backend_recovery07'/(stage+'_shared_capacity06_admission.json'),dict(utc=110,capacity=dict(ready=True)))
    reg=dict(version='paired_backend_only_recovery07_v1',root=str(root),original_deadline=1795684571.2261844,
        source_sha256=after,old_production_sha256=before,activated=False,
        owner=str(root/'owners/backend_recovery07'),logs=str(root/'cycle_logs_recovery07'),
        watcher=str(audit/'watch_shared_capacity07'),job=str(audit/'proposed_backend_recovery07.json'),
        job_sha256=common.sha(audit/'proposed_backend_recovery07.json'),
        prior_registration_sha256=common.sha(audit/'driver_main_recovery05/registration.json'),
        orchestration_sha256={str(audit/name):common.sha(audit/name) for name in common.RUNTIME_SOURCES},
        old_terminal_sha256={str(path):common.sha(path) for path in old_receipts},
        source_amendment=dict(changes=changes,changed_paths=sorted(common.AMENDED_PATHS),
            receipt_sha256=common.sha(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json')),
        stages=stages,preserved_dp=preserved,
        scientific_invariants=dict(dp_new_updates=0,dp_reruns=0,collection_retries=0,new_opportunities=0,
            optimizer_epochs=60,deploy_epoch=60,rear_updates_per_role=38400,caches_own_dp=True,
            encoder_frozen=True,development_descriptive_only=True,sealed_test_families=64,external_presence_is_veto=False))
    add(audit/'BACKEND_RECOVERY07_REGISTRATION.json',reg)
    add(audit/'driver_backend_recovery07/registration.json',dict(reg,activated=True,activated_utc=30,
        registry_sha256=common.sha(audit/'BACKEND_RECOVERY07_REGISTRATION.json')))
    for phase,count,start in [('serial_caches',2,40),('rear_training',1,50),('serial_offline',1,60)]:
        events=[]
        for role in 'AB':
            for stage in range(count):
                events += [dict(role=role,stage=stage,pid=1,command=['synthetic'],utc=start),
                           dict(role=role,stage=stage,returncode=0,utc=start+1)]
        add(audit/'parallel_rear_recovery07_runtime'/phase/'pipeline_events.json',events)
    add(audit/'parallel_rear_recovery07_runtime/admission.json',dict(parallel=True,pair=dict(allowed=True),single=dict(allowed=True)))
    add(audit/'parallel_rear_recovery07_runtime/complete.json',dict(utc=70,parallel=True,caches_serial=True,offline_serial=True))
    binding=dict(version='paired_final_recovery07_bindings_v1',root=str(root),
        original_deadline=1795684571.2261844,small_file_sha256={},preserved_dp=preserved,
        stage_exit_paths=stage_paths,watcher_exit_path=str(audit/'watch07/exit.json'),
        production=dict(before_sha256=before,after_sha256=after,changed_paths=sorted(snapshots),snapshot_paths=snapshots))
    fixture=SimpleNamespace(project=project,audit=audit,root=root,pinned=pinned,binding=binding,old_receipts=old_receipts)
    repin(fixture);return fixture


def test_closed07_accepts_truthful_failed06_ancestry_without_payload_reads(closed,monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.suffix!='.pt','Lifecycle gate must not open checkpoint payloads'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['checkpoint_reads']==0
    assert result['failed06_preserved'] is True and len(result['production_amendment_paths'])==3


@pytest.mark.parametrize('missing',[
    'driver_backend_recovery07/status.json','FINAL_RECOVERY07_BINDINGS.json','ROOT_EVIDENCE_TOOLS_REVIEW07.json'])
def test_missing_final_metadata_remains_pending(closed,missing):
    (closed.audit/missing).unlink()
    assert common.closure_gate(closed.root,closed.audit)['status']=='pending'


@pytest.mark.parametrize('change',[
    'old_success','old_unreaped','old_exit_zero','new_failure','new_unreaped','new_wrong_job',
    'stage_failure','source_changed','dp_stat_changed','dp_wrong_age','source_population',
    'wrong_deployment_sha','wrong_deadline','late_closure','overlapping_owner','review_source_changed'])
def test_lifecycle_rejects_rewritten_failures_wrong_sources_or_incomplete_recovery(closed,change):
    r,a=closed.root,closed.audit
    if change=='old_success':write(a/'driver_main_recovery05/status.json',dict(stage='main_stages_complete'));repin(closed)
    elif change=='old_unreaped':write(r/'owners/main_train/project_exit.json',dict(utc=20,children_reaped=False));repin(closed)
    elif change=='old_exit_zero':write(a/'wait_main_shared06/reviewed_main_exit.json',dict(returncode=0,owned_helper_returned_after_cleanup=True));repin(closed)
    elif change=='new_failure':write(a/'driver_backend_recovery07/failure.json',dict(error='failed07'))
    elif change=='new_unreaped':write(r/'owners/backend_recovery07/project_exit.json',dict(utc=110,children_reaped=False))
    elif change=='new_wrong_job':
        p=r/'owners/backend_recovery07/project_job_started.json';d=common.read_small(p);d['job']['stages']=[dict(entry='wrong')];write(p,d)
    elif change=='stage_failure':write(r/'cycle_logs_recovery07/summarize_test_exit.json',dict(returncode=1));repin(closed)
    elif change=='source_changed':(closed.project/'tools/fixture_source_004.py').write_text('changed')
    elif change=='dp_stat_changed':(r/'learning/A/training/best.pt').write_bytes(b'mutated payload')
    elif change=='dp_wrong_age':write(r/'learning/A/training/finished.json',dict(epochs=60,updates=9860,complete=True));repin(closed)
    elif change=='source_population':closed.binding['production']['before_sha256'].pop('tools/fixture_source_294.py');repin(closed)
    elif change=='wrong_deployment_sha':closed.binding['preserved_dp']['A']['checkpoint_sha256']='0'*64;repin(closed)
    elif change=='wrong_deadline':closed.binding['original_deadline']+=1;repin(closed)
    elif change=='late_closure':write(a/'driver_backend_recovery07/status.json',dict(stage='main_stages_complete',utc=1795684572))
    elif change=='overlapping_owner':
        p=r/'owners/backend_recovery07/project_job_started.json';d=common.read_small(p);d['utc']=19;write(p,d)
    else:(a/'final_compute_recovery07.py').write_text('changed final source')
    with pytest.raises((ValueError,FileNotFoundError)):common.closure_gate(r,a)


def test_watcher_current_restore_and_old_reaped_are_both_required(closed):
    assert common.watcher_closed(closed.audit)['receipt']['modified_schedules']==0
    write(closed.audit/'watch07/exit.json',dict(modified_schedules=1,conflicts=[]))
    with pytest.raises(ValueError):common.watcher_closed(closed.audit)


def test_extractor_enriches_real_validated_counts_and_stat_identity(tmp_path,monkeypatch):
    fixtures=module('old_extract_fixtures07','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    status=extractor.extract(root,audit,audit/'selected07',loader=loader)
    assert status['status']=='passed' and len(owner.loads)==8
    row=common.read_small(audit/'selected07/learning_A_frontend.json')
    assert row['checkpoint_stat_identity']==common.identity((root/'learning/A/training/best.pt').stat())
    assert row['minimum_tick_steps']==60 and row['ancestors_counted'] is False
    assert status['extractor_sha256']==common.sha(HERE/'extract_selected_ledgers_recovery07.py')


def test_extractor_still_rejects_tick49_without_partial_publication(tmp_path,monkeypatch):
    fixtures=module('old_extract_fixtures07b','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    key=str(root/'learning/B/backend/B/training/best.pt')
    states[key]['effective_exposure_state']['counts']['suffix'][2]=49
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    status=extractor.extract(root,audit,audit/'failed07',loader=loader)
    assert status['status']=='failed'
    assert [p.name for p in (audit/'failed07').iterdir()]==['extraction_status.json']


def test_assembler_pending_gate_never_enters_original_science(tmp_path,monkeypatch):
    monkeypatch.setattr(assembler.common,'closure_gate',lambda *_:dict(status='pending',missing=['real07 closure']))
    monkeypatch.setattr(assembler.base,'assemble',lambda *_:pytest.fail('Must not read checkpoint before closure'))
    result=assembler.assemble(tmp_path/'run',tmp_path/'output',tmp_path/'audit',tmp_path/'audit/selected')
    assert result['status']=='pending' and result['actual_checkpoint_reads']==0


def test_assembler_owner_costs_include_evaluation_and_failed_owner_union(closed):
    result=assembler.attempt_costs(closed.root)
    assert any(row['owner']=='main_train' for row in result['owner_attempts'])
    assert {row['owner'] for row in result['owner_attempts']} >= {'evaluation/development','evaluation/test','backend_recovery07'}
    assert result['closed_owner_wall_union_seconds']==90


def test_compute_union_includes_failed06_and_new07_without_overlap_double_count(closed,monkeypatch):
    r,a=closed.root,closed.audit
    write(r/'owners/main_cache/project_job_started.json',dict(utc=5))
    write(r/'owners/main_cache/project_exit.json',dict(utc=15,children_reaped=True))
    write(r/'owners/collection/project_job_started.json',dict(utc=1))
    write(r/'collection/collection_summary.json',dict(opportunities=96))
    write(r/'evaluation/test/summary/primary_summary.json',dict(n=64,episodes=256))
    for stage in common.STANDALONE_STAGES:
        write(a/'driver_backend_recovery07'/(stage+'_shared_capacity06_admission.json'),dict(utc=110,capacity=dict(ready=True)))
    monkeypatch.setattr(compute,'time',SimpleNamespace(time=lambda:130))
    result=compute.inventory(r,a)
    assert result['owner_and_standalone_interval_union']['seconds']==105
    assert result['committed_parent_updates']==59926 and result['committed_main_updates']==97200
    assert result['gpu_hours'] is None
    unknown=next(x for x in result['owners'] if x['path'].endswith('/collection'))
    assert unknown['end_utc'] is None and unknown['wall_time_certified'] is False
    failed=next(x for x in result['owners'] if x['path'].endswith('/main_train'))
    assert failed['failure']['error']=='Stage2 returned1'


@pytest.mark.parametrize('owner',['main_cache','main_train','backend_recovery07','development','test'])
def test_compute_rejects_missing_required_real_owner(closed,owner,monkeypatch):
    monkeypatch.setattr(compute.common,'closure_gate',lambda *_:dict(status='closed'))
    monkeypatch.setattr(compute.common,'watcher_closed',lambda *_:{})
    path=closed.root/('evaluation' if owner in ('development','test') else 'owners')/owner
    if (path/'project_job_started.json').exists():(path/'project_job_started.json').unlink()
    with pytest.raises(ValueError,match='start receipt missing'):compute.inventory(closed.root,closed.audit)


def test_export_refuses_missing_selected_evidence_before_destination(tmp_path):
    project=tmp_path/'project';run=project/'run';audit=project/'audit';assembly=audit/'assembly';destination=project/'destination'
    result=exporter.export(project,run,audit,assembly,destination,selected_ledgers=None)
    assert result['status']=='blocked' and result['copied_files']==0 and not destination.exists()


def test_private_base_sources_and_single_path_substitutions_are_pinned():
    for value,filename in [(extractor,'extract_selected_ledgers_recovery05.py'),
                           (assembler,'final_report_assembler.py'),
                           (compute,'final_compute_shared06.py'),
                           (exporter,'export_final_evidence_recovery05.py')]:
        assert common.sha(HERE/filename)==value.BASE_SHA256
    with pytest.raises(ValueError,match='exactly once'):
        common.pinned_module('final_report_assembler.py',assembler.BASE_SHA256,'bad_substitution',
                             replacements=(('absent source string','replacement'),))


@pytest.mark.parametrize('change',['pipeline_failed','pipeline_missing','phase_order','wrong_registry_source',
    'wrong_backend_source','wrong_budget','wrong_activation','capacity_denied','wrong_guard'])
def test_real_runtime_registry_pipeline_and_capacity_bindings_are_required(closed,change):
    a=closed.audit
    if change in ('pipeline_failed','pipeline_missing','phase_order'):
        path=a/'parallel_rear_recovery07_runtime/rear_training/pipeline_events.json';rows=common.read_small(path)
        if change=='pipeline_failed':rows[-1]['returncode']=1
        elif change=='pipeline_missing':rows.pop()
        else:rows[0]['utc']=39
        write(path,rows)
    elif change=='wrong_backend_source':write(a/'driver_backend_recovery07/backend_source_identity.json',dict(sources={}))
    elif change=='capacity_denied':
        write(a/'driver_backend_recovery07/summarize_test_shared_capacity06_admission.json',dict(utc=110,capacity=dict(ready=False)))
    else:
        path=a/'BACKEND_RECOVERY07_REGISTRATION.json';reg=common.read_small(path)
        if change=='wrong_registry_source':reg['source_sha256']={}
        elif change=='wrong_budget':reg['scientific_invariants']['dp_reruns']=1
        elif change=='wrong_guard':reg['stages'][1]['command']=['old_non07_guard.json']
        else:
            path=a/'driver_backend_recovery07/registration.json';reg=common.read_small(path);reg['activated']=False
        write(path,reg)
        if change!='wrong_activation':
            write(a/'driver_backend_recovery07/registration.json',dict(reg,activated=True,activated_utc=30,registry_sha256=common.sha(path)))
    repin(closed)
    with pytest.raises(ValueError):common.closure_gate(closed.root,a)


def test_successful_extraction_is_reused_eight_times_without_more_payload_reads(tmp_path,monkeypatch):
    fixtures=module('old_extract_reuse_fixtures07','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    (audit/'extract_selected_ledgers_recovery07.py').write_bytes((HERE/'extract_selected_ledgers_recovery07.py').read_bytes())
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    selected=audit/'selected07';assert extractor.extract(root,audit,selected,loader=loader)['status']=='passed'
    assembler.CURRENT_ROOT=root;assembler.CURRENT_AUDIT=audit;assembler.SELECTED_LEDGERS=selected
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.pt','Reusing selected ledgers must never reopen a checkpoint'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            for component in ('frontend','backend'):
                directory=root/scope/role/('training' if component=='frontend' else f'backend/{role}/training')
                (directory/'family_phase_curves.csv').write_text('synthetic curve')
                row=assembler.actual_component(directory,component,None,None)
                assert row['additional_checkpoint_reads']==0 and row['exposure']['minimum_tick_steps']==60
    assert len(owner.loads)==8
    with original(root/'learning/A/training/best.pt','wb') as stream:stream.write(b'changed after SHA/mmap pass')
    with pytest.raises(ValueError,match='metadata changed'):
        common.extracted_component(root,audit,selected,'learning','A','frontend')


def test_auxiliary_accounting_retains_zero_failed08_and_partial09(closed,monkeypatch):
    r,a=closed.root,closed.audit
    write(r/'owners/main_cache/project_job_started.json',dict(utc=5));write(r/'owners/main_cache/project_exit.json',dict(utc=15,children_reaped=True))
    write(r/'collection/collection_summary.json',dict(opportunities=96))
    write(r/'evaluation/test/summary/primary_summary.json',dict(n=64,episodes=256))
    for attempt,count in [('leased_archive_shared08',0),('leased_archive_shared09',2)]:
        directory=a/attempt;write(directory/'registration.json',dict(utc=1,binding='synthetic'))
        for i in range(count):write(directory/f'file_{i:05d}.json',dict(saved_bytes=10))
        write(directory/'failure.json',dict(utc=2,error='capacity wait retained',completed_files=count,saved_bytes=10*count))
    result=compute.inventory(r,a)
    assert [(row['attempt'],row['completed_files'],row['saved_bytes']) for row in result['cold_archives']]==[
        ('leased_archive_shared08',0,0),('leased_archive_shared09',2,20)]


@pytest.mark.parametrize('changed',['version','cuda_initialized','read_only_inputs'])
def test_export_collection_integrity_semantics_remain_strict(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(exporter.common,'closure_gate',lambda *_:dict(status='closed'))
    monkeypatch.setattr(exporter.common,'watcher_closed',lambda *_:{})
    audit=tmp_path/'audit';payload=dict(version='paired_collection_integrity_recovery05_v1',passed=True,
        opportunities=96,paired_opportunities=48,observable_pair_checks=dict(passed=47,unverified=1,failed=0,all_verified=False),
        full48_observed_initial_states_verified=False,cuda_initialized=False,read_only_inputs=True)
    payload[changed]='wrong' if changed=='version' else not payload[changed]
    write(audit/'collection_integrity_recovery05_actual01/integrity.json',payload)
    with pytest.raises(ValueError,match='integrity differs'):exporter.closed_receipt_gate(tmp_path,tmp_path/'run',audit)


def test_build_binding_selects_real_before_and_separate_after_snapshot_sets(closed):
    candidate=common.build_binding(closed.root,closed.audit)
    assert candidate.get('status')!='pending'
    assert candidate['production']['snapshot_paths']==closed.binding['production']['snapshot_paths']
    for relative,path in candidate['production']['snapshot_paths'].items():
        assert common.sha(path)==candidate['production']['before_sha256'][relative]
        after=Path(candidate['production']['after_snapshot_paths'][relative])
        assert common.sha(after)==candidate['production']['after_sha256'][relative]
        assert common.sha(path)!=common.sha(after)
        assert candidate['small_file_sha256'][str(Path(path).resolve())]==common.sha(path)
        assert candidate['small_file_sha256'][str(after.resolve())]==common.sha(after)
    closed.binding=candidate;closed.pinned={Path(p) for p in candidate['small_file_sha256']};repin(closed)
    assert common.closure_gate(closed.root,closed.audit)['status']=='closed'


@pytest.mark.parametrize('direction',['before','after'])
def test_build_binding_rejects_swapped_source_snapshot_sha_before_publication(closed,direction):
    relative=sorted(common.AMENDED_PATHS)[0]
    before=closed.audit/'frozen_source'/relative;after=closed.audit/'recovery07_source_snapshot'/relative
    wrong=after.read_bytes() if direction=='before' else before.read_bytes()
    (before if direction=='before' else after).write_bytes(wrong)
    with pytest.raises(ValueError,match='source snapshot SHA differs|amendment snapshot SHA differs'):
        common.build_binding(closed.root,closed.audit)
