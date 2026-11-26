"""Synthetic small-file tests only; never open actual weights or sealed results."""
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


common=module('fixture_common08','final_recovery10_common.py')
extractor=module('fixture_extract08','extract_selected_ledgers_recovery10.py')
assembler=module('fixture_assembler08','final_report_assembler_recovery10.py')
compute=module('fixture_compute08','final_compute_recovery10.py')
exporter=module('fixture_export08','export_final_evidence_recovery10.py')
oldtests=module('retained_test07_fixture08','test_final_evidence_recovery07.py')


def write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row))


def fullstat(path):
    value=path.stat()
    return dict(dev=value.st_dev,ino=value.st_ino,size=value.st_size,nlink=value.st_nlink,
        mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,allocated_bytes=value.st_blocks*512)


def repin(f):
    f.binding['small_file_sha256']={str(p.resolve()):common.sha(p) for p in f.pinned}
    write(f.audit/common.BINDING_NAME,f.binding)
    write(f.audit/common.REVIEW_NAME,dict(status='cpu_review_passed_not_activated',
        binding_sha256=common.sha(f.audit/common.BINDING_NAME),
        source_sha256={name:common.sha(f.audit/name) for name in common.FINAL_TOOLS}))


@pytest.fixture
def closed(tmp_path):
    old=oldtests.closed.__wrapped__(tmp_path);r,a=old.root,old.audit;pinned=old.pinned
    def add(path,row):write(path,row);pinned.add(path);return path
    pinned.add(a/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json')
    for name in common.FINAL_TOOLS:(a/name).write_text('synthetic final10 '+name)
    for name in common.RUNTIME_SOURCES:
        if not (a/name).exists():(a/name).write_text('synthetic08 runtime '+name)
        pinned.add(a/name)
    for name in ('closure.json',):
        path=a/'driver_backend_recovery07'/name
        path.unlink();pinned.discard(path)
    add(a/'driver_backend_recovery07/status.json',dict(stage='stopped_with_evidence',utc=130))
    add(a/'driver_backend_recovery07/failure.json',dict(error='retained07 cache failure'))
    add(a/'driver_backend_recovery07/driver_exit.json',dict(returned_after_owned_helpers=True,utc=132))
    add(r/'owners/backend_recovery07/project_failure.json',dict(error='retained07 failed body'))
    add(r/'owners/backend_recovery07/project_exit.json',dict(utc=131,children_reaped=True))
    add(r/'cycle_logs_recovery07/train_backend_recovery07_exit.json',dict(returncode=1,utc=130))
    add(a/'OLD_WATCHER07_CLOSED_FOR_RECOVERY08.json',dict(source_pid_reaped=True,backend07_failure_preserved=True,
        restored_verified=[{} for _ in range(29)],service='MainPID=0\nActiveState=inactive\n'))
    add(a/'watch_shared_capacity10/resource_watch_v4_exit.json',dict(modified_schedules=0,conflicts=[],exact_original_byte_restoration=True))
    add(a/'OLD_WATCHER08_CLOSED_FOR_RECOVERY09.json',dict(source_pid_reaped=True,
        restored_verified=[{} for _ in range(32)],service='MainPID=0\nActiveState=inactive\n'))
    add(a/'BACKEND_RECOVERY08_ACTUAL_TERMINAL_BOUNDARY09.json',dict(status='failed',A_error_origin='retained07'))
    add(a/'driver_backend_recovery08/status.json',dict(stage='stopped_with_evidence',utc=134))
    add(a/'driver_backend_recovery08/failure.json',dict(error='atomic mutable telemetry replaced'))
    add(a/'driver_backend_recovery08/driver_exit.json',dict(returned_after_owned_helpers=True,utc=135))
    add(r/'owners/backend_recovery08/project_job_started.json',dict(utc=132))
    add(r/'owners/backend_recovery08/project_failure.json',dict(error='atomic telemetry replaced'))
    add(r/'owners/backend_recovery08/project_exit.json',dict(utc=134,children_reaped=True))
    add(r/'cycle_logs_recovery08/train_backend_recovery08_exit.json',dict(returncode=1,utc=134))
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            for component,sub in [('frontend','training'),('backend',f'backend/{role}/training')]:
                directory=r/scope/role/sub
                (directory/'family_phase_curves.csv').write_text('family,phase,loss\nfixture,near,1\n')
    dedup=[];applied=[]
    for role in 'AB':
        directory=r/'learning'/role/'training';best,last=directory/'best.pt',directory/'last.pt'
        last.write_bytes(best.read_bytes());before_best,before_last=fullstat(best),fullstat(last)
        dedup.append(dict(role=role,before_best=before_best,before_last=before_last))
        if role=='A':
            last.unlink();os.link(best,last)
            applied.append(dict(role='A',before_best=before_best,before_last=before_last,
                after_best=fullstat(best),after_last=fullstat(last),potential_reclaimed_after_lease_close_bytes=before_last['allocated_bytes']))
    add(a/'CLOSED_MAIN_DP_DEDUP08_REVIEW.json',dict(rows=dedup))
    add(a/'CLOSED_MAIN_DP_DEDUP08_APPLY01.json',dict(status='failed_with_prior_completed_rows_preserved',rows=applied,
        checkpoint_payload_reads=0,error='synthetic lease lifecycle failure'))
    add(a/'CLOSED_MAIN_DP_DEDUP08_PARTIAL_DIAGNOSIS01.json',dict(status='partial_A_completed_B_untouched_no_real_retry'))
    oldreg=common.read_small(a/'BACKEND_RECOVERY07_REGISTRATION.json')
    prior=copy.deepcopy(oldreg)
    prior['prior_registration_sha256']=common.sha(a/'BACKEND_RECOVERY07_REGISTRATION.json')
    add(a/'BACKEND_RECOVERY08_REGISTRATION.json',prior)
    prior09=copy.deepcopy(prior)
    prior09['prior_registration_sha256']=common.sha(a/'BACKEND_RECOVERY08_REGISTRATION.json')
    add(a/'BACKEND_RECOVERY09_REGISTRATION.json',prior09)
    add(a/'driver_backend_recovery09/status.json',dict(stage='stopped_with_evidence',utc=136))
    add(a/'driver_backend_recovery09/failure.json',dict(error='native null pause sentinel'))
    add(a/'driver_backend_recovery09/driver_exit.json',dict(returned_after_owned_helpers=True,utc=138))
    add(r/'owners/backend_recovery09/project_job_started.json',dict(utc=135))
    add(r/'owners/backend_recovery09/project_failure.json',dict(error='native null pause sentinel'))
    add(r/'owners/backend_recovery09/project_exit.json',dict(utc=137,children_reaped=True))
    add(r/'cycle_logs_recovery09/train_backend_recovery09_exit.json',dict(returncode=1,utc=136))
    add(a/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',dict(source_pid_reaped=True,
        restored_verified=[{} for _ in range(32)],service='MainPID=0\nActiveState=inactive\n'))
    add(a/'BACKEND_RECOVERY09_ACTUAL_TERMINAL_BOUNDARY10.json',dict(status='failed',B_error_origin='retained08'))
    # Historical09 really restored33 identities. Current10 uses a deliberately
    # different35-entry population so no code can silently assume the old count.
    def fixture_watch(watch,pid,count):
        directory=a/watch;backup=directory/f'resource_watch_v4_backups_1_{pid}'
        records=[]
        for i in range(count):
            target=r/f'fixture_{watch}_{i:03d}'/'concurrency.json'
            existed=watch.endswith('10') and i==0
            raw=b'{"fixture":true}\n' if existed else None
            if existed:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            row=dict(path=str(target),existed=existed,
                sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None,
                base64=base64.b64encode(raw).decode() if raw is not None else None)
            add(backup/(hashlib.sha256(str(target).encode()).hexdigest()+'.json'),row)
            records.append({key:row[key] for key in ('path','existed','sha256')})
        exitpath=add(directory/'resource_watch_v4_exit.json',dict(pid=pid,modified_schedules=0,
            conflicts=[],exact_original_byte_restoration=True,restored=[row['path'] for row in records]))
        return records,exitpath
    priorrows,priorexit=fixture_watch('watch_shared_capacity09',99,33)
    fixture_watch('watch_shared_capacity10',100,35)
    add(a/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',dict(source_pid_reaped=True,pid=99,
        restored_verified=priorrows,service='MainPID=0\nActiveState=inactive\n',exit_sha256=common.sha(priorexit)))
    preserved=copy.deepcopy(old.binding['preserved_dp'])
    for role in 'AB':preserved[role]['checkpoint_stat_identity']=common.identity((r/'learning'/role/'training/best.pt').stat())
    owner=r/'owners/backend_recovery10';job=dict(resource_root=str(owner),stages=[dict(entry='synthetic08')])
    add(a/'proposed_backend_recovery10.json',job)
    for directory in (owner,r/'evaluation/development',r/'evaluation/test'):
        actualjob=job if directory==owner else dict(resource_root=str(directory),stages=[dict(entry='fixture')])
        add(directory/'project_job_started.json',dict(utc=140 if directory==owner else 190,job=actualjob,commands=[['fixture']]))
        add(directory/'project_complete.json',dict(stages=1));add(directory/'project_exit.json',dict(utc=220,children_reaped=True))
    nominal={f'episode{i}':dict(sha256=hashlib.sha256(str(i).encode()).hexdigest(),size_bytes=1) for i in range(234)}
    for role in 'AB':
        actual=dict(nominal)
        if role=='B': actual.update({f'episode{i}':dict(sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(234,237)})
        add(r/'learning'/role/'backend'/role/'nominal/index.json',dict(episodes=actual))
        add(r/'learning'/role/'backend/prepared.json',dict(source_frontend_sha256=preserved[role]['checkpoint_sha256'],train=1,validation=1,epochs=60))
    commands={}
    for key in ('A_cache','A_train','A_offline','B_cache','B_train','B_offline'):
        commands[key]=dict(argv=['-m','tools.tight_finish',key],worker_root=str(r/'learning'/key[0]/'backend'),lease_roots=[str(r/'learning'/key[0]/'backend')])
    stages=[]
    for stage in common.REMAINING_STAGES:
        gpu=stage not in ('verify_final_actual_training','prepare_development','prepare_test')
        command=[str(a/'presealed_guard_main_recovery10.json')] if stage=='verify_final_actual_training' else [stage]
        stages.append(dict(name=stage,command=command,gpu=gpu))
        add(r/'cycle_logs_recovery10'/(stage+'_exit.json'),dict(returncode=0,utc=225))
        if gpu:add(a/'driver_backend_recovery10'/(stage+'_shared_capacity06_admission.json'),dict(utc=180,capacity=dict(ready=True)))
    reg=dict(version='paired_backend_missing_cache_recovery10_v1',activated=False,root=str(r),owner=str(owner),
        original_deadline=1795684571.2261844,logs=str(r/'cycle_logs_recovery10'),watcher=str(a/'watch_shared_capacity10'),
        job=str(a/'proposed_backend_recovery10.json'),job_sha256=common.sha(a/'proposed_backend_recovery10.json'),
        source_sha256=oldreg['source_sha256'],old_production_sha256=oldreg['old_production_sha256'],
        prior_registration_sha256=common.sha(a/'BACKEND_RECOVERY09_REGISTRATION.json'),
        orchestration_sha256={str(a/name):common.sha(a/name) for name in common.RUNTIME_SOURCES},
        preserved_receipts_sha256={str(a/'driver_backend_recovery07/failure.json'):common.sha(a/'driver_backend_recovery07/failure.json')},
        scientific_configuration_sha256={str(r/'session.json'):common.sha(r/'session.json')},preserved_dp=preserved,
        preserved_cache={role:dict(count=count,episodes={k:v for k,v in nominal.items() if int(k[7:])<count}) for role,count in [('A',174),('B',16)]},
        preserved_prepared_backends={role:dict(common.read_small(r/'learning'/role/'backend/prepared.json'),prepared_sha256=common.sha(r/'learning'/role/'backend/prepared.json')) for role in 'AB'},
        scoped_resource_wait10=dict(version='scoped_preentry_resource_wait10_atomicfd_nullpause_v1',wrapper_sha256=common.sha(a/'scoped_resource_wait10.py'),commands=commands),stages=stages,
        scientific_invariants=dict(dp_reruns=0,dp_new_updates=0,collection_retries=0,new_opportunities=0,
            deploy_epoch=60,optimizer_epochs=60,rear_updates_per_role=38400,caches_own_dp=True,
            retained_complete_cache_episodes={'A':174,'B':16},prepare_reruns=0,derived_unpersisted_episode_restart_from_tick0=True,
            external_presence_is_veto=False,encoder_frozen=True,sealed_test_families=64))
    add(a/'BACKEND_RECOVERY10_REGISTRATION.json',reg)
    add(a/'driver_backend_recovery10/registration.json',dict(reg,activated=True,activated_utc=140,
        registry_sha256=common.sha(a/'BACKEND_RECOVERY10_REGISTRATION.json')))
    add(a/'driver_backend_recovery10/status.json',dict(stage='main_stages_complete',utc=230))
    add(a/'driver_backend_recovery10/closure.json',dict(stage='main_stages_complete',status='backend_and_remaining_main_stages_complete',utc=230))
    add(a/'driver_backend_recovery10/driver_exit.json',dict(returned_after_owned_helpers=True,utc=231))
    add(a/'driver_backend_recovery10/backend_source_identity.json',dict(sources=reg['source_sha256']))
    add(a/'presealed_guard_main_recovery10.json',dict(status='passed',mode='main',checkpoint_reads=8,cuda_initialized=False,observed_utc=185))
    runtime=a/'parallel_rear_recovery10_runtime'
    for phase,pop,start in [('caches',{'A':[0],'B':[0]},150),('rear_training',{'A':[0],'B':[0]},160),('serial_offline',{'A':[0],'B':[0]},170)]:
        events=[]
        for role,values in pop.items():
            for stage in values:
                key=role+('_cache' if phase=='caches' else '_train' if phase=='rear_training' else '_offline')
                events.extend([dict(role=role,stage=stage,pid=1,command=['python','--command-key',key],utc=start),dict(role=role,stage=stage,returncode=0,utc=start+1)])
        add(runtime/phase/'pipeline_events.json',events)
    for name in ('caches','rear_trainers'):add(runtime/(name+'_admission.json'),dict(parallel=True,pair=dict(allowed=True),single=dict(allowed=True)))
    add(runtime/'complete.json',dict(utc=180,caches_parallel=True,rears_parallel=True,offline_serial=True,completed_cache_episodes_recomputed=0))
    for key,row in commands.items():
        add(runtime/('wait_'+key)/'registration.json',dict(version='scoped_preentry_resource_wait10_atomicfd_nullpause_v1',command_key=key,
            original_command=row['argv'],registration_sha256=common.sha(a/'BACKEND_RECOVERY10_REGISTRATION.json'),
            wrapper_sha256=reg['scoped_resource_wait10']['wrapper_sha256'],worker_root=row['worker_root'],retry_scope='original_context_manager_enter_only'))
    proof=a/'BACKEND_RECOVERY10_ACTUAL_PREFLIGHT.json'
    add(proof,dict(version='backend_recovery10_actual_preflight_v1',
        status='new10prepared_preserved_payload_checks_passed_not_activated',utc=2e9,
        registration_sha256=common.sha(a/'BACKEND_RECOVERY10_REGISTRATION.json'),
        wrapper_version='scoped_preentry_resource_wait10_atomicfd_nullpause_v1',source_files=295,
        preserved_cache_counts={'A':174,'B':16},optimizer_updates=0,activated=False,
        preserved_dp={role:dict(checkpoint_sha256=row['checkpoint_sha256'],
            deployment_sha256=common.sha(r/'learning'/role/'training/deployment.json'),
            finished_sha256=common.sha(r/'learning'/role/'training/finished.json'),epoch=60,updates=10200)
            for role,row in preserved.items()}))
    binding=copy.deepcopy(old.binding);binding.update(version='paired_final_recovery10_bindings_v1',preserved_dp=preserved,
        preserved_dp_preflight_path=str(proof),stage_exit_paths={s:str(r/'cycle_logs_recovery10'/(s+'_exit.json')) for s in common.REMAINING_STAGES},
        watcher_exit_path=str(a/'watch_shared_capacity10/resource_watch_v4_exit.json'))
    binding['production']['after_snapshot_paths']={relative:str(a/'recovery07_source_snapshot'/relative) for relative in common.AMENDED_PATHS}
    pinned.add(r/'evaluation/test/registration.json');pinned.add(r/'learning/training_source_identity.json')
    f=SimpleNamespace(root=r,audit=a,project=old.project,pinned=pinned,binding=binding,reg=reg)
    repin(f);return f


def test_truthful_failed06_07_plus_actual10_closure_reads_no_weights(closed,monkeypatch):
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.pt','Lifecycle must not read payloads'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=common.closure_gate(closed.root,closed.audit)
    assert result['status']=='closed' and result['checkpoint_reads']==0
    assert all(result[k] is True for k in ('failed06_preserved','failed07_preserved','failed08_preserved','failed09_preserved'))
    assert result['partial_dp_dedup']['status']=='A_completed_B_aborted'
    assert result['payload_read_accounting']['presealed_main_guard_checkpoint_reads']==8
    assert result['payload_read_accounting']['final_detailed_ledger_checkpoint_reads']==8
    assert result['watcher_closure']['previous09_restoration_audit']['backup_count']==33
    assert result['watcher_closure']['current_restoration_audit']['backup_count']==35


@pytest.mark.parametrize('change',['07_success','07_unreaped','07_rc0','08_failure','08_unreaped','08_stage_failure',
    'source_changed','wrong_snapshot','dp_changed','B_deduped','wait_body_retry','wait_failed','cache_recomputed',
    'cache159_changed','guard4','guard_late','proof_wrong_registry','proof_predates_ctime','current_owner_overlaps08',
    'test_wrong_sources','deadline_changed','late_closure'])
def test_rejects_false_ancestry_incomplete_science_and_changed_identity(closed,change):
    r,a=closed.root,closed.audit
    def mutate(path,fn):
        value=common.read_small(path);fn(value);write(path,value)
    if change=='07_success':write(a/'driver_backend_recovery07/status.json',dict(stage='main_stages_complete'))
    elif change=='07_unreaped':write(r/'owners/backend_recovery07/project_exit.json',dict(children_reaped=False,utc=131))
    elif change=='07_rc0':write(r/'cycle_logs_recovery07/train_backend_recovery07_exit.json',dict(returncode=0))
    elif change=='08_failure':write(a/'driver_backend_recovery10/failure.json',dict(error='failure08'))
    elif change=='08_unreaped':write(r/'owners/backend_recovery10/project_exit.json',dict(children_reaped=False,utc=220))
    elif change=='08_stage_failure':write(r/'cycle_logs_recovery10/evaluate_test_exit.json',dict(returncode=1))
    elif change=='source_changed':(closed.project/'tools/fixture_source_004.py').write_text('unexpected source')
    elif change=='wrong_snapshot':closed.binding['production']['after_snapshot_paths']=closed.binding['production']['snapshot_paths']
    elif change=='dp_changed':(r/'learning/A/training/best.pt').write_bytes(b'changed synthetic')
    elif change=='B_deduped':
        last=r/'learning/B/training/last.pt';last.unlink();os.link(r/'learning/B/training/best.pt',last)
    elif change=='wait_body_retry':write(a/'parallel_rear_recovery10_runtime/wait_A_train/exit.json',dict(success=True,body_retries=1));closed.pinned.add(a/'parallel_rear_recovery10_runtime/wait_A_train/exit.json')
    elif change=='wait_failed':write(a/'parallel_rear_recovery10_runtime/wait_B_cache/exit.json',dict(success=False));closed.pinned.add(a/'parallel_rear_recovery10_runtime/wait_B_cache/exit.json')
    elif change=='cache_recomputed':mutate(a/'parallel_rear_recovery10_runtime/complete.json',lambda v:v.update(completed_cache_episodes_recomputed=159))
    elif change=='cache159_changed':mutate(r/'learning/A/backend/A/nominal/index.json',lambda v:v['episodes']['episode1'].update(sha256='bad'))
    elif change=='guard4':mutate(a/'presealed_guard_main_recovery10.json',lambda v:v.update(checkpoint_reads=4))
    elif change=='guard_late':mutate(a/'presealed_guard_main_recovery10.json',lambda v:v.update(observed_utc=200))
    elif change=='proof_wrong_registry':mutate(a/'BACKEND_RECOVERY10_ACTUAL_PREFLIGHT.json',lambda v:v.update(registration_sha256='bad'))
    elif change=='proof_predates_ctime':mutate(a/'BACKEND_RECOVERY10_ACTUAL_PREFLIGHT.json',lambda v:v.update(utc=0))
    elif change=='current_owner_overlaps08':mutate(r/'owners/backend_recovery10/project_job_started.json',lambda v:v.update(utc=120))
    elif change=='test_wrong_sources':mutate(r/'evaluation/test/registration.json',lambda v:v.update(source_sha256={}))
    elif change=='deadline_changed':closed.binding['original_deadline']+=1
    elif change=='late_closure':mutate(a/'driver_backend_recovery10/status.json',lambda v:v.update(utc=1795684572))
    # Repinning emulates a root reviewing the changed bytes; semantic gates still must reject.
    repin(closed)
    with pytest.raises((ValueError,FileNotFoundError)):common.closure_gate(r,a)


def test_builder_pending_never_opens_weight_or_sealed_result(tmp_path,monkeypatch,closed):
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.suffix!='.pt'
        assert path!=closed.root/'evaluation/test/registration.json','Do not open sealed registration before closure'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    (closed.audit/'driver_backend_recovery10/closure.json').unlink()
    result=common.build_binding(closed.root,closed.audit)
    assert result['status']=='pending' and result['checkpoint_reads']==0


def test_watcher10_requires_restore_and_old09_reaped(closed):
    assert common.watcher_closed(closed.audit)['receipt']['modified_schedules']==0
    write(closed.audit/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',dict(source_pid_reaped=False))
    with pytest.raises(ValueError):common.watcher_closed(closed.audit)


def test_extractor_keeps_eight_actual_counts_with_stat_identity(tmp_path,monkeypatch):
    fixtures=module('fixture_selected08','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    result=extractor.extract(root,audit,audit/'selected08',loader=loader)
    assert result['status']=='passed' and len(owner.loads)==8
    row=common.read_small(audit/'selected08/learning_A_frontend.json')
    assert row['checkpoint_stat_identity']==common.identity((root/'learning/A/training/best.pt').stat())
    assert result['extractor_sha256']==common.sha(HERE/'extract_selected_ledgers_recovery10.py')


def test_extractor_retains_below50_rejection_without_component_publication(tmp_path,monkeypatch):
    fixtures=module('fixture_selected08_bad','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    states[str(root/'learning/B/backend/B/training/best.pt')]['effective_exposure_state']['counts']['suffix'][2]=49
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    result=extractor.extract(root,audit,audit/'failed08',loader=loader)
    assert result['status']=='failed' and [p.name for p in (audit/'failed08').iterdir()]==['extraction_status.json']


def test_pending_assembler_does_not_enter_scientific_reader(tmp_path,monkeypatch):
    monkeypatch.setattr(assembler.common,'closure_gate',lambda *_:dict(status='pending',missing=['actual10 closure']))
    monkeypatch.setattr(assembler.base,'assemble',lambda *_:pytest.fail('No science before closure'))
    result=assembler.assemble(tmp_path/'run',tmp_path/'out',tmp_path/'audit',tmp_path/'audit/selected')
    assert result['status']=='pending' and result['actual_checkpoint_reads']==0


def test_pending_export_never_creates_destination(tmp_path):
    result=exporter.export(tmp_path,tmp_path/'run',tmp_path/'audit',tmp_path/'assembly',tmp_path/'public',selected_ledgers=tmp_path/'selected')
    assert result['status']=='blocked' and result['destination_created'] is False and not (tmp_path/'public').exists()


def test_successful_extraction_is_reused_eight_times_without_more_payload_reads(tmp_path,monkeypatch):
    fixtures=module('old_extract_reuse_fixtures08','test_extract_selected_ledgers_recovery05.py')
    owner=fixtures.SelectedLedgers();root,audit,states,loader=owner.fixture(str(tmp_path))
    (audit/'extract_selected_ledgers_recovery10.py').write_bytes((HERE/'extract_selected_ledgers_recovery10.py').read_bytes())
    monkeypatch.setattr(extractor.base,'closure_gate',lambda *_:dict(status='closed',checkpoint_reads=0))
    selected=audit/'selected08';assert extractor.extract(root,audit,selected,loader=loader)['status']=='passed'
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



def prepare_compute(closed):
    r,a=closed.root,closed.audit
    write(r/'owners/main_cache/project_job_started.json',dict(utc=5))
    write(r/'owners/main_cache/project_exit.json',dict(utc=15,children_reaped=True))
    write(r/'owners/collection/project_job_started.json',dict(utc=1))
    write(r/'collection/collection_summary.json',dict(opportunities=96))
    write(r/'evaluation/test/summary/primary_summary.json',dict(n=64,episodes=256))


def test_compute_retains_failed06_07_and_partial_auxiliary_archive08_09(closed,monkeypatch):
    prepare_compute(closed);r,a=closed.root,closed.audit
    for attempt,count in [('leased_archive_shared08',0),('leased_archive_shared09',14)]:
        directory=a/attempt;write(directory/'registration.json',dict(utc=1,binding='synthetic'))
        for i in range(count):write(directory/f'file_{i:05d}.json',dict(saved_bytes=10))
        write(directory/'failure.json',dict(utc=2,error='retained auxiliary failure',completed_files=count,saved_bytes=count*10))
    monkeypatch.setattr(compute,'time',SimpleNamespace(time=lambda:240))
    result=compute.inventory(r,a)
    assert result['committed_parent_updates']==59926 and result['committed_main_updates']==97200
    assert result['owner_and_standalone_interval_union']['seconds']==205
    assert result['gpu_hours'] is None and result['checkpoint_read_accounting']['presealed_main_guard_checkpoint_reads']==8
    assert result['dp_dedup_partial']['rows'][0]['role']=='A'
    assert [(row['attempt'],row['completed_files'],row['saved_bytes']) for row in result['cold_archives']]==[
        ('leased_archive_shared08',0,0),('leased_archive_shared09',14,140)]
    failed={Path(row['path']).name for row in result['owners'] if row.get('failure')}
    assert failed >= {'main_train','backend_recovery07','backend_recovery08','backend_recovery09'}
    unknown=next(row for row in result['owners'] if row['path'].endswith('/collection'))
    assert unknown['wall_time_certified'] is False and unknown['end_utc'] is None


@pytest.mark.parametrize('phase',['caches','rear_training','serial_offline'])
def test_pipeline_stage_failure_rejected_even_if_root_repins(closed,phase):
    path=closed.audit/'parallel_rear_recovery10_runtime'/phase/'pipeline_events.json'
    rows=common.read_small(path);rows[-1]['returncode']=1;write(path,rows);repin(closed)
    with pytest.raises(ValueError,match='phase populations'):common.closure_gate(closed.root,closed.audit)


@pytest.mark.parametrize('owner',['main_cache','main_train','backend_recovery07','backend_recovery08','backend_recovery09','backend_recovery10','development','test'])
def test_compute_requires_all_actual_scientific_owner_starts(closed,owner,monkeypatch):
    prepare_compute(closed)
    monkeypatch.setattr(compute.common,'closure_gate',lambda *_:dict(status='closed'))
    monkeypatch.setattr(compute.common,'watcher_closed',lambda *_:{})
    path=closed.root/('evaluation' if owner in ('development','test') else 'owners')/owner/'project_job_started.json'
    path.unlink()
    with pytest.raises(ValueError,match='start receipt missing'):compute.inventory(closed.root,closed.audit)


def test_builder_keeps_before_after_snapshots_and_roundtrips_actual_binding(closed):
    candidate=common.build_binding(closed.root,closed.audit)
    assert candidate.get('status')!='pending'
    for relative,path in candidate['production']['snapshot_paths'].items():
        after=candidate['production']['after_snapshot_paths'][relative]
        assert common.sha(path)==candidate['production']['before_sha256'][relative]
        assert common.sha(after)==candidate['production']['after_sha256'][relative]
        assert common.sha(path)!=common.sha(after)
    closed.binding=candidate;closed.pinned={Path(p) for p in candidate['small_file_sha256']};repin(closed)
    assert common.closure_gate(closed.root,closed.audit)['status']=='closed'


def test_private_scientific_engines_and_exact_single_path_replacements_remain_pinned():
    for item,filename in [(extractor,'extract_selected_ledgers_recovery05.py'),(assembler,'final_report_assembler.py'),
        (compute,'final_compute_shared06.py'),(exporter,'export_final_evidence_recovery05.py')]:
        assert common.sha(HERE/filename)==item.BASE_SHA256
    with pytest.raises(ValueError,match='exactly once'):
        common.pinned_module('final_report_assembler.py',assembler.BASE_SHA256,'wrong08',replacements=(('absent','bad'),))


@pytest.mark.parametrize('changed',['none','wrong_registry','wrong_dp_sha','old_timestamp','wrong_cache_population'])
def test_direct10_actual_payload_proof_is_required(closed,changed):
    path=closed.audit/'BACKEND_RECOVERY10_ACTUAL_PREFLIGHT.json'
    proof=common.read_small(path)
    if changed=='wrong_registry':proof['registration_sha256']='bad'
    elif changed=='wrong_dp_sha':proof['preserved_dp']['B']['checkpoint_sha256']='bad'
    elif changed=='old_timestamp':proof['utc']=0
    elif changed=='wrong_cache_population':proof['preserved_cache_counts']['B']=0
    write(path,proof);repin(closed)
    binding,bindings=common.reviewed_binding(closed.root,closed.audit)
    if changed=='none':assert common.checkpoint_proof_gate(closed.root,closed.audit,binding,bindings)['checkpoint_payload_reads']==0
    else:
        with pytest.raises(ValueError):common.checkpoint_proof_gate(closed.root,closed.audit,binding,bindings)


@pytest.mark.parametrize('change',['failure_hidden','unreaped','rc0','watcher31','prepared_changed','B_cache236','A160_changed','B1_changed'])
def test10_rejects_fabricated08_success_or_lost_preserved10_boundary(closed,change):
    r,a=closed.root,closed.audit
    if change=='failure_hidden':write(a/'driver_backend_recovery08/status.json',dict(stage='main_stages_complete'))
    elif change=='unreaped':write(r/'owners/backend_recovery08/project_exit.json',dict(children_reaped=False))
    elif change=='rc0':write(r/'cycle_logs_recovery08/train_backend_recovery08_exit.json',dict(returncode=0))
    elif change=='watcher31':write(a/'OLD_WATCHER08_CLOSED_FOR_RECOVERY09.json',dict(source_pid_reaped=True,restored_verified=[{}]*31))
    elif change=='prepared_changed':write(r/'learning/B/backend/prepared.json',dict(changed=True))
    elif change=='B_cache236':
        path=r/'learning/B/backend/B/nominal/index.json';v=common.read_small(path);v['episodes'].pop('episode236');write(path,v)
    else:
        role='A' if change=='A160_changed' else 'B';key='episode159' if role=='A' else 'episode0'
        path=r/'learning'/role/'backend'/role/'nominal/index.json';v=common.read_small(path);v['episodes'][key]['sha256']='bad';write(path,v)
    repin(closed)
    with pytest.raises(ValueError):common.closure_gate(r,a)


def test_native_os_exit_wrapper_finalizer_absence_is_explicit_and_parent_success_required(closed):
    result=common.closure_gate(closed.root,closed.audit)
    assert result['wrapper_closure']['wrapper_finalizer_receipts_present']=={k:False for k in closed.reg['scoped_resource_wait10']['commands']}
    assert 'os._exit' in result['wrapper_closure']['closure_proof']
    path=closed.audit/'parallel_rear_recovery10_runtime/rear_training/pipeline_events.json'
    rows=common.read_small(path);rows[-1]['returncode']=1;write(path,rows);repin(closed)
    with pytest.raises(ValueError):common.closure_gate(closed.root,closed.audit)


@pytest.mark.parametrize('change',['hidden_success','unreaped','rc0','watcher31','lost_boundary'])
def test_failed09_is_retained_as_failure_with_exact_closed_watch09(closed,change):
    r,a=closed.root,closed.audit
    if change=='hidden_success':write(a/'driver_backend_recovery09/status.json',dict(stage='main_stages_complete'))
    elif change=='unreaped':write(r/'owners/backend_recovery09/project_exit.json',dict(children_reaped=False))
    elif change=='rc0':write(r/'cycle_logs_recovery09/train_backend_recovery09_exit.json',dict(returncode=0))
    elif change=='watcher31':write(a/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',dict(source_pid_reaped=True,restored_verified=[{}]*31))
    else:
        path=a/'BACKEND_RECOVERY09_ACTUAL_TERMINAL_BOUNDARY10.json';path.unlink();closed.pinned.discard(path)
    repin(closed)
    with pytest.raises((ValueError,FileNotFoundError)):common.closure_gate(r,a)


@pytest.mark.parametrize('key',['A_train','B_train','A_offline','B_offline'])
def test_normal_native_return_requires_real_finalizer_while_only_tight_finish_can_os_exit(closed,key):
    a=closed.audit;reg=closed.reg;row=reg['scoped_resource_wait10']['commands'][key]
    row['argv']=['-m','tools.tight_paired_learning','rear-train'] if key.endswith('train') else ['-m','visuotactile.anchor_continue','residual-offline']
    write(a/'BACKEND_RECOVERY10_REGISTRATION.json',reg)
    active=common.read_small(a/'driver_backend_recovery10/registration.json')
    write(a/'driver_backend_recovery10/registration.json',dict(reg,activated=True,activated_utc=active['activated_utc'],registry_sha256=common.sha(a/'BACKEND_RECOVERY10_REGISTRATION.json')))
    runtime=a/'parallel_rear_recovery10_runtime'
    for name in reg['scoped_resource_wait10']['commands']:
        path=runtime/('wait_'+name)/'registration.json';start=common.read_small(path)
        start['registration_sha256']=common.sha(a/'BACKEND_RECOVERY10_REGISTRATION.json')
        start['original_command']=reg['scoped_resource_wait10']['commands'][name]['argv'];write(path,start)
    proofpath=a/'BACKEND_RECOVERY10_ACTUAL_PREFLIGHT.json';proof=common.read_small(proofpath)
    proof['registration_sha256']=common.sha(a/'BACKEND_RECOVERY10_REGISTRATION.json');write(proofpath,proof)
    repin(closed)
    with pytest.raises(ValueError,match='Missing wrapper finalizer'):common.closure_gate(closed.root,a)
    path=runtime/('wait_'+key)/'exit.json'
    write(path,dict(version='scoped_preentry_resource_wait10_atomicfd_nullpause_v1',success=True,
        body_retries=0,original_command_replayed=False,method_restored=True,source_files_modified=0))
    closed.pinned.add(path);repin(closed)
    result=common.closure_gate(closed.root,a)
    assert result['wrapper_closure']['wrapper_finalizer_receipts_present'][key] is True


@pytest.mark.parametrize('change',['conflict','modified','no_exact_restore'])
def test_selected_payload_closure_always_requires_current_watch10_exact_restore(closed,change):
    path=closed.audit/'watch_shared_capacity10/resource_watch_v4_exit.json'
    row=common.read_small(path)
    if change=='conflict':row['conflicts']=['changed schedule']
    elif change=='modified':row['modified_schedules']=1
    else:row['exact_original_byte_restoration']=False
    write(path,row);repin(closed)
    with pytest.raises(ValueError,match='Watcher10 must restore'):common.closure_gate(closed.root,closed.audit)


@pytest.mark.parametrize('change',['drop_one','duplicate_identity','changed_path','changed_sha','wrong_exit_sha','wrong_backup_count','backup_content_tamper'])
def test_all33_historical09_restoration_identities_match_frozen_backup_population(closed,change):
    a=closed.audit;path=a/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json';row=common.read_small(path)
    if change=='drop_one':row['restored_verified'].pop()
    elif change=='duplicate_identity':row['restored_verified'][1]=copy.deepcopy(row['restored_verified'][0])
    elif change=='changed_path':row['restored_verified'][0]['path']=str(closed.root/'invented/concurrency.json')
    elif change=='changed_sha':row['restored_verified'][0]['sha256']='wrong'
    elif change=='wrong_exit_sha':row['exit_sha256']='wrong'
    elif change=='wrong_backup_count':
        backup=next((a/'watch_shared_capacity09').glob('resource_watch_v4_backups_*/*.json'))
        backup.unlink();closed.pinned.discard(backup)
    elif change=='backup_content_tamper':
        backup=next((a/'watch_shared_capacity09').glob('resource_watch_v4_backups_*/*.json'))
        raw=common.read_small(backup);raw['base64']='not-null';write(backup,raw)
    write(path,row);repin(closed)
    with pytest.raises((ValueError,FileNotFoundError)):common.closure_gate(closed.root,a)


@pytest.mark.parametrize('change',['existence','bytes','exit_unknown_path','ambiguous_backup_attempt'])
def test_current10_restoration_checks_actual_backup_population_and_original_bytes(closed,change):
    a,r=closed.audit,closed.root
    target=r/'fixture_watch_shared_capacity10_000/concurrency.json'
    if change=='existence':target.unlink()
    elif change=='bytes':target.write_bytes(b'changed original schedule')
    elif change=='exit_unknown_path':
        path=a/'watch_shared_capacity10/resource_watch_v4_exit.json';row=common.read_small(path)
        row['restored'].append(str(r/'unknown/concurrency.json'));write(path,row)
    else:(a/'watch_shared_capacity10/resource_watch_v4_backups_2_100').mkdir()
    repin(closed)
    with pytest.raises(ValueError):common.closure_gate(r,a)


def test_current10_restoration_count_is_audited_not_assumed33_or35(closed):
    a,r=closed.audit,closed.root
    backup=next((a/'watch_shared_capacity10').glob('resource_watch_v4_backups_*'))
    target=r/'fixture_dynamic_watch10_035/concurrency.json'
    path=backup/(hashlib.sha256(str(target).encode()).hexdigest()+'.json')
    write(path,dict(path=str(target),existed=False,sha256=None,base64=None))
    closed.pinned.add(path);repin(closed)
    proof=common.closure_gate(r,a)['watcher_closure']
    assert proof['current_restoration_audit']['backup_count']==36
    assert proof['previous09_restoration_audit']['backup_count']==33



def test_each_of_all33_historical_identity_changes_is_rejected(closed):
    a=closed.audit;path=a/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json'
    original=common.read_small(path);rejected=[]
    try:
        for index in range(33):
            changed=copy.deepcopy(original);changed['restored_verified'][index]['sha256']='0'*64
            write(path,changed);repin(closed)
            bindings={'small_file_sha256':closed.binding['small_file_sha256']}
            with pytest.raises(ValueError,match='Every historical restoration identity'):
                common.watcher_restoration_proof(a,'watch_shared_capacity09',bindings,
                    historical_receipt=path,expected_count=33,run_root=closed.root)
            rejected.append(index)
    finally:write(path,original);repin(closed)
    assert rejected==list(range(33))


@pytest.fixture
def scoped_export_metadata(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir()
    path=project/exporter.CACHE_PHASE_METADATA_REL
    write(path,[])
    binding={'small_file_sha256':{str(path.resolve()):common.sha(path)}}
    calls=[]
    def reviewed(run,audit):
        assert run==project/exporter.RUN_REL and audit==project/exporter.AUDIT_REL
        calls.append((str(run),str(audit)));return binding,{'synthetic_root_review':True}
    monkeypatch.setattr(exporter.common,'reviewed_binding',reviewed)
    return SimpleNamespace(project=project,path=path,binding=binding,calls=calls)


def test_exact_reviewed_cache_phase_metadata_passes_real_pinned_inventory(scoped_export_metadata):
    fixture=scoped_export_metadata;entry=dict(source=exporter.CACHE_PHASE_METADATA_REL,
        expected_sha256=common.sha(fixture.path),category='root_reviewed_real_recovery10_source_or_receipt')
    view=exporter.base.inventory(fixture.project,[entry])
    assert view['file_count']==1 and view['total_bytes']==fixture.path.stat().st_size
    assert view['files'][0]['source']==entry['source']
    assert exporter.base.stable_bytes(fixture.project,entry['source'])==fixture.path.read_bytes()
    assert len(fixture.calls)==2
    assert exporter.base.MAX_FILE==16*2**20 and exporter.base.MAX_TOTAL==256*2**20


@pytest.mark.parametrize('suffix',['payload.pt','payload.npy','payload.npz','payload.hdf5','payload.json','sibling_events.json'])
def test_metadata_exception_does_not_admit_any_cache_sibling(scoped_export_metadata,suffix):
    fixture=scoped_export_metadata;path=fixture.path.parent/suffix;path.write_bytes(b'[]\n')
    relative=str(path.relative_to(fixture.project))
    with pytest.raises(ValueError):exporter.base.inventory(fixture.project,[dict(source=relative)])
    assert not fixture.calls


@pytest.mark.parametrize('change',['unbound','wrong_sha','oversize','symlink','nonregular','wrong_json_shape','malformed_json','directory_symlink'])
def test_exact_exception_retains_binding_size_regular_json_and_symlink_gates(scoped_export_metadata,change):
    fixture=scoped_export_metadata;path=fixture.path
    if change=='unbound':fixture.binding['small_file_sha256']={}
    elif change=='wrong_sha':fixture.binding['small_file_sha256'][str(path.resolve())]='0'*64
    elif change=='oversize':
        with path.open('wb') as stream:stream.truncate(16*2**20+1)
    elif change=='symlink':
        target=path.parent/'other.json';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
    elif change=='nonregular':path.unlink();path.mkdir()
    elif change=='directory_symlink':
        directory=path.parent;outside=fixture.project/'other_cache_metadata';directory.rename(outside);directory.symlink_to(outside,target_is_directory=True)
    else:
        path.write_bytes(b'{}\n' if change=='wrong_json_shape' else b'{malformed')
        fixture.binding['small_file_sha256'][str(path.resolve())]=common.sha(path)
    with pytest.raises(ValueError):exporter.base.inventory(fixture.project,[dict(source=exporter.CACHE_PHASE_METADATA_REL)])


def test_scoped_inventory_does_not_relax_aggregate256mib(scoped_export_metadata,monkeypatch):
    fixture=scoped_export_metadata
    # The real pinned inventory sums every entry, including this scoped JSON.
    monkeypatch.setattr(exporter.base,'MAX_TOTAL',fixture.path.stat().st_size)
    with pytest.raises(ValueError,match='Combined256MiB'):
        exporter.base.inventory(fixture.project,[dict(source=exporter.CACHE_PHASE_METADATA_REL)]*2)


def test_non_scoped_payload_directories_remain_rejected(scoped_export_metadata):
    fixture=scoped_export_metadata
    for part in ('cache','caches','visual','visuals','video','videos','features','demonstration'):
        path=fixture.project/'outputs'/part/'pipeline_events.json';write(path,[])
        with pytest.raises(ValueError,match='Payload directory forbidden'):
            exporter.base.inventory(fixture.project,[dict(source=str(path.relative_to(fixture.project)))])
    assert not fixture.calls


def test_collection_noise_gate_retains_actual7091_seed(closed,monkeypatch):
    a,r=closed.audit,closed.root
    monkeypatch.setattr(exporter.common,'closure_gate',lambda *_:dict(status='closed'))
    monkeypatch.setattr(exporter.common,'watcher_closed',lambda *_:{})
    observed=dict(passed=47,unverified=1,failed=0,all_verified=False)
    write(a/'collection_integrity_recovery05_actual01/integrity.json',dict(
        version='paired_collection_integrity_recovery05_v1',cuda_initialized=False,read_only_inputs=True,
        passed=True,opportunities=96,paired_opportunities=48,observable_pair_checks=observed,
        full48_observed_initial_states_verified=False))
    for directory,cuda in [('collection_integrity_recovery05_actual01',False),('collection_noise_recovery05_actual01',True)]:
        write(a/directory/'noise_audit.json',dict(version='paired_collection_native_noise_recovery05_v1',
            rows=[],unavailable=[dict(case=f'case{i}',retained_in_denominator=True) for i in range(96)],
            pairs=[{} for _ in range(48)],assignments=96,paired_assignments=48,observable_pair_checks=observed,
            sampler_seed=7091,per_native_plan_draw_shape=[1,24,29],commit_control_ticks=12,cuda_used=cuda,
            cpu_receipt_checks_passed=0,uncertified_execution_endpoints=96,all96_execution_endpoints_certified=False,
            exact_cuda_full_endpoint_checks=0,student_prefix_cuda_reconstructed=0,observed_takeover_rng_states=0,
            independent_observed_takeover_rng_claim=False))
    assert exporter.closed_receipt_gate(closed.project,r,a)['status']=='closed'
    path=a/'collection_noise_recovery05_actual01/noise_audit.json';row=common.read_small(path);row['sampler_seed']=7101;write(path,row)
    with pytest.raises(ValueError,match='noise certification differs'):exporter.closed_receipt_gate(closed.project,r,a)


def test_actual10_inherited09_metadata_path_passes_same_reviewed_inventory(scoped_export_metadata):
    fixture=scoped_export_metadata
    path=fixture.project/exporter.INHERITED_CACHE_PHASE_METADATA_REL
    write(path,[])
    fixture.binding['small_file_sha256'][str(path.resolve())]=common.sha(path)
    entries=[dict(source=relative,expected_sha256=common.sha(fixture.project/relative))
             for relative in sorted(exporter.CACHE_PHASE_METADATA_PATHS)]
    view=exporter.base.inventory(fixture.project,entries)
    assert view['file_count']==2
    assert {row['source'] for row in view['files']}==exporter.CACHE_PHASE_METADATA_PATHS
    fixture.binding['small_file_sha256'].pop(str(path.resolve()))
    with pytest.raises(ValueError,match='root-reviewed binding'):
        exporter.base.inventory(fixture.project,entries)


@pytest.mark.parametrize('attempt',[7,8,11,19,90])
def test_no_other_runtime_cache_phase_receipt_uses_exception(scoped_export_metadata,attempt):
    fixture=scoped_export_metadata
    relative=str(Path(exporter.AUDIT_REL)/f'parallel_rear_recovery{attempt:02d}_runtime/caches/pipeline_events.json')
    write(fixture.project/relative,[])
    with pytest.raises(ValueError,match='Payload directory forbidden'):
        exporter.base.inventory(fixture.project,[dict(source=relative)])
    assert not fixture.calls


def test_inherited09_metadata_payload_sibling_and_symlink_stay_rejected(scoped_export_metadata):
    fixture=scoped_export_metadata
    path=fixture.project/exporter.INHERITED_CACHE_PHASE_METADATA_REL
    write(path,[])
    fixture.binding['small_file_sha256'][str(path.resolve())]=common.sha(path)
    sibling=path.with_name('payload.json');write(sibling,[])
    with pytest.raises(ValueError,match='Payload directory forbidden'):
        exporter.base.inventory(fixture.project,[dict(source=str(sibling.relative_to(fixture.project)))])
    target=path.with_name('real_events.json');path.rename(target);path.symlink_to(target)
    with pytest.raises(ValueError,match='cannot use symlinks'):
        exporter.base.inventory(fixture.project,[dict(source=exporter.INHERITED_CACHE_PHASE_METADATA_REL)])
