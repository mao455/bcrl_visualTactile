#!/usr/bin/env python3
"""Inactive CPU-only allowlisted lightweight export. Default is read-only inventory.

No discovery/copy recursion; no tensor/HDF5/cache/video/JSONL reads. An actual
export requires a validated final assembler and complete registered64/256 tables.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
RUN_REL = 'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
AUDIT_REL = 'outputs/diagnostics/tight_paired_dagger_20261122_01'
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
SUFFIXES = {'.json', '.csv', '.png', '.md', '.py', '.yaml', '.yml'}
DENIED_SUFFIXES = {'.h5', '.hdf5', '.pt', '.pth', '.pkl', '.pickle', '.npz', '.npy', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.log', '.jsonl', '.gz', '.zip', '.tar', '.bin'}
ROLES = ('A_new', 'B_new', 'A_parent', 'B_parent')
CONFIG_NAMES = ('experiment.json','experts.json','experts_admission01.json','parent_train.json','parent_train_recovery01.json','shared_demo_cache.json','main_cache.json','main_train.json','evaluate_development.json','evaluate_test.json')
RUN_NAMES = ('registration.json','session.json','cohorts.json','parent_pairs.json','actual_parent_ages.json','protected.json','protected_historical_references.json','historical_entry50_rear60_pairs.json','parent_reconstruction_amendment.json','parent_recovery01_registration.json','training_expansion_registry.json','training_expansion_registry_admission01.json','active_training_expansion.json','training_family_admission_amendment.json','collection_job.json','collection_recovery05_job.json','source_ready.json','concurrency.json','cycle_status.json','closure.json','final_decision.json')
COMPONENT_NAMES = ('configuration.json','deployment.json','selected.json','finished.json','initial_metrics.json','validation.json','sampling_coverage.json','effective_exposure.json','entry_exposure.json','family_phase_definition.json','family_phase_curves.csv','family_objective_loss.csv','family_phase_curves.png','validation.png','training.png')
OWNER_NAMES = ('session.json','project_job_started.json','project_status.json','project_complete.json','project_exit.json','project_failure.json','monitor_exit.json','cuda_allocator_policy.json')
AUDIT_NAMES = ('source_ready_identity.json','encoder_probe_feasibility.json','collection_noise_reference_cpu.json','collection_integrity_recovery05.json','collection_noise_recovery05.json','recovery05_integrity.json','recovery05_noise.json','COMPUTE_RECORD_RECOVERY05.json','COMPUTE_RECORD_RECOVERY05.md','interruption_observation_20261123_0935.json','collection_interruption_census09.json','reconstruction_age_audit.json','parent_recovery_readonly_metadata.json','MAIN_CONTRACT_REVIEW05.json','MAIN_CONTRACT_REVIEW05.md','MAIN_RECOVERY05_REVIEW.json','MAIN_RECOVERY05_REVIEW.md','WATCH_RECOVERY05_REVIEW.json','WATCH_RECOVERY05_REVIEW.md','EVAL_REPORT_RECOVERY05_REVIEW.json','EVAL_REPORT_RECOVERY05_REVIEW.md','RECOVERY05_AUDIT_HANDOFF.md','EXPORT_RECOVERY05_REVIEW.json','EXPORT_RECOVERY05_REVIEW.md','DP_PIPELINE_FEASIBILITY05.md','DP_PIPELINE_FEASIBILITY05.json','REPRODUCTION_PACKAGE_REVIEW05.md','REPRODUCTION_PACKAGE_REVIEW05.json')
ASSEMBLY_NAMES = ('assembly_status.json','DRAFT_REPORT.md','actual_exposure_and_age.json','curve_trends.json','pool_partitions.json','attempt_wall_accounting.json')
CASE_NAMES = ('result.json','failure.json','execution_complete.json','capture_complete.json','recovery_verification_failure.json','camera_acceptance.json','initial_collision.json','initial_observable.json','paired_initial_check.json','sampler_rng_start.json','sampler_rng_end.json','native_inference_timing.json')

AUDIT_SOURCE_ALLOWLIST = (('admission_amendment.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('audit_collection_integrity.py', '9fe68a6f834f7059adb490016a78bb1fd80aa4d7d1d2e63aa16bf285fd052dc5', 'Actual frozen waiter registration'), ('audit_collection_noise.py', 'efe27eeb9ea4a66d448fc1843893f6bee65ffa3706def3666c073c7543f8f257', 'Actual frozen waiter registration'), ('audit_collection_recovery05.py', '2c50641db3cae82210905fb09ea69702a3dff6709c8752d9963010f40d3ddedc', 'Actual frozen waiter registration'), ('baseline_preflight.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('coexistence_watch_recovery05.py', '9a58af179e568ebe706fcf069361ed21b215d17d5e3cfdc5c15882ab4d85c5c0', 'Actual frozen waiter registration'), ('collection_recovery05.py', '5e2c5f365d43f91f431896e22b56a29a6ee7b743f5ee9062c23e155a2128e357', 'Actual frozen waiter registration'), ('collection_recovery05_aggregate.py', 'bb28612d6bf416597a60753f1b28314aaa7eee60649d6fb9fd169071a64d81e1', 'Actual frozen waiter registration'), ('collection_recovery05_finalize.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('collection_recovery05_scope.py', '642a006861b0004aba62d4928de5a62c23d463408c8d5592e5c1aaad354efaaa', 'Actual frozen waiter registration'), ('driver_support.py', 'c24cda034190519bd317add5027e40cb240baca7de2281b2363871122fc886c6', 'Actual frozen waiter registration'), ('driver_support_v4.py', 'e99dbdc5f0973c16f49730fbbe0a18b9a2eb1ee693f1a6de0bc3024078e4150e', 'Actual frozen waiter registration'), ('encoder_probe_inventory.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('final_report_assembler.py', '1eca4843d9e7e2d1ac6287b220bf8d429fe36458cb58ddcf63a63e9ab2611b2f', 'Actual frozen waiter registration'), ('parallel_rear_proposal.py', '5d314b23d935cf82672e3a1a3ba2d53d68aebe12c308fb224b47aafa7b4c944b', 'Actual frozen waiter registration'), ('presealed_guard.py', '39569316f9ec58a6bce58f7452cd9a9464c017e3f673b7db81bc41ccde55c1cf', 'Actual frozen waiter registration'), ('read_progress_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('reconstruction_age_audit.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('refresh_compute_record_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('resource_watch.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('resource_watch_parallel_proposal.py', '005ee0dbfe13c08c88e7e7ae15d8b65e0d17ec210c9399992ee6912fd5244fc7', 'Actual frozen waiter registration'), ('resource_watch_v2.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('resource_watch_v3.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('resource_watch_v4.py', '1ad88f4ab761455cbd7031ffa8d6c24b83eef572dd5b35caeb3da56a51488f0d', 'Actual frozen waiter registration'), ('run_collection_recovery05.py', 'c88e9e55c2c9339e438837f092f9522df6f3c1634d125f597d2f7ab3fe3bcefb', 'Actual frozen waiter registration'), ('run_cycle.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_cycle_recovery01.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_cycle_recovery02.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_cycle_recovery03.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_cycle_recovery04.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_cycle_v1_preserved.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_cycle_v2_preserved.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_main_recovery05.py', '3edbb158e0845cc4f51bfee0e27e1552c79fc8455613d8e610a40036a5e9002f', 'Actual frozen waiter registration'), ('run_parent_queue.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_parent_queue_recovery01.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_parent_queue_v1_preserved.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_parent_queue_v2_preserved.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_parent_queue_v3_preserved.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('run_parent_queue_v4_preserved.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_audit_collection_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_audit_collection_recovery05_independent.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_coexistence_watch_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_collection_integrity.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_collection_noise.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_collection_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_collection_recovery05_aggregate.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_collection_recovery05_finalize.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_final_report_assembler.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_parallel_rear_proposal.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_presealed_guard.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_resource_watch_v2.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_resource_watch_v3.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_resource_watch_v4.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_run_main_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('test_wait_then_main_recovery05.py', None, 'Preserved inactive source current bytes; runtime identity only where original receipts bind it'), ('wait_then_main_recovery05.py', 'cb3e1791b693f8f60364544f4ea517457fd628bb80e25094b34bc6371b54a28e', 'Actual frozen waiter registration'))
PRODUCTION_TEST_ALLOWLIST = ('tests/test_paired_collection.py', 'tests/test_paired_curve_diagnostics.py', 'tests/test_paired_evaluate.py', 'tests/test_paired_training_decks.py')
ADDITIONAL_AUDIT_ALLOWLIST = ('source_ready_identity.json', 'all_frozen_families.json', 'all_historical_partitions.json', 'parallel_rear_registration.json', 'proposed_main_train_parallel.json', 'wait_main_recovery05/registration.json', 'wait_main_recovery05/observed_collection_closure.json', 'wait_main_recovery05/aggregate_exit.json', 'driver_main_recovery05/registration.json', 'driver_collection_recovery05/registration.json', 'durable_collection_recovery05_launch.json', 'durable_service_recovery05_launch.json', 'main_wait_service_activation.json', 'presealed_guard_main_recovery05.json', 'export_source_closure_proposal.json')
SCOPE_METADATA_ALLOWLIST = ('registration.json', 'metadata.json', 'training_source_identity.json', 'recovery01_source_identity.json', 'matched_learning_rates.json', 'matched_backend_learning_rates.json', 'backend_episode_sampling.json', 'full_trajectory_batch.json', 'paired_group_deck_A.json', 'paired_group_deck_B.json', 'family_phase_diagnostics.json', 'training_sampling.json', 'entry_exposure.json', 'effective_exposure.json', 'frozen_visual_io.json', 'frozen_visual_hotset.json', 'visual_arithmetic.json', 'cuda_allocator_policy.json', 'complete_exposure_preflight.json', 'budget_registration.json', 'preflight.json', 'gate.json', 'bounded_extension.json')
COLLECTION_METADATA_ALLOWLIST = ('plan.json', 'session.json', 'registration.json', 'results.json', 'recovered_results.json', 'result.json', 'correction_entries.json', 'expert_entries.json', 'collection_summary.json', 'paired_initial_check.json', 'paired_queue_assignments.json', 'recovery05_binding.json', 'recovery05_runtime_scope.json', 'interruption_census.json', 'recovery05_aggregate_binding.json', 'recovery05_active_aggregate.json', 'capture_audit_memo.json')

# These exact paths are emitted by the frozen recovery05 main and waiter.
# storage_gate writes ready metadata, not a cycle stage-exit receipt.
RECOVERY05_STAGE_NAMES = (
    'validate_collection_recovery05', 'audit_collection_integrity_recovery05',
    'prepare_learning_recovery05', 'preflight_learning', 'register_learning_jobs',
    'cache_own_observations', 'train_four_components_parallel_rears',
    'verify_final_actual_training', 'prepare_development', 'evaluate_development',
    'summarize_development', 'prepare_test', 'evaluate_test', 'summarize_test',
    'audit_collection_cuda_noise_recovery05',
)
RECOVERY05_AUDIT_RECEIPTS = (
    'collection_integrity_recovery05_actual01/integrity.json',
    'collection_integrity_recovery05_actual01/noise_audit.json',
    'collection_noise_recovery05_actual01/noise_audit.json',
    'driver_main_recovery05/status.json', 'driver_main_recovery05/storage_gate.json',
    'wait_main_recovery05/status.json', 'wait_main_recovery05/reviewed_main_exit.json',
)
RECOVERY05_REVIEW_NAMES = ('ROOT_EVIDENCE_TOOLS_REVIEW05.json',
    'EXTRACT_SELECTED_LEDGERS_REVIEW05.md', 'EXTRACT_SELECTED_LEDGERS_REVIEW05.json')
RECOVERY05_WAIT_METADATA = ('wait_main_recovery05/validated_aggregate.json',
    'wait_main_recovery05/reviewed_main_started.json',
    'wait_main_recovery05/aggregate_and_publish_once_started.json',
    'wait_main_recovery05/aggregate_and_publish_once_exit.json')
CLOSED_OWNER_PATHS = ('owners/main_cache', 'owners/main_train',
    'evaluation/development', 'evaluation/test')


def require(value, message):
    if not value:
        raise ValueError(message)


def relpath(value):
    p = PurePosixPath(str(value))
    require(not p.is_absolute() and p.parts and '..' not in p.parts, 'Only project-relative safe source paths')
    return str(p)


def source_path(project, relative):
    relative = relpath(relative)
    p = project / relative
    require(p.resolve().is_relative_to(project.resolve()), 'Source escapes project: '+relative)
    require(p.is_file(), 'Missing regular source: '+relative)
    require(stat.S_ISREG(p.stat().st_mode), 'Nonregular source: '+relative)
    require(p.suffix.lower() in SUFFIXES and not any(s.lower() in DENIED_SUFFIXES for s in p.suffixes), 'Unsupported/payload suffix: '+relative)
    require(not any(part.lower() in ('visual','visuals','cache','caches','videos','video','demonstration','features') for part in p.parts), 'Payload directory forbidden: '+relative)
    require(p.stat().st_size <= MAX_FILE, 'Individual16MiB limit: '+relative)
    return p


def stable_bytes(project, relative):
    p = source_path(project, relative)
    before = p.stat()
    raw = p.read_bytes()
    after = p.stat()
    identity = lambda x: (x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    require(identity(before) == identity(after) and len(raw) == before.st_size, 'Source changed during read: '+relative)
    return raw


def json_at(project, relative):
    return json.loads(stable_bytes(project, relative))


def strict_truth(value):
    if value is True or value == 'True' or value == 'true' or value == '1': return True
    if value is False or value == 'False' or value == 'false' or value == '0': return False
    raise ValueError('Invalid boolean outcome')


def closed_receipt_gate(project, run, audit):
    """Bind actual closed metadata only; never inspect tensors or rerun audits."""
    bindings = {}
    def receipt(path):
        relative = str(path.relative_to(project))
        require(path.is_file(), 'Missing required final receipt: '+relative)
        raw = stable_bytes(project, relative)
        bindings[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    for directory in (audit/'driver_main_recovery05', audit/'wait_main_recovery05'):
        require(not (directory/'failure.json').exists(), 'Failed main/waiter attempt cannot close export: '+str(directory))
    for name in RECOVERY05_REVIEW_NAMES:
        path = audit/name
        relative = str(path.relative_to(project))
        require(path.is_file(), 'Missing required final receipt: '+relative)
        bindings[relative] = hashlib.sha256(stable_bytes(project,relative)).hexdigest()
    main = receipt(audit/'driver_main_recovery05/status.json')
    require(main.get('stage') == 'main_stages_complete'
            and main.get('training_or_evaluation_reruns') == 0, 'Actual recovery05 main must be closed')
    waiter = receipt(audit/'wait_main_recovery05/status.json')
    require(waiter.get('stage') == 'reviewed_main_closed'
            and waiter.get('collection_retries') == 0, 'Actual recovery05 waiter must be closed')
    exit_row = receipt(audit/'wait_main_recovery05/reviewed_main_exit.json')
    require(type(exit_row.get('returncode')) is int and exit_row['returncode'] == 0
            and exit_row.get('owned_helper_returned_after_cleanup') is True,
            'Actual reviewed main exit/cleanup required')
    require(receipt(audit/'driver_main_recovery05/storage_gate.json').get('ready') is True,
            'Actual final storage admission required')
    for stage in RECOVERY05_STAGE_NAMES:
        row = receipt(run/'cycle_logs'/(stage+'_exit.json'))
        require(type(row.get('returncode')) is int and row['returncode'] == 0,
                'Actual recovery05 stage did not close successfully: '+stage)
    for relative in CLOSED_OWNER_PATHS:
        owner = run/relative
        require(not (owner/'project_failure.json').exists(), 'Failed final owner: '+relative)
        started = receipt(owner/'project_job_started.json')
        complete = receipt(owner/'project_complete.json')
        closed = receipt(owner/'project_exit.json')
        stages = started.get('job', {}).get('stages')
        commands = started.get('commands')
        require(isinstance(stages,list) and bool(stages) and isinstance(commands,list)
                and type(complete.get('stages')) is int
                and complete['stages'] == len(stages) == len(commands)
                and Path(started['job'].get('resource_root','')).resolve() == owner.resolve()
                and closed.get('children_reaped') is True,
                'Final owner completion/registered job/cleanup differs: '+relative)
    integrity = receipt(audit/'collection_integrity_recovery05_actual01/integrity.json')
    observed = dict(passed=47,unverified=1,failed=0,all_verified=False)
    require(integrity.get('version') == 'paired_collection_integrity_recovery05_v1'
            and integrity.get('passed') is True and integrity.get('opportunities') == 96
            and integrity.get('paired_opportunities') == 48
            and integrity.get('observable_pair_checks') == observed
            and integrity.get('full48_observed_initial_states_verified') is False
            and integrity.get('cuda_initialized') is False and integrity.get('read_only_inputs') is True,
            'Actual recovery05 integrity audit did not pass')
    noise_sets = []
    for relative, cuda in (
        ('collection_integrity_recovery05_actual01/noise_audit.json',False),
        ('collection_noise_recovery05_actual01/noise_audit.json',True),
    ):
        noise = receipt(audit/relative)
        rows, unavailable, pairs = (noise.get(k) for k in ('rows','unavailable','pairs'))
        require(isinstance(rows,list) and isinstance(unavailable,list) and isinstance(pairs,list),
                'Actual recovery05 noise audit populations missing')
        certified = {r.get('case') for r in rows}; absent = {r.get('case') for r in unavailable}
        require(noise.get('version') == 'paired_collection_native_noise_recovery05_v1'
                and noise.get('assignments') == 96 and noise.get('paired_assignments') == 48
                and noise.get('observable_pair_checks') == observed
                and noise.get('sampler_seed') == 7091 and noise.get('per_native_plan_draw_shape') == [1,24,29]
                and noise.get('commit_control_ticks') == 12 and noise.get('cuda_used') is cuda
                and noise.get('cpu_receipt_checks_passed') == len(rows)
                and noise.get('uncertified_execution_endpoints') == len(unavailable)
                and len(rows)+len(unavailable) == 96 and len(pairs) == 48
                and len(certified) == len(rows) and len(absent) == len(unavailable)
                and not certified&absent and None not in certified|absent
                and all(r.get('retained_in_denominator') is True for r in unavailable)
                and noise.get('all96_execution_endpoints_certified') is (not unavailable)
                and noise.get('exact_cuda_full_endpoint_checks') == (len(rows) if cuda else 0)
                and noise.get('student_prefix_cuda_reconstructed') == (len(rows) if cuda else 0)
                and noise.get('observed_takeover_rng_states') == 0
                and noise.get('independent_observed_takeover_rng_claim') is False,
                'Actual recovery05 noise audit certification differs: '+relative)
        noise_sets.append((certified,absent))
    require(noise_sets[0] == noise_sets[1], 'CPU/CUDA recovery05 audited case populations differ')
    return {'status':'closed','receipt_sha256':bindings,
            'scope':'Actual fixed-path final receipts; software failures and the unverified initial pair remain explicit'}


def gate(project, run, assembly):
    status_path = assembly/'assembly_status.json'
    status = json_at(project, str(status_path.relative_to(project)))
    require(status.get('status') == 'draft_ready_for_root_review', 'Final assembler draft_ready required')
    closure_gate = closed_receipt_gate(project,run,project/AUDIT_REL)
    paths = {'final_summary':run/'evaluation/test/summary/primary_summary.json',
             'final_registration':run/'evaluation/test/registration.json',
             'per_initial_state.csv':run/'evaluation/test/summary/per_initial_state.csv',
             'rollouts.csv':run/'evaluation/test/summary/rollouts.csv'}
    data = {}
    for name,p in paths.items():
        relative = str(p.relative_to(project)); raw = stable_bytes(project,relative)
        binding = status.get('inputs',{}).get(name,{})
        require(Path(binding.get('path','')).resolve() == p.resolve(), 'Assembler input path mismatch: '+name)
        require(binding.get('sha256') == hashlib.sha256(raw).hexdigest(), 'Assembler input SHA mismatch: '+name)
        data[name] = list(csv.DictReader(io.StringIO(raw.decode()))) if name.endswith('.csv') else json.loads(raw)
    summary, registration = data['final_summary'],data['final_registration']
    pairs,rollouts = data['per_initial_state.csv'],data['rollouts.csv']
    require(isinstance(registration.get('source_sha256'),dict) and registration['source_sha256'],'Frozen production source identity required')
    require(summary.get('cohort') == registration.get('cohort') == 'test', 'Independent test cohort required')
    require(summary.get('n') == registration.get('n') == 64 and summary.get('episodes') == 256 and summary.get('full_denominator_preserved') is True, '64/256 full denominator required')
    seeds=registration.get('seeds',[])
    require(len(seeds)==len(set(seeds))==64, '64 distinct registered seeds required')
    require(len(pairs)==64 and {int(x['seed']) for x in pairs}==set(seeds) and len({x['family'] for x in pairs})==64, '64 paired family outcomes required')
    require(len(rollouts)==256 and {(int(x['seed']),x['role']) for x in rollouts}=={(s,r) for s in seeds for r in ROLES}, '256 unique four-student outcomes required')
    idx={(int(x['seed']),x['role']):x for x in rollouts}
    rates={x['role']:x for x in summary.get('absolute_success',[])}
    require(set(rates)==set(ROLES),'All four absolute success rates required')
    for role in ROLES:
        success=sum(strict_truth(x[role+'_success']) for x in pairs)
        rate=rates[role]
        require(rate['n']==64 and rate['success']==success and math.isfinite(rate['success_rate']) and abs(rate['success_rate']-success/64)<1e-12,'Absolute outcome mismatch')
        for row in pairs:
            require(strict_truth(row[role+'_success'])==strict_truth(idx[int(row['seed']),role]['success']), 'Pair/rollout outcome mismatch')
    closure=json_at(project,str((run/'cycle_logs/summarize_test_exit.json').relative_to(project)))
    require(closure.get('returncode')==0,'Successful final summary exit required')
    require(summary.get('registration_sha256')==hashlib.sha256(stable_bytes(project,str(paths['final_registration'].relative_to(project)))).hexdigest(),'Final registration identity mismatch')
    require((assembly/'DRAFT_REPORT.md').is_file(),'Assembler draft missing')
    return {'status_sha256':hashlib.sha256(stable_bytes(project,str(status_path.relative_to(project)))).hexdigest(),
            'actual_recovery05_closure':closure_gate,
            'registered_families':64,'outcomes':256,'software_incomplete':'Retained in full outcome tables; never filtered during export'}


def allowlist(project, run, audit, assembly):
    """Finite exact basenames and registered case identifiers; no recursive traversal."""
    entries={}
    def add(path,category,expected=None,*,required=False):
        relative=relpath(path.relative_to(project))
        require(not required or path.is_file(), 'Missing required final receipt: '+relative)
        if path.exists():
            previous=entries.get(relative)
            if previous and previous.get('expected_sha256') and expected:
                require(previous['expected_sha256']==expected,'Conflicting frozen source identity')
            entries[relative]={'source':relative,'category':category,'expected_sha256':expected or (previous or {}).get('expected_sha256')}
    for name in CONFIG_NAMES: add(project/'configs/experiments/paired_dagger_20261122'/name,'experiment_config')
    for name in RUN_NAMES: add(run/name,'registration_and_closure')
    for name in ('TIGHT_PREMATURE_SCOPE_20261122.md','TIGHT_ENTRY_FINAL24_20261122.md','TIGHT_ENTRY_PHASE_20261122.md','TIGHT_ADAPTIVE_DAGGER_20261119_RESULTS.md','TIGHT_REPRESENTATION_AUDIT_20261119.md'): add(project/'research_docs'/name,'historical_reference_report')
    for name in AUDIT_NAMES: add(audit/name,'audit_and_compute')
    for name in ADDITIONAL_AUDIT_ALLOWLIST: add(audit/name,'registered_orchestration_metadata')
    for name in RECOVERY05_AUDIT_RECEIPTS: add(audit/name,'actual_recovery05_audit_and_closure',required=True)
    for name in RECOVERY05_REVIEW_NAMES: add(audit/name,'final_evidence_tool_review',required=True)
    for name in RECOVERY05_WAIT_METADATA: add(audit/name,'actual_recovery05_waiter_metadata')
    for name,expected,identity_scope in AUDIT_SOURCE_ALLOWLIST:
        add(audit/name,'frozen_runtime_audit_python' if expected else 'preserved_inactive_audit_python',expected)
        key=str((audit/name).relative_to(project))
        if key in entries: entries[key]['identity_scope']=identity_scope
    for relative in PRODUCTION_TEST_ALLOWLIST: add(project/relative,'production_tests')
    for name in ASSEMBLY_NAMES: add(assembly/name,'final_assembly')
    if Path(__file__).resolve().is_relative_to(project): add(Path(__file__).resolve(),'exporter_source')
    add(audit/'test_export_final_evidence_recovery05.py','exporter_tests')
    add(audit/'extract_selected_ledgers_recovery05.py','selected_exposure_extractor_source')
    add(audit/'test_extract_selected_ledgers_recovery05.py','selected_exposure_extractor_tests')
    for scope in ('common180_parents','learning'):
        for name in SCOPE_METADATA_ALLOWLIST: add(run/scope/name,'training_scope_metadata')
        for name in ('budget_registration.json','complete_exposure_preflight.json','gate.json','bounded_extension.json','matched_backend_learning_rates.json','matched_learning_rates.json','backend_episode_sampling.json','full_trajectory_batch.json','training_source_identity.json','paired_group_deck_A.json','paired_group_deck_B.json','learning.json'):
            add(run/scope/name,'budget_and_preflight')
        for role in 'AB':
            for name in ('continuation.json','cache_index.json'): add(run/scope/role/name,'own_component_cache_binding')
            for name in ('base_identities.json','cache_complete.json','fixed_epoch_registration.json','configuration.json','prepared.json'): add(run/scope/role/'backend'/name,'own_nominal_cache_binding')
            for name in ('data.json','statistics.json','normalization.json'):
                add(run/scope/role/'state_data'/name,'data_and_normalization_config')
            for component in ('training',f'backend/{role}/training'):
                for name in COMPONENT_NAMES: add(run/scope/role/component/name,'training_curves_and_exposure')
                for epoch in range(1,61): add(run/scope/role/component/f'deck_{epoch:03d}.json','actual_registered_training_decks')
    for cohort in ('development','test'):
        directory=run/'evaluation'/cohort
        for name in ('registration.json','plan.json','result.json','closure.json','evaluation_job.json',
                     'project_job_started.json','project_complete.json','project_exit.json','project_failure.json'):
            add(directory/name,'evaluation_registration')
        for name in ('primary_summary.json','per_initial_state.csv','rollouts.csv','protocol_audit.json','initial_pairs.json','summary.json','funnels.csv','paired_contrasts.csv','noise_audit.json','distinct_observed_initials.json'):
            add(directory/'summary'/name,'paired_evaluation_outcomes')
        # Every wave/case is named by its frozen registration/plan, no tree walk.
        freeze_path=directory/'registration.json'
        if freeze_path.exists():
            freeze_data=json_at(project,str(freeze_path.relative_to(project)))
            for wave_name,expected in freeze_data.get('wave_plan_sha256',{}).items():
                wave=Path(wave_name)
                if not wave.is_absolute(): wave=project/wave
                wave=wave.resolve()
                require(wave.parent==directory and wave.name.startswith('wave') and wave.name[4:].isdigit(),'Invalid registered evaluation wave path')
                add(wave/'plan.json','frozen_evaluation_wave_plan',expected)
                for name in ('session.json','result.json','results.json','paired_initial_check.json','paired_worker_failure.json','project_failure.json','project_exit.json','monitor_exit.json'): add(wave/name,'evaluation_wave_metadata')
                plan=json_at(project,str((wave/'plan.json').relative_to(project)))
                for case in plan.get('cases',[]):
                    name=case.get('name')
                    require(isinstance(name,str) and Path(name).name==name and name not in ('.','..'),'Unsafe registered eval case identifier')
                    for filename in CASE_NAMES: add(wave/name/filename,'registered_eval_case_metadata')
    for directory in ('experts','experts_admission01','experts_shared','collection','collection_recovery05','collection_consumed_recovered05','collection_aggregate_recovery05'):
        for name in ('registration.json','plan.json','result.json','results.json','recovered_results.json','collection_summary.json','correction_entries.json','entries.json','capture_audit_memo.json','recovery05_aggregate_binding.json','recovery05_active_aggregate.json','initial_pair_checks.json','initial_observable_pairs.json','paired_initial_check.json','paired_queue_assignments.json','expert_entries.json','finalization.json'):
            add(run/directory/name,'collection_and_shared_expert_metadata')
        for name in COLLECTION_METADATA_ALLOWLIST: add(run/directory/name,'collection_bound_recovery_metadata')
    for stage in ('collect_96','verify_actual_parents','prepare_learning','preflight_learning','train_learning','summarize_development','summarize_test','evaluate_development','evaluate_test'):
        add(run/'cycle_logs'/f'{stage}_exit.json','cycle_stage_failure_or_closure')
    for stage in RECOVERY05_STAGE_NAMES:
        add(run/'cycle_logs'/f'{stage}_exit.json','actual_recovery05_stage_closure',required=True)
    # Expert retries are enumerated only by their saved attempt ledger.
    for attempt in ('experts','experts_admission01'):
        result_path=run/attempt/'results.json'
        if result_path.exists():
            rows=json_at(project,str(result_path.relative_to(project)))
            for row in rows:
                case=row.get('case')
                if not case: continue
                require(isinstance(case,str) and Path(case).name==case and case not in ('.','..'),'Unsafe expert attempt name')
                for name in CASE_NAMES: add(run/attempt/case/name,'expert_attempt_metadata')
    # Enumerate owner names only, never their payload descendants or general files.
    if (run/'owners').exists():
        for owner in sorted((run/'owners').iterdir()):
            if owner.is_dir():
                for name in OWNER_NAMES: add(owner/name,'owner_failure_and_closure')
    # Case paths come exclusively from the preregistered collection plan.
    plan_path=run/'collection/plan.json'
    if plan_path.exists():
        plan=json_at(project,str(plan_path.relative_to(project)))
        tasks=plan if isinstance(plan,list) else plan.get('cases',plan.get('tasks',plan.get('opportunities',[])))
        if isinstance(tasks,list):
            for task in tasks:
                case=task.get('name',task.get('case',task.get('identity')))
                require(isinstance(case,str) and case, 'Missing registered case identifier')
                require(Path(case).name==case and case not in ('.','..'),'Unsafe registered case identifier')
                for attempt in ('collection','collection_recovery05','collection_consumed_recovered05','collection_aggregate_recovery05'):
                    for name in CASE_NAMES: add(run/attempt/case/name,'registered_case_failure_and_capture_metadata')
    # Production Python sources must match the final frozen evaluation identity.
    freeze=run/'evaluation/test/registration.json'
    if freeze.exists():
        frozen=json_at(project,str(freeze.relative_to(project))).get('source_sha256',{})
        require(isinstance(frozen,dict),'Invalid final source registration')
        for relative,expected in frozen.items():
            relative=relpath(relative)
            require(relative.endswith('.py') and relative.split('/')[0] in ('tools','visuotactile'),'Only registered production Python source allowed')
            require(isinstance(expected,str) and len(expected)==64,'Invalid frozen Python SHA')
            require((project/relative).is_file(),'Missing registered frozen production source: '+relative)
            add(project/relative,'frozen_production_python',expected)
    return [entries[k] for k in sorted(entries)]


def inventory(project, entries):
    total=0;records=[]
    for entry in entries:
        p=source_path(project,entry['source']); size=p.stat().st_size
        total+=size
        require(total<=MAX_TOTAL,'Combined256MiB limit')
        records.append({**entry,'bytes':size})
    return {'files':records,'total_bytes':total,'file_count':len(records),'max_file_bytes':MAX_FILE,'max_total_bytes':MAX_TOTAL}



def selected_ledger_gate(project,audit,assembly,selected_ledgers):
    """Require eight independently extracted selected checkpoint count tables."""
    require(selected_ledgers is not None,'Explicit --selected-ledgers successful eight-component extraction required')
    directory=Path(selected_ledgers).resolve()
    require(directory.is_relative_to(audit) and directory!=audit,'Selected-ledger attempt must be a child of this AUDIT')
    status_path=directory/'extraction_status.json';summary_path=directory/'selected_ledger_summary.json'
    status=json_at(project,str(status_path.relative_to(project)))
    summary_raw=stable_bytes(project,str(summary_path.relative_to(project)));summary=json.loads(summary_raw)
    for record in (status,summary):
        require(record.get('status')=='passed' and record.get('checkpoint_reads')==8 and record.get('cuda_initialized') is False,'Successful CPU-only eight-selected-ledger extraction required')
        for name in ('all_legal_ticks_at_least50','all_native_entry_slots_at_least50','paired_actual_updates_equal','own_frontend_rear_mask_and_base_binding'):
            require(record.get(name) is True,'Selected-ledger gate failed: '+name)
        require(record.get('all_selected_epoch')==60,'Selected-ledger epoch60 required')
    require(status.get('files')==summary.get('files'),'Selected summary/status component binding differs')
    require(Path(status.get('output','')).resolve()==directory and Path(status.get('root','')).resolve()==project/RUN_REL,'Selected-ledger root/output binding differs')
    require(status.get('raw_data_opened') is False and status.get('model_forward_calls')==0,'Only closed CPU metadata extraction accepted')
    actual=json_at(project,str((assembly/'actual_exposure_and_age.json').relative_to(project)))
    expected={(scope,role,component) for scope in ('common180_parents','learning') for role in 'AB' for component in ('frontend','backend')}
    filenames={f'{scope}_{role}_{component}.json' for scope,role,component in expected}
    require(set(summary.get('files',{}))==filenames,'Exactly eight selected component count files required')
    records=[];loaded={}
    for scope,role,component in sorted(expected):
        filename=f'{scope}_{role}_{component}.json';item=summary['files'][filename]
        p=directory/filename;raw=stable_bytes(project,str(p.relative_to(project)));sha=hashlib.sha256(raw).hexdigest()
        require(item.get('sha256')==sha and item.get('bytes')==len(raw),'Selected component JSON identity differs')
        value=json.loads(raw);loaded[scope,role,component]=value
        require((value.get('scope'),value.get('role'),value.get('component'))==(scope,role,component),'Component/table identity differs')
        reference=actual[scope][role][component]
        require(value.get('epoch')==reference.get('epoch')==60 and value.get('updates')==reference.get('updates'),'Selected ledger deployment age differs')
        require(value.get('checkpoint_sha256')==reference.get('checkpoint_sha256'),'Selected ledger checkpoint differs from final assembler')
        require(value.get('minimum_tick_steps',-1)>=50 and value.get('below50_ticks')==0 and value.get('ancestors_counted') is False,'Selected legal tick exposure incomplete')
        episodes=value.get('episodes',{});all_counts=[];masks={}
        require(value.get('duplicate_within_step')=='count_once' and isinstance(episodes,dict) and len(episodes)==value.get('trajectories') and bool(episodes),'Complete actual selected episode counters required')
        for identity,episode in episodes.items():
            ticks=episode.get('legal_ticks',[]);counts=episode.get('successful_optimizer_steps',[]);length=episode.get('complete_history_ticks')
            require(type(length) is int and 0<length<=20000 and len(ticks)==len(counts)>0,'Selected tick/count population differs')
            require(all(type(t) is int and 0<=t<length for t in ticks) and ticks==sorted(set(ticks)),'Invalid selected legal tick indices')
            require(all(type(n) is int and 50<=n<=value['updates'] for n in counts),'Selected actual tick count below50 or invalid')
            require(type(episode.get('visits')) is int and episode['visits']>=50,'Selected trajectory visits incomplete')
            legal=set(ticks);mask=bytes(i in legal for i in range(length));require(hashlib.sha256(mask).hexdigest()==episode.get('mask_sha256'),'Selected legal mask identity differs')
            masks[identity]=(ticks,counts,length);all_counts.extend(counts)
        require(len(all_counts)==value.get('legal_ticks') and min(all_counts)==value['minimum_tick_steps'] and max(all_counts)==value.get('maximum_tick_steps'),'Selected actual count summary differs')
        if component=='frontend':
            entries=value.get('native_entry_rows');require(isinstance(entries,list),'Native entry count rows missing')
            keys=set()
            for entry in entries:
                identity=entry['identity'];tick=entry['absolute_tick'];key=(identity,tick)
                require(key not in keys,'Duplicate selected native-entry counter');keys.add(key)
                ticks,counts,length=masks[identity];anchor=(min(ticks)//12)*12
                require(entry['native_anchor']==anchor and entry['slot']==tick-anchor and 0<=entry['slot']<12 and tick in ticks,'Selected native-entry phase/population differs')
                n=entry['successful_optimizer_steps'];require(type(n) is int and 50<=n<=counts[ticks.index(tick)],'Selected native-entry count below50 or invalid')
            expected_entries={(identity,t) for identity,(ticks,counts,length) in masks.items() if 0 not in ticks for t in ticks if t//12==min(ticks)//12}
            require(keys==expected_entries,'Selected native-entry population incomplete')
        records.append({'source':str(p.relative_to(project)),'category':'selected_actual_ledger_counts','expected_sha256':sha})
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            front,rear=(loaded[scope,role,c] for c in ('frontend','backend'))
            require(rear.get('base_sha256')==front['checkpoint_sha256'] and front.get('data_sha256')==rear.get('data_sha256') and front.get('train_mask_sha256')==rear.get('train_mask_sha256'),'Selected DP/rear data/mask/base differs')
        for component in ('frontend','backend'):
            require(loaded[scope,'A',component]['updates']==loaded[scope,'B',component]['updates'],'Selected A/B update budget differs')
    for p in (status_path,summary_path):
        records.append({'source':str(p.relative_to(project)),'category':'selected_actual_ledger_counts','expected_sha256':hashlib.sha256(stable_bytes(project,str(p.relative_to(project)))).hexdigest()})
    return {'status':'passed','components':8,'summary_sha256':hashlib.sha256(summary_raw).hexdigest(),'files':records}


def export(project,run,audit,assembly,destination,*,apply=False,selected_ledgers=None):
    project=project.resolve();run=run.resolve();audit=audit.resolve();assembly=assembly.resolve()
    for p in (run,audit,assembly): require(p.is_relative_to(project),'Input root outside project')
    require(assembly.is_relative_to(audit) and assembly!=audit,'Assembly must be a child of this AUDIT')
    # Gate FIRST: no destination mkdir, parent mkdir, copy, payload reads or source scan before final evidence.
    try: final_gate=gate(project,run,assembly)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        if apply: raise ValueError('Export blocked before destination creation: '+str(exc)) from exc
        return {'status':'blocked','reason':str(exc),'destination_created':False,'copied_files':0}
    try: extracted=selected_ledger_gate(project,audit,assembly,selected_ledgers)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        if apply: raise ValueError('Export blocked before destination creation: '+str(exc)) from exc
        return {'status':'blocked','reason':str(exc),'destination_created':False,'copied_files':0}
    entries=allowlist(project,run,audit,assembly)+extracted['files']; view=inventory(project,entries)
    if not apply:
        return {'status':'dry_inventory_ready','gate':final_gate,'selected_ledger_gate':{k:v for k,v in extracted.items() if k!='files'},**view,'destination_created':False,'copied_files':0}
    require(destination is not None,'Explicit destination required for actual export')
    destination=Path(destination)
    require(destination.is_absolute(),'Absolute destination required')
    evidence=(project/'research_docs/evidence').resolve()
    require(destination.parent.resolve()==evidence and destination.name not in ('.','..'),'New direct research_docs/evidence child required')
    require(not destination.exists() and not destination.is_symlink(),'Immutable destination already exists')
    # Read/verify all bounded allowlisted bytes before creating destination. This
    # makes changed files, stale frozen Python SHA and payload policy fail early.
    payloads=[]
    for entry in view['files']:
        raw=stable_bytes(project,entry['source']);sha=hashlib.sha256(raw).hexdigest()
        require(entry.get('expected_sha256') in (None,sha),'Frozen source SHA mismatch: '+entry['source'])
        payloads.append((entry,raw,sha))
    require(sum(len(raw) for _,raw,_ in payloads)<=MAX_TOTAL,'Combined256MiB changed during preparation')
    # Source/index gate must still hold after preparing bytes.
    require(gate(project,run,assembly)==final_gate,'Final gate changed during export preparation')
    require(selected_ledger_gate(project,audit,assembly,selected_ledgers)==extracted,'Selected ledger gate changed during export preparation')
    manifest=[]
    for entry,raw,sha in payloads:
        manifest.append({'source':entry['source'],'destination':'files/'+entry['source'],'bytes':len(raw),'sha256':sha,'category':entry['category'],'registered_sha256':entry.get('expected_sha256'),'identity_scope':entry.get('identity_scope')})
    result={'version':'paired_lightweight_final_evidence_recovery05_v1','created_utc':time.time(),'status':'export_complete','gate':final_gate,'selected_ledger_gate':{k:v for k,v in extracted.items() if k!='files'},'files':manifest,'total_bytes':sum(x['bytes'] for x in manifest),'file_count':len(manifest),'excluded':'Raw HDF5, visual/features/cache, model tensors, RNG tensors, pickle/npz, videos, general logs and unsupported suffixes remain local; raw failures and historical baselines are preserved', 'source_relative_project':str(project),'immutable_destination':str(destination)}
    readme='# 配对 DAgger 轻量终态证据\n\nMANIFEST.json逐文件登记项目相对源路径、SHA256和字节数。files保留原相对目录。全部64初态、256结果保留，包括软件未完成。DRAFT_REPORT仍需主执行者科学审阅。大载荷、权重、缓存和视频留本机；没有删除或替换任何原始证据。\n'
    require(result['total_bytes']+len((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())+len(readme.encode())<=MAX_TOTAL,'Combined256MiB includes MANIFEST and README')
    evidence.mkdir(parents=True,exist_ok=True)
    destination.mkdir(exist_ok=False)
    try:
        for entry,raw,sha in payloads:
            target=destination/'files'/entry['source'];target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(raw)
            target.chmod(0o444)
        with (destination/'MANIFEST.json').open('x') as stream: json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
        (destination/'MANIFEST.json').chmod(0o444)
        with (destination/'README.md').open('x') as stream: stream.write(readme)
        (destination/'README.md').chmod(0o444)
        # Walk only paths CREATED by this exporter, never any source tree.
        directories={destination}
        for item in manifest:
            p=(destination/item['destination']).parent
            while p!=destination: directories.add(p);p=p.parent
        for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True): directory.chmod(0o555)
        return result
    except Exception:
        # Never overwrite or remove incomplete evidence; caller sees exception.
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=PROJECT);ap.add_argument('--assembly',type=Path,required=True);ap.add_argument('--destination',type=Path);ap.add_argument('--selected-ledgers',type=Path);ap.add_argument('--apply',action='store_true')
    args=ap.parse_args();project=args.project.resolve()
    result=export(project,project/RUN_REL,project/AUDIT_REL,args.assembly,args.destination,apply=args.apply,selected_ledgers=args.selected_ledgers)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
