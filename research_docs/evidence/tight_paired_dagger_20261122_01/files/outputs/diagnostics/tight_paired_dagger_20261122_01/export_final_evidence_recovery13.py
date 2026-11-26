"""Inactive bounded final13 exporter; old05/06/07 failures remain immutable evidence.

The pinned05 engine supplies the unchanged64/256 outcome, selected-counter,
16MiB/file and256MiB aggregate gates and no-overwrite export. This adapter uses
truthful failed10/12 plus closed13 lifecycle bindings and actual cycle_logs_recovery12.
No export, GPU or runtime action occurs on import.
"""
from __future__ import annotations
import argparse
import hashlib
import stat
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_final13_common_export', HERE/'final_recovery13_common.py')
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)
BASE_SHA256 = '56a8212ac5494a8ccda908fc390e926508d035694fb3a85a9ba0c7c410e89d6b'
base = common.pinned_module('export_final_evidence_recovery05.py',BASE_SHA256,'export',
    replacements=(("run/'cycle_logs/summarize_test_exit.json'",
                   "Path(SUMMARY_EXIT_PATH)"),))
PROJECT,RUN_REL,AUDIT_REL = base.PROJECT,base.RUN_REL,base.AUDIT_REL
base.RECOVERY05_STAGE_NAMES = common.OLD_COMPLETED_STAGES + ('train_four_components_parallel_rears',)
base.RECOVERY05_AUDIT_RECEIPTS = (
    'collection_integrity_recovery05_actual01/integrity.json',
    'collection_integrity_recovery05_actual01/noise_audit.json',
    'collection_noise_recovery05_actual01/noise_audit.json',
    'driver_main_recovery05/status.json','driver_main_recovery05/failure.json',
    'driver_main_recovery05/storage_gate.json',
    'wait_main_shared06/status.json','wait_main_shared06/failure.json',
    'wait_main_shared06/reviewed_main_exit.json')


# Only these two exact, mandatory lightweight JSON receipts may traverse a
# directory named caches. The inherited09 receipt is pinned by the actual10
# registration; current10 is pinned by its final root-reviewed binding. Every
# sibling and all payload formats retain the pinned base policy.
CACHE_PHASE_METADATA_REL=str(Path(AUDIT_REL)/'parallel_rear_recovery10_runtime/caches/pipeline_events.json')
INHERITED_CACHE_PHASE_METADATA_REL=str(Path(AUDIT_REL)/'parallel_rear_recovery09_runtime/caches/pipeline_events.json')
CACHE_PHASE_METADATA_PATHS=frozenset((CACHE_PHASE_METADATA_REL,INHERITED_CACHE_PHASE_METADATA_REL))
ORIGINAL_SOURCE_PATH=base.source_path


def source_path(project,relative):
    relative=base.relpath(relative)
    if relative not in CACHE_PHASE_METADATA_PATHS:
        return ORIGINAL_SOURCE_PATH(project,relative)
    project=Path(project).resolve();path=project/relative
    common.require(path.resolve().is_relative_to(project),'Metadata source escapes project')
    current=path
    while current!=project:
        common.require(not current.is_symlink(),'Scoped metadata source cannot use symlinks')
        current=current.parent
    common.require(path.is_file() and stat.S_ISREG(path.stat().st_mode),
        'Scoped metadata source must be one regular JSON file')
    common.require(path.suffix=='.json' and path.stat().st_size<=base.MAX_FILE,
        'Scoped metadata retains the16MiB file limit')
    binding,_=common.reviewed_binding(project/RUN_REL,project/AUDIT_REL)
    expected=binding.get('small_file_sha256',{}).get(str(path.resolve()))
    common.require(isinstance(expected,str) and len(expected)==64,
        'Scoped cache-phase metadata must be in the root-reviewed binding')
    raw=common.stable_small(path,maximum=base.MAX_FILE)
    common.require(hashlib.sha256(raw).hexdigest()==expected,
        'Scoped cache-phase metadata differs from the root-reviewed SHA')
    rows=json.loads(raw)
    common.require(isinstance(rows,list),'Scoped cache-phase metadata must be a JSON event list')
    return path


base.source_path=source_path


def closed_receipt_gate(project,run,audit):
    lifecycle = common.closure_gate(run,audit)
    common.require(lifecycle['status'] == 'closed', 'Actual fragment13 closure required')
    watcher = common.watcher_closed(audit)
    integrity = common.read_small(audit/'collection_integrity_recovery05_actual01/integrity.json')
    observed = dict(passed=47,unverified=1,failed=0,all_verified=False)
    common.require(integrity.get('version') == 'paired_collection_integrity_recovery05_v1'
        and integrity.get('cuda_initialized') is False and integrity.get('read_only_inputs') is True
        and integrity.get('passed') is True and integrity.get('opportunities') == 96
        and integrity.get('paired_opportunities') == 48 and integrity.get('observable_pair_checks') == observed
        and integrity.get('full48_observed_initial_states_verified') is False,
        'Actual collection96/47+1 integrity differs')
    sets = []
    for directory,cuda in [('collection_integrity_recovery05_actual01',False),
                           ('collection_noise_recovery05_actual01',True)]:
        noise = common.read_small(audit/directory/'noise_audit.json')
        rows,unavailable,pairs = (noise.get(key) for key in ('rows','unavailable','pairs'))
        common.require(isinstance(rows,list) and isinstance(unavailable,list) and isinstance(pairs,list),
                       'Collection noise populations missing')
        certified,absent = {r.get('case') for r in rows},{r.get('case') for r in unavailable}
        common.require(noise.get('version') == 'paired_collection_native_noise_recovery05_v1'
            and noise.get('assignments') == 96 and noise.get('paired_assignments') == 48
            and noise.get('observable_pair_checks') == observed
            and noise.get('sampler_seed') == 7091 and noise.get('per_native_plan_draw_shape') == [1,24,29]
            and noise.get('commit_control_ticks') == 12 and noise.get('cuda_used') is cuda
            and noise.get('cpu_receipt_checks_passed') == len(rows)
            and noise.get('uncertified_execution_endpoints') == len(unavailable)
            and len(rows)+len(unavailable) == 96 and len(pairs) == 48
            and len(certified) == len(rows) and len(absent) == len(unavailable)
            and not certified&absent and None not in certified|absent
            and all(row.get('retained_in_denominator') is True for row in unavailable)
            and noise.get('all96_execution_endpoints_certified') is (not unavailable)
            and noise.get('exact_cuda_full_endpoint_checks') == (len(rows) if cuda else 0)
            and noise.get('student_prefix_cuda_reconstructed') == (len(rows) if cuda else 0)
            and noise.get('observed_takeover_rng_states') == 0
            and noise.get('independent_observed_takeover_rng_claim') is False,
            'Actual collection CPU/CUDA noise certification differs')
        sets.append((certified,absent))
    common.require(sets[0] == sets[1], 'CPU/CUDA certified/uncertified endpoints differ')
    return dict(status='closed',version='actual_recovery13_final_receipts_v1',
                lifecycle=lifecycle,watcher=watcher,failed06_preserved=True,failed07_preserved=True,failed08_preserved=True,failed09_preserved=True,
                actual_initial_pairs=observed,failed10_preserved=True,failed12_preserved=True,
                evaluation_execution=lifecycle['evaluation_execution'])


base.closed_receipt_gate = closed_receipt_gate
ORIGINAL_GATE,ORIGINAL_ALLOWLIST,ORIGINAL_EXPORT = base.gate,base.allowlist,base.export


def gate(project,run,assembly):
    status = common.read_small(assembly/'assembly_status.json')
    common.require(status.get('assembler_sha256') == common.sha(project/AUDIT_REL/'final_report_assembler_recovery13.py')
        and status.get('actual_checkpoint_reads') == 0
        and status.get('actual_selected_ledger_tables_reused') == 8,
        'Actual10 assembler must reuse the successful selected-ledger pass')
    binding,_=common.reviewed_binding(run,project/AUDIT_REL)
    base.SUMMARY_EXIT_PATH=binding['stage_exit_paths']['summarize_test']
    value = ORIGINAL_GATE(project,run,assembly)
    value['actual_recovery13_closure'] = value.pop('actual_recovery05_closure')
    value['adapter_version'] = 'paired_final_evidence_recovery13_v1'
    value['pinned_export_engine_sha256'] = BASE_SHA256
    return value


def allowlist(project,run,audit,assembly):
    entries = {row['source']:row for row in ORIGINAL_ALLOWLIST(project,run,audit,assembly)}
    binding,review = common.reviewed_binding(run,audit)
    def add(path,category,expected=None):
        path = Path(path).resolve(); common.require(path.is_relative_to(project), 'Export source escapes project')
        common.require(path.is_file(), 'Required final10 export source missing: '+str(path))
        relative = str(path.relative_to(project))
        previous = entries.get(relative,{})
        common.require(expected is None or previous.get('expected_sha256') in (None,expected),
                       'Conflicting frozen final10 export source')
        entries[relative] = dict(source=relative,category=category,
                                expected_sha256=expected or previous.get('expected_sha256'))
    for name in common.FINAL_TOOLS:
        add(audit/name,'reviewed_recovery13_final_tool',common.sha(audit/name))
    for name in (common.BINDING_NAME,common.REVIEW_NAME,'COMPUTE_RECORD_RECOVERY13.json',
                 'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json','OLD_WATCHER07_CLOSED_FOR_RECOVERY08.json',
                 'OLD_WATCHER08_CLOSED_FOR_RECOVERY09.json','OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',
                 'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json','EVAL_STARTUP12_OWNED_TERMINATION.json',
                 'WATCHER12_ACTUAL_CLOSED.json','WATCHER13_ACTUAL_CLOSED.json',
                 'ROOT_TEST_FRAGMENT13_BOUNDARY.json','ROOT_TEST_FRAGMENT13_REVIEW.json',
                 'TEST_FRAGMENT13_REGISTRATION.json','test_fragment13_job.json','WATCHER08_START_DIRECTORY_FAILURE.json',
                 'CLOSED_MAIN_DP_DEDUP08_APPLY01.json','CLOSED_MAIN_DP_DEDUP08_PARTIAL_DIAGNOSIS01.json'):
        add(audit/name,'actual_recovery13_final_binding_and_compute')
    # These are bounded optional source/review/transaction receipts. General logs
    # and the source patch stay local, with their SHA retained in the registry.
    for name in ('ROOT_BACKEND_RECOVERY10_REVIEW.json','BACKEND_RECOVERY10_REVIEW.json',
                 'PAIRED_CONTRACT07_ACTUAL_PREFLIGHT.json','FINAL_RECOVERY10_TOOLS_REVIEW.json',
                 'FINAL_RECOVERY10_TOOLS_REVIEW.md','FINAL_RECOVERY10_TEST_REVIEW_NOTE.txt',
                 'FINAL_RECOVERY10_TOOLS_REVIEW_V2.json','FINAL_EVIDENCE10_HANDOFF_V2.json',
                 'WATCHER09_RESTORE_METADATA10_V2_AUDIT.json','FINAL_RECOVERY10_RECIPE.md',
                 'FINAL_RECOVERY10_TOOLS_REVIEW_V3.json','FINAL_EVIDENCE10_HANDOFF_V3.json',
                 'FINAL10_V3_PATCH_REVIEW.json','FINAL_RECOVERY10_TOOLS_REVIEW_V4.json',
                 'FINAL_EVIDENCE10_HANDOFF_V4.json','ROOT_FINAL10_V3_ADDITIONAL_EXPORT_FINDING.json',
                 'FINAL_RECOVERY12_RECIPE.md','FINAL_RECOVERY12_TOOLS_REVIEW.json',
                 'FINAL_RECOVERY13_RECIPE.md','FINAL_RECOVERY13_TOOLS_REVIEW.json',
                 'TEST_FRAGMENT13_PREPARATION_RECIPE.md','TEST_FRAGMENT13_SOURCE_REVIEW.md'):
        if (audit/name).is_file():add(audit/name,'bounded_recovery10_review_or_preflight')
    attempts=sorted(p.name for p in audit.iterdir() if p.is_dir() and p.name.startswith('leased_archive_shared') and p.name[len('leased_archive_shared'):].isdigit())
    for attempt in attempts:
        directory=audit/attempt
        for path in sorted(directory.glob('*.json')):
            common.require(path.name in ('registration.json','failure.json','complete.json')
                or path.name.startswith('file_'), 'Unexpected auxiliary archive metadata')
            add(path,'auxiliary_archive_attempt_retained')
    for path,digest in binding['small_file_sha256'].items():
        add(Path(path),'root_reviewed_real_recovery13_source_or_receipt',digest)
    for path in binding['stage_exit_paths'].values():
        add(Path(path),'actual_resolved_stage12_or13_exit')
    add(Path(binding['watcher_exit_path']),'actual_watcher13_restored_closure')
    for name in ('finished.json','deployment.json','selected.json'):
        for role in 'AB':
            add(run/'learning'/role/'training'/name,'preserved_completed_DP60_metadata')
    return [entries[key] for key in sorted(entries)]


base.gate,base.allowlist = gate,allowlist


def export(project,run,audit,assembly,destination,*,apply=False,selected_ledgers=None):
    # The engine's manifest version identifies its preserved bounded-copy format;
    # gate.adapter_version records10 lifecycle semantics without rewriting files.
    try:
        common.require(selected_ledgers is not None, 'Explicit actual selected-ledger10 output required')
        status = common.read_small(Path(assembly)/'assembly_status.json')
        common.require(Path(status.get('selected_ledgers_directory','')).resolve() == Path(selected_ledgers).resolve(),
                       'Assembler/export selected-ledger directory differs')
        for scope in ('common180_parents','learning'):
            for role in 'AB':
                for component in ('frontend','backend'):
                    common.extracted_component(run,audit,selected_ledgers,scope,role,component)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        if apply: raise ValueError('Export10 blocked before destination creation: '+str(exc)) from exc
        return dict(status='blocked',reason=str(exc),destination_created=False,copied_files=0)
    return ORIGINAL_EXPORT(project,run,audit,assembly,destination,apply=apply,selected_ledgers=selected_ledgers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--assembly',type=Path,required=True)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--selected-ledgers',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args(); project = args.project.resolve()
    print(json.dumps(export(project,project/RUN_REL,project/AUDIT_REL,args.assembly,args.destination,
        apply=args.apply,selected_ledgers=args.selected_ledgers),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
