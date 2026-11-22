"""Close the completed retrospective audit and isolated rescue pilot."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[2]
PUBLIC = PROJECT/'research_docs/evidence/tight_premature_scope_20261122_01'
DIAG = PROJECT/'outputs/diagnostics/tight_premature_scope_20261122_01'
PILOT = PROJECT/'outputs/visuotactile/run/tight_alignment_pilot_20261122_01'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pilot = read(PUBLIC/'pilot/summary.json')
    assert pilot['opportunities'] == 16 and len(pilot['rows']) == 16
    assert pilot['software_incomplete'] == pilot['illegal'] == pilot['optimizer_updates'] == 0
    assert pilot['training_pool_admitted_trajectories'] == 0
    assert all(v['matched'] for v in pilot['protected_before_after'].values())
    assert pilot['frozen_source_hashes_matched'] and pilot['frozen_wrapper_hashes_matched']
    lifecycle = read(PUBLIC/'pilot/lifecycle.json')
    assert lifecycle['children_reaped'] and len(lifecycle['process_checks']) == 6
    assert not any(v['matching_owned_process_still_alive'] for v in lifecycle['process_checks'].values())
    assert lifecycle['timing']['project_complete_utc'] <= lifecycle['timing']['frozen_deadline_utc']
    owner_exit = read(PILOT/'project_exit.json')
    assert owner_exit['children_reaped']
    assert (PILOT/'monitor_exit.json').exists()
    for name in ('rules_registration.json', 'rules_addendum.json'):
        shutil.copy2(DIAG/name, PUBLIC/name)
    sources = [*sorted((PROJECT/'scripts/tight_premature_scope_20261122').glob('*.py')),
               *sorted((PROJECT/'configs/experiments/premature_scope_pilot').glob('*.json'))]
    for path in sources:
        dest = PUBLIC/'source'/path.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    receipt = read(PUBLIC/'train_dev/receipt.json')
    assert sha(PROJECT/'scripts/tight_premature_scope_20261122/trajectory_audit.py') == receipt['script_sha256']
    assert sha(PUBLIC/'rules_registration.json') == receipt['rules_sha256']
    shutil.copy2(PROJECT/'scripts/tight_premature_scope_20261122/verify_scope.py', PUBLIC/'verify_scope.py')
    report = PROJECT/'research_docs/TIGHT_PREMATURE_SCOPE_20261122.md'
    if not (PUBLIC/'transition_report.md').exists():
        shutil.copy2(PUBLIC/'REPORT.md', PUBLIC/'transition_report.md')
    shutil.copy2(report, PUBLIC/'REPORT.md')
    write(PUBLIC/'closure.json', dict(
        complete=True, closed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        original_main_evaluation_repeated=False, train_development_audited=276,
        frozen_terminal_audited=96, retrospective_total_ticks=166219,
        pilot_complete=True, pilot_opportunities=16,
        pilot_actual_expert_switches=sum(r['actual_expert_switch_tick'] is not None for r in pilot['rows']),
        pilot_owners_reaped=True, pilot_monitor_reaped=True,
        optimizer_updates=0, new_training_pool_trajectories=0,
        expert_diagnostic_captures_isolated=True, protected_files_unchanged=True,
        terminal_rules_frozen=True, terminal_used_for_pilot_selection=False,
        pilot_registration_sha256=sha(PILOT/'registration.json'),
        report_sha256=sha(report), publication_scope='All reviewable lightweight evidence; raw payloads remain local as confirmed by user'))
    files = {str(p.relative_to(PUBLIC)): sha(p) for p in sorted(PUBLIC.rglob('*'))
             if p.is_file() and p.name != 'published_sha256.json' and '__pycache__' not in p.parts}
    write(PUBLIC/'published_sha256.json', files)
    print(json.dumps(dict(files=len(files), bytes=sum((PUBLIC/n).stat().st_size for n in files),
        pilot=pilot['counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
