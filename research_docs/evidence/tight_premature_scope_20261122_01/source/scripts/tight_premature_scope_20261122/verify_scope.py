"""Offline verification of the scope audit and isolated pilot evidence.

Only the standard library is needed. No simulation, inference, or training.
"""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def csvrows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def native_category(success, subgoals):
    if success:
        assert subgoals >= 1
        return 'complete_success'
    return 'subgoal_without_completion' if subgoals >= 1 else 'no_subgoal'


def verify_core(root):
    """Check actual artifact schemas before or after the publication manifest."""
    transitions = csvrows(root/'transitions_by_seed.csv')
    assert len(transitions) == 48
    assert len({(r['group'], r['seed']) for r in transitions}) == 48
    categories = ('no_subgoal', 'subgoal_without_completion', 'complete_success')
    native_tables = {}
    seed_sets = {}
    for group in 'AB':
        pairs = [r for r in transitions if r['group'] == group]
        seed_sets[group] = {int(r['seed']) for r in pairs}
        assert len(pairs) == len(seed_sets[group]) == 24
        native_tables[group] = [[sum(r['original_category'] == original
            and r['entry_category'] == entry for r in pairs)
            for entry in categories] for original in categories]
        for r in pairs:
            for version in ('original', 'entry'):
                assert r[version+'_success'] in ('True', 'False')
                assert r[version+'_legal'] == r[version+'_software_complete'] == 'True'
                assert r[version+'_category'] == native_category(
                    r[version+'_success'] == 'True', int(r[version+'_native_subgoals']))
    assert seed_sets['A'] == seed_sets['B'] == (
        set(range(28851000, 28851008)) | set(range(28852000, 28852016)))
    assert native_tables == {
        'A': [[11, 5, 2], [1, 0, 0], [5, 0, 0]],
        'B': [[10, 5, 1], [2, 2, 3], [1, 0, 0]]}
    transition_summary = read(root/'summary.json')
    assert transition_summary['tables'] == native_tables
    assert tuple(transition_summary['categories']) == categories
    a_success = [r for r in transitions if r['group'] == 'A'
                 and r['original_category'] == 'complete_success']
    a_postfail = [r for r in transitions if r['group'] == 'A'
                 and r['entry_category'] == 'subgoal_without_completion']
    assert len(a_success) == len(a_postfail) == 5
    assert all(r['entry_category'] == 'no_subgoal' for r in a_success)
    assert all(r['original_category'] == 'no_subgoal' for r in a_postfail)
    assert {r['seed'] for r in a_success}.isdisjoint(r['seed'] for r in a_postfail)
    assert transition_summary['selected_sets_overlap'] == 0
    assert transition_summary['A_original_success_seeds'] == [int(r['seed']) for r in a_success]
    assert transition_summary['A_entry_post_subgoal_failure_seeds'] == [int(r['seed']) for r in a_postfail]
    train_dev = read(root/'train_dev/trajectories.json')
    assert len(train_dev) == 276
    assert sum(r['ticks'] for r in train_dev) == 111468
    train = [r for r in train_dev if r['cohort'] == 'train']
    development = [r for r in train_dev if r['cohort'] == 'development']
    assert len(train) == 180 and len(development) == 96
    assert len({(r['path'], r['group']) for r in train}) == 180
    assert Counter(r['trajectory_kind'] for r in train) == {
        'expert_demonstration': 30, 'expert_takeover_v1': 150}
    assert {r['policy'] for r in train} == {'expert_or_takeover'}
    assert len({r['seed'] for r in development}) == 16
    assert len({(r['policy'], r['seed']) for r in development}) == 96
    assert Counter(r['policy'] for r in development) == dict.fromkeys(
        ('A_130_50', 'B_130_50', 'A_R2', 'B_R2', 'A_entry', 'B_entry'), 16)
    assert all(r['native_success'] for r in train)
    assert sum(r['overlap_flags']['entered_left_before10'] for r in train) == 55
    assert Counter(r['exclusive_category'] for r in development) == {
        'never15mm': 47, 'entered_left_before10': 22,
        'premature_withdrawn_fixed_final10': 6, 'other': 21}
    assert sum(r['native_success'] for r in development) == 20
    # Training successes are demonstrations/takeovers, never autonomous counts.
    failed = [r for r in development if not r['native_success']]
    entry_failed = [r for r in failed if r['policy'] in ('A_entry', 'B_entry')]
    entry_no_subgoal = [r for r in entry_failed if r['native_first_subgoal_tick'] is None]
    assert len(failed) == 76 and sum(r['overlap_flags']['never15mm'] for r in failed) == 47
    assert len(entry_failed) == 21 and sum(r['overlap_flags']['never15mm'] for r in entry_failed) == 11
    assert len(entry_no_subgoal) == 14 and sum(r['overlap_flags']['never15mm'] for r in entry_no_subgoal) == 11
    assert sum(r['overlap_flags']['entered_left_before10']
               for r in development if r['native_success']) == 7
    for name, expected_train, expected_development in (
        ('object_down12_after_gate', 7, 2),
        ('palm_down12_after_gate', 22, 2),
        ('command_palm_down12_after_gate', 93, 13)):
        assert sum(r['events'][name]['onset_tick'] is not None for r in train) == expected_train
        assert sum(r['events'][name]['onset_tick'] is not None
                   for r in development if r['native_success']) == expected_development
    for r in train_dev:
        assert r['events']['native_open_command']['known'] is False
        assert r['native_force_consecutive'] is False
        assert r['native_success_steps'] == 10
        assert not r['missing_fields']
    train_dev_summary = read(root/'train_dev/summary.json')
    for name, pool in [('train', train), ('development', development)]:
        assert train_dev_summary[name]['n'] == len(pool)
        assert train_dev_summary[name]['exclusive'] == Counter(r['exclusive_category'] for r in pool)
        assert train_dev_summary[name]['outcomes'] == Counter(r['native_outcome'] for r in pool)
    terminal = read(root/'terminal/trajectories.json')
    assert len(terminal) == 96
    assert len({(r['policy'], r['seed']) for r in terminal}) == 96
    assert all(r['software_complete'] and r['legal'] for r in terminal)
    assert sum(r['native_success'] for r in terminal) == 12
    assert sum(r['ticks'] for r in terminal) == 54751
    terminal_map = {(r['policy'], str(r['seed'])): r for r in terminal}
    for r in terminal:
        assert r['native_force_consecutive'] is False and r['native_success_steps'] == 10
        assert r['events']['native_open_command']['known'] is False
        assert not r['missing_fields']
    for pair in transitions:
        for version, suffix in (('original', '_130_50'), ('entry', '_entry')):
            row = terminal_map[pair['group']+suffix, pair['seed']]
            assert row['native_success'] == (pair[version+'_success'] == 'True')
            assert row['native_max_subgoals'] == int(pair[version+'_native_subgoals'])
            assert row['ticks'] == int(pair[version+'_steps'])
            assert pair[version+'_category'] == native_category(
                row['native_success'], row['native_max_subgoals'])
    terminal_summary = read(root/'terminal/summary.json')
    for role in ('A_130_50', 'A_entry', 'B_130_50', 'B_entry'):
        pool = [r for r in terminal if r['policy'] == role]
        assert terminal_summary[role]['n'] == len(pool) == 24
        assert terminal_summary[role]['exclusive'] == Counter(r['exclusive_category'] for r in pool)
        assert terminal_summary[role]['outcomes'] == Counter(r['native_outcome'] for r in pool)
    receipt = read(root/'train_dev/receipt.json')
    assert receipt['new_simulation'] == receipt['new_policy_inference'] == receipt['optimizer_updates'] == receipt['new_dagger'] == 0
    assert receipt['terminal_trajectories_read'] == 0
    terminal_receipt = read(root/'terminal/receipt.json')
    assert terminal_receipt['trajectories'] == 96 and terminal_receipt['ticks'] == 54751
    for key in ('optimizer_updates', 'new_dagger_trajectories', 'new_physics_steps', 'new_policy_inferences'):
        assert terminal_receipt[key] == 0
    assert terminal_receipt['description_only'] and terminal_receipt['scripts_and_rules_unchanged']
    assert not terminal_receipt['thresholds_changed']
    assert not terminal_receipt['triggers_selected'] and not terminal_receipt['pilot_seeds_selected']
    assert receipt['script_sha256'] in terminal_receipt['pinned_sha256'].values()
    assert receipt['rules_sha256'] in terminal_receipt['pinned_sha256'].values()
    for folder, count in [('train_dev', 111468), ('terminal', sum(r['ticks'] for r in terminal))]:
        with gzip.open(root/folder/'derived_ticks.csv.gz', 'rt', newline='') as stream:
            assert sum(1 for _ in csv.DictReader(stream)) == count


def main():
    root = Path(__file__).resolve().parent
    if not (root/'published_sha256.json').is_file():
        root = Path(__file__).resolve().parents[2]/'research_docs/evidence/tight_premature_scope_20261122_01'
    manifest = read(root/'published_sha256.json')
    for name, expected in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    verify_core(root)
    receipt = read(root/'train_dev/receipt.json')
    assert hashlib.sha256((root/'source/scripts/tight_premature_scope_20261122/trajectory_audit.py').read_bytes()).hexdigest() == receipt['script_sha256']
    assert hashlib.sha256((root/'rules_registration.json').read_bytes()).hexdigest() == receipt['rules_sha256']
    closure = read(root/'closure.json')
    assert closure['complete'] and closure['optimizer_updates'] == 0
    assert closure['new_training_pool_trajectories'] == 0
    assert closure['pilot_complete'] and closure['pilot_owners_reaped']
    pilot = read(root/'pilot/summary.json')
    assert pilot['opportunities'] == len(pilot['rows']) == 16
    assert pilot['software_incomplete'] == pilot['illegal'] == pilot['optimizer_updates'] == 0
    assert pilot['training_pool_admitted_trajectories'] == 0
    assert pilot['diagnostic_expert_success_is_autonomous'] is False
    assert all(v['matched'] and v['expected'] == v['actual'] for v in pilot['protected_before_after'].values())
    assert pilot['frozen_source_hashes_matched'] and pilot['frozen_wrapper_hashes_matched']
    for name in ('source_hashes_before_after.json', 'wrapper_hashes_before_after.json'):
        checks = read(root/'pilot'/name)
        assert checks and all(v['matched'] and v['expected'] == v['actual'] for v in checks.values())
    lifecycle = read(root/'pilot/lifecycle.json')
    assert lifecycle['children_reaped'] and len(lifecycle['process_checks']) == 6
    assert not any(v['matching_owned_process_still_alive'] for v in lifecycle['process_checks'].values())
    assert lifecycle['timing']['project_complete_utc'] <= lifecycle['timing']['frozen_deadline_utc']
    assert len({(r['role'], r['seed'], r['diagnostic_arm']) for r in pilot['rows']}) == 16
    assert len({r['seed'] for r in pilot['rows']}) == 4
    for r in pilot['rows']:
        assert r['legal'] and r['software_complete'] and r['accepted_for_training'] is False
        if r['diagnostic_arm'] == 'autonomous':
            assert r['expert_ticks'] == r['expert_valid_ticks'] == 0
            assert r['actual_expert_switch_tick'] is None
            assert r['autonomous_success'] == r['native_success']
        elif r['actual_expert_switch_tick'] is not None:
            g = r['trigger_geometry']
            assert g['confirmation_transition_tick'] + 1 == g['expert_first_action_tick'] == r['actual_expert_switch_tick']
            assert g['native_goal_before'] == g['native_goal_after'] == 0
            assert g['keypoint_mm'] > 15 and g['object_goal_root_mm'] <= 90
            assert g['minimum_contact_fingers_both_substeps'] >= 2
    for role in 'AB':
        controls = [r for r in pilot['rows'] if r['role'] == role and r['diagnostic_arm'] == 'autonomous']
        rescues = [r for r in pilot['rows'] if r['role'] == role and r['diagnostic_arm'] == 'alignment_rescue']
        triggered = [r for r in rescues if r['actual_expert_switch_tick'] is not None]
        c = pilot['counts'][role]
        assert len(controls) == c['autonomous_opportunities'] == len(rescues) == c['rescue_opportunities'] == 4
        assert sum(r['native_success'] for r in controls) == c['autonomous_native_successes']
        assert len(triggered) == c['triggered']
        assert sum(r['expert_rescue_success'] for r in triggered) == c['expert_successes']
    assert len(pilot['paired']) == 8
    numeric = sorted((root/'pilot').glob('wave*/*/numeric_ticks.csv.gz'))
    assert len(numeric) == 16
    for path in numeric:
        schema = read(path.with_suffix('.schema.json'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == schema['gzip_sha256']
        row = next(r for r in pilot['rows'] if r['wave'] == path.parents[1].name and r['case'] == path.parent.name)
        assert schema['rows'] == row['steps'] and schema['columns'] > 1
    decoded = subprocess.run([sys.executable, str(root/'audit_inputs/decode_audit_inputs.py'),
        str(root/'audit_inputs'), '--verify-only'], check=True, capture_output=True, text=True)
    inputs = json.loads(decoded.stdout)
    assert inputs['passed'] and inputs['records'] == 372 and inputs['arrays'] == 5658
    assert inputs['cohorts'] == {'train': 180, 'development': 96, 'terminal': 96}
    print(f'PASS: {len(manifest)} hashes; 48 paired states; 276 train/development + 96 frozen terminal trajectories; isolated completed pilot; zero training.')


if __name__ == '__main__':
    main()
