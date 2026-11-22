"""Isolated diagnostic dispatch; production task, controller and success stay intact."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def validate(plan):
    from visuotactile.adaptive_takeover import profile
    from visuotactile.state_benchmark import dagger_configuration
    assert plan['diagnostic_contract'] == 'frozen_alignment_rescue_no_admission_v1'
    assert plan['kind'] == 'takeover' and plan['test_used'] is False
    assert len(plan['cases']) == plan['batch']['lanes'] == 4
    assert {c['diagnostic_arm'] for c in plan['cases']} == {'autonomous', 'alignment_rescue'}
    for case in plan['cases']:
        assert case['spec']['split'] == 'train' and case['spec']['takeover_tick'] == 10**9
        expected = profile('alignment') if case['diagnostic_arm'] == 'alignment_rescue' else None
        assert case['dagger']['trigger'] == expected
        dagger_configuration(case['dagger'], case['spec'], read(case['source_data']))


def worker(output):
    from tools import tight_fresh_batch_run as runner
    from visuotactile.paired_execution import install
    from visuotactile.state_benchmark import audit_dagger_capture
    from tools.tight_fresh_report import audit_trigger_case
    output = Path(output).resolve()
    plan = read(output/'plan.json')
    freeze = read(Path(plan['resource_root'])/'registration.json')
    assert sha256(output/'plan.json') == freeze['wave_plan_sha256'][str(output)]
    for path, digest in freeze['wrapper_sha256'].items():
        assert sha256(path) == digest, path
    validate(plan)
    original_validate, original_admit = runner.validate_takeover_plan, runner.admit_capture

    def diagnostic_admission_blocked(output, case, result, pair, plan, entries, *, audit_memo=None):
        assert not entries
        folder = output/case['name']
        audit = (audit_dagger_capture(folder, pair, **({} if audit_memo is None else {'audit_memo': audit_memo}))
                 if result.get('legal') else None)
        trigger = audit_trigger_case(folder, result) if result.get('legal') else None
        if audit is not None:
            assert audit['passed']
        if case['diagnostic_arm'] == 'autonomous':
            assert result.get('dagger', {}).get('expert_ticks', 0) == 0
            assert result.get('dagger', {}).get('expert_valid_ticks', 0) == 0
        result.update(accepted_for_training=False, diagnostic_only=True,
                      diagnostic_arm=case['diagnostic_arm'], training_pool_admission='blocked')
        atomic_json(folder/'diagnostic_audit.json', dict(provenance=audit, trigger=trigger,
            training_pool_admission='blocked', actual_expert_success_is_autonomous=False,
            source_plan_sha256=sha256(output/'plan.json'), utc=time.time()))
        atomic_json(folder/'result.json', result)
        atomic_json(output/'correction_entries.json', {})

    runner.validate_takeover_plan = validate
    runner.admit_capture = diagnostic_admission_blocked
    install(output)
    try:
        runner.run(output)
    finally:
        runner.validate_takeover_plan, runner.admit_capture = original_validate, original_admit


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument('worker')
        p.add_argument('--output', type=Path, required=True)
        a = p.parse_args()
        try:
            worker(a.output)
        except BaseException as error:
            atomic_json(a.output/'diagnostic_worker_failure.json', dict(error=repr(error), utc=time.time()))
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
    else:
        from tools import tight_project
        tight_project.ENTRIES['diagnostic_alignment_pilot'] = (
            'evaluate', 'scripts.tight_premature_scope_20261122.pilot_entry', ['worker'])
        tight_project.main()


if __name__ == '__main__':
    main()
