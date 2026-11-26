"""Execute a newly registered frozen test without historical experiment globals."""
import argparse
from pathlib import Path

from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    plan = read(output / 'plan.json')
    root = Path(plan['resource_root']).resolve()
    freeze = read(root / 'registration.json')
    assert freeze['version'] == 'tight_entry_final24_frozen_v1'
    assert plan['kind'] == 'autonomous' and plan['test_used'] is True
    assert plan['holdout_contract'] == 'layout_final_test_v1'
    assert plan['pairs'] == freeze['pairs']
    assert str(output) in freeze['wave_plan_sha256']
    assert sha256(output / 'plan.json') == freeze['wave_plan_sha256'][str(output)]
    assert {c['spec']['seed'] for c in plan['cases']} <= set(freeze['seeds'])
    assert len(plan['cases']) == 4 and {c['role'] for c in plan['cases']} == set(freeze['pairs'])
    from visuotactile.paired_execution import install
    from tools.tight_evaluate import run
    install(output)
    run(output, allow_heldout=True)


if __name__ == '__main__':
    main()
