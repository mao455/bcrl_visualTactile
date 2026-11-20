"""Second explicit set of inactive, reproducible nominal derivatives; protected roots excluded."""
import argparse
import importlib.util
import json
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retired_nominals', PUBLIC / 'cleanup_retired_nominals.py')
retired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retired)
retired.LEDGER = retired.ROOT / 'resource_cleanup_20261120/attempt09_nominal'
retired.ALLOWED = (
    'tight_long_20261117_01', 'tight_recovery_20261118_01',
    'tight_cycle_20261118_01', 'tight_phase_20261116_01',
    'tight_fresh_20261114_01', 'tight_fresh_20261115_01',
    'tight_layout10_20261116_learning01', 'tight_layout10_20261116_dagger01',
)
original_safe = retired.safe
PROTECTED_ROOTS = tuple(Path(p).resolve() for p in json.loads(
    (retired.ROOT / 'ACTIVE_DEPENDENCIES.json').read_text())['protected_roots_include_future_files'])


def safe(path):
    path = Path(path)
    assert not any(path.resolve().is_relative_to(root) for root in PROTECTED_ROOTS), path
    return original_safe(path)


retired.safe = safe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-sha256')
    args = parser.parse_args()
    if args.apply_sha256:
        retired.apply(args.apply_sha256)
    else:
        retired.plan()
