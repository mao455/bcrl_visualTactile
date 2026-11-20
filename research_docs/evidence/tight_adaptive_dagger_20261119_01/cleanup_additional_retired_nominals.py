"""Additional explicit historical derivative scope; inherit all existing safeguards."""
import argparse
import importlib.util
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('protected_nominals', PUBLIC / 'cleanup_more_retired_nominals.py')
protected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protected)
retired = protected.retired
retired.LEDGER = retired.ROOT / 'resource_cleanup_20261120/attempt11_nominal'
retired.ALLOWED = (
    'tight_fresh_20261114_02', 'tight_fresh_20261114_03',
    'tight_fresh_20261114_04', 'tight_fresh_20261114_05', 'tight_fresh_20261114_06',
    'tight_fresh_20261115_02', 'tight_fresh_20261115_03', 'tight_fresh_20261115_04',
    'tight_fresh_20261115_06', 'tight_fresh_20261115_07', 'tight_fresh_20261115_08',
    'tight_fresh_20261115_09', 'tight_fresh_20261115_10', 'tight_fresh_20261115_11',
    'tight_fresh_20261115_12', 'tight_fresh_20261115_13',
    'tight_fresh_20261116_01', 'tight_fresh_20261116_03',
    'tight_balance76_20261117_01_recovery01', 'tight_balance76_20261117_01_recovery02',
    'tight_sampling76_A_20261117_01',
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-sha256')
    args = parser.parse_args()
    if args.apply_sha256:
        retired.apply(args.apply_sha256)
    else:
        retired.plan()
