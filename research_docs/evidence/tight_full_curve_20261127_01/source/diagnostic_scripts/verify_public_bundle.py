"""Offline verification of exported identities and aggregate numeric results.

Usage: python source/diagnostic_scripts/verify_public_bundle.py [bundle_directory]
Only Python's standard library is required. This verifies exported evidence,
not the local raw sensor payloads or hidden simulator state.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import sys


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name, identity in manifest['files'].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root), name
        assert path.stat().st_size == identity['bytes'], name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        assert digest.hexdigest() == identity['sha256'], name
    summary = json.loads((root / 'audit/primary_summary.json').read_text())
    with (root / 'audit/per_initial_state.csv').open(newline='') as stream:
        cases = list(csv.DictReader(stream))
    assert len(cases) == summary['complete_opportunities'] == 224
    assert summary['old64_used'] is False
    assert summary['actual_layout_all_passed'] is True
    assert not any(28912000 <= int(row['seed']) <= 28912063 for row in cases)
    groups = collections.defaultdict(list)
    seen = set()
    for row in cases:
        key = (int(row['epoch']), row['role'], row['condition'], row['cohort'])
        identity = (*key, int(row['seed']))
        assert identity not in seen, identity
        seen.add(identity)
        assert row['actual_layout_passed'] == 'True', identity
        groups[key].append(row)
    field_map = {'strict_success': 'success', 'stable_hold': 'stable_hold',
                 'stable_near_hole': 'stable_near_hole',
                 'first_15mm_gate': 'first_15mm_gate',
                 'native_first_subgoal': 'native_first_subgoal'}
    for row in summary['rows']:
        key = (row['epoch'], row['role'], row['condition'], row['cohort'])
        records = groups[key]
        assert len(records) == row['registered']
        for aggregate, field in field_map.items():
            assert sum(r[field] == 'True' for r in records) == row[aggregate], (key, aggregate)
        assert sum(r['software_complete'] == 'True' for r in records) == row['completed']
        n, k = row['registered'], row['strict_success']
        z = 1.959963984540054
        center = (k/n + z*z/(2*n)) / (1+z*z/n)
        half = z*math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n)) / (1+z*z/n)
        assert all(abs(a-b) < 1e-9 for a,b in zip(row['wilson95'], [max(0, center-half), min(1, center+half)]))
    for pair in summary['paired_contrasts']:
        n = pair['n']
        assert pair['both_success'] + pair['gained'] + pair['lost'] + pair['both_failure'] == n
        assert pair['left_success'] == pair['both_success'] + pair['gained']
        assert pair['right_success'] == pair['both_success'] + pair['lost']
        assert abs(pair['paired_difference_pp'] - 100*(pair['gained']-pair['lost'])/n) < 1e-10
    print(json.dumps({'verified_files': len(manifest['files']), 'unique_opportunities': len(cases),
                      'aggregate_rows': len(summary['rows']), 'paired_contrasts': len(summary['paired_contrasts']),
                      'scope': 'Export hashes, case counts, phase numerators, Wilson intervals and paired arithmetic; raw RGB-D/models remain local'}))


if __name__ == '__main__':
    default = Path(__file__).resolve().parents[2]
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
