"""Inactive CPU-only final ledger adapter for admission14.

The pinned05 validator still reads exactly eight actual selected ledgers. Only
the lifecycle gate is replaced: failed06/07/08/09 attempts remain truthful ancestry,
and the actual14 native suffix must finish before any checkpoint can be opened.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_final14_common_extract', HERE/'final_recovery14_common.py')
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)
BASE_SHA256 = '197eb8881358740ecba907042d4dc65ed93b268a70c96d32eb0bc4132b58f678'
base = common.pinned_module('extract_selected_ledgers_recovery05.py', BASE_SHA256, 'extract')
base.closure_gate = common.closure_gate
base.VERSION = 'selected_successful_step_ledger_export_recovery14_v1'
ORIGINAL_SELECTED_COMPONENT = base.selected_component


def selected_component(root, scope, role, component, *, loader=base.load_checkpoint):
    path = base.component_directory(root,scope,role,component)/'best.pt'
    before = common.identity(path.stat())
    value = ORIGINAL_SELECTED_COMPONENT(root,scope,role,component,loader=loader)
    after = common.identity(path.stat())
    common.require(before == after, 'Checkpoint changed during selected SHA/mmap/count pass')
    value['checkpoint_stat_identity'] = after
    value['checkpoint_content_verification'] = 'Actual selected SHA256 and CPU mmap validated in this extraction; stat identity only guards subsequent immutable reuse.'
    return value


base.selected_component = selected_component
# The unchanged private writer records the adapter that actually controls entry.
base.__file__ = __file__
ROOT = base.ROOT


def extract(root, audit, output, *, loader=base.load_checkpoint):
    return base.extract(root, audit, output, loader=loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status = extract(args.root, args.audit, args.output)
    print(json.dumps(dict(status=status['status'], output=str(args.output),
                          checkpoint_reads=status.get('checkpoint_reads')), ensure_ascii=False))
    raise SystemExit(0 if status['status'] == 'passed' else 2)


if __name__ == '__main__': main()
