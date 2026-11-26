"""Proposed future gates only; never imported by active recovery scripts.

Uses the exact same external project predicate as watcher v4. Original owned
child cleanup is reused unchanged. No signal/process launch occurs on import.
"""
import json
from pathlib import Path
import time

from driver_support import run_owned
from resource_watch_v4 import external_test_processes


def external_test_scan():
    return external_test_processes()


def wait_external(deadline, stop, note, *, scan=external_test_scan):
    idle = 0
    while idle < 2:
        if time.time() >= deadline or Path(stop).exists():
            raise TimeoutError('Finite deadline or own stop while awaiting external test')
        try:
            state = scan()
            if (not isinstance(state, dict) or not isinstance(state.get('processes'), list)
                    or not isinstance(state.get('errors'), list)):
                raise ValueError('Malformed independent process scan')
        except Exception as exc:
            state = dict(processes=[], errors=[repr(exc)])
        note(state)
        idle = idle+1 if not state['processes'] and not state['errors'] else 0
        if idle < 2:
            time.sleep(min(30., max(0., deadline-time.time())))
