"""Opt-in probe startup evidence; no simulator or control changes."""
import faulthandler
import os
from pathlib import Path
import time

from .training_resources import atomic_json


class StartupDiagnostics:
    """Keep a repeating Python-stack timer only while environment startup runs.

    Native frames may remain opaque. Owner timeout/SIGTERM is still authoritative;
    the last durable stage then records where Python entered the blocked operation.
    """
    def __init__(self, root, *, interval_s=120):
        if type(interval_s) not in (int, float) or interval_s <= 0:
            raise ValueError('Positive startup traceback interval required')
        self.root = Path(root)
        self.interval_s = interval_s
        self.events = []
        self.started = time.perf_counter()
        self.stream = None

    def mark(self, stage, **details):
        self.events.append(dict(stage=stage, utc=time.time(), elapsed_s=time.perf_counter()-self.started,
            **details))
        atomic_json(self.root/'environment_initialization.json', dict(
            version='probe_environment_startup_v1', diagnostic_only=True, pid=os.getpid(),
            interval_s=self.interval_s, current_stage=stage, events=self.events,
            traceback_file='startup_tracebacks.log',
            semantics='Python stacks and host stages only; no render, physics step or retry. '
                'A native call can remain opaque. Owner status determines interrupted attempts.'))

    def __enter__(self):
        self.stream = (self.root/'startup_tracebacks.log').open('x')
        try:
            self.mark('startup_diagnostics_started')
            faulthandler.dump_traceback_later(self.interval_s, repeat=True, file=self.stream, exit=False)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, error_type, error, tb):
        faulthandler.cancel_dump_traceback_later()
        try:
            self.mark('environment_ready' if error is None else 'environment_startup_failed',
                error=None if error is None else repr(error))
        finally:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
        return False
