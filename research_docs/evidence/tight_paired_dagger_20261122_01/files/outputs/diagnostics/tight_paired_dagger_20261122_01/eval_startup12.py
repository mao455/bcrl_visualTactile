"""Run the original never-started test with startup-only diagnostics.

The original worker, autonomous scope, plans, leases and scientific sources are
called unchanged. There is no simulation retry or replacement in this module.
"""
from pathlib import Path
import faulthandler
import json
import os
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12'
REGISTRY = HERE/'EVALUATION_RECOVERY12_REGISTRATION.json'


def read(path):
    return json.loads(Path(path).read_text())


def validate(output):
    from visuotactile.compact import sha256
    from tools.tight_paired_evaluate import validate_wave
    output = Path(output).resolve()
    registry = read(REGISTRY)
    if output.parent != ROOT/'evaluation/test' or output.name not in {f'wave{i:02d}' for i in range(64)}:
        raise ValueError('Only original sealed64 test wave paths are eligible')
    if registry['original_deadline'] != read(ROOT/'session.json')['deadline'] or time.time() >= registry['original_deadline']:
        raise ValueError('Original finite session expired or changed')
    for path, expected in registry['orchestration_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Reviewed recovery12 source changed: '+path)
    plan, freeze = validate_wave(output)
    if freeze['cohort'] != 'test' or freeze['pairs'] != registry['frozen_pairs']:
        raise ValueError('Original test/four-policy identity differs')
    if freeze['cohorts_sha256'] != registry['cohorts_sha256']:
        raise ValueError('Sealed cohort identity differs')
    return plan, freeze


def instrumented_environment(original, expected_output):
    def environment(root, *args, **kwargs):
        if Path(root).resolve() != expected_output:
            raise ValueError('Startup diagnostic scope escaped exact wave')
        if kwargs.get('startup_diagnostics') not in (None, False, True):
            raise ValueError('Unexpected diagnostic argument')
        kwargs['startup_diagnostics'] = True
        return original(root, *args, **kwargs)
    return environment


def worker(output):
    output = Path(output).resolve()
    validate(output)
    from tools import tight_state
    from tools.tight_paired_evaluate import worker as native_worker
    from visuotactile.training_resources import atomic_json
    # Installed before Kit/environment construction. This distinct file avoids
    # colliding with StartupDiagnostics' exclusive startup_tracebacks.log.
    stream = (output/'boot_signal_tracebacks12.log').open('x')
    faulthandler.register(signal.SIGUSR2, file=stream, all_threads=True, chain=False)
    atomic_json(output/'boot_diagnostics_ready12.json', dict(pid=os.getpid(), utc=time.time(),
        signal='SIGUSR2', diagnostic_only=True, original_worker=True, retries=0,
        environment_diagnostics='existing StartupDiagnostics, interval120s'))
    original = tight_state.environment
    tight_state.environment = instrumented_environment(original, output)
    try:
        native_worker(output)
    finally:
        tight_state.environment = original
        # Native successful Kit termination uses os._exit and legitimately
        # bypasses this finalizer. No successful receipt is manufactured.
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()


def project():
    from tools import tight_project
    # Original frozen job arguments already contain the worker subcommand.
    tight_project.ENTRIES['paired_evaluate'] = ('evaluate', MODULE, [])
    sys.argv.pop(1)
    tight_project.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'project':
        return project()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['worker'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worker(args.output)


if __name__ == '__main__':
    main()
