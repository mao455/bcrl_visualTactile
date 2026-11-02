"""One bounded owner: legal freeze -> expert/control -> BC -> small StateDP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .compact import sha256
from .native_v2_repair import RepairRunner
from .training_resources import atomic_json


def initialize(root):
    old = json.loads(Path('outputs/visuotactile/native_dp_v2/repair_20261102_01/session.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    source = Path(old['source_root']); started = time.time()
    specs = []
    for cohort, identities in [('train', old['bc_train_ids']), ('validation', old['bc_validation_ids'])]:
        for identity in identities:
            specs.append(dict(identity=identity, cohort=cohort, seed=982401+len(specs)))
    files = list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks').rglob('*.py'))
    session = dict(protocol_version=4, started_epoch=started, deadline=started+6*3600,
        training_deadline=started+6*3600, source_root=str(source), previous_root=old['previous_root'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256={str(p): sha256(p) for p in files}, saved_specs=specs,
        fresh_candidates=list(range(982501, 982533)), fresh_count=4,
        expert_checkpoint=old['expert_checkpoint'], expert_sha256=old['expert_sha256'],
        old_state_checkpoint=old['old_state_checkpoint'], data_sha256=sha256(source/'data.json'),
        bc_updates=2000, dp_updates=1000, maximum_controlled_trajectories=38,
        expert_minimum_successes=10, saved_expert_minimum_successes=6,
        automatic_expansion=False, automatic_retry=False, test_calls=0,
        admission='new pre-action legal expert>=10/12 and saved>=6/8; same-command replay exact; numeric tests pass',
        initialization='first4 legal from32 fresh candidates plus predeclared train4/val4 observable states; no outcome filtering')
    atomic_json(root/'session.json', session)
    atomic_json(root/'gpu_schedule.json', dict(target_percent=40, admission_percent=45, min_gpu_free_mib=6144, min_ram_gib=14))
    atomic_json(root/'status.json', dict(state='registered', new_updates=0))


class Runner(RepairRunner):
    def run(self):
        try:
            with (self.root/'monitor.log').open('a') as log:
                self.monitor = subprocess.Popen(['bash', 'visuotactile/run_cpu.sh', '-m', 'visuotactile.native_resources',
                    '--root', str(self.root)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.stage('preservation', 'visuotactile.native_v2_preservation', [])
            numeric = self.stage('parity', 'visuotactile.state_history_audit', [])
            self.stage('freeze', 'visuotactile.state_benchmark', ['--mode', 'freeze'])
            expert = self.stage('expert', 'visuotactile.state_benchmark', ['--mode', 'expert'])
            replay = self.stage('replay', 'visuotactile.state_benchmark', ['--mode', 'replay'])
            self.stage('old', 'visuotactile.state_benchmark', ['--mode', 'old'])
            passed = (numeric['passed'] and len(expert['trials']) == 12 and all(r['legal'] for r in expert['trials'])
                and sum(r['success'] for r in expert['trials']) >= self.session['expert_minimum_successes']
                and sum(r['success'] for r in expert['trials'] if r['cohort'] != 'fresh') >= self.session['saved_expert_minimum_successes']
                and len(replay['trials']) == 2 and all(all(r.get('bytewise_fields', {}).values())
                    and bool(r.get('bytewise_fields')) for r in replay['trials']))
            names = ['preservation/result.json', 'parity/result.json', 'freeze/benchmark.json', 'expert/result.json', 'replay/result.json']
            atomic_json(self.root/'foundation.json', dict(passed=passed,
                evidence_sha256={name: sha256(self.root/name) for name in names},
                historical_open_loop_is_training_veto=False, expert_successes=sum(r['success'] for r in expert['trials'])))
            if not passed:
                raise RuntimeError('Frozen legal expert/parity foundation failed; no new training')
            for kind in ('bc', 'dp'):
                self.stage(kind+'_train', 'visuotactile.state_history_training', ['--kind', kind])
                self.stage(kind, 'visuotactile.state_benchmark', ['--mode', kind])
            atomic_json(self.root/'status.json', dict(state='diagnostics_complete', utc=time.time(),
                expansion=False, reason='Review actual State native outcomes before any larger training'))
        except BaseException as error:
            atomic_json(self.root/'blocked.json', dict(reason=repr(error), utc=time.time()))
            atomic_json(self.root/'status.json', dict(state='blocked', reason=repr(error), utc=time.time()))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                import signal
                os.killpg(self.worker.pid, signal.SIGTERM)
                try:
                    self.worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.worker.pid, signal.SIGKILL); self.worker.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch(); self.monitor.wait(timeout=20)
            atomic_json(self.root/'owner_exit.json', dict(utc=time.time(), owned_children_exited=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['initialize', 'run'])
    p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    if a.command == 'initialize':
        initialize(a.root.resolve())
    else:
        import fcntl
        with (a.root/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            Runner(a.root.resolve()).run()
