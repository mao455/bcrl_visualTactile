"""Inactive CPU waiter: close exact suffix, aggregate once, run reviewed main.

Root owns activation. Importing this module launches nothing and changes nothing.
No interrupted opportunity is retried. A launched failure is terminal.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
OUT = HERE/'wait_main_recovery05'
MAIN_SHA = '3edbb158e0845cc4f51bfee0e27e1552c79fc8455613d8e610a40036a5e9002f'
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.collection_recovery05'
ORCHESTRATION = ('wait_then_main_recovery05.py','run_main_recovery05.py',
    'run_collection_recovery05.py','collection_recovery05.py','collection_recovery05_aggregate.py',
    'collection_recovery05_scope.py','parallel_rear_proposal.py','parallel_rear_registration.json',
    'proposed_main_train_parallel.json','coexistence_watch_recovery05.py',
    'resource_watch_parallel_proposal.py','resource_watch_v4.py','driver_support_v4.py','driver_support.py',
    'audit_collection_recovery05.py','audit_collection_integrity.py','audit_collection_noise.py',
    'presealed_guard.py','final_report_assembler.py')


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def original_command(root=ROOT):
    return [sys.executable,'-m',MODULE,'project','dagger','--job',str(Path(root)/'collection_recovery05_job.json')]


def aggregate_command(root=ROOT):
    root = Path(root)
    return [sys.executable,str(HERE/'collection_recovery05_aggregate.py'),
        '--original',str(root/'collection'),'--recovered',str(root/'collection_consumed_recovered05'),
        '--continued',str(root/'collection_recovery05'),'--output',str(root/'collection_aggregate_recovery05'),
        '--publish-index']


def closure(root=ROOT):
    """Actual driver and owner terminal receipts, not inferred worker progress."""
    root = Path(root); driver = HERE/'driver_collection_recovery05'; owner = root/'owners/collection_recovery05'
    registration = read(driver/'registration.json')
    require(registration['command'] == original_command(root)
            and registration['deadline'] == read(root/'session.json')['deadline']
            and registration['driver_sha256'] == digest(HERE/'run_collection_recovery05.py'),
            'Exact frozen suffix driver registration differs')
    require(registration.get('scope') == '81 previously unassigned opportunities; no retry', 'Suffix scope changed')
    failure = driver/'failure.json'
    require(not failure.exists(), 'Suffix driver failed; do not aggregate or launch main')
    if not (driver/'exit.json').exists():
        return None
    exit = read(driver/'exit.json')
    require(exit.get('returncode') == 0, 'Suffix driver exited nonzero; no retry')
    if not (owner/'project_exit.json').exists():
        return None
    closed = read(owner/'project_exit.json')
    require(closed.get('children_reaped') is True, 'Suffix owner has not reaped all children')
    require(not (owner/'project_failure.json').exists()
            and read(owner/'project_complete.json').get('stages') == 1, 'Suffix owner has no exact successful one-stage completion')
    status = read(driver/'status.json')
    if status.get('stage') == 'collection_running':
        return None  # exit.json precedes the driver's final status by one atomic write.
    require(status.get('stage') == 'collection_suffix_closed', 'Suffix driver terminal status differs')
    job = read(root/'collection_recovery05_job.json')
    started = read(owner/'project_job_started.json')
    require(started['job'] == job and job['resource_root'] == str(owner)
            and job['operation'] == 'dagger'
            and job['stages'] == [dict(entry='paired_collection_recovery05',args=['--output',str(root/'collection_recovery05')])],
            'Closed suffix owner is not the exact registered81 job')
    worker = read(owner/'project_status.json')
    expected = [sys.executable,'-m',MODULE,'worker','--output',str(root/'collection_recovery05')]
    require(started['commands'] == [expected] and worker.get('command') == expected and worker.get('stage') == 0,
            'Closed suffix worker command differs')
    require(started['utc'] <= closed['utc'] <= exit['utc'] <= status['utc'], 'Suffix terminal receipt times differ')
    return dict(driver_exit=exit,owner_exit=closed,driver_status=status,owner_started=started)


def index_state(root=ROOT):
    root = Path(root); original=root/'collection'; aggregate=root/'collection_aggregate_recovery05'
    names = ('results.json','correction_entries.json','collection_summary.json',
             'recovery05_aggregate_binding.json','recovery05_active_aggregate.json')
    existing = [name for name in names if (original/name).exists() or (original/name).is_symlink()]
    if not aggregate.exists() and not aggregate.is_symlink() and not existing:
        return 'absent'
    require(aggregate.is_dir() and not aggregate.is_symlink() and set(existing) == set(names),
            'Partial/redirected aggregate or index exists; preserve it without overwrite')
    return 'published'


def validate_index(root=ROOT):
    from collection_recovery05_scope import validated_summary
    root=Path(root);original=root/'collection';aggregate=root/'collection_aggregate_recovery05'
    require(index_state(root) == 'published', 'Complete aggregate/index required')
    pointer=read(original/'recovery05_active_aggregate.json');binding=read(aggregate/'recovery05_aggregate_binding.json')
    require(pointer == dict(path=str(aggregate),binding_sha256=digest(aggregate/'recovery05_aggregate_binding.json'),
        indexed_files=['results.json','correction_entries.json','collection_summary.json']), 'Exact aggregate index pointer differs')
    expected = [original,root/'collection_consumed_recovered05',root/'collection_recovery05']
    require(binding['source_directories'] == list(map(str,expected))
            and binding.get('no_initiated_opportunity_reexecuted') is True, 'Aggregate original15/continued81 source binding differs')
    require(binding['original_plan_sha256'] == digest(original/'plan.json')
        and binding['recovered_results_sha256'] == digest(expected[1]/'recovered_results.json')
        and binding['continued_results_sha256'] == digest(expected[2]/'results.json'), 'Aggregate source outcome/plan changed')
    require((original/'recovery05_aggregate_binding.json').read_bytes() == (aggregate/'recovery05_aggregate_binding.json').read_bytes(),
            'Published binding differs')
    summary=validated_summary(original)
    require(summary['opportunities']==96 and summary['observable_pair_checks'] ==
        dict(passed=47,unverified=1,failed=0,all_verified=False), 'Truthful96 and47/1 denominator required')
    assignments=read(aggregate/'paired_queue_assignments.json')
    require([a['attempt'] for a in assignments] == ['original']*15+['continued']*81,
            'Aggregate consumed15/continued81 assignments differ')
    return summary


def frozen_identity(root=ROOT):
    from tools.tight_paired_dagger import sources
    root=Path(root)
    require(digest(HERE/'run_main_recovery05.py') == MAIN_SHA, 'Reviewed main driver SHA changed')
    production=sources()
    require(production == read(root/'common180_parents/training_source_identity.json')['sources'],
            'Production differs from actual matched-parent training')
    paths=[HERE/name for name in ORCHESTRATION]
    paths += [root/name for name in ('session.json','parent_pairs.json','actual_parent_ages.json',
        'collection/plan.json','collection/registration.json','collection_recovery05_job.json',
        'collection_recovery05/plan.json','collection_recovery05/registration.json',
        'collection_recovery05/recovery05_binding.json')]
    return dict(production_sha256=production,orchestration_and_inputs_sha256={str(p):digest(p) for p in paths})


def verify_identity(frozen):
    from tools.tight_paired_dagger import sources
    require(sources()==frozen['production_sha256'], 'Production source changed during finite wait')
    for path,value in frozen['orchestration_and_inputs_sha256'].items():
        require(digest(path)==value, 'Frozen orchestration/input changed: '+path)


class Waiter:
    def __init__(self):
        from visuotactile.training_resources import atomic_json
        self.write=atomic_json;self.deadline=read(ROOT/'session.json')['deadline'];self.stop=ROOT/'STOP_CYCLE'

    def finite(self):
        require(time.time()<self.deadline and not self.stop.exists(), 'Original session deadline/own STOP reached')

    def note(self,stage,**extra):
        self.write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.time(),**extra))

    def run(self,name,command):
        from driver_support_v4 import run_owned
        self.finite();self.note(name,command=command)
        with (OUT/(name+'.log')).open('x') as stream:
            self.write(OUT/(name+'_started.json'),dict(command=command,utc=time.time(),
                owned_process_group_helper='frozen driver_support.run_owned',
                scope='Starts one child in a new session; deadline/STOP cleanup signals its group and captured descendants only'))
            try:
                rc=run_owned(command,stream,self.deadline,self.stop)
            except BaseException as exc:
                self.write(OUT/(name+'_exception.json'),dict(utc=time.time(),error=repr(exc),
                    scope='Frozen owned helper raised after its finally cleanup; no replacement attempt is launched'))
                raise
        self.write(OUT/(name+'_exit.json'),dict(returncode=rc,utc=time.time(),owned_helper_returned_after_cleanup=True))
        require(rc==0,name+' failed; no retry or replacement is launched')

    def main(self):
        self.finite();require(not OUT.exists(), 'Preserve previous waiter attempt')
        frozen=frozen_identity();OUT.mkdir(exist_ok=False)
        self.write(OUT/'registration.json',dict(utc=time.time(),deadline=self.deadline,poll_seconds=30,
            frozen=frozen,aggregate_command=aggregate_command(),main_command=[sys.executable,str(HERE/'run_main_recovery05.py')],
            scope='CPU-only wait for existing81 suffix; aggregate once, then exact reviewed main; no physical retry'))
        while True:
            self.finite();closed=closure()
            if closed is not None:break
            self.note('waiting_exact_collection_suffix_closure');time.sleep(30)
        verify_identity(frozen);self.write(OUT/'observed_collection_closure.json',closed)
        if index_state() == 'absent':
            self.run('aggregate_and_publish_once',aggregate_command())
        summary=validate_index();verify_identity(frozen)
        self.write(OUT/'validated_aggregate.json',dict(utc=time.time(),summary=summary,overwrite_attempts=0))
        self.run('reviewed_main',[sys.executable,str(HERE/'run_main_recovery05.py')])
        self.note('reviewed_main_closed',report_required=True,collection_retries=0)


def main():
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def interrupted(signum,frame):raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    waiter=Waiter()
    try:waiter.main()
    except BaseException as exc:
        if OUT.exists():
            waiter.write(OUT/'failure.json',dict(utc=time.time(),error=repr(exc),traceback=traceback.format_exc()))
            waiter.note('stopped_with_evidence',error=repr(exc))
        raise


if __name__=='__main__':main()
