"""INACTIVE exact continuation15 owner/worker identity and actual-capacity watcher.

The original13 global stop marker remains untouched. This watcher has its own
STOP_RESOURCE_WATCH15 and lock, and runs only after actual12 closure/review.
All telemetry comes from the native fragment owner monitor, without rewriting.
"""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import signal
import time

from outputs.diagnostics.tight_paired_dagger_20261122_01 import shared_capacity06 as base
from outputs.diagnostics.tight_paired_dagger_20261122_01 import run_test_continuation15 as fragment

MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.shared_capacity15'
AUDIT=Path(__file__).resolve().parent/'watch_shared_capacity15'
watch=base.watch
inspect=base.inspect


def exact_spec(root):
    return dict(name='test_continuation15',owner=Path(root).resolve()/'owners/test_continuation15',
                module=fragment.MODULE,operation='evaluate',job=fragment.JOB)


def specs(root,config):
    return [*base.specs(root,config),exact_spec(root)]


def registered_worker(pid,registrations,root,lookup=watch.process):
    proc=lookup(pid)
    if proc:
        for reg in registrations:
            spec=reg['spec']
            if spec['name']!='test_continuation15':
                continue
            try:
                root=Path(root).resolve(); owner=root/'owners/test_continuation15'; test=root/'evaluation/test'
                if spec!=exact_spec(root) or owner.is_symlink() or test.is_symlink():
                    continue
                binding=watch.read(fragment.REGISTRY)
                review=watch.read(fragment.REVIEW)
                if (review.get('status')!='source_and_actual_missing47_review_passed_inactive'
                        or review.get('registration_sha256')!=fragment.digest(fragment.REGISTRY)
                        or binding['job_sha256']!=fragment.digest(fragment.JOB)
                        or binding['original_test_registration_sha256']!=fragment.digest(test/'registration.json')
                        or reg['receipt'].get('job')!=reg['job']):
                    continue
                status=watch.read(owner/'project_status.json'); stage=status['stage']
                if type(stage) is not int or not 0<=stage<len(binding['eligible_indices']):
                    continue
                index=binding['eligible_indices'][stage]; output=test/f'wave{index:02d}'
                expected_stage=dict(entry='paired_evaluate_continuation15',args=['worker','--output',str(output)])
                if reg['job']['stages'][stage]!=expected_stage:
                    continue
                expected=['python','-m',fragment.MODULE,'worker','--output',str(output)]
                if (status.get('pid')!=pid or not watch.born_before_receipt(proc,status.get('utc'))
                        or not watch.descends(proc,reg['owner_process'],lookup)
                        or not watch.matching_command(proc['command'],expected,proc['cwd'])
                        or not watch.matching_command(status['command'],expected,proc['cwd'])
                        or not watch.matching_command(reg['receipt']['commands'][stage],expected,proc['cwd'])):
                    continue
                plan=watch.read(output/'plan.json'); freeze=watch.read(test/'registration.json')
                fence=watch.read(owner/'dispatches'/f'wave{index:02d}.json')
                if (Path(plan['resource_root']).resolve()!=test or plan['pairs']!=binding['frozen_pairs']
                        or fragment.digest(output/'plan.json')!=freeze['wave_plan_sha256'][str(output)]
                        or fence.get('pid')!=pid or fence.get('index')!=index or fence.get('dispatch_attempt')!=1
                        or fence.get('registration_sha256')!=fragment.digest(fragment.REGISTRY)
                        or fence.get('original13_registration_sha256')!=binding['original13_registry_sha256']
                        or fence.get('plan_sha256')!=freeze['wave_plan_sha256'][str(output)]):
                    continue
                return dict(pid=pid,created=proc['created'],name=spec['name'],output=str(output),
                            owner_pid=reg['owner_process']['pid'],owner_created=reg['owner_process']['created'])
            except (OSError,KeyError,IndexError,TypeError,ValueError):
                continue
    return base.identity.registered_worker(pid,registrations,root,lookup)


def live_identity(row,process,now):
    age=now-float(row['utc'])
    if not math.isfinite(age) or not -2<=age<10:
        raise ValueError('Watcher15 heartbeat stale')
    if row.get('schedule_conflicts')!=[]:
        raise ValueError('Watcher15 schedule conflict')
    expected=['-m',MODULE,'--audit',str(AUDIT),'--apply']
    if (process.pid!=row['pid'] or not process.is_running() or process.cmdline()[1:]!=expected
            or Path(process.cwd()).resolve()!=watch.PROJECT or process.create_time()>row['utc']+1):
        raise ValueError('Exact live watcher15 identity required')
    return dict(pid=process.pid,utc=row['utc'])


def startup_row(row,process,now,*,allow_gap=False):
    live_identity(row,process,now)
    if row.get('errors')!=[]:
        raise ValueError('Watcher15 errors')
    idle=(row.get('active_resource_roots')==[] and row.get('raw_monitor_rows')==[]
          and row.get('registered_workers_excluded')==[])
    normal=row.get('paused') is False and row.get('reason')=='shared_capacity_available'
    gap=allow_gap and idle and row.get('paused') is True and row.get('reason')=='no_fresh_owned_telemetry'
    if not(normal or gap):
        raise ValueError('Capacity watcher15 not ready')
    return dict(starting_without_owned_telemetry=bool(gap),external_presence_is_not_veto=True,
                active_resource_roots=row['active_resource_roots'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,default=AUDIT)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.audit.resolve()!=AUDIT:
        raise ValueError('Exact continuation15 watcher audit required')
    fragment.validate()
    # Original watcher marker and closure remain intact and bound. A new marker
    # is intentionally used for this finite writer rather than removing12 STOP.
    stop=fragment.ROOT/'STOP_RESOURCE_WATCH15'
    if stop.exists():
        raise ValueError('Preserve attempted watcher15 stop identity')
    watch.specs=specs; watch.registered_worker=registered_worker; watch.inspect=inspect
    if not args.apply:
        print(json.dumps(inspect(fragment.ROOT,watch.DEFAULT_CONFIG),indent=2)); return
    AUDIT.mkdir(exist_ok=False)
    with (AUDIT/'resource_watch_v4.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        schedule=watch.Schedules(fragment.ROOT,AUDIT/f'resource_watch_v4_backups_{int(time.time())}_{os.getpid()}')
        def interrupt(signum,frame):
            raise KeyboardInterrupt(signum)
        for sig in (signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,interrupt)
        try:
            last=None
            while time.time()<fragment.DEADLINE and not stop.exists():
                state=inspect(fragment.ROOT,watch.DEFAULT_CONFIG)
                with fragment.schedule_lock():
                    schedule.pause() if state['paused'] else schedule.restore()
                    state.update(pid=os.getpid(),modified_schedules=len(schedule.changed),schedule_conflicts=list(schedule.conflicts))
                    watch.atomic_json(AUDIT/'resource_watch_v4_status.json',state)
                signature={k:v for k,v in state.items() if k not in ('utc','raw_monitor_rows')}
                if signature!=last:
                    with (AUDIT/'resource_watch_v4_decisions.jsonl').open('a') as log:
                        log.write(json.dumps(state)+'\n')
                    last=signature
                time.sleep(min(.5,max(0.,fragment.DEADLINE-time.time())))
        finally:
            with fragment.schedule_lock():
                restored=schedule.restore()
            watch.atomic_json(AUDIT/'resource_watch_v4_exit.json',dict(utc=time.time(),pid=os.getpid(),restored=restored,
                conflicts=schedule.conflicts,modified_schedules=len(schedule.changed),no_process_signals=True,
                exact_original_byte_restoration=True,original12_13_14_stop_markers_preserved=True))


if __name__=='__main__':
    main()
