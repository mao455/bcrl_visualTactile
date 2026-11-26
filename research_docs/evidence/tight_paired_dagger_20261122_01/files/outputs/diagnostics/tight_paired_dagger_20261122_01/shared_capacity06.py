"""User-authorized concurrent GPU use; measured free capacity, exact own identity.

External process existence is recorded, never a pause predicate. Production,
task, optimizer and successful-update contracts remain frozen and unchanged.
"""
import math
import subprocess
import time
from pathlib import Path
import psutil
from outputs.diagnostics.tight_paired_dagger_20261122_01 import resource_watch_v4 as watch
from outputs.diagnostics.tight_paired_dagger_20261122_01 import coexistence_watch_recovery05 as identity

HERE=Path(__file__).resolve().parent
ROOT=watch.DEFAULT_ROOT
AUDIT=HERE/'watch_shared_capacity06'
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.shared_capacity06'
ORIGINAL_INSPECT=watch.inspect
GPU_RESERVE_MIB=4096
RAM_RESERVE_BYTES=10*2**30
DISK_RESERVE_BYTES=12*2**30

def specs(root,config):
    rows=identity.specs(root,config)
    for row in rows:
        if row['name']=='main_train':
            row.update(module='outputs.diagnostics.tight_paired_dagger_20261122_01.parallel_rear_shared06',
                job=HERE/'proposed_main_train_shared06.json')
    return rows


def finite_number(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0


def live_capacity():
    """Actual nvidia/proc measurements; no fabricated monitor or CUDA context."""
    import shutil
    started=time.time()
    raw=subprocess.check_output(['nvidia-smi','--id=0',
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
    if len(raw)!=1:raise ValueError('One selected GPU required')
    free,util=map(float,raw[0].split(','))
    return dict(query_started_utc=started,utc=time.time(),gpu_free_mib=free,gpu_util_percent=util,
        ram_available_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(ROOT).free,
        source='independent nvidia-smi/virtual_memory/disk_usage; no owned telemetry claim')


def capacity_decision(row,*,gpu_resident_mib=0,ram_resident_bytes=0,now=None):
    now=time.time() if now is None else now
    for key in ('utc','query_started_utc','gpu_free_mib','ram_available_bytes','disk_free_bytes'):
        if not finite_number(row.get(key)):raise ValueError('Invalid real capacity: '+key)
    if not -2<=now-row['utc']<3 or not 0<=row['utc']-row['query_started_utc']<12:
        raise ValueError('Stale or slow real capacity')
    if not finite_number(gpu_resident_mib) or not finite_number(ram_resident_bytes):
        raise ValueError('Invalid resident budget')
    required=dict(gpu_free_mib=GPU_RESERVE_MIB+gpu_resident_mib,
        ram_available_bytes=RAM_RESERVE_BYTES+ram_resident_bytes,disk_free_bytes=DISK_RESERVE_BYTES)
    insufficient= [key for key,value in required.items() if row[key]<value]
    return dict(ready=not insufficient,insufficient=insufficient,required=required,observed=row,
        external_process_presence_is_not_a_veto=True,
        utilization_is_recorded_not_subtracted_from_memory=True)


def inspect(root,config,now=None,lookup=watch.process,external_scan=watch.external_test_processes):
    row=ORIGINAL_INSPECT(root,config,now,lookup,external_scan)
    # All identity, telemetry and scan errors remain fail-closed. External
    # records remain exact raw observations, including unowned GPU processes.
    errors=list(row['errors'])
    capacity=None
    # Cooperative early margin above production's hard4GiB/10GiB guard.
    try:capacity=capacity_decision(live_capacity(),gpu_resident_mib=1024,ram_resident_bytes=2*2**30)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:errors.append(dict(error='real_capacity_query',detail=repr(exc)))
    if errors:reason='telemetry_or_capacity_error'
    elif not row['raw_monitor_rows']:reason='no_fresh_owned_telemetry'
    elif not capacity['ready']:reason='remaining_capacity_below_reserve'
    else:reason='shared_capacity_available'
    row.update(paused=bool(errors or not row['raw_monitor_rows'] or not capacity or not capacity['ready']),
        reason=reason,errors=errors,capacity=capacity,
        resource_amendment='USER_RESOURCE_STEERING06.json',external_presence_pauses=False)
    return row


def startup_row(row,proc,now,*,allow_gap=False):
    age=now-float(row['utc'])
    if not math.isfinite(age) or not -2<=age<10:raise ValueError('Watcher heartbeat stale')
    if row.get('errors')!=[] or row.get('schedule_conflicts')!=[]:raise ValueError('Watcher errors/conflicts')
    expected=['-m',MODULE,'--audit',str(AUDIT),'--apply']
    if proc.pid!=row['pid'] or not proc.is_running() or proc.cmdline()[1:]!=expected:
        raise ValueError('Exact capacity watcher PID/command required')
    if Path(proc.cwd()).resolve()!=watch.PROJECT or proc.create_time()>row['utc']+1:
        raise ValueError('Capacity watcher cwd/create time differs')
    idle=row.get('active_resource_roots')==[] and row.get('raw_monitor_rows')==[] and row.get('registered_workers_excluded')==[]
    normal=row.get('paused') is False and row.get('reason')=='shared_capacity_available'
    gap=allow_gap and idle and row.get('paused') is True and row.get('reason')=='no_fresh_owned_telemetry'
    if not(normal or gap):raise ValueError('Capacity watcher not ready')
    return dict(starting_without_owned_telemetry=bool(gap),watcher_remains_paused=bool(gap),
        active_resource_roots=row['active_resource_roots'],external_gpu_processes=row['external_gpu_processes'],
        external_test_python_processes=row['external_test_python_processes'],external_presence_is_not_veto=True)


def main():
    watch.specs=specs
    watch.registered_worker=identity.registered_worker
    watch.inspect=inspect
    watch.main()


if __name__=='__main__':main()
