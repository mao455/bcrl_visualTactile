"""Finalize an explicitly requested legacy boundary pause, without killing Kit."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time

from tools.runtime import read,write,sha,proc,gpu_pids


def finalize_pause(root):
    request=read(root/'PAUSE_AFTER_CURRENT.json');archive=root/'maintenance/priority_change_20261105'
    assert proc(request['owner']['pid']) is None,'Old owner still active'
    assert proc(read(root/'monitor.json')['pid']) is None,'Old resource monitor still active'
    attempts=read(root/'attempts.json')
    assert attempts[-1]['name']==request['current_stage']
    for a in attempts:
        assert a.get('valid') and a.get('resources_released') and 'exit_code' in a
        assert not a.get('remaining_owned_pids') and not a.get('remaining_owned_gpu_pids')
        assert read(root/'stages'/a['name']/'committed.json')==a
    assert not {a['pid'] for a in attempts}&gpu_pids(),'Owned GPU PID remains'
    barrier=Path(request['barrier']);assert barrier.is_dir() and not list(barrier.iterdir())
    error=read(root/'owner_error.json')['error']
    assert error.splitlines()[-1].startswith('FileExistsError:') and str(barrier) in error
    for name in ('owner.lock','gpu.lock'):
        with (root/name).open('r') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    writers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            for fd in (p/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target.startswith(str(root)):
                        info=dict(x.split(':',1) for x in (p/'fdinfo'/fd.name).read_text().splitlines() if ':' in x)
                        if int(info['flags'].strip(),8)&3:writers.append(dict(pid=int(p.name),path=target))
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    assert not writers,writers
    checkpoint=read(root/'checkpoint.json');assert sha(root/'checkpoint.pt')==checkpoint['sha256']
    for name in ('status.json','owner_exit.json','owner_error.json','summary.json'):
        shutil.copyfile(root/name,archive/('boundary_'+name))
    (root/'owner_error.json').rename(archive/'intentional_boundary_error.json')
    receipt=dict(state='paused',evaluation_class='early_stopped_exploratory',planned_valid=200,
        valid=read(root/'summary.json')['valid'],reason='User prioritized frozen expert slowdown and visual-tactile research',
        pause_requested=request['requested'],finished=time.time(),next_stage=request['next_stage'],
        current_stage_committed=request['current_stage'],resources_released=True,owned_gpu_pids=[],open_writers=[],
        checkpoint_sha256=checkpoint['sha256'],attempts_sha256=sha(root/'attempts.json'),
        sampling_manifest_sha256=sha(root/'sampling_manifest.json'),session_sha256=sha(root/'session.json'),
        resume='Explicit request only: remove the recorded empty barrier directory, preserve committed stages/checkpoint/sampling, review remaining time budget, resume original owner. No automatic restart.',
        barrier=str(barrier),raw_boundary_error=str(archive/'intentional_boundary_error.json'))
    write(root/'pause_receipt.json',receipt)
    state=read(root/'status.json');state.update(state='paused',pid=None,worker_pid=None,finished=receipt['finished'],
        evaluation_class='early_stopped_exploratory',pause_receipt=str(root/'pause_receipt.json'))
    write(root/'status.json',state);write(root/'owner_exit.json',state)
    return receipt


def timing_audit(root):
    """Old aggregate timings use saved clocks, never invent unavailable phases."""
    attempts=[a for a in read(root/'attempts.json') if 'finished' in a]
    latest={}
    with (root/'work.jsonl').open() as f:
        import json
        for line in f:
            row=json.loads(line);latest[row['pid']]=row
    rows=[]
    for a in attempts:
        out=Path(a['output']);r=dict(stage=a['name'],wall_s=a['finished']-a['started'],
            formal=a['name'].startswith('random_'),valid=a.get('native_outcome_valid',False),
            initialization_rejected=a.get('initialization_rejected',False))
        def utc(name):return read(out/name)['utc'] if (out/name).exists() else None
        start=utc('runtime_before.json');ready=utc('app_ready.json');exit_time=utc('process_exit_requested.json')
        trial=a.get('result',{}).get('trials',[{}])[0];folder=out/trial.get('identity','missing')
        if start is not None and ready is not None:r['app_startup_including_admission_pacing_s']=ready-start
        if ready is not None and (folder/'started.json').exists():r['scene_and_expert_setup_including_wait_s']=read(folder/'started.json')['utc']-ready
        if exit_time is not None:r['exit_reaping_s']=a['finished']-exit_time
        if (folder/'inference_timing.json').exists():r['dp_inference_host_wall_s']=sum(x['wall_s'] for x in read(folder/'inference_timing.json')['rows'])
        w=latest.get(a['pid'])
        if w:r.update(resource_admission_wait_s=w['total_wait_s'],resource_pacing_s=w['total_pacing_s'],measured_lease_work_s=w['total_work_s'])
        rows.append(r)
    fields=sorted({k for r in rows for k,v in r.items() if k.endswith('_s')})
    stats={k:dict(n=len(values),mean=sum(values)/len(values),total=sum(values),maximum=max(values))
        for k in fields if (values:=[r[k] for r in rows if k in r])}
    return dict(rows=rows,statistics=stats,wall_elapsed_s=read(root/'status.json')['finished']-read(root/'session.json')['started_epoch'],
        limitation='Old logs do not separately time physics/render, RL shadow, numeric recording or video encoding. Startup/setup include lease waits; components overlap and must not be added to resource wait totals. New workers record non-overlapping host categories.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:print(finalize_pause(a.root))
    write(a.root/'timing_audit.json',timing_audit(a.root))
