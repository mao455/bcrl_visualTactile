"""Finite read-only warming of the next registered frozen-input files.

This process consumes no model RNG and changes no file, tensor or optimizer.
Its small read buffers are discarded; the OS controls the shared page cache.
"""
import argparse
import json
import os
from pathlib import Path
import time

import psutil

from tools.tight_adaptive_dagger import ROOT, read, write


def run(output, seconds):
    output.mkdir(exist_ok=False)
    started=time.time();deadline=min(started+seconds,read(ROOT/'session.json')['deadline'])
    write(output/'registration.json',dict(started_utc=started,deadline=deadline,
        method='Sequential read-only warming of actual upcoming registered DP B2 inputs',
        maximum_poll_bytes=2*2**30,read_buffer_bytes=2**21,minimum_available_ram_gib=16,
        future_batches=2,model_rng_draws=0,optimizer_updates=0,source_changes=False,
        limitation='OS page cache is advisory and shared; throughput comparisons are observational'))
    seen=set();events=0;total_bytes=0;last=None
    with (output/'reads.jsonl').open('x') as log:
        while time.time()<deadline:
            owner=ROOT/'owners/r1_train'
            if (owner/'project_complete.json').exists() or (owner/'project_failure.json').exists():break
            state=read(owner/'project_status.json')
            command=state.get('command',[])
            if 'tools.tight_fresh_train' not in command or '--role' not in command:break
            role=command[command.index('--role')+1];target=ROOT/f'round01/learning/{role}'
            path=target/'training/progress.json'
            if not path.exists():time.sleep(1);continue
            p=read(path);last={k:p[k] for k in ('epoch','batch_index','updates','utc')}
            if psutil.virtual_memory().available<16*2**30:time.sleep(2);continue
            deck=read(target/f"training/deck_{p['epoch']:03d}.json")['order']
            index=read(target/'cache_index.json')['entries'];remaining=2*2**30
            for position in range(2*p['batch_index'],min(len(deck),2*(p['batch_index']+2))):
                identity=deck[position];key=(role,p['epoch'],position)
                if key in seen:continue
                source=Path(index[identity]['path']);before=source.stat()
                if before.st_size>remaining:continue
                remaining-=before.st_size;count=0;begin=time.monotonic()
                with source.open('rb',buffering=0) as stream:
                    while time.time()<deadline:
                        block=stream.read(2**21)
                        if not block:break
                        count+=len(block)
                        if psutil.virtual_memory().available<16*2**30:break
                after=source.stat()
                assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
                if count==before.st_size:seen.add(key)
                row=dict(role=role,epoch=p['epoch'],upcoming_position=position,observed_update=p['updates'],
                    path=str(source),expected_sha256=index[identity]['sha256'],bytes=count,
                    seconds=time.monotonic()-begin,utc=time.time())
                log.write(json.dumps(row)+'\n');log.flush();events+=1;total_bytes+=count
            time.sleep(.5)
    write(output/'complete.json',dict(started_utc=started,ended_utc=time.time(),events=events,
        total_read_bytes=total_bytes,last_observed_progress=last,model_updates=0,physics_updates=0,
        inputs_modified=False,source_changes=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1200);a=p.parse_args()
    assert 0<a.seconds<=1200
    run(a.output,a.seconds)
