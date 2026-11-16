"""Bounded sequential OS-page warming for an already running owned B2 trainer.

A host-only scheduling workaround for measured mapped read stalls. It does not
change any file, tensor, RNG, gradient or optimizer ordering. Always resumes the
owned PID on ordinary errors/signals; a 25s watchdog bounds each pause.
"""
import os,time,signal,threading,argparse
from pathlib import Path
from scripts.tight_fulltask_20261116.run import ROOT,DEADLINE,read,atomic_json
p=argparse.ArgumentParser();p.add_argument('--batches',type=int,default=12);p.add_argument('--stall-only',action='store_true');a=p.parse_args()
pid=read(ROOT/'frontend_B_worker.json')['pid'];assert b'tools.tight_fresh_train' in Path('/proc',str(pid),'cmdline').read_bytes()
base=ROOT/'update/B';buf=bytearray(16*2**20);paused=False;lock=threading.Lock();rows=[];seen=set();started=time.time()
def resume():
    global paused
    with lock:
        if paused:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
            paused=False
signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(SystemExit('stop')))
try:
    while len(rows)<a.batches and time.time()<DEADLINE-660:
        try:
            if b'tools.tight_fresh_train' not in Path('/proc',str(pid),'cmdline').read_bytes():break
        except FileNotFoundError:break
        row=read(base/'training/progress.json');cursor=(row['epoch'],row['batch_index'])
        if a.stall_only and time.time()-row['utc']<3:time.sleep(.2);continue
        if cursor in seen:time.sleep(.05);continue
        deck=read(base/'training'/f"deck_{row['epoch']:03d}.json")['order'];ids=deck[row['batch_index']*2:(row['batch_index']+1)*2]
        if not ids:seen.add(cursor);time.sleep(.1);continue
        began=time.time();total=0;watch=threading.Timer(25,resume);watch.daemon=True
        try:
            with lock:os.kill(pid,signal.SIGSTOP);paused=True
            watch.start()
            for identity in ids:
                path=(base/'state_data/perception'/(identity+'.pt')).resolve()
                with path.open('rb',buffering=0) as f:
                    while True:
                        n=f.readinto(buf)
                        if not n:break
                        total+=n
        finally:resume();watch.cancel()
        seen.add(cursor);rows.append(dict(after_update=row['updates'],epoch=row['epoch'],bytes=total,seconds=time.time()-began))
        atomic_json(ROOT/'serialized_warm_progress.json',dict(pid=os.getpid(),trainer_pid=pid,utc=time.time(),rows=rows,extra_disk_bytes=0))
finally:
    resume();atomic_json(ROOT/f'serialized_warm_{os.getpid()}.json',dict(utc=time.time(),seconds=time.time()-started,rows=rows,extra_disk_bytes=0))
