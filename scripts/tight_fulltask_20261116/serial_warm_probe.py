"""One bounded actual B2 read, with only owned reader/trainer paused temporarily."""
import os,signal,time,threading
from pathlib import Path
from scripts.tight_fulltask_20261116.run import ROOT,read,atomic_json
pid=read(ROOT/'frontend_B_worker.json')['pid'];reader=read(ROOT/'read_warm_progress.json')['pid']
assert b'tools.tight_fresh_train' in Path('/proc',str(pid),'cmdline').read_bytes()
assert b'warm_current' in Path('/proc',str(reader),'cmdline').read_bytes()
os.kill(reader,signal.SIGTERM)
paused=False;lock=threading.Lock()
def resume():
    global paused
    with lock:
        if paused:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
            paused=False
signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(SystemExit('stop')))
watch=threading.Timer(25,resume);watch.daemon=True;start=time.time();rows=[]
try:
    with lock:os.kill(pid,signal.SIGSTOP);paused=True
    watch.start();base=ROOT/'update/B';row=read(base/'training/progress.json');deck=read(base/'training'/f"deck_{row['epoch']:03d}.json")['order']
    buf=bytearray(16*2**20)
    for identity in deck[row['batch_index']*2:(row['batch_index']+1)*2]:
        p=(base/'state_data/perception'/(identity+'.pt')).resolve();began=time.time();total=0
        with p.open('rb',buffering=0) as f:
            while True:
                n=f.readinto(buf)
                if not n:break
                total+=n
        rows.append(dict(identity=identity,bytes=total,seconds=time.time()-began))
finally:resume();watch.cancel()
atomic_json(ROOT/'serial_warm_probe.json',dict(utc=time.time(),seconds=time.time()-start,rows=rows,after_update=row['updates'],trainer_pid=pid,
    scope='One temporary pause of our trainer only; no weights/order/history changed; cached and uncached bytes mixed; no throughput claim from this alone'))
print(rows,flush=True)
