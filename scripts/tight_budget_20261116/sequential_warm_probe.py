import os,signal,json,time,threading
from pathlib import Path
R=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_budget_20261116_01')
assert not (R/'sequential_warm_probe.json').exists(),'Completed one-off probe; do not repeat'
def terminate(*args):raise SystemExit('Requested stop; resume owned PIDs in finally')
signal.signal(signal.SIGTERM,terminate)
read=lambda p:json.loads(p.read_text())
pid=read(R/'logs/dp_B/state.json')['pid'];reader=725999
assert b'tools.tight_fresh_train' in Path(f'/proc/{pid}/cmdline').read_bytes()
assert str(R).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'read_ahead_serial.py' in Path(f'/proc/{reader}/cmdline').read_bytes()
paused=[];lock=threading.Lock();released=threading.Event()
def resume():
 with lock:
  for p in paused:
   try:os.kill(p,signal.SIGCONT)
   except ProcessLookupError:pass
  paused.clear();released.set()
watch=threading.Timer(30,resume);watch.daemon=True
start=time.time();rows=[]
try:
 with lock:
  for p in [pid,reader]:os.kill(p,signal.SIGSTOP);paused.append(p)
 watch.start()
 with (R/'B/training/learning.jsonl').open('rb') as f:
  f.seek(0,2);f.seek(max(0,f.tell()-4*2**20));raw=f.read()
 lines=raw.splitlines();lines=lines if raw.endswith(b'\n') else lines[:-1]
 current=json.loads(lines[-1]);u=current['updates'];epoch=u//196+1;batch=u%196
 deck=read(R/'B/training'/f'deck_{epoch:03}.json')['order'][batch*2:(batch+1)*2]
 buf=bytearray(16*2**20)
 for identity in dict.fromkeys(deck):
  p=(R/'B/state_data/perception'/(identity+'.pt')).resolve();began=time.time();nbytes=0
  with p.open('rb',buffering=0) as f:
   while True:
    n=f.readinto(buf)
    if not n:break
    nbytes+=n
  rows.append(dict(identity=identity,bytes=nbytes,seconds=time.time()-began))
finally:
 resume();watch.cancel()
result=dict(utc=time.time(),pid=pid,reader=reader,epoch=epoch,batch=batch,update=u,rows=rows,seconds=time.time()-start,maximum_pause_seconds=30,scope='Single warm of actual current B2 while ONLY owned trainer/reader temporarily paused, no file or optimizer mutation; cached and cold bytes mixed.')
p=R/'sequential_warm_probe.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
