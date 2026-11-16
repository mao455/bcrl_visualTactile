"""Bounded immutable-file read-ahead into OS cache, zero new feature files.
Read only current/next B2 files, in 16MiB disposable buffers. Does not load a
network, change sampling or expose observations to policy ahead of their time.
"""
import json,time,signal,os,hashlib
from pathlib import Path
R=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_budget_20261116_01')
read=lambda p:json.loads(p.read_text())
stop=False
signal.signal(signal.SIGTERM,lambda *a:globals().__setitem__('stop',True))
deadline=read(R/'session.json')['deadline'];completed=set();previous=[];total=0;began=time.time()
registration=dict(utc=began,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 profile='current_plus_next_B2_file_read_ahead_v1',buffer_bytes=16*2**20,maximum_unique_paths_per_cursor=4,
 scope='Only kernel file cache warming; no modified data, training RNG, tensors, models or algorithm')
(R/'read_ahead_registration.json').write_text(json.dumps(registration,indent=2)+'\n')
while not stop and time.time()<deadline-300:
 if (R/'dp_complete.json').exists():break
 p=R/'aligned_io_state.json'
 if not p.exists():time.sleep(1);continue
 state=read(p)
 if time.time()-state['utc']>10:break
 for role,pid in state['pids'].items():
  try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  except FileNotFoundError:stop=True;break
  assert b'tools.tight_fresh_train' in cmd and str(R).encode() in cmd
 if stop:break
 update=min(state['updates'].values());epoch=update//196+1;batch=update%196;cursor=(epoch,batch)
 if cursor in completed:time.sleep(.2);continue
 deck=R/'A/training'/f'deck_{epoch:03}.json'
 if not deck.exists():time.sleep(.2);continue
 order=read(deck)['order'];ids=order[batch*2:(batch+2)*2];paths=[]
 for i in ids:
  path=(R/'A/state_data/perception'/(i+'.pt')).resolve()
  if path not in paths:paths.append(path)
 started=time.time();bytes_read=0
 for path in paths:
  if path in previous:continue
  with path.open('rb',buffering=0) as f:
   while not stop:
    chunk=f.read(16*2**20)
    if not chunk:break
    bytes_read+=len(chunk)
   del chunk
 previous=paths;completed.add(cursor);total+=bytes_read
 row=dict(utc=time.time(),epoch=epoch,batch=batch,current_updates=state['updates'],files=len(paths),bytes_read=bytes_read,
  seconds=time.time()-started,total_bytes=total,elapsed=time.time()-began)
 with (R/'read_ahead_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 tmp=R/'read_ahead_state.partial.json';tmp.write_text(json.dumps(row));tmp.replace(R/'read_ahead_state.json')
(R/'read_ahead_exit.json').write_text(json.dumps(dict(utc=time.time(),total_bytes=total,elapsed=time.time()-began))+'\n')
