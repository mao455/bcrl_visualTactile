"""Current/next B2 immutable file pre-read for one active GPU trainer only."""
import json,time,signal,os,hashlib
from pathlib import Path
R=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_budget_20261116_01')
read=lambda p:json.loads(p.read_text());stop=False
signal.signal(signal.SIGTERM,lambda *a:globals().__setitem__('stop',True))
started=time.time();deadline=read(R/'session.json')['deadline'];done=set();previous=[];total=0;buffer=bytearray(16*2**20)
reg=dict(utc=started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),read_buffer_bytes=len(buffer),maximum_current_and_next_files=4)
(R/'serial_read_ahead_registration.json').write_text(json.dumps(reg,indent=2)+'\n')
def latest(path):
 with path.open('rb') as f:
  f.seek(0,2);f.seek(max(0,f.tell()-4*2**20));raw=f.read()
 lines=raw.splitlines()
 if not raw.endswith(b'\n'):lines=lines[:-1]
 for line in reversed(lines):
  try:return json.loads(line)
  except json.JSONDecodeError:pass
 return None
try:
 while not stop and time.time()<deadline-180 and not (R/'dp_complete.json').exists():
  active=[]
  for role in 'AB':
   p=R/'logs'/('dp_'+role)/'state.json'
   if not p.exists():continue
   s=read(p)
   try:cmd=Path(f"/proc/{s['pid']}/cmdline").read_bytes()
   except FileNotFoundError:continue
   if b'tools.tight_fresh_train' in cmd and str(R).encode() in cmd:active.append(role)
  if not active:time.sleep(.3);continue
  if len(active)>1:raise RuntimeError('Serial read-ahead refuses overlapping DP jobs')
  role=active[0];row=latest(R/role/'training/learning.jsonl')
  if row is None:time.sleep(.2);continue
  update=row['updates'];epoch=update//196+1;batch=update%196;cursor=(role,epoch,batch)
  if cursor in done:time.sleep(.2);continue
  deck=R/role/'training'/f'deck_{epoch:03}.json'
  if not deck.exists():time.sleep(.2);continue
  ids=read(deck)['order'][batch*2:(batch+2)*2];paths=[]
  for i in ids:
   p=(R/role/'state_data/perception'/(i+'.pt')).resolve()
   if p not in paths:paths.append(p)
  began=time.time();amount=0
  for p in paths:
   if p in previous:continue
   with p.open('rb',buffering=0) as f:
    while not stop:
     n=f.readinto(buffer)
     if not n:break
     amount+=n
  previous=paths;done.add(cursor);total+=amount
  v=dict(utc=time.time(),role=role,epoch=epoch,batch=batch,bytes_read=amount,seconds=time.time()-began,total_bytes=total,elapsed=time.time()-started)
  with (R/'serial_read_ahead_events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  p=R/'serial_read_ahead_state.partial.json';p.write_text(json.dumps(v));p.replace(R/'serial_read_ahead_state.json')
finally:(R/'serial_read_ahead_exit.json').write_text(json.dumps(dict(utc=time.time(),total_bytes=total,elapsed=time.time()-started))+'\n')
