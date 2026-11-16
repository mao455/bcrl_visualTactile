"""Bounded scheduling only: align identical B2 decks to share immutable OS pages.
No model/data/sampler changes. Signals only the two explicitly bound owned workers;
resume both in finally, abort on identity/deck mismatch or excessive pause.
"""
import json,os,signal,time
from pathlib import Path
R=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_budget_20261116_01')
read=lambda p:json.loads(p.read_text())
def interrupted(*args):raise KeyboardInterrupt('Bounded scheduling stopped')
signal.signal(signal.SIGTERM,interrupted)
pids={r:read(R/'logs'/('dp_'+r)/'state.json')['pid'] for r in 'AB'}
paused=set();events=R/'aligned_io_events.jsonl';started=time.time();deadline=read(R/'session.json')['deadline'];pause_since={}
def valid(role):
 try:
  cmd=Path(f'/proc/{pids[role]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  return 'tools.tight_fresh_train train' in cmd and '--root '+str(R)+' ' in cmd and '--role '+role+' ' in cmd
 except FileNotFoundError:return False
def row(role):
 with (R/role/'training/learning.jsonl').open('rb') as f:
  f.seek(0,2);f.seek(max(0,f.tell()-4*1024**2));raw=f.read();lines=raw.splitlines()
  if not raw.endswith(b'\n'):lines=lines[:-1]
  for line in reversed(lines):
   try:return json.loads(line)
   except json.JSONDecodeError:continue
  raise ValueError('No completed update in bounded tail')
def event(role,action,**kw):
 with events.open('a') as f:f.write(json.dumps(dict(utc=time.time(),role=role,pid=pids[role],action=action,**kw))+'\n')
def resume(role):
 if role in paused:
  if valid(role):os.kill(pids[role],signal.SIGCONT)
  event(role,'resume',paused_s=time.time()-pause_since[role]);paused.remove(role)
try:
 for role in 'AB':assert valid(role),'Owned worker identity changed'
 while time.time()<deadline-300:
  if any(not valid(r) or (R/r/'training/finished.json').exists() for r in 'AB'):break
  rows={r:row(r) for r in 'AB'}
  # Same stored sampling stream is required, including independent role contracts.
  if rows['A']['epoch']==rows['B']['epoch']:
   assert rows['A']['deck_sha256']==rows['B']['deck_sha256'],'Cannot align different decks'
  for role,other in [('A','B'),('B','A')]:
   lag=rows[role]['updates']-rows[other]['updates']
   if role in paused:
    if lag<=2:resume(role)
    elif time.time()-pause_since[role]>600:raise TimeoutError('Alignment pause exceeded ten minutes')
   elif lag>6 and not paused:
    assert valid(role);os.kill(pids[role],signal.SIGSTOP);paused.add(role);pause_since[role]=time.time();event(role,'pause',lag_updates=lag)
  state=dict(utc=time.time(),pids=pids,paused=sorted(paused),updates={r:rows[r]['updates'] for r in 'AB'},scope='Only scheduling; fixed models, masks, optimizer, EMA and RNG')
  tmp=R/'aligned_io_state.partial.json';tmp.write_text(json.dumps(state));tmp.replace(R/'aligned_io_state.json')
  time.sleep(1)
finally:
 for role in list(paused):resume(role)
 with events.open('a') as f:f.write(json.dumps(dict(utc=time.time(),action='exit',elapsed=time.time()-started))+'\n')
