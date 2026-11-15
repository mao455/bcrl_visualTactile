"""Bounded actual capture followed by registered scale probes; no hidden retries."""
import json,subprocess,time
from pathlib import Path
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_04')
RENDER=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_render_20261115')
SCALE=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_throughput2_20261115')
LEDGER=BASE/'throughput_render_scale15_supervisor';LEDGER.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
def write(name,value):
 p=LEDGER/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
deadline=min(read(BASE/'session.json')['deadline'],time.time()+2400);rows=[]
write('started.json',dict(started=time.time(),deadline=deadline,render_commit='478b606',scale_commit='1942593'))
try:
 while not (BASE/'throughput_queue15_03/owner_finished.json').exists():
  if time.time()>=deadline:raise TimeoutError('Waiting for previous queue admission')
  time.sleep(5)
 if not read(BASE/'throughput_queue15_03/owner_finished.json')['software_complete']:raise RuntimeError('Queue incomplete; inspect failure before another run')
 for name,tree,owner,budget in [('throughput_render15_01',RENDER,'tight_batch_owner',900),('throughput_tiled15_01',SCALE,'tight_batch_probe_owner',600),('throughput_standard32_15_01',SCALE,'tight_batch_probe_owner',600)]:
  if (BASE/'STOP_MONITOR').exists():raise RuntimeError('Original monitor closed; preserve unstarted attempts')
  cmd=[str(tree/'.venv_isaacsim/bin/python'),'-m','tools.'+owner,'--output',str(BASE/name),'--worker-tree',str(tree),'--startup-seconds','300']
  if owner=='tight_batch_probe_owner':cmd+=['--wall-seconds',str(budget)]
  write('status.json',dict(stage=name,command=cmd,utc=time.time()))
  with (LEDGER/(name+'.log')).open('x') as log:
   child=subprocess.run(cmd,cwd=tree,stdout=log,stderr=subprocess.STDOUT,timeout=min(budget+45,max(1,deadline-time.time())))
  row=dict(name=name,returncode=child.returncode,receipt=read(BASE/name/'owner_finished.json'));rows.append(row);write('stages.json',rows)
  if owner=='tight_batch_owner' and not row['receipt']['software_complete']:raise RuntimeError('Actual render collector failed; stop and inspect')
 write('finished.json',dict(complete=True,ended=time.time(),stages=rows))
except BaseException as error:
 write('finished.json',dict(complete=False,error=repr(error),ended=time.time(),stages=rows));raise
