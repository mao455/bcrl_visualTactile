"""Two bounded real loads after the collector comparison; each identity runs once."""
import json,subprocess,time
from pathlib import Path
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_04')
TREE=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_throughput2_20261115')
LEDGER=BASE/'throughput_scale15_supervisor';LEDGER.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
def write(name,value):
 p=LEDGER/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
deadline=min(read(BASE/'session.json')['deadline'],time.time()+1800);rows=[]
write('started.json',dict(started=time.time(),deadline=deadline,source_commit='19425937f15ecdfd8c90c6aed2ea630d9fe9d6cc'))
try:
 while not (BASE/'throughput_queue15_02/owner_finished.json').exists():
  if time.time()>=deadline:raise TimeoutError('Wait for completed queue comparison')
  time.sleep(5)
 if not read(BASE/'throughput_queue15_02/owner_finished.json')['software_complete']:raise RuntimeError('Queue incomplete; no scaling')
 for name in ['throughput_tiled15_01','throughput_standard32_15_01']:
  if (BASE/'STOP_MONITOR').exists():raise RuntimeError('Original monitor closed; preserve unstarted probe')
  command=[str(TREE/'.venv_isaacsim/bin/python'),'-m','tools.tight_batch_probe_owner','--output',str(BASE/name),
   '--worker-tree',str(TREE),'--startup-seconds','300','--wall-seconds','600']
  write('status.json',dict(stage=name,command=command,utc=time.time()))
  with (LEDGER/(name+'.log')).open('x') as log:
   child=subprocess.run(command,cwd=TREE,stdout=log,stderr=subprocess.STDOUT,timeout=min(650,max(1,deadline-time.time())))
  rows.append(dict(name=name,returncode=child.returncode,receipt=read(BASE/name/'owner_finished.json')))
  write('stages.json',rows)
 write('finished.json',dict(complete=True,ended=time.time(),stages=rows))
except BaseException as error:
 write('finished.json',dict(complete=False,error=repr(error),ended=time.time(),stages=rows));raise
