"""Bounded ordinary supervisor: sequential matched load, no automatic retries."""
import json,subprocess,time
from pathlib import Path
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_04')
TREE=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_throughput_20261115')
OUT=BASE/'throughput_queue15_01';LEDGER=BASE/'throughput_pair15_supervisor'
LEDGER.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
def write(name,value):
 p=LEDGER/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
deadline=min(read(BASE/'session.json')['deadline'],time.time()+1800)
write('started.json',dict(started=time.time(),deadline=deadline,source_commit='b0e7da09febc17db9a45ca4e757c0451056842c3'))
try:
 while not (BASE/'throughput_baseline15_01/owner_finished.json').exists():
  if time.time()>=deadline:raise TimeoutError('Waiting for actual baseline completion')
  write('status.json',dict(stage='waiting_baseline',utc=time.time()));time.sleep(5)
 old=read(BASE/'throughput_baseline15_01/owner_finished.json')
 if not old['software_complete']:raise RuntimeError('Baseline failed; preserve, no automatic candidate launch')
 command=[str(TREE/'.venv_isaacsim/bin/python'),'-m','tools.tight_batch_owner','--output',str(OUT),'--worker-tree',str(TREE),'--startup-seconds','300']
 write('status.json',dict(stage='queue_comparison',command=command,utc=time.time()))
 subprocess.run(command,cwd=TREE,check=True,timeout=max(1,deadline-time.time()))
 write('finished.json',dict(complete=True,ended=time.time(),baseline=read(BASE/'throughput_baseline15_01/result.json'),candidate=read(OUT/'result.json')))
except BaseException as error:
 write('finished.json',dict(complete=False,error=repr(error),ended=time.time()));raise
