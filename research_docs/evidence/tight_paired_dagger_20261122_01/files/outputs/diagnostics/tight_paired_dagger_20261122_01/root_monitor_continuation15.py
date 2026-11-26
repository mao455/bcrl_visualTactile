"""Finite read-only lifecycle monitor; no outcomes or signals."""
from pathlib import Path
import json,time,os,psutil
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS')
A=BASE/'outputs/diagnostics/tight_paired_dagger_20261122_01'
O=BASE/'outputs/visuotactile/run/tight_paired_dagger_20261122_01/owners/test_continuation15'
D=1795684571.2261844
activation=json.loads((A/'TEST_CONTINUATION15_ACTIVATED.json').read_text())
identity=activation['actors']['tight-paired-test-continuation15.service']
seen={};start=time.time()
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def capture(pid,role):
 try:
  p=psutil.Process(pid);row=dict(pid=pid,created=p.create_time(),command=p.cmdline(),role=role)
  seen[str(pid)]=row;return row
 except psutil.Error:return dict(pid=pid,absent=True,role=role)
try:
 with (A/'ROOT_LIVE15_METRICS.jsonl').open('x',buffering=1) as log:
  while time.time()<D:
   driver_exit=(A/'driver_test_continuation15/driver_exit.json').exists()
   status=read(O/'project_status.json');owner=read(O/'project_job_started.json');monitor=read(O/'monitor.json')
   for x,label in ((identity,'driver'),(owner,'owner'),(monitor,'native_monitor'),(status,'worker')):
    if isinstance(x.get('pid'),int):capture(x['pid'],label)
   row=dict(utc=time.time(),stage=status.get('stage'),index=17+status['stage'] if type(status.get('stage')) is int else None,launches=len(list((O/'launches').glob('wave*.json'))),dispatches=len(list((O/'dispatches').glob('wave*.json'))),reaped_stages=len(list((O/'stage_exits').glob('wave*.json'))),driver_exit=driver_exit,available_ram_bytes=psutil.virtual_memory().available,identities=dict(seen))
   log.write(json.dumps(row)+'\n')
   tmp=A/'ROOT_LIVE15_IDENTITIES.tmp';tmp.write_text(json.dumps(dict(utc=time.time(),identities=seen),indent=2));tmp.replace(A/'ROOT_LIVE15_IDENTITIES.json')
   print(json.dumps({k:v for k,v in row.items() if k!='identities'}),flush=True)
   if driver_exit:break
   time.sleep(45)
finally:
 (A/'ROOT_MONITOR15_EXIT.json').write_text(json.dumps(dict(pid=os.getpid(),started=start,ended=time.time(),read_only=True,signals=0,outcomes_read=False),indent=2))
