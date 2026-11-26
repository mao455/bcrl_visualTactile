"""Finite durable driver for only the unassigned collection suffix."""
import json,os,signal,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from driver_support_v4 import run_owned,wait_external
from visuotactile.training_resources import atomic_json
PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
AUDIT=Path(__file__).resolve().parent
OUT=AUDIT/'driver_collection_recovery05'
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.collection_recovery05'
def read(p):return json.loads(Path(p).read_text())
def note(stage,**extra):atomic_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.time(),**extra))
def interrupt(s,f):raise KeyboardInterrupt(s)
def main():
 OUT.mkdir(exist_ok=False)
 for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,interrupt)
 deadline=read(ROOT/'session.json')['deadline']
 command=[sys.executable,'-m',MODULE,'project','dagger','--job',str(ROOT/'collection_recovery05_job.json')]
 atomic_json(OUT/'registration.json',dict(utc=time.time(),deadline=deadline,command=command,scope='81 previously unassigned opportunities; no retry',driver_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
 try:
  assert not (ROOT/'owners/collection_recovery05/project_job_started.json').exists()
  wait_external(deadline,ROOT/'STOP_CYCLE',lambda x:note('quiet_external_gate',scan=x))
  target=ROOT/'shared_resources.json'
  assert target.is_symlink(),'Only existing telemetry symlink may be rebound'
  atomic_json(OUT/'telemetry_transition.json',dict(old=os.readlink(target),new=str(ROOT/'owners/collection_recovery05/shared_resources.json'),utc=time.time()))
  target.unlink();target.symlink_to(ROOT/'owners/collection_recovery05/shared_resources.json')
  note('collection_running',command=command)
  with (OUT/'project.log').open('x') as stream:rc=run_owned(command,stream,deadline,ROOT/'STOP_CYCLE')
  atomic_json(OUT/'exit.json',dict(returncode=rc,utc=time.time()))
  if rc:raise RuntimeError('Collection exited nonzero; preserve all consumed opportunities')
  note('collection_suffix_closed',aggregation_required=True)
 except BaseException as exc:
  atomic_json(OUT/'failure.json',dict(utc=time.time(),error=repr(exc),traceback=traceback.format_exc()))
  note('stopped_with_evidence',error=repr(exc));raise
if __name__=='__main__':main()
