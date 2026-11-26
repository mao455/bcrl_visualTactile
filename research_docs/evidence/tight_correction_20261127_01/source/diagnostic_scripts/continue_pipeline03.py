from pathlib import Path
import json,time,subprocess,sys,os
D=Path('outputs/diagnostics/tight_correction_20261127_01');R=Path('outputs/visuotactile/run/tight_correction_20261127_01');C=Path('configs/experiments/correction_20261127');read=lambda p:json.loads(p.read_text())
(D/'pipeline03_owner.json').write_text(json.dumps(dict(pid=os.getpid(),utc=time.time()),indent=2))
def run(name,args):
 (D/'pipeline_progress.json').write_text(json.dumps(dict(stage=name,utc=time.time(),args=args,attempt='recovery03'),indent=2))
 with (D/(name+'.log')).open('x') as stream:r=subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed; preserve first failure and inspect')
def wait_capacity():
 deadline=read(R/'short_old_only/session.json')['deadline'];candidate=R/'short_with_corrections'
 while True:
  if (candidate/'project_failure.json').exists():raise RuntimeError('Correction job failed before old-B recovery; inspect exact committed update')
  if time.time()>=deadline:raise TimeoutError('Original training deadline expired')
  ready=(candidate/'B/training/finished.json').exists()
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
  (D/'pipeline_progress.json').write_text(json.dumps(dict(stage='wait_B_DP_complete_and_capacity',utc=time.time(),candidate_B_DP_complete=ready,free_mib=free,required_mib=11000,not_external_presence_gate=True),indent=2))
  if ready and free>=11000:return
  time.sleep(10)
try:
 wait_capacity()
 run('pipeline03_old_B_rear',['-m','outputs.diagnostics.tight_correction_20261127_01.resume_old_backend','project','train','--job',str(C/'short_old_backend_retry03.json')])
 run('pipeline03_join_correction',['-m','outputs.diagnostics.tight_correction_20261127_01.short_train','project','train','--job',str(C/'short_with_corrections.json')])
 run('pipeline03_prepare_retest',[str(D/'prepare_retest.py')])
 for role in 'AB':run('pipeline03_retest_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('development_after_'+role+'.json'))])
 (D/'pipeline_complete.json').write_text(json.dumps(dict(complete=True,utc=time.time(),attempt='recovery03'),indent=2))
except BaseException as e:
 (D/'pipeline03_failure.json').write_text(json.dumps(dict(error=repr(e),utc=time.time()),indent=2));raise
