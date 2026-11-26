from pathlib import Path
import json,time,subprocess,sys,os
D=Path('outputs/diagnostics/tight_correction_20261127_01');C=Path('configs/experiments/correction_20261127')
(D/'pipeline04_owner.json').write_text(json.dumps(dict(pid=os.getpid(),utc=time.time()),indent=2))
try:
 for role in 'AB':
  name='pipeline04_retest_'+role;args=['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('development_after_'+role+'.json'))]
  (D/'pipeline_progress.json').write_text(json.dumps(dict(stage=name,utc=time.time(),args=args,attempt='recovery04_no_training'),indent=2))
  with (D/(name+'.log')).open('x') as stream:r=subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(name+' failed; preserve first record')
 (D/'pipeline_complete.json').write_text(json.dumps(dict(complete=True,utc=time.time(),attempt='recovery04',additional_training_updates=0),indent=2))
except BaseException as e:
 (D/'pipeline04_failure.json').write_text(json.dumps(dict(error=repr(e),utc=time.time()),indent=2));raise
