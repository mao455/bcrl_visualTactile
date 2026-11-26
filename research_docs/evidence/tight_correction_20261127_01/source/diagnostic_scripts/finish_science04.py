from pathlib import Path
import json,subprocess,sys,time,os
D=Path('outputs/diagnostics/tight_correction_20261127_01');R=Path('outputs/visuotactile/run/tight_correction_20261127_01');C=Path('configs/experiments/correction_20261127');began=time.time()
def run(name,args):
 (D/'science04_progress.json').write_text(json.dumps(dict(stage=name,utc=time.time(),args=args),indent=2))
 with (D/(name+'.log')).open('x') as stream:r=subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed; failure retained')
def wait_for(paths):
 while not all(p.exists() for p in paths):
  if time.time()-began>12*3600:raise TimeoutError('Finite scientific follow-up wait expired')
  if not (Path('/proc')/str(json.loads((D/'pipeline04_owner.json').read_text())['pid'])/'cmdline').exists() and not (D/'pipeline_complete.json').exists():raise RuntimeError('Main pipeline exited before completion; inspect original failures')
  time.sleep(20)
try:
 wait_for([R/'short_old_only/short_complete.json',R/'short_with_corrections/short_complete.json'])
 assert (D/'short_optimization_audit.json').exists(),'Actual exposure audit must already have passed'
 wait_for([D/'pipeline_complete.json'])
 run('science04_prepare_own_history',[str(D/'own_history.py')])
 for role in 'AB':run('science04_own_history_'+role,['-m','outputs.diagnostics.tight_correction_20261127_01.own_history','project','evaluate','--job',str(C/('own_history_'+role+'.json'))])
 run('science04_analyze',[str(D/'analyze_round.py')])
 run('science04_sampler',[str(D/'audit_sampler.py')])
 run('science04_attempts',[str(D/'attempts.py')])
 run('science04_close',[str(D/'close_audit.py')])
 (D/'science_complete.json').write_text(json.dumps(dict(complete=True,started_utc=began,utc=time.time(),pid=os.getpid()),indent=2))
except BaseException as e:
 (D/'science04_failure.json').write_text(json.dumps(dict(error=repr(e),utc=time.time(),pid=os.getpid()),indent=2));raise
