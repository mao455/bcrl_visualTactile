"""Finite task scheduler only; existing owners execute every research stage."""
from pathlib import Path
import subprocess,time,json,sys
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';D=P/'outputs/diagnostics/tight_correction_20261127_01';py=sys.executable
until=time.time()+8*3600
while not (R/'b72_fit/fit_complete.json').exists():
 failure=R/'b72_fit_retry02/project_failure.json'
 if failure.exists():raise RuntimeError('Fit owner failed; preserve failure and inspect before new work')
 if time.time()>until:raise RuntimeError('Finite queue fit wait expired')
 time.sleep(2)
for name,command in [
 ('b72_history_owner',[py,'-m','outputs.diagnostics.tight_correction_20261127_01.b72_history','project','evaluate','--job','configs/experiments/correction_20261127/b72_fit_history.json']),
 ('development_retry01_owner',[py,'-m','tools.tight_machine_review','project','evaluate','--job','configs/experiments/correction_20261127/development_retry01.json'])]:
 with (D/(name+'.log')).open('x') as stream:
  r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed; first attempt preserved')
