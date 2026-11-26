from pathlib import Path
import time,subprocess,sys,json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01');deadline=time.time()+4*3600
while not (R/'development_retry01/project_exit.json').exists():
 if (R/'development_retry01/project_failure.json').exists():raise RuntimeError('Development failed; preserve and inspect')
 if time.time()>deadline:raise TimeoutError('Finite wait expired')
 time.sleep(2)
with (D/'grasp_expert_owner.log').open('x') as stream:
 result=subprocess.run([sys.executable,'-m','tools.tight_machine_review','project','evaluate','--job','configs/experiments/correction_20261127/grasp_expert_controls.json'],stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError('Expert diagnostic failed')
