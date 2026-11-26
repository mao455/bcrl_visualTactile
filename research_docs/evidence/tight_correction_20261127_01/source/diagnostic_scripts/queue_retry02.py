from pathlib import Path
import subprocess,sys
D=Path('outputs/diagnostics/tight_correction_20261127_01')
for name in ['grasp_expert_controls','development_retry02']:
 with (D/(name+'_owner02.log')).open('x') as stream:
  result=subprocess.run([sys.executable,'-m','tools.tight_machine_review','project','evaluate','--job','configs/experiments/correction_20261127/'+name+'.json'],stdout=stream,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(name+' failed; attempted identity retained')
