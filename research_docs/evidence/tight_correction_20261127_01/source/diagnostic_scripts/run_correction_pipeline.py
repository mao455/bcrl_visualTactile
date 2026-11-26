"""Finite serial heavy stages; CPU audits/publication can proceed alongside."""
from pathlib import Path
import subprocess,sys,time,json
P=Path.cwd();D=P/'outputs/diagnostics/tight_correction_20261127_01';R=P/'outputs/visuotactile/run/tight_correction_20261127_01';deadline=time.time()+10*3600

def run(name,args):
 (D/'pipeline_progress.json').write_text(json.dumps(dict(stage=name,utc=time.time(),args=args),indent=2))
 with (D/(name+'.log')).open('x') as stream:r=subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed; attempted identity retained')

while not (R/'development_retry03/project_exit.json').exists():
 if (R/'development_retry03/project_failure.json').exists():raise RuntimeError('Development failed; inspect before retries')
 if time.time()>deadline:raise TimeoutError('Finite wait expired')
 time.sleep(2)
if (R/'development_retry03/project_failure.json').exists():raise RuntimeError('Development failed')
for name in ['development_summary','apply_capture','apply_queue']:
 run('pipeline_'+name,[str(D/(name+'.py'))])
run('pipeline_related_checks',['-m','pytest','-q','tests/test_machine_review.py','tests/test_reset_admission.py','tests/test_paired_collection.py','tests/test_paired_evaluate.py','tests/test_paired_training_decks.py','tests/test_paired_fresh_contract.py'])
run('pipeline_prepare_capture',[str(D/'prepare_capture.py')])
run('pipeline_current_capture',['-m','tools.tight_machine_review','project','evaluate','--job','configs/experiments/correction_20261127/current_takeovers.json'])
run('pipeline_prepare_short',[str(D/'prepare_short_learning.py')])
for condition in ['old_only','with_corrections']:
 run('pipeline_train_'+condition,['-m','outputs.diagnostics.tight_correction_20261127_01.short_train','project','train','--job','configs/experiments/correction_20261127/short_'+condition+'.json'])
run('pipeline_prepare_retest',[str(D/'prepare_retest.py')])
for role in 'AB':
 run('pipeline_retest_'+role,['-m','tools.tight_machine_review','project','evaluate','--job','configs/experiments/correction_20261127/development_after_'+role+'.json'])
(D/'pipeline_complete.json').write_text(json.dumps(dict(utc=time.time(),complete=True),indent=2))
