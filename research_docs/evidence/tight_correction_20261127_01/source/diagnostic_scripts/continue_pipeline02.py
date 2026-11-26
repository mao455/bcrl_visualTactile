from pathlib import Path
import subprocess,sys,time,json
D=Path('outputs/diagnostics/tight_correction_20261127_01');C=Path('configs/experiments/correction_20261127')
def run(name,args):
 (D/'pipeline_progress.json').write_text(json.dumps(dict(stage=name,utc=time.time(),args=args,attempt='recovery02'),indent=2))
 with (D/(name+'.log')).open('x') as stream:r=subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed; original and current attempts retained')
run('pipeline_train_old_only_retry02',['-m','outputs.diagnostics.tight_correction_20261127_01.short_train','project','train','--job',str(C/'short_old_only_retry02.json')])
run('pipeline_train_with_corrections',['-m','outputs.diagnostics.tight_correction_20261127_01.short_train','project','train','--job',str(C/'short_with_corrections.json')])
run('pipeline_prepare_retest',[str(D/'prepare_retest.py')])
for role in 'AB':run('pipeline_retest_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('development_after_'+role+'.json'))])
(D/'pipeline_complete.json').write_text(json.dumps(dict(complete=True,utc=time.time(),attempt='recovery02'),indent=2))
