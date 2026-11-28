from pathlib import Path
# Load the scheduler helpers without executing the consumed initial schedule.
p=Path('outputs/diagnostics/tight_full_curve_20261127_01/pipeline.py')
source=p.read_text();prefix=source[:source.index('\ntry:\n')]
exec(compile(prefix,str(p),'exec'))
try:
 assert (R/'all_frontends_complete.json').exists()
 for epochs in (30,60):
  rears=[]
  for c,cpus in [('old_only','8-15'),('plus_corrected24','16-23')]:
   root=R/c;j=job(root,'rear'+str(epochs)+'_resume02','train','curve_recovery',['rear','--root',str(root),'--epochs',str(epochs)])
   rears.append((run(c+'_rear'+str(epochs)+'_resume02',['-m','outputs.diagnostics.tight_full_curve_20261127_01.train_recovery','project','train','--job',str(j)],cpus),c))
  for p,n in rears:join(p,n+' rear'+str(epochs))
  prep=run('prepare_evaluation'+str(epochs),[str(D/'prepare_evaluation.py'),'--epoch',str(epochs)]);join(prep,'prepare evaluation')
  for role in 'AB':
   p=run(f'evaluation_e{epochs:02d}_{role}',['-m','tools.tight_machine_review','project','evaluate','--job',str(C/f'evaluation_e{epochs:02d}_{role}.json')]);join(p,'evaluation')
 finalprep=run('prepare_final16',[str(D/'prepare_evaluation.py'),'--epoch','60','--holdout']);join(finalprep,'prepare final16')
 for role in 'AB':
  p=run(f'final16_{role}',['-m','tools.tight_machine_review','project','evaluate','--job',str(C/f'final16_{role}.json')]);join(p,'independent final16')
 atomic_json(R/'curve_complete.json',dict(utc=time.time(),milestones=[0,30,60],architecture_changed=False))
except BaseException as e:
 atomic_json(D/'continuation07_failure.json',dict(utc=time.time(),error=repr(e)));raise
