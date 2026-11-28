from pathlib import Path
# Load the scheduler helpers without executing the consumed initial schedule.
p=Path('outputs/diagnostics/tight_full_curve_20261127_01/pipeline.py')
source=p.read_text();prefix=source[:source.index('\ntry:\n')]
exec(compile(prefix,str(p),'exec'))
try:
 assert (R/'all_frontends_complete.json').exists()
 # Existing30 workers keep running. Start independent required60 work now.
 background=[]
 for c,cpus in [('old_only','24-31'),('plus_corrected24','0-7')]:
  root=R/c;j=job(root,'rear60_overlap08','train','curve_recovery',['rear','--root',str(root),'--epochs','60'])
  background.append((run(c+'_rear60_overlap08',['-m','outputs.diagnostics.tight_full_curve_20261127_01.train_recovery','project','train','--job',str(j)],cpus),c))
 wait_for([R/c/'owner_rear30_resume02' for c in ['old_only','plus_corrected24']])
 prep=run('prepare_evaluation30',[str(D/'prepare_evaluation.py'),'--epoch','30']);join(prep,'prepare evaluation30')
 for role in 'AB':
  p=run('evaluation_e30_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('evaluation_e30_'+role+'.json'))]);join(p,'evaluation30')
 for p,n in background:join(p,n+' rear60')
 prep=run('prepare_evaluation60',[str(D/'prepare_evaluation.py'),'--epoch','60']);join(prep,'prepare evaluation60')
 for role in 'AB':
  p=run('evaluation_e60_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('evaluation_e60_'+role+'.json'))]);join(p,'evaluation60')
 finalprep=run('prepare_final16',[str(D/'prepare_evaluation.py'),'--epoch','60','--holdout']);join(finalprep,'prepare final16')
 for role in 'AB':
  p=run('final16_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('final16_'+role+'.json'))]);join(p,'independent final16')
 atomic_json(R/'curve_complete.json',dict(utc=time.time(),milestones=[0,30,60],architecture_changed=False))
except BaseException as e:
 atomic_json(D/'continuation08_failure.json',dict(utc=time.time(),error=repr(e)));raise
