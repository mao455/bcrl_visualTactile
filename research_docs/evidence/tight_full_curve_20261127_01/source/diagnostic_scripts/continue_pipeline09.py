from pathlib import Path
# Load the scheduler helpers without executing the consumed initial schedule.
p=Path('outputs/diagnostics/tight_full_curve_20261127_01/pipeline.py')
source=p.read_text();prefix=source[:source.index('\ntry:\n')]
exec(compile(prefix,str(p),'exec'))

# Concurrent stages never mutate a shared parent telemetry pointer.
def job(root,name,operation,entry,args):
 owner=root/('owner_'+name);owner.mkdir();s=copy.deepcopy(read(R/'session.json'));s['resource_root']=str(owner);atomic_json(owner/'session.json',s)
 path=C/(root.name+'_'+name+'.json');atomic_json(path,dict(version='tight_project_job_v1',operation=operation,resource_root=str(owner),stages=[dict(entry=entry,args=args)]));return path

def run(name,args,cpus='8-15'):
 env=dict(os.environ,PYTHONPATH=str(P),RESEARCH_CPUS=cpus,OMP_NUM_THREADS='2')
 if '--job' in args:env['CURVE_STAGE_OWNER']=read(Path(args[args.index('--job')+1]))['resource_root']
 with (D/(name+'.log')).open('x') as stream:
  process=subprocess.Popen([py,*args],cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 atomic_json(D/(name+'_pid.json'),dict(pid=process.pid,args=args,utc=time.time()))
 return process
try:
 assert (R/'all_frontends_complete.json').exists()
 rears=[]
 for c,cpus in [('old_only','8-15'),('plus_corrected24','16-23')]:
  root=R/c;j=job(root,'rear30_resume03','train','curve_recovery',['rear','--root',str(root),'--epochs','30'])
  rears.append((run(c+'_rear30_resume03',['-m','outputs.diagnostics.tight_full_curve_20261127_01.train_recovery','project','train','--job',str(j)],cpus),c))
 for p,n in rears:join(p,n+' rear30')
 prep=run('prepare_evaluation30',[str(D/'prepare_evaluation.py'),'--epoch','30']);join(prep,'prepare evaluation30')
 for role in 'AB':
  p=run('evaluation_e30_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('evaluation_e30_'+role+'.json'))]);join(p,'evaluation30')
 wait_for([R/c/'owner_rear60_overlap08' for c in ['old_only','plus_corrected24']])
 prep=run('prepare_evaluation60',[str(D/'prepare_evaluation.py'),'--epoch','60']);join(prep,'prepare evaluation60')
 for role in 'AB':
  p=run('evaluation_e60_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('evaluation_e60_'+role+'.json'))]);join(p,'evaluation60')
 finalprep=run('prepare_final16',[str(D/'prepare_evaluation.py'),'--epoch','60','--holdout']);join(finalprep,'prepare final16')
 for role in 'AB':
  p=run('final16_'+role,['-m','tools.tight_machine_review','project','evaluate','--job',str(C/('final16_'+role+'.json'))]);join(p,'independent final16')
 atomic_json(R/'curve_complete.json',dict(utc=time.time(),milestones=[0,30,60],architecture_changed=False))
except BaseException as e:
 atomic_json(D/'continuation09_failure.json',dict(utc=time.time(),error=repr(e)));raise
