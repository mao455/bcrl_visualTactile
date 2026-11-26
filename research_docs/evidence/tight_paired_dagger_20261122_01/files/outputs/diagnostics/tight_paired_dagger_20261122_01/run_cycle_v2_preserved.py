"""Sequential finite work after the external-test-aware parent owner closes."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from tools.tight_paired_dagger import ROOT,CONFIG,AUDIT,read,sources
from visuotactile.training_resources import atomic_json
def interrupted(signum,frame):raise KeyboardInterrupt('Owned cycle interrupted '+str(signum))
for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
status=ROOT/'cycle_status.json';deadline=read(ROOT/'session.json')['deadline']
logroot=ROOT/'cycle_logs';logroot.mkdir(exist_ok=False)

def note(stage,**extra):atomic_json(status,dict(pid=os.getpid(),stage=stage,utc=time.time(),**extra))
def run(name,args):
    assert time.time()<deadline
    note(name,command=args)
    with (logroot/(name+'.log')).open('x') as stream:
        child=subprocess.Popen([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if time.time()>=deadline or (ROOT/'STOP_CYCLE').exists():
                    raise TimeoutError('Finite deadline or owned STOP_CYCLE')
                time.sleep(5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
    atomic_json(logroot/(name+'_exit.json'),dict(returncode=child.returncode,utc=time.time()))
    if child.returncode:raise RuntimeError(name+' failed; preserve evidence before any explicit recovery')
def telemetry(container,owner):
    path=container/'shared_resources.json'
    assert path.is_symlink()
    path.unlink();path.symlink_to(ROOT/'owners'/owner/'shared_resources.json')
try:
    note('waiting_parent_owner')
    while not (ROOT/'parent_queue_result.json').exists():
        if time.time()>=deadline or (ROOT/'STOP_CYCLE').exists():raise TimeoutError('Finite deadline or owned STOP_CYCLE while waiting')
        time.sleep(30)
    assert read(ROOT/'parent_queue_result.json')['returncode']==0
    run('verify_actual_parents',['-m','tools.tight_paired_prepare','complete-parents'])
    run('prepare_collection',['-m','tools.tight_paired_collection','prepare','--lanes','8'])
    telemetry(ROOT,'collection')
    run('collect_96',['-m','tools.tight_paired_collection','project','dagger','--job',str(ROOT/'collection_job.json')])
    run('summarize_collection',['-m','tools.tight_paired_collection','summarize'])
    run('prepare_learning',['-m','tools.tight_paired_prepare','prepare-learning'])
    run('preflight_learning',['-m','tools.tight_paired_prepare','preflight-learning'])
    run('register_learning_jobs',['-m','tools.tight_paired_prepare','learning-jobs'])
    run('cache_own_observations',['-m','tools.tight_project','prepare','--job',str(CONFIG/'main_cache.json')])
    telemetry(ROOT/'learning','main_train')
    atomic_json(ROOT/'learning/training_source_identity.json',dict(utc=time.time(),sources=sources()))
    run('train_four_components',['-m','tools.tight_paired_learning','project','train','--job',str(CONFIG/'main_train.json')])
    for cohort in ('development','test'):
        run('prepare_'+cohort,['-m','tools.tight_paired_evaluate','prepare','--cohort',cohort])
        run('evaluate_'+cohort,['-m','tools.tight_paired_evaluate','project','evaluate','--job',str(CONFIG/('evaluate_'+cohort+'.json'))])
        run('summarize_'+cohort,['-m','tools.tight_paired_evaluate','summarize','--output',str(ROOT/'evaluation'/cohort)])
    note('main_stages_complete',report_required=True)
except BaseException as error:
    atomic_json(ROOT/'cycle_failure.json',dict(utc=time.time(),error=repr(error),traceback=traceback.format_exc()))
    note('stopped_with_evidence',error=repr(error))
    raise
