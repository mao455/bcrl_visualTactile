"""Resume after a pre-physics telemetry binding failure; preserve finite identity."""
import json, os, signal, shutil, sys, time, traceback
from pathlib import Path
from tools.tight_paired_dagger import ROOT, CONFIG, AUDIT, read, sources
from visuotactile.training_resources import atomic_json
from driver_support_v4 import run_owned, wait_external

def interrupted(signum, frame): raise KeyboardInterrupt('Owned cycle interrupted '+str(signum))
for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, interrupted)
status=ROOT/'cycle_status.json'; deadline=read(ROOT/'session.json')['deadline']
logroot=ROOT/'cycle_logs'; assert logroot.is_dir()
def note(stage, **extra): atomic_json(status, dict(pid=os.getpid(), stage=stage, utc=time.time(), recovery='recovery04', **extra))
def run(name,args,gpu=False):
    assert time.time()<deadline and not (ROOT/'STOP_CYCLE').exists()
    if gpu: wait_external(deadline, ROOT/'STOP_CYCLE', lambda state:note('waiting_external_before_'+name,external_test_scan=state))
    note(name,command=args)
    with (logroot/(name+'.log')).open('x') as stream:
        command=[sys.executable,*args]
        if name in ('collect_96','evaluate_development','evaluate_test'):
            command=['taskset','-c','8-23',*command]
        rc=run_owned(command,stream,deadline,ROOT/'STOP_CYCLE')
    atomic_json(logroot/(name+'_exit.json'),dict(returncode=rc,utc=time.time()))
    if rc: raise RuntimeError(name+' failed; preserve evidence before explicit recovery')
def telemetry(container, owner):
    path=container/'shared_resources.json'
    if path.is_symlink(): path.unlink()
    else: assert not path.exists(), 'Refuse replacing a real telemetry file'
    path.symlink_to(ROOT/'owners'/owner/'shared_resources.json')
def storage_gate():
    # Entry.length covers the complete real student history and expert suffix.
    entries=read(ROOT/'collection/correction_entries.json')
    ticks=sum(int(row['length']) for row in entries.values())
    features=ticks*1334200*1.06
    # Covers main checkpoints, caches, logs, preparation and288 autonomous artifacts.
    required=features+35*2**30+12*2**30
    while True:
        assert time.time()<deadline and not (ROOT/'STOP_CYCLE').exists()
        free=shutil.disk_usage(ROOT).free
        row=dict(utc=time.time(),admitted_trajectories=len(entries),complete_history_ticks=ticks,
                 estimated_feature_bytes=features,remaining_fixed_headroom_bytes=35*2**30,
                 operational_reserve_bytes=12*2**30,required_free_bytes=required,
                 actual_free_bytes=free,ready=free>=required,forecast=True)
        atomic_json(ROOT/'post_collection_storage_gate.json',row)
        if row['ready']: return
        note('waiting_verified_storage_capacity',storage=row)
        time.sleep(30)
try:
    assert read(ROOT/'parent_queue_result.json')['returncode']==0
    assert read(ROOT/'actual_parent_ages.json')['roles']['A']['backend_updates']==22763
    run('prepare_collection_recovery04',['-m','tools.tight_paired_collection','prepare','--lanes','8'])
    assert not (ROOT/'collection/results.json').exists()
    assert not (ROOT/'owners/collection/project_job_started.json').exists()
    telemetry(ROOT,'collection')
    run('collect_96',['-m','tools.tight_paired_collection','project','dagger','--job',str(ROOT/'collection_job.json')],True)
    run('summarize_collection',['-m','tools.tight_paired_collection','summarize'])
    run('audit_collection_integrity',[str(AUDIT/'audit_collection_integrity.py'),'--collection',str(ROOT/'collection'),'--output',str(AUDIT/'collection_integrity_actual01')])
    storage_gate()
    run('prepare_learning',['-m','tools.tight_paired_prepare','prepare-learning'])
    run('preflight_learning',['-m','tools.tight_paired_prepare','preflight-learning'])
    run('register_learning_jobs',['-m','tools.tight_paired_prepare','learning-jobs'])
    run('cache_own_observations',['-m','tools.tight_project','prepare','--job',str(CONFIG/'main_cache.json')],True)
    telemetry(ROOT/'learning','main_train')
    atomic_json(ROOT/'learning/training_source_identity.json',dict(utc=time.time(),sources=sources()))
    run('train_four_components',['-m','tools.tight_paired_learning','project','train','--job',str(CONFIG/'main_train.json')],True)
    run('verify_final_actual_training',[str(AUDIT/'presealed_guard.py'),'--mode','main'])
    for cohort in ('development','test'):
        run('prepare_'+cohort,['-m','tools.tight_paired_evaluate','prepare','--cohort',cohort])
        run('evaluate_'+cohort,['-m','tools.tight_paired_evaluate','project','evaluate','--job',str(CONFIG/('evaluate_'+cohort+'.json'))],True)
        run('summarize_'+cohort,['-m','tools.tight_paired_evaluate','summarize','--output',str(ROOT/'evaluation'/cohort)])
    note('main_stages_complete',report_required=True)
except BaseException as error:
    atomic_json(ROOT/'cycle_failure.json',dict(utc=time.time(),error=repr(error),traceback=traceback.format_exc(),recovery='recovery04'))
    note('stopped_with_evidence',error=repr(error))
    raise
