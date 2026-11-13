"""One pre-registered conditioning experiment; research decisions remain external."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.runtime import RUN,kill_owned,proc,read,sha,supervised_owner,write
from tools.tight_execution_follow import residual_command,select_safe_candidate
from tools.tight_execution_learning import GROUPS,HISTORY,run_child
from visuotactile.execution_learning import PROJECTION_FY_LOSS
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT

ORIGINAL=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_execution_learning_20261113_01')
COMPUTE_DEADLINE=1794585315.106664
ABSOLUTE_DEADLINE=1794585915.106664


def checked_root(root):
    root=root.resolve()
    if root.parent!=ORIGINAL or root.name not in ('adaptive_centered_01','adaptive_tick_window_01'):
        raise ValueError('Unique registered continuation root required')
    return root


def configuration_for(source, *, tick_window=False):
    configuration=copy.deepcopy(read(source/'configuration.json'))
    pair=configuration['residual_pair']
    pair.update(output_contract=REACHABLE_CENTERED_OUTPUT,
        loss_contract=dict(PROJECTION_FY_LOSS,regularizer_weight=0.),history_contract=HISTORY,
        learning_rate=1e-4,maximum_control_tick_epochs=1,minimum_control_tick_epochs=1,
        comparison_control_tick_epochs=[1],validation_epoch_interval=1,
        initialization_seed=2026111301)
    if tick_window:
        pair['tick_window_contract']=dict(TICK_WINDOW_CONTRACT)
    return configuration


def prepare(root):
    from tools.tight_dagger import reuse_nominal_cache,share_monitor
    from tools.tight_finish import residual_prepare
    root=checked_root(root)
    if root.exists():raise ValueError('Existing attempt must not be reset or overwritten')
    now=time.time()
    if now>=COMPUTE_DEADLINE-7200:raise TimeoutError('Insufficient original budget for registered complete experiment')
    previous=ORIGINAL/'adaptive_recovery_01'
    tick_window=root.name=='adaptive_tick_window_01'
    boundary=read(ORIGINAL/'agent_supervisor/initial_boundary.json')
    if not boundary.get('complete'):raise ValueError('Previous complete-policy boundary required')
    if tick_window and not (ORIGINAL/'adaptive_centered_01/finished.json').is_file():
        raise ValueError('Completed centered comparator required')
    root.mkdir()
    write(root/'registration.json',dict(started_utc=now,compute_deadline=COMPUTE_DEADLINE,
        absolute_deadline=ABSOLUTE_DEADLINE,stage_deadline=min(now+7200,COMPUTE_DEADLINE-300),
        hypothesis=('Fixed128tick gradient windows restore explicit recovery weighting and reduce exact current-weight prefix replay; benefit unproven'
            if tick_window else 'Reachable-normalized proposal removes cancellation of large absolute nominal error; not proven causal'),
        source_attempt=str(previous),epochs_each=1,frontend_updates=0,new_collection=False,test22=False,
        prior_results={group:{key:boundary['summaries'][group][key] for key in
            ('successes','completed','native_lifts','alignments','software_failures')} for group in GROUPS},
        original_failure='Intentional coordinator handoff, not evaluation failure',
        comparison_limit=('Same centered output, original round01 parameters, sampling deck, exposure, FY reg0 LR1e-4; optimizer count and loss normalization change together, not equal-update'
            if tick_window else 'Output map and exposure differ; same original round01 parameters, FY reg0 LR1e-4 and cache; no success claim'),
        comparator=str(ORIGINAL/'adaptive_centered_01') if tick_window else str(previous),
        tick_window_contract=dict(TICK_WINDOW_CONTRACT) if tick_window else None))
    session=copy.deepcopy(read(previous/'formal/QG01/backend/session.json'))
    session.update(resource_root=str(root),matrix_owner=str(root),owned_worker_roots=[str(root)],
        started=now,deadline=ABSOLUTE_DEADLINE,compute_deadline=COMPUTE_DEADLINE)
    write(root/'session.json',session)
    for group in GROUPS:
        source=previous/'formal'/group/'backend'
        destination=root/'formal'/group/'backend'
        configuration=configuration_for(source,tick_window=tick_window)
        pair=configuration['residual_pair']
        residual_prepare(destination,configuration=configuration)
        reuse_nominal_cache(source/group/'nominal',destination/group/'nominal',
            pair['bases'][group],destination/'state_data',pair['residual_statistics_sha256'])
        shutil.copy2(source/group/'nominal/scales.json',destination/group/'nominal/scales.json')
        shutil.copy2(source/'evaluation_plan.json',destination/'evaluation_plan.json')
        share_monitor(root,destination)
        end=read(root/'registration.json')['stage_deadline']
        for path in (destination/'session.json',destination/group/'session.json'):
            local=read(path)
            local.update(resource_root=str(root),matrix_owner=str(root),owned_worker_roots=[str(root)],
                started=now,deadline=ABSOLUTE_DEADLINE,compute_deadline=COMPUTE_DEADLINE,
                training_deadline=end-2400,offline_deadline=end-1800,evaluation_deadline=end,
                cwd=str(Path.cwd()))
            write(path,local)
        write(destination/'cache_complete.json',dict(complete=True,exact_reuse=True,
            source=str(source/group/'nominal'),new_frontend_updates=0))
    write(root/'status.json',dict(state='PREPARED',stage='awaiting_committed_launch',utc=time.time()))


@supervised_owner
def run(root):
    root=checked_root(root)
    registration=read(root/'registration.json')
    if (root/'launch.json').exists():raise ValueError('No implicit repeat of existing attempt')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Reviewed committed source required')
    if time.time()>=registration['stage_deadline']-3600:raise TimeoutError('Registered stage budget exhausted')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    files=subprocess.check_output(['git','ls-files','*.py','visuotactile/run_cpu.sh','configs/tight_state.json'],text=True).splitlines()
    write(root/'source_freeze.json',{str(Path(path).resolve()):sha(path) for path in files})
    write(root/'launch.json',dict(pid=os.getpid(),process=proc(os.getpid()),source_head=head,
        cwd=str(Path.cwd()),utc=time.time(),compute_deadline=COMPUTE_DEADLINE))
    for path in [root/'session.json',*root.glob('formal/*/backend/session.json'),*root.glob('formal/*/backend/*/session.json')]:
        session=read(path);session['source_head']=head;write(path,session)
    monitor=None
    def status(stage,**fields):
        write(root/'status.json',dict(state='RUNNING',stage=stage,pid=os.getpid(),utc=time.time(),**fields))
    try:
        status('resource_admission')
        with (root/'monitor.log').open('a') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.time()
        while not (root/'shared_resources.json').exists():
            if monitor.poll() is not None or time.time()-started>30:raise RuntimeError('Fresh resource monitor unavailable')
            time.sleep(.2)
        end=registration['stage_deadline']
        for group in GROUPS:
            destination=root/'formal'/group/'backend'
            status('tick_window_backend_training' if registration.get('tick_window_contract') else 'centered_backend_training',group=group)
            run_child(root,group+'_train',residual_command('residual-train',destination/group),end-2400)
            choice=select_safe_candidate(destination/group/'training')
            write(destination/'candidate_selection.json',choice)
            status('centered_offline_validation',group=group)
            run_child(root,group+'_offline',residual_command('residual-offline',destination/group),end-1800)
        for group in GROUPS:
            status('complete_student_development',group=group)
            run_child(root,group+'_eval',['-m','tools.tight_finish','residual_evaluate','--root',str(root/'formal'/group/'backend')],end)
        results={}
        for group in GROUPS:
            destination=root/'formal'/group/'backend'
            results[group]=dict(exposure=read(destination/group/'training/finished.json'),
                selection=read(destination/'candidate_selection.json'),
                evaluation_receipt=read(destination/'evaluation_complete.json'),
                summary={key:value for key,value in read(destination/group/'evaluation_summary.json').items() if key!='results'})
        write(root/'finished.json',dict(groups=results,utc=time.time(),test22=False,
            research_decision='Return completed experiment to real Agent; no automatic DAgger or adoption'))
        write(root/'status.json',dict(state='COMPLETE',stage='awaiting_research_decision',utc=time.time()))
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=time.time()))
        write(root/'status.json',dict(state='FAILED',stage='requires_actual_diagnosis',error=repr(error),utc=time.time()))
        raise
    finally:
        (root/'STOP_MONITOR').touch()
        if monitor is not None:kill_owned(monitor)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    arguments=parser.parse_args()
    (prepare if arguments.mode=='prepare' else run)(arguments.root)


if __name__=='__main__':main()
