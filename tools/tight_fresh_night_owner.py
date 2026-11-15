"""Finite supervised-to-unattended grasp DAgger, with explicit review barrier.

The first two complete real cycles require Codex observation. No receipt created
by this owner releases that barrier. Recoverable training stages retry once from
durable state; interrupted physical trajectories are preserved for review.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tools.tight_fresh_collect_cycle_owner import run_stage, collection_admission
from tools.tight_fresh_grasp_evidence import behavior
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

read=lambda path:json.loads(Path(path).read_text())


def execution_identity(directory):
    from visuotactile.execution_backend import normalize_backend
    plan=read(Path(directory)/'plan.json');batch=plan.get('batch',{})
    return dict(backend=normalize_backend(plan.get('execution_backend')),
        queue=batch.get('queue_profile'),gpu_pipeline=batch.get('gpu_pipeline'),render=batch.get('render_profile'))


def source_decision(candidate, sources, evaluations, *, candidate_evaluation=None):
    """Conservative operational choice on identical registered seed subsets.

    Small cohorts guide next collection only; never certify 90% grasp reliability.
    """
    candidate_evaluation=candidate/'development' if candidate_evaluation is None else Path(candidate_evaluation)
    new=behavior(candidate_evaluation,horizon=300);decision={}
    for role in 'AB':
        if execution_identity(candidate_evaluation)!=execution_identity(evaluations[role]):
            raise ValueError('Source/candidate execution backend or queue differs; measure the source under the current pipeline first')
        old=behavior(evaluations[role],role,horizon=300)
        a={r['seed']:r for r in old['rows'] if r['legal'] and r.get('diagnostic_cohort')!='training_family_recheck'}
        b={r['seed']:r for r in new['rows'] if r['role']==role and r['legal'] and r.get('diagnostic_cohort')!='training_family_recheck'}
        common=sorted(a.keys()&b.keys())
        old_held=sum(a[s]['hold60']['sustained'] for s in common)
        new_held=sum(b[s]['hold60']['sustained'] for s in common)
        adopt=len(common)>=3 and new_held>=old_held+2
        decision[role]=dict(common_legal_seeds=common,source_hold60=old_held,candidate_hold60=new_held,
            use_candidate=adopt,selected_root=str(candidate if adopt else sources[role]),
            execution_identity=execution_identity(evaluations[role]),
            explanation='Two additional 1s holds on shared legal seeds under the same backend/queue contracts required; otherwise retain the previous complete pair. Lane populations can differ; no exact hidden-state replay, statistical proof or stage completion is claimed.')
    return decision


def recoverable_training_failure(candidate):
    ledger=candidate.with_name(candidate.name+'_stages')
    if (candidate/'development').exists():return False
    failure=ledger/'failure.json'
    if not failure.exists():return False
    text=failure.read_text()
    for log in ledger.glob('*.log'):
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-8192));text+=stream.read().decode(errors='replace')
    bad=('Nonfinite','nonfinite','SHA256 differs','source drift','configuration changed','corrupt','No space left')
    transient=('VRAM reserve','RAM reserve','Stale resource telemetry','stage_progress_timeout')
    probe=ledger/'controlled_resume_probe.json'
    if 'controlled_completed_frontend_resume_probe' in text and probe.exists() and not any(s in text for s in bad):
        receipt=read(probe)
        return receipt.get('complete_frontends') is True and all(
            sha256(candidate/role/'training/finished.json')==receipt['frontends'][role]['finished_sha256'] for role in 'AB')
    return not any(s in text for s in bad) and any(s in text for s in transient)


def run(root,configuration):
    cfg=read(configuration);session=read(root/'session.json');ledger=root/'night';ledger.mkdir(exist_ok=True)
    deadline=min(session['deadline'],cfg['absolute_wall_deadline'])-120;started=time.time();tree=Path.cwd();python=str(tree/'.venv_isaacsim/bin/python')
    assert cfg['version']=='supervised_grasp_night_v1' and len(cfg['rounds'])<=4
    assert cfg['supervised_cycles']==2 and deadline>started and (root/'shared_resources.json').is_file()
    binding=sha256(configuration);source_head=cfg['source_commit']
    import subprocess
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source_head
    state_path=ledger/'state.json'
    state=read(state_path) if state_path.exists() else dict(configuration_sha256=binding,
        sources=cfg['sources'],evaluations=cfg['source_evaluations'],data_root=cfg['data_root'],rounds=[],complete=False)
    assert state['configuration_sha256']==binding
    def save(stage,**extra):
        state.update(stage=stage,heartbeat=time.time(),owner_pid=os.getpid(),deadline=deadline,**extra)
        atomic_json(state_path,state)
    stops=(root/'STOP',ledger/'STOP')
    def check():
        if any(p.exists() for p in stops):raise InterruptedError('requested_stop')
        if time.time()>=deadline:raise TimeoutError('bounded_night_deadline')
        if sha256(configuration)!=binding:raise ValueError('Night configuration changed')
    def stage(name,command,collection,candidate):
        check();attempt=ledger/(name+'_'+str(time.time_ns()));attempt.mkdir()
        save(name,stage_ledger=str(attempt))
        context=dict(ledger=attempt,paths=dict(worker_tree=tree,collection=collection,output=candidate))
        receipt=run_stage(name,command,context,deadline,stops)
        if not receipt['complete']:raise RuntimeError(name+' failed; '+str(attempt))
        return receipt
    def cmd(module,*args):return [python,'-m',module,*map(str,args)]
    def prepare_once(path,name,command,collection,candidate):
        marker=ledger/(name+'_complete.json')
        if marker.exists():
            assert path.exists() and read(marker)['sha256']==sha256(path)
        else:
            stage(name,command,collection,candidate)
            atomic_json(marker,dict(path=str(path),sha256=sha256(path)))
    result=None
    with (root/'night_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if cfg.get('refresh_source_backend_comparison') and not state.get('source_backend_bridge_complete'):
                reference=Path(cfg['candidate_reference']);pair=root/'source_backend_pair';baseline=root/'source_backend_baseline'
                prepare_once(pair/'prepared.json','source_backend_pair',cmd('tools.tight_fresh_collection_source',
                    '--session-root',root,'--data-source',cfg['data_root'],'--output',pair,
                    '--role-source','A='+cfg['sources']['A'],'--role-source','B='+cfg['sources']['B'],
                    '--role-evaluation','A='+cfg['source_evaluations']['A'],'--role-evaluation','B='+cfg['source_evaluations']['B']),baseline,reference)
                prepare_once(baseline/'plan.json','source_backend_prepare',cmd('tools.tight_fresh_batch_run','prepare',
                    '--root',pair,'--output',baseline,'--count',8,'--seed',28500000,'--lanes',16,'--vision-microbatch',8,
                    '--queued-gpu','--grasp-evaluation-ticks',300,'--backend-json',cfg['backend_json']),baseline,reference)
                result_path=baseline/'result.json'
                if not result_path.exists():
                    if (baseline/'run_started.json').exists():raise RuntimeError('Interrupted source baseline needs unresolved-case review')
                    stage('source_backend_run',cmd('tools.tight_fresh_batch_run','run','--output',baseline),baseline,reference)
                assert read(result_path)['software_complete']
                decisions=source_decision(reference,cfg['sources'],dict(A=baseline,B=baseline))
                proof=dict(utc=time.time(),source_baseline=str(baseline),candidate=str(reference),decisions=decisions,
                    previous_execution_identities={r:execution_identity(cfg['source_evaluations'][r]) for r in 'AB'},
                    reason='Historical A11/B09 evaluation used CPU physics; current candidate uses GPU physics. Old rates alone cannot isolate learning from this execution change.')
                atomic_json(ledger/'source_backend_bridge.json',proof)
                for role in 'AB':
                    state['sources'][role]=decisions[role]['selected_root']
                    state['evaluations'][role]=str(reference/'development' if decisions[role]['use_candidate'] else baseline)
                save('source_backend_comparison_complete',source_backend_bridge_complete=True)
            for spec in cfg['rounds']:
                check();index=spec['index']
                if any(r['index']==index and r['complete'] for r in state['rounds']):continue
                if len(state['rounds'])>=2:
                    release=ledger/'supervised_release.json'
                    save('awaiting_two_cycle_review')
                    # This is a finite persisted barrier, not a claim of agent observation.
                    while not release.exists():
                        check();save('awaiting_two_cycle_review');time.sleep(5)
                    proof=read(release)
                    assert proof['configuration_sha256']==binding and proof['reviewed_rounds']==[r['index'] for r in state['rounds'][:2]]
                    assert proof['two_real_complete_cycles_observed'] is True
                if deadline-time.time()<cfg['minimum_round_seconds']:
                    result='insufficient_remaining_time';break
                if shutil.disk_usage(root).free<cfg['minimum_free_bytes']:
                    result='insufficient_disk_headroom';break
                collection=root/f'new_family_{index:02d}';candidate=root/f'round_{index:02d}'
                source=root/f'collection_source_{index:02d}'
                prepare_once(source/'prepared.json',f'source_{index}',cmd('tools.tight_fresh_collection_source',
                    '--session-root',root,'--data-source',state['data_root'],'--output',source,
                    '--role-source','A='+state['sources']['A'],'--role-source','B='+state['sources']['B'],
                    '--role-evaluation','A='+state['evaluations']['A'],'--role-evaluation','B='+state['evaluations']['B']),collection,candidate)
                prepare_once(collection/'plan.json',f'prepare_{index}',cmd('tools.tight_fresh_expansion',
                    '--source',source,'--session-root',root,'--output',collection,'--registry-id',f'night20261116_r{index}',
                    '--initial-seeds','--revisit-families',*spec['families'],
                    '--partition-manifest',cfg['partition_manifest'],'--frozen-families',cfg['frozen_families'],
                    '--schedule-cycle',*spec['schedules'],'--lanes',4,'--event-seed',spec['event_seed'],
                    '--round',index,'--backend-json',cfg['backend_json'],'--writer-mib',512,'--vision-microbatch',4,
                    '--verification-workers',2,'--verification-max-pending',4,'--capture-audit-memo','--queued-gpu'),collection,candidate)
                if not (collection/'result.json').exists():
                    if (collection/'run_started.json').exists():raise RuntimeError('Interrupted physical collection: preserve and inspect unresolved opportunities')
                    stage(f'collection_{index}',cmd('tools.tight_batch_owner','--output',collection,'--worker-tree',tree,
                        '--startup-seconds',300),collection,candidate)
                audit=collection_admission(dict(paths=dict(collection=collection),plan=read(collection/'plan.json'),
                    registration=dict(collection_plan_sha256=sha256(collection/'plan.json'),deadline=deadline,
                        minimum_accepted=spec['minimum_accepted'],minimum_cycle_remaining_seconds=1200)),time.time())
                atomic_json(ledger/f'collection_{index}_audit.json',audit)
                if not audit['admitted']:raise RuntimeError('Collection admission: '+audit['reason'])
                cycle=cmd('tools.tight_fresh_cycle','--root',source,'--collection',collection,'--output',candidate,
                    '--epochs',spec['epochs'],'--rear-epochs',spec['rear_epochs'],'--count',4,'--seed',28500000,
                    '--latest-repeat',4,'--backend-latest-repeat',4,'--evaluation-lanes',8,'--vision-microbatch',4,
                    '--parallel-frontends','--parallel-backends','--full-trajectory-batch','--continue-rear',
                    '--payload-audit-memo','--release-unused-cuda-cache','--reuse-index',source/'A/cache_index.json',
                    '--offline-anchor-batch',32,'--backend-offline-anchor-batch',32,'--queued-gpu-evaluation',
                    '--grasp-evaluation-ticks',300,'--recheck-training-families','--backend-json',cfg['backend_json'],
                    '--training-sampling-json',cfg['training_sampling_json'],'--frozen-visual-io-json',cfg['frozen_visual_io_json'])
                if spec.get('exercise_resume_after_frontends'):cycle+=['--exercise-resume-after-frontends']
                if cfg.get('backend_phase_weighting_json'):cycle+=['--backend-phase-weighting-json',cfg['backend_phase_weighting_json']]
                completed=candidate.with_name(candidate.name+'_stages')/'finished.json'
                if not completed.exists():
                    previous=candidate.with_name(candidate.name+'_stages')/'registration.json'
                    if previous.exists():
                        if not recoverable_training_failure(candidate):raise RuntimeError('Existing cycle needs explicit diagnosis before resume')
                        cycle+=['--resume']
                    try:stage(f'cycle_{index}',cycle,collection,candidate)
                    except RuntimeError:
                        retry=ledger/f'cycle_{index}_retry.json'
                        if retry.exists() or '--resume' in cycle or not recoverable_training_failure(candidate):raise
                        atomic_json(retry,dict(utc=time.time(),limit=1,
                            reason='Declared completed-DP coordinator recovery probe' if spec.get('exercise_resume_after_frontends') else 'Transient resource guard',
                            recovery='Serial durable-stage resume; original completed stage hashes/commands must match',command=cycle+['--resume']))
                        stage(f'cycle_{index}_retry',cycle+['--resume'],collection,candidate)
                assert read(completed)['software_complete']
                decisions=source_decision(candidate,state['sources'],state['evaluations'])
                row=dict(index=index,complete=True,collection=str(collection),candidate=str(candidate),
                    accepted=audit['accepted'],decisions=decisions,ended=time.time(),
                    training_family_rechecks=[r for r in behavior(candidate/'development',horizon=300)['rows']
                        if r['diagnostic_cohort']=='training_family_recheck'])
                atomic_json(ledger/f'round_{index}_complete.json',row)
                for role in 'AB':
                    if decisions[role]['use_candidate']:
                        state['sources'][role]=str(candidate);state['evaluations'][role]=str(candidate/'development')
                state['data_root']=str(candidate);state['rounds'].append(row);save('round_complete')
                if len(state['rounds'])>=2 and not any(d['use_candidate'] for r in state['rounds'][-2:] for d in r['decisions'].values()):
                    result='two_cycles_no_clear_grasp_gain_requires_new_diagnosis';break
            else:result='registered_rounds_complete'
            save(result,complete=result=='registered_rounds_complete',safe_stop=True)
        except BaseException as error:
            save('safe_failure',complete=False,safe_stop=True,error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            (root/'STOP_MONITOR').touch(exist_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,required=True);args=parser.parse_args()
    run(args.root.resolve(),args.configuration.resolve())
