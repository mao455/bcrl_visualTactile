"""Finite takeover aggregation, using the existing collector and two trainers.

This module coordinates dependencies only. Sensor recording, target semantics,
causal replay, optimization and deployment remain in their existing modules.
The public entry is ``tools.tight_finish dagger_*``.
"""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, proc, kill_owned, supervised_owner


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CODE, text=True, stderr=subprocess.STDOUT, timeout=90).strip()


def takeover_plan(data, families, policy):
    """Training families only; independent event seeds and no outcome selection."""
    import random
    train = set(data['splits']['train'])
    family_ids = {data['entries'][i]['family'] for i in train}
    available = [f for f in families if f['family'] in family_ids and f['split'] == 'train']
    random.Random(policy['plan_seed']).shuffle(available)
    require(bool(available), 'No registered training families')
    jobs = []
    for index in range(policy['attempts_per_group']):
        family = available[index % len(available)]
        round_index=policy.get('round_index',1);group=policy.get('source_group')
        identity = f'dagger_r{round_index:02d}_{group+"_" if group else ""}{index:03d}'
        source = next(i for i in data['splits']['train'] if data['entries'][i]['family'] == family['family'])
        development=index in policy.get('development_indices',[])
        jobs.append(dict(family, identity=identity, source_identity=source, split='train', cohort='dagger_train',
            seed=policy['seed_start'] + index, event_seed=policy['event_seed_start'] + index,
            takeover_tick=policy['takeover_ticks'][index % len(policy['takeover_ticks'])],
            pool_partition='correction_dev' if development else 'train',round_index=round_index,source_group=group))
    return jobs


def pool_mode(cfg):
    return cfg.get('version')=='successful_takeover_pool_v1'


def template_root(root):
    cfg=read(root/'configuration.json')
    return (CODE/cfg['template_root']).resolve() if cfg.get('template_root') else Path(read(root/'registration.json')['predecessor'])


def prepare(root):
    cfg = read(CODE/'configs/tight_state.json')['dagger']
    require(not (root/'configuration.json').exists(), 'DAgger already registered; no overwrite')
    root.mkdir(parents=True, exist_ok=True)
    predecessor = (CODE/cfg['predecessor']).resolve()
    require((predecessor/'launch.json').is_file(), 'Missing actual predecessor launch')
    template=(CODE/cfg.get('template_root',cfg['predecessor'])).resolve()
    data = read(template/'configuration/state_data/data.json')
    parent = Path(read(CODE/cfg['family_source'])['parent'])
    plan = takeover_plan(data, read(parent/'frozen_families.json'), cfg['collection'])
    # The autonomous developer cohort remains the original frozen ten specs.
    evaluation = read(template/'command_residual/evaluation_plan.json')
    require(len(evaluation) == 10, 'Expected existing ten-case development protocol')
    require(not ({j['seed'] for j in plan} & {e['spec']['seed'] for e in evaluation}), 'Collection/development seed overlap')
    write(root/'configuration.json', cfg)
    write(root/'collection_plan.json', plan)
    write(root/'evaluation_plan.json', evaluation)
    write(root/'registration.json', dict(predecessor=str(predecessor), predecessor_launch=read(predecessor/'launch.json'),
        predecessor_source=read(predecessor/'session.json')['source_head'], preparing_worktree=str(CODE),
        predecessor_process=proc(read(predecessor/'launch.json')['pid']),
        configuration_sha256=sha(root/'configuration.json'), collection_plan_sha256=sha(root/'collection_plan.json'),
        evaluation_plan_sha256=sha(root/'evaluation_plan.json'),
        waiting_deadline=read(predecessor/'session.json')['deadline'] + 7200, created=time.time(),
        start_rule='terminal actual predecessor plus exited owner/owned GPU workers; valid selected paired checkpoints',
        new_deadline_rule='starts when the new finite stage is admitted after predecessor exit; never inherits predecessor absolute deadline',
        prepared_source=git('rev-parse','HEAD'),template_root=str(template)))
    write(root/'status.json', dict(state='PREPARED', stage='awaiting_registered_successor', groups={}, gpu_started=False))


def predecessor_finished(root):
    registration = read(root/'registration.json'); parent = Path(registration['predecessor'])
    status = read(parent/'status.json')
    terminal = status.get('state') in ('COMPLETE', 'FAILED', 'BUDGET_STOP')
    owner = proc(registration['predecessor_launch']['pid'])
    original = registration.get('predecessor_process')
    owner_alive = owner is not None and (original is None or owner['start_ticks'] == original['start_ticks'])
    cleanup = read(parent/'owned_cleanup.json') if (parent/'owned_cleanup.json').exists() else {}
    # The owner may still be publishing after setting the terminal result.
    return terminal and status.get('owned_workers_exited') is True and cleanup.get('verified') is True and not owner_alive


def integrate_predecessor(root):
    """Old owner publishes first. Only its scoped report delta may be merged."""
    registration = read(root/'registration.json'); base = registration['predecessor_source']
    require(git('rev-parse','HEAD')==registration['prepared_source'],'Registered successor source changed before admission')
    require(not git('status', '--porcelain'), 'Successor worktree must be committed before integration')
    require(git('remote', 'get-url', 'origin') in ('https://github.com/mao455/bcrl_visualTactile.git',
        'git@github.com:mao455/bcrl_visualTactile.git'), 'Unexpected origin')
    git('fetch', 'origin', 'feature/tight-state-foundation')
    remote = git('rev-parse', 'FETCH_HEAD')
    git('merge-base', '--is-ancestor', base, remote)
    changed = git('diff', '--name-only', base, remote).splitlines()
    prefix = 'research_docs/evidence/tight_transformer_20261110_02/'
    require(all(p == 'research_docs/STATE_DP_RESULTS.md' or p.startswith(prefix) for p in changed),
        'Remote includes unrelated changes; preserve both worktrees for review')
    git('merge', '--no-edit', remote)
    parent = Path(registration['predecessor'])
    published = read(parent/'public_git.json') if (parent/'public_git.json').exists() else {}
    if not published.get('pushed'):
        # A report-only failure does not invalidate usable trained models.
        from tools.tight_diagnostics import matrix_publish
        output = CODE/prefix
        matrix_publish(parent, output)
        git('add', '--', 'research_docs/STATE_DP_RESULTS.md', prefix)
        if git('diff', '--cached', '--name-only'):
            git('commit', '-m', 'Publish predecessor evidence before takeover aggregation')
    git('push', 'origin', 'HEAD:refs/heads/feature/tight-state-foundation')
    write(root/'integration.json', dict(predecessor_remote=remote, successor_source=git('rev-parse', 'HEAD'),
        old_report_pushed=bool(published.get('pushed')), integrated=time.time(), target='feature/tight-state-foundation'))


@supervised_owner
def follow(root):
    with (root/'successor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root/'launch.json').exists(), 'Already launched; never repeat collection or updates')
        registration = read(root/'registration.json')
        require(git('rev-parse','HEAD')==registration['prepared_source'],'Registered successor source differs')
        require(sha(root/'configuration.json') == registration['configuration_sha256'], 'Registered configuration changed')
        require(sha(root/'collection_plan.json') == registration['collection_plan_sha256'], 'Registered attempts changed')
        while not predecessor_finished(root):
            if (root/'STOP').exists():
                write(root/'status.json', dict(state='STOPPED', stage='waiting', gpu_started=False));return
            require(time.time() < registration['waiting_deadline'], 'Predecessor did not exit within bounded waiting period')
            write(root/'status.json', dict(state='WAITING', stage='predecessor_exit_and_publication', pid=os.getpid(),
                dependency=registration['predecessor'], gpu_started=False, utc=time.time()))
            time.sleep(15)
        integrate_predecessor(root)
        run(root)


def selected_pair(root, group):
    cfg = read(root/'configuration.json'); parent = Path(read(root/'registration.json')['predecessor'])
    return validate_selected_pair(cfg['groups'][group],parent,group)


def validate_selected_pair(spec,parent,group):
    """Load an existing complete pair without coercing its observation or target identity."""
    import torch
    frontend = (parent/spec['frontend']).resolve(); backend = (parent/spec['backend']).resolve()
    require(frontend.is_file() and backend.is_file(), 'Missing selected initialization for ' + group)
    front = torch.load(frontend, map_location='cpu', weights_only=False)
    back = torch.load(backend, map_location='cpu', weights_only=False)
    require(front.get('updates', 0) > 0 and back.get('updates', 0) > 0, 'Initialization has no real optimizer updates')
    require(not front.get('disposable') and not back.get('disposable'), 'Disposable model cannot initialize DAgger')
    require(back.get('residual_version') == 'target_residual_transformer60_v1', 'Expected existing Transformer backend')
    require(back['configuration']['base_sha256'] == sha(frontend), 'Selected frontend/backend are not a paired policy')
    require(front['contract'].get('supervision', 'u_expert') == spec['supervision'], 'Frontend supervision identity changed')
    require(front['contract']['modality'] == spec['modality'], 'Frontend observation modality changed')
    data_root = Path(spec['data_root']).resolve()
    require(front['configuration']['statistics_sha256'] == sha(data_root/'statistics.pt'), 'Frontend normalizer changed')
    require(all(bool(torch.isfinite(v).all()) for v in front['ema'].values() if torch.is_floating_point(v)), 'Invalid frontend tensors')
    require(all(bool(torch.isfinite(v).all()) for v in back['model'].values() if torch.is_floating_point(v)), 'Invalid backend tensors')
    return dict(frontend=str(frontend), frontend_sha256=sha(frontend), backend=str(backend), backend_sha256=sha(backend),
        frontend_updates=front['updates'], backend_updates=back['updates'], supervision=spec['supervision'], modality=spec['modality'],
        data_root=str(data_root), statistics_sha256=sha(data_root/'statistics.pt'), scales=back['model']['scales'].tolist(),
        residual_statistics_path=back['configuration']['residual_statistics_path'],
        residual_statistics_sha256=back['configuration']['residual_statistics_sha256'],
        normalization_identity=front['contract'].get('normalization_identity', 'u_expert_train166_affine_v1'),
        strategy_role=spec.get('strategy_role'),geometry=bool(spec.get('geometry',group=='QG01')),
        evaluation_root=spec.get('evaluation_root',str(parent/'round_01'/group/'backend'/group/'evaluation')))


def revise(root):
    """Explicit reviewed scope transition at an exited boundary, in the same run.

    Preserve the prior immutable metadata and all stage/data/model evidence.
    The new source freeze is a recorded revision, never a disabled hash gate or
    a fresh session/deadline. This entry does not launch any worker.
    """
    root=Path(root).resolve();cfg=read(CODE/'configs/tight_state.json')['dagger']
    require(cfg.get('scope_revision')=='complete_strategy_ab_v1','Expected explicit complete-policy revision')
    require(not (root/'revisions'/cfg['scope_revision']).exists(),'Scope revision already recorded')
    require((root/'scope_revision_request.json').exists(),'Missing explicit scope transition request')
    require(not git('status','--porcelain'),'Commit reviewed scope changes before migration')
    require(read(root/'status.json').get('state')=='STOPPED','Scope changes require the saved STOP boundary')
    require(read(root/'status.json').get('owned_workers_exited') is True,'Owned workers have not exited')
    from tools.tight_finish import matrix_worker_absence
    require(matrix_worker_absence(read(root/'worker_processes.json'))[0],'Live owned worker')
    old_session=read(root/'session.json');old_registration=read(root/'registration.json');old_launch=read(root/'launch.json')
    require(sha(root/'session.json')==old_launch['session_sha256'],'Original session receipt changed')
    require(time.time()<old_session['compute_deadline'],'Original compute deadline already passed')
    owner=read(root/'resume.json') if (root/'resume.json').exists() else old_launch
    live=proc(owner['pid']);require(live is None or live['start_ticks']!=owner.get('start_ticks'),'Owner still live')
    old_freeze=read(root/'source_freeze.json')
    import hashlib
    for path,digest in old_freeze.items():
        blob=subprocess.check_output(['git','show',old_session['source_head']+':'+path],cwd=CODE)
        require(hashlib.sha256(blob).hexdigest()==digest,'Prior frozen source is not preserved in git: '+path)
    require(cfg['predecessor']==read(root/'configuration.json')['predecessor'],'Predecessor changed')
    require(cfg['first_round']==2 and cfg['rounds']==3,'Expected finite two remaining round identities')
    require(not (root/'round_02').exists(),'Started future round cannot be re-registered')
    parent=Path(old_registration['predecessor'])
    pairs={g:validate_selected_pair(spec,parent,g) for g,spec in cfg['groups'].items()}
    require({p['strategy_role'] for p in pairs.values()}=={'A','B'},'Exactly two core strategies required')
    require(all(p['modality']=='G00' for p in pairs.values()),'A/B frontend input modalities differ')
    revision=root/'revisions'/cfg['scope_revision'];require(not revision.exists(),'Scope revision already recorded')
    revision.mkdir(parents=True)
    preserved={}
    for name in ('configuration.json','session.json','registration.json','launch.json','source_freeze.json',
                 'status.json','rounds.json','public_git.json','active_stage.json','scope_revision_request.json'):
        if (root/name).exists():shutil.copyfile(root/name,revision/name);preserved[name]=sha(revision/name)
    source=git('rev-parse','HEAD')
    session=dict(old_session,source_head=source,dagger=cfg,scope_revision=cfg['scope_revision'])
    session['owned_worker_roots']=list(dict.fromkeys(old_session['owned_worker_roots']+
        [str(root/f'round_{i:02d}'/g/p) for i in range(cfg['first_round'],cfg['rounds']+1)
         for g in cfg['groups'] for p in ('','dp','dp/training','backend','backend/'+g)]))
    require(all(session[k]==old_session[k] for k in ('started','deadline','compute_deadline')),'Deadline reset forbidden')
    write(root/'configuration.json',cfg);write(root/'session.json',session)
    write(root/'registration.json',dict(old_registration,configuration_sha256=sha(root/'configuration.json'),
        scope_revision=cfg['scope_revision'],revision_source=source,original_started=old_session['started'],original_deadline=old_session['deadline']))
    write(root/'launch.json',dict(old_launch,session_sha256=sha(root/'session.json'),source_head=source,
        scope_revision=cfg['scope_revision'],revision_source=source,original_source_head=old_launch['source_head'],
        original_session_sha256=old_launch['session_sha256']))
    files=git('ls-files','*.py','configs/tight_state.json','visuotactile/run_cpu.sh').splitlines()
    write(root/'source_freeze.json',{p:sha(CODE/p) for p in files})
    receipt=dict(revision=cfg['scope_revision'],same_root=str(root),preserved_metadata=preserved,
        previous_source=old_session['source_head'],source_head=source,started=session['started'],
        deadline=session['deadline'],compute_deadline=session['compute_deadline'],initial_pairs=pairs,
        active_groups=list(cfg['groups']),retired_groups=cfg['retired_groups'],round_schedule=cfg['round_schedule'],
        original_data_and_weights_preserved=True,deadline_reset=False,utc=time.time())
    write(revision/'applied.json',receipt);write(root/'scope_revision.json',receipt)
    write(root/'status.json',dict(read(root/'status.json'),stage='reviewed_scope_revision_ready',source_head=source,
        scope_revision=cfg['scope_revision'],deadline=session['deadline'],utc=time.time()))


class Stages:
    """Owned worker ledger and bounded waits; all compute uses existing entries."""
    def __init__(self, root):
        self.root = root; self.children = []
        self.records = read(root/'worker_processes.json') if (root/'worker_processes.json').exists() else []

    def execute(self, name, args, seconds, completion=None):
        session = read(self.root/'session.json')
        allocation=self.root/'stages'/(name+'_allocation.json')
        limit=read(allocation)['deadline'] if allocation.exists() else min(session.get('compute_deadline',session['deadline']), time.time()+seconds)
        if not allocation.exists():write(allocation,dict(deadline=limit,seconds=seconds,created=time.time()))
        frozen = read(self.root/'source_freeze.json')
        require(all(sha(CODE/p) == digest for p, digest in frozen.items()), 'Frozen successor source changed')
        durable=pool_mode(read(self.root/'configuration.json'))
        receipt=self.root/'stages'/(name+'.json')
        if durable and receipt.exists():
            prior=read(receipt)
            require(prior['args']==args and prior['source_head']==session['source_head'],'Completed stage identity differs')
            if prior.get('completion'):
                require(sha(prior['completion'])==prior['completion_sha256'],'Completed receipt changed')
            return prior
        if durable and completion is not None and Path(completion).exists():
            value=read(completion)
            if value.get('complete',bool(value.get('updates',0))):
                prior=dict(args=args,source_head=session['source_head'],completion=str(completion),
                    completion_sha256=sha(completion),recovered_completed_receipt=True,utc=time.time())
                write(receipt,prior);return prior
        if limit<=time.time():raise TimeoutError('Original stage deadline exhausted: '+name)
        with (self.root/(name+'.log')).open('a') as stream:
            worker = subprocess.Popen(RUN+args, cwd=CODE, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(worker)
        record=dict(pid=worker.pid,start_ticks=None,process_group=worker.pid,phase=name);self.records.append(record)
        try:
            identity = proc(worker.pid)
            if identity is not None:record.update(identity)
            write(self.root/'worker_processes.json', self.records)
            while worker.poll() is None:
                if (self.root/'STOP').exists():
                    # Trainers save at their own complete boundary before the
                    # owner resorts to terminating its child process group.
                    worker_root=Path(args[args.index('--root')+1]) if '--root' in args else None
                    if worker_root is not None:
                        (worker_root/'STOP').touch()
                        if (worker_root/'training').is_dir():(worker_root/'training/STOP').touch()
                    try:worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    raise InterruptedError('DAgger STOP; preserve durable stage state')
                if time.time() >= limit:raise TimeoutError(name+' exceeded finite budget')
                write(self.root/'active_stage.json', dict(name=name, pid=worker.pid, deadline=limit, utc=time.time()))
                time.sleep(2)
            require(worker.returncode == 0, name+' exited '+str(worker.returncode))
        finally:
            kill_owned(worker)
            from tools.tight_finish import matrix_worker_absence
            gone, rows = matrix_worker_absence([self.records[-1]])
            write(self.root/(name+'_reaped.json'), dict(verified=gone, workers=rows))
            require(gone, 'Owned worker exit unverified: '+name)
        if durable:
            record=dict(args=args,source_head=session['source_head'],utc=time.time(),exit_code=0)
            if completion is not None:
                require(Path(completion).is_file(),'Successful stage missing completion receipt')
                record.update(completion=str(completion),completion_sha256=sha(completion))
            write(receipt,record)
            return record

    def close(self):
        errors=[]
        for worker in self.children:
            try:kill_owned(worker)
            except BaseException as error:errors.append(repr(error))
        from tools.tight_finish import matrix_worker_absence
        gone, rows = matrix_worker_absence(self.records)
        write(self.root/'owned_cleanup.json', dict(verified=gone and not errors, workers=rows,errors=errors))
        return gone and not errors


def share_monitor(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('shared_resources.json', 'shared_resources.jsonl'):
        path=destination/name
        if not path.exists() and not path.is_symlink():path.symlink_to(root/name)


@supervised_owner
def collect(root):
    """One persistent Kit, the same finite twelve opportunities for one group."""
    import torch
    from tools.tight_state import environment
    from visuotactile.state_benchmark import trial, dagger_configuration, audit_dagger_capture
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.narrow_data import dagger_record
    from visuotactile.anchor_sampling import supervision_population
    from visuotactile.native_evaluation import exit_completed_kit_worker
    s=read(root/'session.json');cfg=read(root/'configuration.json');pair=read(root/'initial_pair.json')
    plan=read(root/'collection_plan.json');source=read(Path(pair['data_root'])/'data.json')
    require(len(plan)==cfg['collection']['attempts_per_group'] and len({p['identity'] for p in plan})==len(plan),
        'Finite unique collection plan differs from registered budget')
    resumable=pool_mode(cfg)
    if resumable:
        registered=read(root/'collector_registration.json')
        require(all(sha(root/name)==digest for name,digest in registered.items()),'Frozen collection pair/plan/configuration changed')
    if not resumable:require(not (root/'attempts.jsonl').exists(), 'Collection already attempted; no implicit replay')
    if resumable and (root/'collection_finished.json').exists() and read(root/'collection_finished.json').get('complete'):
        return read(root/'collection_finished.json')
    for spec in plan:dagger_configuration(cfg['bridge'],spec,source)
    for key in ('frontend','backend'):
        require(sha(pair[key])==pair[key+'_sha256'], 'Frozen collector initialization drift: '+key)
    require(sha(s['expert_checkpoint'])==s['expert_sha256'], 'Frozen collector expert drift')
    saved=torch.load(pair['backend'],map_location='cpu',weights_only=False)
    results=read(root/'collection_results.json') if resumable and (root/'collection_results.json').exists() else []
    entries=read(root/'correction_entries.json') if resumable and (root/'correction_entries.json').exists() else {}
    if resumable and (root/'attempts.jsonl').exists():
        started_rows=[json.loads(line) for line in (root/'attempts.jsonl').read_text().splitlines() if line.strip()]
        present={r['identity'] for r in results}
        for attempt in started_rows:
            if attempt['identity'] not in present:
                # Never repeat a physically started opportunity after a crash.
                results.append(dict(attempt,result=dict(software_complete=False,success=False),accepted=False,
                    error='interrupted_started_attempt',exclusion_reason='uncommitted outcome retained as software interruption'))
        write(root/'collection_results.json',results)
    started=time.time();row=None;app=None;collection_stop='registered_attempts_resolved'
    finished_ids={r['identity'] for r in results}
    gate_count=cfg['collection'].get('gate_attempts',2)
    require(0<gate_count<=len(plan), 'First takeover gate must be inside the finite attempt budget')
    try:
        app,env,player,lease=environment(root,evaluation=True,camera_content_validation=True)
        # Evaluation construction avoids the old speed wrapper; the bridge owns
        # the one expert conversion. Every wait still consumes collection time.
        lease.deadline=min(s['deadline'],s['collection_deadline'])
        for index,spec in enumerate(plan):
            if spec['identity'] in finished_ids:continue
            row=None
            if cfg['collection'].get('soft_stop_reserve_seconds') and time.time()+cfg['collection']['soft_stop_reserve_seconds']>=s['collection_deadline']:
                collection_stop='finite_collection_window_preserves_training';break
            require(time.time()<s['collection_deadline'], 'Finite collection deadline')
            require(shutil.disk_usage(root).free>=s['reserve_bytes'], 'Disk reserve before new collection attempt')
            out=root/'episodes'/spec['identity'];require(not out.exists(),'Attempt already exists; never retry')
            attempt=dict(identity=spec['identity'],index=index,started=time.time(),seed=spec['seed'],
                event_seed=spec['event_seed'],takeover_tick=spec['takeover_tick'],family=spec['family'])
            with (root/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(attempt)+'\n');stream.flush();os.fsync(stream.fileno())
            row=dict(attempt,result=dict(software_complete=False,success=False),accepted=False)
            result=trial(env,lease,out,spec,source,player,pair['backend'],
                saved=saved,position=True,video=index<2,expert_checkpoint=s['expert_checkpoint'],capture_dataset=True,
                capture_compression='gzip4_lossless_v1',dagger=cfg['bridge'])
            row['result']=result
            if result.get('legal') and result.get('software_complete'):
                if cfg.get('scope_revision'):
                    acceptance_path=root/'complete_policy_acceptance.json'
                    acceptance=read(acceptance_path) if acceptance_path.exists() else dict(attempts=[],passed=False)
                    if len(acceptance['attempts'])<gate_count:
                        acceptance['attempts'].append(audit_dagger_capture(out,pair))
                        acceptance['passed']=len(acceptance['attempts'])>=gate_count and all(x['passed'] for x in acceptance['attempts'])
                        write(acceptance_path,acceptance)
                path=Path(result['dataset_capture']['path']);takeover=result['dagger']
                require(takeover['provenance_passed'] and takeover['position_executor_continuous'], 'Takeover provenance failed')
                row['exposure']=dict(episodes=1,history_ticks=result['steps'],supervised_ticks=0,
                    supervised_anchors=0,supervised_window_targets=0)
                # The strict training reader intentionally rejects empty masks.
                # Early student success/failure and switch-flush termination are
                # valid attempts, but have no eligible expert supervision.
                if takeover['expert_valid_ticks']>0:
                    entry=dict(path=str(path),episode=spec['identity'],length=result['steps'],family=spec['family'],split='train',
                        source_sha256=sha(path),trajectory_kind='expert_takeover_v1',
                        source_group=spec.get('source_group'),round_index=spec.get('round_index',1),
                        pool_partition=spec.get('pool_partition','train'),seed=spec['seed'],takeover_tick=spec['takeover_tick'],
                        frontend_sha256=pair['frontend_sha256'],backend_sha256=pair['backend_sha256'])
                    record,_,_=dagger_record(entry,include_geometry=False)
                    exposure=supervision_population({spec['identity']:record});row['exposure']=exposure
                    require(exposure['supervised_ticks']==takeover['expert_valid_ticks'], 'Recorded expert denominator differs')
                    if cfg['collection'].get('align_clean_labels_to_commit'):
                        first=int(torch.nonzero(record['supervision_mask'],as_tuple=False)[0])
                        anchors=[t for t in range(0,len(record['time']),12) if bool(record['supervision_mask'][t])]
                        row['supervision_timing']=dict(takeover_tick=spec['takeover_tick'],first_valid_expert_tick=first,
                            first_dp_anchor=anchors[0] if anchors else None,first_backend_tick=first)
                        expected=spec['takeover_tick']+cfg['collection']['expected_expert_delay_queue']-1
                        require(first==expected and first%12==0 and anchors and anchors[0]==first,
                            'Clean expert supervision/DP phase alignment differs from registered queue contract')
                    row['label_contract_checked']=True
                    if result['success'] and takeover['expert_takeover_success'] and exposure['supervised_anchors']>0:
                        visual=inspect_capture(path)
                        require(visual['passed'],'Successful correction has invalid sensor content')
                        entries[spec['identity']]=entry;row['accepted']=True
                else:row['label_contract_checked']=False;row['exclusion_reason']='no clean executed expert labels'
            else:row['exclusion_reason']='illegal initial state' if result.get('legal') is False else 'incomplete runtime'
            results.append(row)
            write(root/'collection_results.json',results);write(root/'correction_entries.json',entries)
            write(root/'collection_progress.json',dict(completed=len(results),accepted=len(entries),utc=time.time()))
            row=None
            # Natural early termination is an experiment outcome, not an
            # interface failure. Accumulate the first qualified takeovers from
            # this same fixed plan; never add or replace an opportunity.
            write(root/'collection_gate.json',collection_summary(results,cfg['collection'])['physical_gate'])
            if cfg.get('scope_revision'):write(root/'recovery_coverage.json',recovery_coverage(results,pair))
        summary=collection_summary(results,cfg['collection'])
        summary.update(wall_s=time.time()-started,complete=True,stop_reason=collection_stop,
            all_registered_attempts_run=len(results)==len(plan))
        if cfg.get('scope_revision'):
            require((root/'complete_policy_acceptance.json').exists() and read(root/'complete_policy_acceptance.json')['passed'],
                'No real complete-policy collection acceptance; preserve data and stop')
        write(root/'collection_finished.json',summary)
        write(root/'result.json',dict(software_complete=True,stage='dagger_collection_complete',worker_error=None))
        exit_completed_kit_worker(root)
    except BaseException as error:
        # Preserve the real started-attempt denominator even if trial, sensor
        # admission or compact readback failed before the ordinary append.
        if row is not None:
            row.update(accepted=False,error=repr(error),traceback=traceback.format_exc())
            entries.pop(row['identity'],None)
            if not results or results[-1]['identity']!=row['identity']:results.append(row)
        write(root/'collection_results.json',results);write(root/'correction_entries.json',entries)
        summary=collection_summary(results,cfg['collection'])
        summary.update(wall_s=time.time()-started,complete=False,sufficient=False,error=repr(error))
        write(root/'collection_gate.json',summary['physical_gate'])
        write(root/'collection_finished.json',summary)
        write(root/'worker_error.json',dict(error=repr(error),traceback=traceback.format_exc(),attempts=len(results)))
        raise


def recovery_coverage(rows,pair):
    """Observed takeover coverage only; no new controller inputs or success labels."""
    from collections import Counter
    covers=[dict(identity=r['identity'],accepted=bool(r.get('accepted')),
        seed=r.get('seed'),family=r.get('family'),takeover_tick=r.get('takeover_tick'),
        **r.get('result',{}).get('dagger',{}).get('recovery_coverage',{})) for r in rows]
    return dict(source_frontend_sha256=pair['frontend_sha256'],source_backend_sha256=pair['backend_sha256'],
        strategy_role=pair.get('strategy_role'),attempts=len(rows),accepted=sum(x['accepted'] for x in covers),
        unique_seeds=len({r.get('seed') for r in rows}),unique_training_families=len({r.get('family') for r in rows}),
        independent_observed_takeover_starts=len({r['switch_observable_sha256'] for r in covers if r.get('switch_observable_sha256')}),
        phase_counts=dict(Counter(r.get('phase','unknown') for r in covers)),
        observed_deviation_counts=dict(Counter((r.get('deviation') or {}).get('type','unknown') for r in covers)),
        accepted_population={key:sum((r.get('population') or {}).get(key,0) for r in covers if r['accepted'])
            for key in ('valid_expert_ticks','dp_anchors')},
        accepted_recovery_prefix={str(size):{key:sum(((r.get('population') or {}).get('first'+str(size)) or {}).get(key,0)
            for r in covers if r['accepted']) for key in ('valid_expert_ticks','dp_anchors','valid_h24_window_targets')}
            for size in (24,48)},
        rows=covers,scope='Observed switch states and legal recovery-prefix population; stages/deviations are diagnostics, not established causal failure types')


def collection_summary(rows, policy):
    accepted=[r for r in rows if r.get('accepted')]
    ticks=sum(r['exposure']['supervised_ticks'] for r in accepted)
    anchors=sum(r['exposure']['supervised_anchors'] for r in accepted)
    actual=[r['result'].get('dagger',{}) for r in rows]
    gate_count=policy.get('gate_attempts',2)
    qualified=[r for r in rows if r['result'].get('legal') and r['result'].get('software_complete')
        and not r.get('error') and r.get('label_contract_checked')
        and r['result'].get('dagger',{}).get('takeover_triggered')
        and r['result'].get('dagger',{}).get('expert_valid_ticks',0)>0
        and r['result'].get('dagger',{}).get('provenance_passed')
        and r['result'].get('dagger',{}).get('position_executor_continuous')]
    gate=dict(passed=len(qualified)>=gate_count,required_attempts=gate_count,qualified_attempts=len(qualified),
        attempts=[dict(identity=r['identity'],actual_takeover=True,label_contract_checked=True) for r in qualified[:gate_count]],
        counts_inside_budget=True,success_required=False,
        rule='First required number of qualified actual takeovers within the fixed attempt plan; execution provenance, nonempty clean labels and compact readback')
    reasons={}
    for row in rows:
        if row.get('accepted'):continue
        result=row['result']
        if row.get('error'):reason='software_or_admission_error'
        elif result.get('dagger',{}).get('student_autonomous_success'):reason='student_autonomous_success'
        else:reason=row.get('exclusion_reason') or next((k for k,v in result.get('reasons',{}).items()
            if v and k!='max_successes'),'no eligible DP supervision')
        reasons[reason]=reasons.get(reason,0)+1
    return dict(attempts=len(rows),attempt_budget=policy['attempts_per_group'],unstarted=policy['attempts_per_group']-len(rows),
        legal=sum(bool(r['result'].get('legal')) for r in rows),
        rejected=sum(r['result'].get('legal') is False for r in rows),
        software_complete=sum(bool(r['result'].get('software_complete')) for r in rows),
        software_or_admission_errors=sum(bool(r.get('error')) or not r['result'].get('software_complete',False) for r in rows),
        mixed_successes=sum(bool(r['result'].get('success')) for r in rows),
        takeover_triggers=sum(bool(v.get('takeover_triggered')) for v in actual),
        expert_takeover_successes=sum(bool(v.get('expert_takeover_success')) for v in actual),
        expert_takeover_failures=sum(bool(r['result'].get('dagger',{}).get('takeover_triggered')) and not r['result'].get('success',False) for r in rows),
        expert_takeover_timeouts=sum(bool(r['result'].get('dagger',{}).get('takeover_triggered')) and bool(r['result'].get('reasons',{}).get('timeout')) for r in rows),
        student_control_ticks=sum(v.get('student_ticks',0) for v in actual),expert_control_ticks=sum(v.get('expert_ticks',0) for v in actual),
        expert_shadow_queries=sum(v.get('expert_shadow_queries',0) for v in actual),
        unknown_interaction_attempts=sum(not r['result'].get('software_complete',False) for r in rows),
        accepted_episodes=len(accepted),effective_expert_ticks=ticks,effective_dp_anchors=anchors,
        selection_fraction=len(accepted)/len(rows) if rows else None,exclusion_reasons=reasons,
        physical_gate=gate,
        insufficient_interface_evidence=None if gate['passed'] else 'Not enough actual takeovers with verified nonempty clean expert labels within the fixed attempt budget',
        sufficient=gate['passed'] and len(accepted)>=policy['minimum_successful_corrections'] and ticks>=policy['minimum_expert_ticks'] and anchors>=policy['minimum_dp_anchors'],
        student_autonomous_successes=sum(bool(v.get('student_autonomous_success')) for v in actual),
        selection='successful expert-executed corrections only; student prefix history retained with zero supervision')


def discard_incomplete_metadata(directory, allowed_names):
    """Only regenerate this stage's uncommitted derived files; never follow links."""
    directory=Path(directory)
    rows=list(directory.rglob('*'))
    require(all(p.is_dir() and not p.is_symlink() or p.name in allowed_names for p in rows),
        'Unexpected payload in incomplete preparation; preserve it for review')
    removed=[]
    for path in sorted(rows,key=lambda p:len(p.parts),reverse=True):
        if path.is_symlink() or path.is_file():
            removed.append(str(path));path.unlink()
        else:path.rmdir()
    directory.rmdir()
    return removed


def prepare_frontend(root, group, round_index=1, snapshot=None, reuse_index=None):
    import torch
    from visuotactile.narrow_data import build_dagger_aggregate
    from visuotactile.anchor_sampling import AnchorSampler, supervision_population
    cfg=read(root/'configuration.json');parent=template_root(root)
    job=root/f'round_{round_index:02d}'/group;pair=read(job/'initial_pair.json');source=Path(pair['data_root']).parent
    out=job/'dp'
    if (out/'prepared.json').exists():
        prior=read(out/'prepared.json')
        require(prior['pair']==pair and prior['snapshot_sha256']==(sha(snapshot) if snapshot else None),'Frontend preparation identity differs')
        require(all(sha(out/p)==v for p,v in prior['files'].items()),'Prepared frontend immutable data changed')
        return out
    out.mkdir(exist_ok=True);(out/'training').mkdir(exist_ok=True)
    if snapshot and (out/'state_data').exists() and not (out/'state_data/aggregation_complete.json').exists():
        removed=discard_incomplete_metadata(out/'state_data',{'data.json','statistics.pt','tactile_statistics.pt',
            'numeric.pt','tactile.pt','loss_context.pt'})
        write(out/'preparation_recovery.json',dict(regenerated_uncommitted_derived_files=removed,utc=time.time()))
    base=read(source/'configuration.json');session=read(source/'session.json')
    entries=read(job/'correction_entries.json')
    manifest=read(source/'manifest.json')
    if snapshot:
        frozen=read(snapshot);entries={i:e for i,e in frozen['entries'].items()}
    manifest=dict(manifest,dagger_round=round_index,previous_manifest_sha256=sha(source/'manifest.json'),
        episodes=manifest['episodes']+[dict(id=i,**e,sha256=e['source_sha256']) for i,e in entries.items()])
    write(out/'manifest.json',manifest)
    data=build_dagger_aggregate(source/'state_data',{} if snapshot else entries,out/'state_data',pair['supervision'],
        manifest_sha256=sha(out/'manifest.json'),include_geometry=bool(cfg['groups'][group].get('geometry',group=='QG01')),
        snapshot=snapshot,consumer_group=group)
    raw=torch.load(out/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    records={i:raw['records'][i] for i in data['splits']['train']};exposure=supervision_population(records)
    count=len(records);mean_batch=64 if data.get('sampling_strata') else count/math.ceil(count/64);policy=cfg['frontend']
    target=policy.get('maximum_updates') or math.ceil(policy['maximum_anchor_epochs']*exposure['supervised_anchors']/mean_batch)
    minimum=policy.get('minimum_updates') or math.ceil(policy['minimum_anchor_epochs']*exposure['supervised_anchors']/mean_batch)
    interval=policy.get('validation_updates') or max(1,math.ceil(policy['validation_anchor_epochs']*exposure['supervised_anchors']/mean_batch))
    modality=pair['modality'];profile={}
    if cfg['groups'][group].get('geometry',group=='QG01'):profile=read(parent/'configuration/configuration.json')['vision_pair']['geometry_profiles']['QG00']
    base['vision_pair']=dict(base['vision_pair'],groups=[group],modalities={group:modality},warm_starts={},
        continuation_from={group:dict(path=pair['frontend'],sha256=pair['frontend_sha256'])})
    base['dagger']=cfg;base['resources']=cfg['resources']
    session.update(version='expert_takeover_aggregate_v1',vision_pair=base['vision_pair'],dagger_round=round_index,
        source_head=read(root/'session.json')['source_head'],resource_policy='full_gpu_v1',resources=cfg['resources'],
        resource_root=str(out),deadline=read(root/'session.json')['deadline'],training_deadline=read(root/'session.json')['deadline'],
        cache_deadline=read(root/'session.json')['deadline'],reserve_bytes=cfg['resources']['min_disk_gib']*1024**3,
        owned_worker_roots=[str(root)],dino_reuse_index=str(reuse_index or source/'cache_index.json'))
    write(out/'configuration.json',base);write(out/'session.json',session);write(out/'evaluation_plan.json',read(root/'evaluation_plan.json'))
    write(out/'COMPLETE.json',dict(dataset_prepared=True,dagger_round=round_index,exposure=exposure,test22_used=False))
    continuation=dict(path=pair['frontend'],sha256=pair['frontend_sha256'],teacher_supervision=pair['supervision'],
        statistics_sha256=pair['statistics_sha256'],normalization_identity=pair['normalization_identity'],round_index=round_index,source_group=group)
    runtime=dict(resource_policy='full_gpu_v1',external_monitor_root=str(out),modality=modality,random_seed=cfg['training_seed']+round_index-1,
        training_deadline=session['deadline'],continuation_from=continuation,microbatch=4,
        validation_interval=interval,early_stop_minimum_update=minimum,
        comparison_updates=sorted({min(target,interval*k) for k in range(1,5)}),deterministic_validation_loss=True,
        incremental_schedule=dict(maximum_updates=target,minimum_updates=minimum,warmup_updates=100,
            lr_peak=policy['learning_rate'],lr_floor=policy['learning_rate']/10,validation_interval=interval))
    if profile:runtime['hand_geometry']=profile
    if snapshot:runtime['sampling_weights']=cfg['sampling_weights']
    if cfg.get('recovery_prefix'):runtime['recovery_prefix']=cfg['recovery_prefix']
    write(out/'training/session.json',runtime)
    budget=dict(total_updates=target,anchor_population=exposure['supervised_anchors'],mean_batch=mean_batch,
        target_anchor_epochs=target*mean_batch/exposure['supervised_anchors'],early_stop_minimum_update=minimum,
        budget_basis='fixed_incremental_updates' if policy.get('maximum_updates') else 'anchor_exposure',
        validation_interval=interval,comparison_updates=runtime['comparison_updates'],exposure=exposure)
    write(out/'budget.json',budget);write(out/'training/budget.json',budget);share_monitor(root,out)
    write(out/'prepared.json',dict(pair=pair,snapshot_sha256=sha(snapshot) if snapshot else None,
        files={p:sha(out/p) for p in ('manifest.json','state_data/data.json','state_data/numeric.pt','state_data/statistics.pt','budget.json')}))
    return out


def cache_visual(root):
    from visuotactile.native_data import cache_visual_pair
    cache_visual_pair(root,reuse_index=read(root/'session.json')['dino_reuse_index'])


def check(root):
    """Changed-path real batch: historical data plus an actual correction suffix."""
    import torch
    import h5py
    from visuotactile.state_history import load_image_continuation
    from visuotactile.position_control import image_contract
    from visuotactile.native_data import visual_records
    from visuotactile.anchor_sampling import anchor_batch, mask_totals, weighted_parts
    from visuotactile.state_image_loss import GeometryObjective,PALM_LOSS_ID
    from visuotactile.checkpointing import save,training_state,restore_training
    from visuotactile.models.vendor.ema_model import EMAModel
    from visuotactile.state_resources import lease_for
    data=read(root/'state_data/data.json');runtime=read(root/'training/session.json');s=read(root/'session.json')
    source=runtime['continuation_from'];saved=torch.load(source['path'],map_location='cpu',weights_only=False)
    stats=torch.load(root/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    if runtime['modality']=='G10':stats.update(torch.load(root/'state_data/tactile_statistics.pt',map_location='cpu',weights_only=False))
    contract=image_contract('I-HistoryDP',24,runtime['modality'],data['supervision'])
    model,audit=load_image_continuation(saved,contract,stats);del saved
    current=data.get('sampling_strata',{}).get('current_correction',[])
    new_ids=current or [i for i in data['splits']['train'] if data['entries'][i].get('trajectory_kind')=='expert_takeover_v1']
    require(bool(new_ids),'No correction to validate')
    ids=[data['splits']['train'][0],new_ids[0]];records=visual_records(root/'state_data',ids,runtime['modality'])
    selected=[(i,next(t for t in range(0,len(records[i]['time']),12) if records[i]['supervision_mask'][t])) for i in ids]
    lease=lease_for(root,deadline=s['deadline'])
    with h5py.File(data['entries'][ids[0]]['path']) as f:metadata=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(metadata,mode=PALM_LOSS_ID,commit=12);objective.weights=s['loss_weights']
    with lease.block():
        model.cuda().train();records={i:{k:v.cuda() if torch.is_tensor(v) else v for k,v in r.items()} for i,r in records.items()};averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6);generator=torch.Generator().manual_seed(7091);noise=torch.Generator(device='cuda').manual_seed(7092)
        condition,target,mask=anchor_batch(model,records,selected)
        _,parts,_=objective.forward(model,condition,target,mask,noise)
        totals=mask_totals({i:len(records[i]['time']) for i in ids},selected,12,records=records)
        loss,_=weighted_parts(parts,mask,totals,objective.weights,12)
        require(torch.isfinite(loss),'Nonfinite correction forward');loss.backward()
        gradients={name:sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None)
            for name in ('robot_encoder','vision','native_input','native_gru','visual_gru','condition_projection','denoiser')+
                (('tactile_history','tactile_fusion') if runtime['modality']=='G10' else ())}
        require(all(math.isfinite(v) and v>0 for v in gradients.values()),'Frontend trainable component did not receive gradient')
        optimizer.step();ema.step(model)
        checkpoint=training_state(model,averaged,optimizer,ema,generator,noise,group=model.group,updates=1,configuration={'disposable':True})
        path=root/'dagger_check.pt';save(path,checkpoint);restored=torch.load(path,map_location='cpu',weights_only=False)
        with torch.no_grad():next(model.parameters()).add_(.125)
        generator.manual_seed(1);noise.manual_seed(2)
        restore_training(restored,model,averaged,optimizer,ema,generator,noise)
        def same(a,b):
            if torch.is_tensor(a):return torch.equal(a.detach().cpu(),b.detach().cpu())
            if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            return a==b
        restore_checks=dict(model=same(model.state_dict(),restored['model']),ema=same(averaged.state_dict(),restored['ema']),
            optimizer=same(optimizer.state_dict(),restored['optimizer']),schedule_rng=same(generator.get_state(),restored['schedule_rng']),
            noise_rng=same(noise.get_state(),restored['noise_rng']),torch_rng=same(torch.get_rng_state(),restored['torch_rng']),
            cuda_rng=same(torch.cuda.get_rng_state_all(),restored['cuda_rng']),ema_step=ema.optimization_step==restored['ema_optimization_step'])
        exact=all(restore_checks.values());require(exact,'Correction trainer save/restore differs')
        with torch.no_grad():prediction=averaged.predict(condition.detach(),torch.Generator(device='cuda').manual_seed(7091))
        require(prediction.shape==(2,24,29) and torch.isfinite(prediction).all(),'Full DDIM correction check failed')
    path.unlink()
    write(root/'vision_acceptance.json',dict(passed=True,data_sha256=sha(root/'state_data/data.json'),
        statistics_sha256=sha(root/'state_data/statistics.pt'),source_sha256=source['sha256'],continuation=audit,
        real_original_and_correction_ids=ids,selected=selected,valid_supervised_targets=int(mask.sum()),loss=float(loss),
        correction_scope='current_round' if current else 'historical_correction_no_current_training_sample',
        gradients=gradients,save_restore_exact=exact,restore_checks=restore_checks,disposable_updates=1,formal_updates=0,test22_used=False))


def prepare_backend(root, group, round_index=1):
    from tools.tight_finish import residual_prepare
    job=root/f'round_{round_index:02d}'/group;front=job/'dp';pair=read(job/'initial_pair.json');cfg=read(root/'configuration.json')
    out=job/'backend'
    if (out/'prepared.json').exists():
        require(read(out/'prepared.json')['frontend_sha256']==sha(front/'training/best.pt'),'Prepared backend frontend changed')
        return out
    if out.exists():
        removed=discard_incomplete_metadata(out,{'state_data','configuration.json','session.json','base_identities.json',
            'evaluation_plan.json','budget.json','status.json'})
        write(job/'backend_preparation_recovery.json',dict(regenerated_uncommitted_metadata=removed,utc=time.time()))
    configuration=read(front/'configuration.json')
    parent=template_root(root)
    stage='command_residual' if pair['supervision']=='u_expert' else 'configuration_residual'
    configuration['residual_pair']=copy.deepcopy(read(parent/stage/'configuration.json')['residual_pair'])
    v=configuration['residual_pair'];selected=front/'training/best.pt'
    data=read(front/'state_data/data.json');normalizer=job/'command_statistics';normalizer.mkdir(exist_ok=True)
    shutil.copyfile(pair['residual_statistics_path'],normalizer/'statistics.pt')
    write(normalizer/'data.json',dict(data,supervision='u_expert',normalization_identity='u_expert_train166_affine_v1'))
    v.update(source_root=str(front),groups=[group],dagger_round=round_index,
        bases={group:dict(group=pair['modality'],modality=pair['modality'],checkpoint=str(selected),sha256=sha(selected))},
        residual_statistics_path=str(normalizer/'statistics.pt'),fixed_residual_scales=pair['scales'],
        continuation_from=dict(path=pair['backend'],sha256=pair['backend_sha256'],teacher_supervision='u_expert',
            statistics_sha256=pair['residual_statistics_sha256'],normalization_identity='u_expert_train166_affine_v1',round_index=round_index,source_group=group),
        maximum_control_tick_epochs=cfg['backend']['maximum_control_tick_epochs'],
        minimum_control_tick_epochs=cfg['backend']['minimum_control_tick_epochs'],
        comparison_control_tick_epochs=list(range(cfg['backend'].get('validation_epoch_interval',5),cfg['backend']['maximum_control_tick_epochs']+1,cfg['backend'].get('validation_epoch_interval',5))),
        validation_epoch_interval=cfg['backend'].get('validation_epoch_interval',5),learning_rate=cfg['backend']['learning_rate'],
        stop_rule='registered finite incremental exposure, validation trend and absolute wall budget; update0 included',
        cache_definition='current paired frozen DP; complete actual prefixes, separate previous_nominal, expert supervision mask',
        training_hours=1.5,maximum_hours=4,cache_hours=1.5,offline_hours=.5,evaluation_hours=1)
    if pool_mode(cfg):
        v['sampling_weights']=cfg['sampling_weights']
        if cfg.get('recovery_prefix'):v['recovery_prefix']=cfg['recovery_prefix']
        v.update({key:cfg['resources'][key] for key in ('min_gpu_free_mib','min_ram_gib','min_disk_gib')})
        v['dagger_evaluation']=dict(initial_trials=cfg['evaluation']['attempts_per_group'],max_trials=cfg['evaluation']['maximum_attempts_per_group'],
            expand_if_success=1,expand_if_subgoal=1,expand_if_native_lift=2)
    residual_prepare(out,configuration=configuration)
    share_monitor(root,out)
    write(out/'prepared.json',dict(frontend_sha256=sha(selected),snapshot_id=data.get('snapshot_id')))
    return out


def phase_deadlines(root, directory, **values):
    global_session=read(root/'session.json');limit=global_session.get('compute_deadline',global_session['deadline']);s=read(directory/'session.json')
    allocations=read(directory/'phase_allocations.json') if (directory/'phase_allocations.json').exists() else {}
    for key,seconds in values.items():
        if key not in allocations:allocations[key]=min(limit,time.time()+seconds)
        s[key]=allocations[key]
    write(directory/'phase_allocations.json',allocations);s['deadline']=limit
    write(directory/'session.json',s)
    for g in s.get('residual_pair',{}).get('groups',[]):
        if (directory/g/'session.json').exists():write(directory/g/'session.json',dict(s,resource_root=str(directory),modality=g))
    return s


def round_decision(cfg, previous, remaining_seconds, round_index=None):
    """Bounded, declared metric rules; no learned or privileged trigger."""
    d=cfg['decisions'];clean=d['default_clean_ticks'];reason='registered broad initial recovery'
    if previous:
        completed=sum(v.get('collection',{}).get('attempts',0) for v in previous.values())
        accepted=sum(v.get('collection',{}).get('accepted',0) for v in previous.values())
        online=[v.get('selection',{}).get('candidate_metrics',{}) for v in previous.values()]
        if completed and accepted/completed<d['low_takeover_success_fraction']:
            clean=d['earlier_clean_ticks'];reason='measured expert recovery below .5; shift to earlier reachable states'
        elif any(v.get('successes',0) or v.get('subgoals',0) or v.get('lifts',0)>=2 for v in online):
            clean=d['later_clean_ticks'];reason='measured student stage progress; extend student prefixes'
        else:
            clean=d['broader_clean_ticks'];reason='no measured closed-loop stage gain; broaden early recovery coverage'
    attempts=d['default_attempts_per_group'] if remaining_seconds>=5*3600 else d['minimum_attempts_per_group']
    result=dict(attempts_per_group=attempts,clean_ticks=clean,takeover_ticks=[t-cfg['collection']['expected_expert_delay_queue']+1 for t in clean],
        reason=reason,remaining_seconds=remaining_seconds,weights=cfg['sampling_weights'],zero_student_success_is_stop=False)
    if cfg.get('scope_revision'):
        require(round_index is not None,'Scope schedule needs explicit round identity')
        schedule=cfg['round_schedule'][str(round_index)];active=list(schedule['active_groups']);p=cfg['planning']
        future=[v for k,v in cfg['round_schedule'].items() if int(k)>round_index]
        reserve=(len(active)+sum(len(v['active_groups']) for v in future))*p['pair_stage_reserve_seconds']
        reserve+=sum(len(v['active_groups']) for v in future)*p['reserve_future_minimum_attempts']*p['seconds_per_attempt']
        available=remaining_seconds-reserve-p['transition_reserve_seconds'];counts={}
        for i,g in enumerate(active):
            later=sum(schedule['attempts'][other] for other in active[i+1:])*p['seconds_per_attempt']
            counts[g]=max(0,min(schedule['attempts'][g],int((available-later)/p['seconds_per_attempt'])))
            available-=counts[g]*p['seconds_per_attempt']
        require(all(counts[g]>=p['minimum_admitted_attempts'] for g in active),
            'Insufficient original remaining budget for scheduled collection and complete paired updates')
        result.update(scope_revision=cfg['scope_revision'],active_groups=active,attempts_by_group=counts,
            requested_attempts=schedule['attempts'],reserved_compute_seconds=reserve,
            unequal_exposure_exploration=True,not_scheduled=[g for g in cfg['groups'] if g not in active])
    return result


def policy_metrics(directory, plan, checkpoint):
    rows={}
    for item in plan:
        spec=item.get('spec',item);path=Path(directory)/spec['identity']/'result.json'
        if path.exists():
            value=read(path)
            if value.get('software_complete') and value.get('legal') and value.get('checkpoint_sha256')==checkpoint and value.get('seed')==spec['seed']:
                rows[spec['identity']]=value
    return rows


def paired_policy_selection(previous, candidate, old_rows, new_rows, validation_ok):
    """Compare only the same observed developer specs; rollback the whole pair."""
    shared=sorted(set(old_rows)&set(new_rows))
    def metrics(rows):
        values=[rows[i] for i in shared]
        return dict(valid=len(values),successes=sum(bool(v.get('success')) for v in values),
            subgoals=sum(int(v.get('native_subgoals',0)) for v in values),lifts=sum(bool(v.get('native_lift')) for v in values),
            falls=sum(bool(v.get('reasons',{}).get('fall')) for v in values),
            dropped=sum(bool(v.get('reasons',{}).get('dropped')) for v in values),
            timeouts=sum(bool(v.get('reasons',{}).get('timeout')) for v in values))
    old=metrics(old_rows);new=metrics(new_rows)
    def rank(m):return (m['successes'],m['subgoals'],m['lifts'],-m['falls'],-m['dropped'])
    accepted=len(shared)>=4 and validation_ok and rank(new)>=rank(old)
    return dict(selected_pair=candidate if accepted else previous,selected='candidate' if accepted else 'previous_complete_pair',
        candidate_pair=candidate,previous_pair=previous,matching_ids=shared,previous_metrics=old,candidate_metrics=new,
        validation_acceptable=validation_ok,reason='paired stage ranking then recovery-aware offline selection; ties may retain adapted pair' if accepted else
        'rollback complete pair: missing paired evidence, worse stage outcomes or unacceptable validation',
        scope='fixed development specs and seeds; physical hidden state is not claimed bitwise paired; multiple method differences')


def candidate_pair(job):
    import torch
    pair=read(job/'initial_pair.json');group=job.name
    front=job/'dp/training/best.pt';back=job/'backend'/group/'training/best.pt'
    a=torch.load(front,map_location='cpu',weights_only=False);b=torch.load(back,map_location='cpu',weights_only=False)
    require(b['configuration']['base_sha256']==sha(front),'Candidate pair frontend binding differs')
    require(a['contract'].get('supervision','u_expert')==pair['supervision'],'Candidate frontend supervision differs')
    require(a['configuration']['statistics_sha256']==pair['statistics_sha256'],'Candidate frontend normalizer differs')
    require(b['configuration']['residual_statistics_sha256']==pair['residual_statistics_sha256'],'Candidate command normalizer differs')
    return dict(pair,frontend=str(front),frontend_sha256=sha(front),backend=str(back),backend_sha256=sha(back),
        frontend_updates=a['updates'],backend_updates=b['updates'],evaluation_root=str(job/'backend'/group/'evaluation'))


def execute_bounded(stages,name,args,seconds,completion=None,restarts=0):
    """Persist retry counts; only recognized transient process failures retry."""
    path=stages.root/'stages'/(name+'_recovery.json')
    history=read(path) if path.exists() else []
    while True:
        try:return stages.execute(name,args,seconds,completion=completion)
        except (InterruptedError,TimeoutError):raise
        except Exception as error:
            log=(stages.root/(name+'.log')).read_text(errors='replace')[-6000:]
            transient=any(t in str(error)+log for t in ('exited -6','exited -11','exited 134','exited 139','FileThread','CUDA out of memory'))
            history.append(dict(error=repr(error),utc=time.time(),transient=transient))
            write(path,history)
            if not transient or len(history)>restarts:raise


def train_pair(root,group,round_index,stages,snapshot,reuse_index):
    cfg=read(root/'configuration.json');job=root/f'round_{round_index:02d}'/group;tag=f'r{round_index:02d}_{group}'
    front=prepare_frontend(root,group,round_index,snapshot,reuse_index)
    execute_bounded(stages,tag+'_features',['-m','tools.tight_finish','dagger_cache_visual','--root',str(front)],3600)
    stages.execute(tag+'_input_check',['-m','tools.tight_finish','dagger_check','--root',str(front)],900,front/'vision_acceptance.json')
    budget=read(front/'budget.json');runtime=read(front/'training/session.json')
    # The previous same-architecture real measurements determine admission;
    # new per-update timings are recorded by the formal trainer itself.
    allocation=cfg['frontend']['maximum_hours']*3600
    deadlines=phase_deadlines(root,front,training_deadline=allocation)
    runtime['training_deadline']=deadlines['training_deadline'];write(front/'training/session.json',runtime)
    execute_bounded(stages,tag+'_frontend',['-m','visuotactile.anchor_continue','fresh-train','--job','state','--root',str(front/'training'),
        '--data',str(front/'state_data'),'--target',str(budget['total_updates']),'--output',str(front/'training/launch')],allocation,
        front/'training/finished.json',cfg['recovery']['maximum_training_worker_restarts'])
    done=read(front/'training/finished.json');require(done['updates']>0,'No real frontend optimizer updates')
    back=prepare_backend(root,group,round_index);phase_deadlines(root,back,cache_deadline=3600)
    execute_bounded(stages,tag+'_predictions',['-m','tools.tight_finish','residual_cache','--root',str(back)],3600)
    allocation=cfg['backend']['maximum_hours']*3600;phase_deadlines(root,back,training_deadline=allocation)
    execute_bounded(stages,tag+'_backend',['-m','visuotactile.anchor_continue','residual-train','--job','state','--root',str(back/group),
        '--output',str(back/group/'training/launch')],allocation,back/group/'training/finished.json',cfg['recovery']['maximum_training_worker_restarts'])
    require(read(back/group/'training/finished.json')['updates']>0,'No real backend optimizer updates')
    phase_deadlines(root,back,offline_deadline=1800)
    stages.execute(tag+'_offline',['-m','visuotactile.anchor_continue','residual-offline','--job','state','--root',str(back/group),
        '--output',str(back/group/'offline_launch')],1800)
    phase_deadlines(root,back,evaluation_deadline=3600)
    stages.execute(tag+'_evaluation',['-m','tools.tight_finish','residual_evaluate','--root',str(back)],3600,back/'evaluation_complete.json')
    require(read(back/'evaluation_complete.json')['complete'],'Incomplete evaluation; retain previous pair')
    pair=candidate_pair(job);previous=read(job/'initial_pair.json');plan=read(root/'evaluation_plan.json')
    old_rows=policy_metrics(previous['evaluation_root'],plan,previous['backend_sha256'])
    new_rows=policy_metrics(pair['evaluation_root'],plan,pair['backend_sha256'])
    a=read(front/'training/selected.json');b=read(back/group/'training/selected.json')
    validation_ok=a.get('selection_score',float('inf'))<=1.000001 and b.get('validation_score',b.get('selection_score',float('inf')))<=1.000001
    choice=paired_policy_selection(previous,pair,old_rows,new_rows,validation_ok)
    choice.update(frontend_selected=a,backend_selected=b,snapshot_sha256=sha(snapshot),utc=time.time())
    write(job/'selection.json',choice)
    return choice


def resume(root):
    require((root/'launch.json').exists(),'No registered launch to resume')
    require(read(root/'status.json').get('state')!='COMPLETE','Completed finite queue cannot be resumed')
    require(time.time()<read(root/'session.json')['compute_deadline'],'Original deadline passed; cannot reset it')
    owner=read(root/'resume.json') if (root/'resume.json').exists() else read(root/'launch.json')
    process=proc(owner['pid'])
    require(process is None or process['start_ticks']!=owner.get('start_ticks'),'Original owner is still live')
    from tools.tight_finish import matrix_worker_absence
    gone,_=matrix_worker_absence(read(root/'worker_processes.json'))
    require(gone,'Original owned worker is still live')
    for path in [root/'STOP']+list(root.glob('round_*/*/dp/STOP'))+list(root.glob('round_*/*/dp/training/STOP'))+list(root.glob('round_*/*/backend/*/STOP'))+list(root.glob('round_*/*/backend/*/training/STOP')):
        path.unlink(missing_ok=True)
    run(root,recover=True)


@supervised_owner
def run(root,recover=False):
    """Finite shared-pool rounds, durable stage receipts and paired rollback."""
    from visuotactile.narrow_data import ingest_dagger_pool,freeze_dagger_snapshot
    cfg=read(root/'configuration.json');require(pool_mode(cfg),'Only the explicitly registered pool queue is launchable')
    root=Path(root).resolve();lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(recover or not (root/'launch.json').exists(),'Already launched; use explicit deadline-preserving resume')
    require(predecessor_finished(root),'Predecessor owned work has not verifiably exited')
    registration=read(root/'registration.json')
    require(sha(root/'configuration.json')==registration['configuration_sha256'],'Registered queue configuration changed')
    require(sha(root/'evaluation_plan.json')==registration['evaluation_plan_sha256'],'Registered developer cohort changed')
    now=time.time();source=git('rev-parse','HEAD')
    if recover:
        session=read(root/'session.json');source=session['source_head']
        require(sha(root/'session.json')==read(root/'launch.json')['session_sha256'],'Original session/deadline changed')
        require(all(sha(CODE/p)==v for p,v in read(root/'source_freeze.json').items()),'Frozen runtime source changed')
        write(root/'resume.json',dict(pid=os.getpid(),start_ticks=proc(os.getpid())['start_ticks'],utc=now,original_deadline=session['deadline']))
    else:
        require(not git('status','--porcelain'),'Launch requires committed reviewed source')
        session=read(template_root(root)/'configuration/session.json')
        session.update(version='successful_takeover_pool_v1',source_head=source,started=now,deadline=now+cfg['maximum_hours']*3600,
            resource_policy='full_gpu_v1',resource_target_percent=None,resources=cfg['resources'],reserve_bytes=cfg['resources']['min_disk_gib']*1024**3,
            compute_deadline=now+cfg['maximum_hours']*3600-600,report_reserve_seconds=600,
            owned_worker_roots=[str(root)]+[str(root/f'round_{r:02d}'/g/p) for r in range(1,cfg['rounds']+1)
                for g in cfg['groups'] for p in ('','dp','dp/training','backend','backend/'+g)],dagger=cfg)
        write(root/'session.json',session);write(root/'launch.json',dict(pid=os.getpid(),start_ticks=proc(os.getpid())['start_ticks'],
            started=now,deadline=session['deadline'],source_head=source,session_sha256=sha(root/'session.json')))
        files=git('ls-files','*.py','configs/tight_state.json','visuotactile/run_cpu.sh').splitlines()
        write(root/'source_freeze.json',{p:sha(CODE/p) for p in files})
    (root/'STOP_MONITOR').unlink(missing_ok=True)
    stages=Stages(root);monitor=None;rounds=read(root/'rounds.json') if (root/'rounds.json').exists() else {}
    state='RUNNING';stage='pool_admission';failure=None;pool=(CODE/cfg['pool_root']).resolve()
    def status():write(root/'status.json',dict(state=state,stage=stage,rounds=rounds,source_head=source,utc=time.time(),
        deadline=session['deadline'],test22_used=False))
    try:
        pairs={g:selected_pair(root,g) for g in cfg['groups']}
        # Idempotent import preserves all old attempts, successes and masks.
        for old in cfg['initial_correction_runs']:
            for g in cfg.get('pool_import_groups',list(cfg['groups'])):
                receipt=ingest_dagger_pool(pool,CODE/old/'round_01'/g,run_id=Path(old).name,source_group=g,round_index=1,
                    core_data_root=pairs.get(g,next(iter(pairs.values())))['data_root'])
                require(receipt['integrity_passed'],'Historical pool integrity failed')
        with (root/'monitor.log').open('a') as stream:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'monitor_process.json',proc(monitor.pid))
        for _ in range(15):
            if (root/'shared_resources.json').exists() and time.time()-read(root/'shared_resources.json').get('utc_epoch',0)<4:break
            time.sleep(1)
        require((root/'shared_resources.json').exists() and time.time()-read(root/'shared_resources.json').get('utc_epoch',0)<4,'Fresh shared resource monitor missing')
        previous={};reuse_index=Path(cfg['initial_dino_reuse_index']) if cfg.get('initial_dino_reuse_index') else None
        if cfg.get('scope_revision'):
            for g in cfg['groups']:
                prior=root/'round_01'/g
                if (prior/'selection.json').exists():
                    previous[g]=dict(selection=read(prior/'selection.json'),collection=dict(
                        attempts=read(prior/'collection_finished.json')['attempts'],accepted=read(prior/'collection_finished.json')['accepted_episodes']))
        for round_index in range(cfg.get('first_round',1),cfg['rounds']+1):
            key=f'round_{round_index:02d}';directory=root/key;directory.mkdir(exist_ok=True)
            if (directory/'round_finished.json').exists():
                prior=read(directory/'round_finished.json');previous=prior['groups'];rounds[key]=prior
                for g,value in previous.items():
                    if value.get('selection'):pairs[g]=value['selection']['selected_pair']
                    cache=directory/g/'dp/cache_index.json'
                    if cache.exists():reuse_index=cache
                continue
            if (root/'STOP').exists():raise InterruptedError('User STOP')
            remaining=session['compute_deadline']-time.time()
            if not (directory/'decision.json').exists() and remaining<cfg['decisions']['minimum_round_reserve_hours']*3600:
                state='BUDGET_STOP';stage='insufficient_remaining_round_budget';break
            decision_path=directory/'decision.json'
            try:decision=read(decision_path) if decision_path.exists() else round_decision(cfg,previous,remaining,round_index)
            except ValueError as error:
                if cfg.get('scope_revision') and 'remaining budget' in str(error):
                    write(directory/'budget_not_admitted.json',dict(reason=str(error),remaining_seconds=remaining,utc=time.time()))
                    state='BUDGET_STOP';stage='preserve_original_deadline_no_round_admission';break
                raise
            active_groups=decision.get('active_groups',list(cfg['groups']))
            write(decision_path,decision);rounds[key]=dict(state='RUNNING',groups={});stage=key+'_collection';status()
            family_parent=Path(read(CODE/cfg['family_source'])['parent']);families=read(family_parent/'frozen_families.json')
            for g in active_groups:
                job=directory/g;job.mkdir(exist_ok=True);pair=pairs[g]
                if not (job/'collector_registration.json').exists():
                    require(not (job/'attempts.jsonl').exists(),'Started collection cannot be silently re-registered')
                    count=decision.get('attempts_by_group',{}).get(g,decision['attempts_per_group'])
                    policy=dict(cfg['collection'],attempts_per_group=count,takeover_ticks=decision['takeover_ticks'],
                        round_index=round_index,source_group=g,plan_seed=cfg['collection']['plan_seed']+round_index-1,
                        seed_start=cfg['collection']['seed_start']+(round_index-1)*1000,event_seed_start=cfg['collection']['event_seed_start']+(round_index-1)*1000)
                    if cfg.get('scope_revision'):
                        policy['development_indices']=list(range(policy['development_offset'],count,policy['development_stride']))
                    local=dict(cfg,collection=policy);plan=takeover_plan(read(Path(pair['data_root'])/'data.json'),families,policy)
                    write(job/'initial_pair.json',pair);write(job/'configuration.json',local);write(job/'collection_plan.json',plan)
                    write(job/'session.json',dict(session,source_root=pair['data_root'],collection_deadline=session['compute_deadline']))
                    write(job/'collector_registration.json',{name:sha(job/name) for name in ('initial_pair.json','configuration.json','collection_plan.json')})
                collection_seconds=cfg['collection']['maximum_hours']*3600
                if cfg.get('scope_revision'):
                    collection_seconds=min(collection_seconds,read(job/'configuration.json')['collection']['attempts_per_group']*
                        cfg['planning']['seconds_per_attempt']+cfg['collection']['soft_stop_reserve_seconds']+60)
                share_monitor(root,job);phase_deadlines(root,job,collection_deadline=collection_seconds)
                execute_bounded(stages,key+'_'+g+'_collect',['-m','tools.tight_finish','dagger_collect','--root',str(job)],
                    collection_seconds,job/'collection_finished.json',cfg['recovery']['maximum_collection_worker_restarts'])
                rows=read(job/'collection_results.json')
                require(sum(not r['result'].get('software_complete',False) for r in rows)<=cfg['recovery']['maximum_software_failed_attempts_per_group'],
                    'Persistent collection software errors; retain finite evidence and stop')
                receipt=ingest_dagger_pool(pool,job,run_id=root.name,source_group=g,round_index=round_index,core_data_root=pair['data_root'])
                write(job/'ingestion.json',receipt);require(receipt['integrity_passed'],'New correction integrity failed')
                rounds[key]['groups'][g]=dict(state='COLLECTED',collection=dict(attempts=len(rows),accepted=sum(r.get('accepted',False) for r in rows)))
                status()
            snapshot=directory/'snapshot.json';frozen=freeze_dagger_snapshot(pool,snapshot,run_id=root.name,round_index=round_index)
            current=[e for e in frozen['entries'].values() if e['cohort']=='train' and e['pool_source']['run_id']==root.name and e['pool_source']['round_index']==round_index]
            current_ticks=sum(e['exposure']['supervised_ticks'] for e in current)
            current_anchors=sum(e['exposure']['supervised_anchors'] for e in current)
            write(directory/'current_supervision.json',dict(episodes=len(current),supervised_ticks=current_ticks,supervised_anchors=current_anchors,
                minimum_ticks=cfg['collection']['minimum_expert_ticks'],minimum_anchors=cfg['collection']['minimum_dp_anchors']))
            for g in active_groups:
                job=directory/g;value=rounds[key]['groups'][g];stage=key+'_'+g+'_two_stage_training';status()
                if not frozen['correction_dev_ids']:
                    value.update(state='INSUFFICIENT_CORRECTION_DEV',reason='Keep original pair; never replace held development data with training labels');continue
                if current_ticks<cfg['collection']['minimum_expert_ticks'] or current_anchors<cfg['collection']['minimum_dp_anchors']:
                    value.update(state='INSUFFICIENT_NEW_DP_SUPERVISION',reason='Retain old pair; no backend-only substitute for a complete DAgger round');continue
                try:
                    choice=train_pair(root,g,round_index,stages,snapshot,reuse_index)
                    value.update(state='COMPLETE',selection=choice);pairs[g]=choice['selected_pair']
                    reuse_index=job/'dp/cache_index.json'
                except (TimeoutError,InterruptedError):raise
                except Exception as error:
                    value.update(state='FAILED',error=repr(error))
                    write(job/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),previous_pair_retained=read(job/'initial_pair.json')))
                status()
            result=dict(state='RESOLVED',groups=rounds[key]['groups'],snapshot_id=frozen['snapshot_id'],utc=time.time(),
                active_groups=active_groups,not_scheduled=[g for g in cfg['groups'] if g not in active_groups],
                retained_complete_pairs={g:pairs[g] for g in cfg['groups'] if g not in active_groups})
            write(directory/'round_finished.json',result);rounds[key]=result;previous=result['groups'];write(root/'rounds.json',rounds)
            from tools.tight_diagnostics import dagger_publish
            dagger_publish(root)  # CPU-only stage evidence; final push follows owned-worker exit.
        else:
            required=[r for key,r in rounds.items() if int(key.removeprefix('round_'))>=cfg.get('first_round',1)]
            state='COMPLETE' if required and all(v['state']=='COMPLETE' for r in required for v in r['groups'].values()) else 'PARTIAL'
            stage='registered_rounds_resolved'
    except InterruptedError as error:state='STOPPED';failure=repr(error)
    except TimeoutError as error:state='BUDGET_STOP';failure=repr(error)
    except Exception as error:
        state='FAILED';failure=repr(error);write(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),stage=stage))
    finally:
        write(root/'rounds.json',rounds);gone=stages.close();(root/'STOP_MONITOR').touch()
        if monitor is not None:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        gone=gone and (monitor is None or monitor.poll() is not None)
        write(root/'status.json',dict(state=state,stage=stage,rounds=rounds,owned_workers_exited=gone,source_head=source,
            utc=time.time(),deadline=session['deadline'],test22_used=False,automatic_queue_exited=True,failure=failure))
        try:
            from tools.tight_diagnostics import dagger_publish
            dagger_publish(root)
            if gone:publish_commit(root)
        except Exception as error:write(root/'publication_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
        lock.close()


def cleanup_obsolete(root):
    """Delete only a fixed old-weight allowlist after both owned queues exit.

    This is called by the finite owner after reaping. It never sweeps a run,
    follows a candidate symlink, removes source episodes, or touches current
    selected parents. Hard-linked best/last names count once for freed blocks.
    """
    root=Path(root).resolve();report_path=root/'cleanup.json'
    if report_path.exists():return read(report_path)  # One-shot, no implicit retry.
    run=(CODE/'outputs/visuotactile/run').resolve()
    relative=[f'narrow_state_20261105_02/training/{n}.pt' for n in ('best','last')]
    relative+=['narrow_repair_20261106_01/expanded_training/best.pt']
    relative+=[f'tight_state_20261107_01/composition_06/training/{n}.pt' for n in ('best','last')]
    relative+=[f'tight_palm_20261108_01/training/{n}.pt' for n in ('best','last')]
    relative+=[f'tight_vision_20261109_01/{g}/training/last.pt' for g in ('G00','G10')]
    relative+=[f'tight_residual_20261110_01/{g}/training/{n}.pt' for g in ('G01','G11') for n in ('best','last')]
    relative+=['dp_command_20261029/b0/best.pt']
    report=dict(version='dagger_obsolete_weights_v1',utc=time.time(),executed=False,rows=[],
        scope='Fourteen explicit obsolete checkpoint names only; raw400/Core, current parents, source caches and success/video/denominator records retained',
        deleted_files=0,deleted_logical_bytes=0,released_allocated_bytes=0)
    blockers=[]
    try:
        require(predecessor_finished(root),'Predecessor owner/workers have not verifiably exited')
        cfg=read(root/'configuration.json');status=read(root/'status.json');cleanup=read(root/'owned_cleanup.json')
        require(status.get('state') in ('COMPLETE','PARTIAL','FAILED','BUDGET_STOP') and
            status.get('owned_workers_exited') is True and cleanup.get('verified') is True,
            'Successor stage workers have not verifiably exited')
        for worker in cleanup.get('workers',[]):
            current=proc(worker['pid'])
            require(current is None or (worker.get('start_ticks') is not None and
                current['start_ticks']!=worker['start_ticks']),'An original successor worker is still live')
        pair_paths=[root/'round_01'/group/'initial_pair.json' for group in cfg['groups']]
        require(pair_paths and all(path.is_file() for path in pair_paths),'Selected initialization pairs are not fixed')
        for path in pair_paths:
            pair=read(path)
            for key in ('frontend','backend'):
                require(sha(pair[key])==pair[key+'_sha256'],'Fixed initialization changed: '+key)
    except Exception as error:blockers.append(str(error))
    if blockers:
        report.update(blocked=blockers)
        report['rows']=[dict(path=str(run/p),deleted=False,skip_reason='owner or dependency guard blocked') for p in relative]
        write(report_path,report);return report

    parent=Path(read(root/'registration.json')['predecessor']).resolve()
    dependency_files=[]
    names={'configuration.json','session.json','initial_pair.json','execution_plan.json','data.json',
        'manifest.json','cache_index.json','nominal_index.json','protected.json','source_freeze.json'}
    for directory in (root,parent):
        dependency_files.extend(p for p in directory.rglob('*.json') if p.name in names)
    dependency_files.append(CODE/'configs/tight_state.json')
    protected={};reference_errors=[]
    def references(value,origin,key=''):
        if isinstance(value,dict):
            for field,child in value.items():
                if field not in ('owned_worker_roots','source_protection_summary'):
                    references(child,origin,field)
        elif isinstance(value,list):
            for child in value:references(child,origin,key)
        elif isinstance(value,str) and '/' in value:
            candidate=Path(value)
            for path in ([candidate] if candidate.is_absolute() else [CODE/candidate,origin.parent/candidate]):
                if path.is_file():protected.setdefault(str(path.resolve()),[]).append(str(origin))
                elif path.is_dir() and any(tag in key.lower() for tag in ('checkpoint','resume','training_root','weights')):
                    for old in relative:
                        p=run/old
                        if p.is_relative_to(path.resolve()):protected.setdefault(str(p.resolve()),[]).append(str(origin))
    for path in sorted(set(dependency_files)):
        try:references(read(path),path)
        except Exception as error:reference_errors.append(dict(path=str(path),error=repr(error)))
    report['dependency_files_checked']=len(set(dependency_files))
    report['dependency_read_errors']=reference_errors

    def open_files():
        """Inspect only same-user processes; never signal or change a process."""
        opened={};errors=[]
        for entry in os.scandir('/proc'):
            if not entry.name.isdigit():continue
            directory=Path(entry.path)
            try:
                if directory.stat().st_uid!=os.getuid():continue
                for fd in (directory/'fd').iterdir():
                    try:
                        state=fd.stat()
                        opened.setdefault(f'inode:{state.st_dev}:{state.st_ino}',[]).append(int(entry.name))
                        destination=fd.resolve(strict=True)
                        opened.setdefault(str(destination),[]).append(int(entry.name))
                    except (FileNotFoundError,ProcessLookupError):pass
                    except PermissionError:errors.append(dict(pid=int(entry.name),scope='fd link'))
            except (FileNotFoundError,ProcessLookupError):pass
            except PermissionError:errors.append(dict(pid=int(entry.name),scope='fd directory'))
        return opened,errors
    opened,fd_errors=open_files();report['fd_visibility_errors']=fd_errors
    before_free=shutil.disk_usage(run).free
    for name in relative:
        path=run/name;row=dict(path=str(path),deleted=False)
        report['rows'].append(row)
        try:
            if path.is_symlink() or path.resolve()!=path:
                row['skip_reason']='symlink or redirected parent';continue
            if not path.is_file():row['skip_reason']='already absent';continue
            state=path.stat()
            row.update(logical_bytes=state.st_size,allocated_bytes=state.st_blocks*512,device=state.st_dev,
                inode=state.st_ino,links_before=state.st_nlink,sha256=sha(path))
            if reference_errors or fd_errors:
                row['skip_reason']='dependency or open-FD visibility incomplete';continue
            if str(path) in protected:
                row.update(skip_reason='referenced by current or queued dependency',references=protected[str(path)]);continue
            inode_key=f'inode:{state.st_dev}:{state.st_ino}'
            if str(path) in opened or inode_key in opened:
                row.update(skip_reason='open file descriptor',pids=opened.get(str(path),opened.get(inode_key)));continue
            # Recheck immediately before mutation; a resumed writer or changed
            # artifact invalidates this candidate instead of expanding cleanup.
            now=path.stat();current_open,current_errors=open_files()
            if current_errors or str(path) in current_open or inode_key in current_open:
                row['skip_reason']='new or unobservable file reader';continue
            if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns)!=(state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns):
                row['skip_reason']='candidate changed during validation';continue
            if sha(path)!=row['sha256']:
                row['skip_reason']='candidate SHA changed';continue
            path.unlink()
            row.update(deleted=True,links_at_unlink=now.st_nlink,
                released_allocated_bytes=now.st_blocks*512 if now.st_nlink==1 else 0)
            report['deleted_files']+=1;report['deleted_logical_bytes']+=state.st_size
            report['released_allocated_bytes']+=row['released_allocated_bytes']
            write(report_path,report)  # Preserve partial completed deletions on interruption.
        except Exception as error:row.update(skip_reason='validation or deletion error',error=repr(error))
    report.update(executed=True,completed_utc=time.time(),disk_free_delta_bytes=shutil.disk_usage(run).free-before_free,
        allocated_accounting='st_blocks counted only when this unlink removed the final hard link; free-space delta may include unrelated writes')
    retained=[]
    for path in pair_paths:
        pair=read(path)
        for key in ('frontend','backend'):
            retained.append(dict(path=pair[key],sha256=sha(pair[key]),unchanged=sha(pair[key])==pair[key+'_sha256']))
    report['retained_initialization_checks']=retained
    audit=CODE/'research_docs/CLEANUP_AUDIT.md'
    if audit.is_file():
        marker='<!-- dagger-obsolete-cleanup:'+str(root)+' -->'
        content=audit.read_text()
        if marker not in content:
            content+='\n\n'+marker+'\n本轮 DAgger 收尾按明确依赖白名单清理旧 checkpoint：删除 '+str(report['deleted_files'])+' 个目录项，逻辑 '+str(report['deleted_logical_bytes'])+' 字节，按最终硬链接释放的已分配空间 '+str(report['released_allocated_bytes'])+' 字节。原400/Core、当前初始化与源缓存、旧成功视频及分母未列入清理。逐项 SHA、跳过原因与保留核对见本地 `'+str(report_path)+'`。\n'
            audit.write_text(content)
        report['existing_audit_updated']=str(audit)
    else:report['audit_notice']='CLEANUP_AUDIT.md absent; no new markdown report created'
    write(report_path,report)
    return report


def publish_commit(root):
    """Publish only reviewed public evidence after all new owned workers exit."""
    session=read(root/'session.json')
    require(read(root/'status.json').get('owned_workers_exited') is True,'Cannot publish live-stage completion')
    previous=read(root/'public_git.json')['head'] if (root/'public_git.json').exists() else session['source_head']
    require(git('rev-parse','HEAD')==previous,'Successor source moved during this round')
    frozen=read(root/'source_freeze.json')
    require(all(sha(CODE/p)==value for p,value in frozen.items()),'Successor runtime changed')
    require(not git('diff','--cached','--name-only'),'Existing staged changes are not owned by this publisher')
    prefix='research_docs/evidence/'+root.name+'/'
    allowed={'research_docs/STATE_DP_RESULTS.md','research_docs/CLEANUP_AUDIT.md'}
    changed=set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    require(all(p in allowed or p.startswith(prefix) for p in changed),'Unreviewed unrelated working-tree change; preserve for review')
    git('fetch','origin','feature/tight-state-foundation')
    require(git('rev-parse','FETCH_HEAD')==previous,'Remote advanced; do not overwrite other work')
    git('add','--','research_docs/STATE_DP_RESULTS.md','research_docs/CLEANUP_AUDIT.md',prefix)
    staged=git('diff','--cached','--name-only').splitlines()
    require(all(p in allowed or p.startswith(prefix) for p in staged),'Staged publication scope differs')
    if staged:git('commit','-m','Publish finite two-stage takeover DAgger results')
    git('push','origin','HEAD:refs/heads/feature/tight-state-foundation')
    write(root/'public_git.json',dict(pushed=True,head=git('rev-parse','HEAD'),target='feature/tight-state-foundation',utc=time.time()))
