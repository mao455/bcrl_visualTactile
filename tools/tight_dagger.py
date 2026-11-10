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
    require(len(available) >= policy['attempts_per_group'], 'Insufficient distinct registered training families')
    jobs = []
    for index, family in enumerate(available[:policy['attempts_per_group']]):
        identity = f'dagger_r01_{index:03d}'
        source = next(i for i in data['splits']['train'] if data['entries'][i]['family'] == family['family'])
        jobs.append(dict(family, identity=identity, source_identity=source, split='train', cohort='dagger_train',
            seed=policy['seed_start'] + index, event_seed=policy['event_seed_start'] + index,
            takeover_tick=policy['takeover_ticks'][index % len(policy['takeover_ticks'])]))
    return jobs


def prepare(root):
    cfg = read(CODE/'configs/tight_state.json')['dagger']
    require(not (root/'configuration.json').exists(), 'DAgger already registered; no overwrite')
    root.mkdir(parents=True, exist_ok=True)
    predecessor = (CODE/cfg['predecessor']).resolve()
    require((predecessor/'launch.json').is_file(), 'Missing actual predecessor launch')
    data = read(predecessor/'configuration/state_data/data.json')
    parent = Path(read(CODE/cfg['family_source'])['parent'])
    plan = takeover_plan(data, read(parent/'frozen_families.json'), cfg['collection'])
    # The autonomous developer cohort remains the original frozen ten specs.
    evaluation = read(predecessor/'command_residual/evaluation_plan.json')
    require(len(evaluation) == 10, 'Expected existing ten-case development protocol')
    require(not ({j['seed'] for j in plan} & {e['spec']['seed'] for e in evaluation}), 'Collection/development seed overlap')
    write(root/'configuration.json', cfg)
    write(root/'collection_plan.json', plan)
    write(root/'evaluation_plan.json', evaluation)
    write(root/'registration.json', dict(predecessor=str(predecessor), predecessor_launch=read(predecessor/'launch.json'),
        predecessor_source=read(predecessor/'session.json')['source_head'], preparing_worktree=str(CODE),
        predecessor_process=proc(read(predecessor/'launch.json')['pid']),
        configuration_sha256=sha(root/'configuration.json'), collection_plan_sha256=sha(root/'collection_plan.json'),
        waiting_deadline=read(predecessor/'session.json')['deadline'] + 7200, created=time.time(),
        start_rule='terminal actual predecessor plus exited owner/owned GPU workers; valid selected paired checkpoints',
        new_deadline_rule='starts when the new finite stage is admitted after predecessor exit; never inherits predecessor absolute deadline'))
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
    import torch
    cfg = read(root/'configuration.json'); parent = Path(read(root/'registration.json')['predecessor'])
    spec = cfg['groups'][group]
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
        normalization_identity=front['contract'].get('normalization_identity', 'u_expert_train166_affine_v1'))


class Stages:
    """Owned worker ledger and bounded waits; all compute uses existing entries."""
    def __init__(self, root):
        self.root = root; self.children = []; self.records = []

    def execute(self, name, args, seconds):
        session = read(self.root/'session.json'); limit = min(session.get('compute_deadline',session['deadline']), time.time()+seconds)
        require(limit > time.time(), 'DAgger session deadline exhausted')
        frozen = read(self.root/'source_freeze.json')
        require(all(sha(CODE/p) == digest for p, digest in frozen.items()), 'Frozen successor source changed')
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
                if (self.root/'STOP').exists():raise InterruptedError('DAgger STOP')
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
    from visuotactile.state_benchmark import trial, dagger_configuration
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.narrow_data import dagger_record
    from visuotactile.anchor_sampling import supervision_population
    from visuotactile.native_evaluation import exit_completed_kit_worker
    s=read(root/'session.json');cfg=read(root/'configuration.json');pair=read(root/'initial_pair.json')
    plan=read(root/'collection_plan.json');source=read(Path(pair['data_root'])/'data.json')
    require(len(plan)==cfg['collection']['attempts_per_group'] and len({p['identity'] for p in plan})==len(plan),
        'Finite unique collection plan differs from registered budget')
    require(not (root/'attempts.jsonl').exists(), 'Collection already attempted; no implicit replay')
    for spec in plan:dagger_configuration(cfg['bridge'],spec,source)
    for key in ('frontend','backend'):
        require(sha(pair[key])==pair[key+'_sha256'], 'Frozen collector initialization drift: '+key)
    require(sha(s['expert_checkpoint'])==s['expert_sha256'], 'Frozen collector expert drift')
    saved=torch.load(pair['backend'],map_location='cpu',weights_only=False)
    results=[];entries={};started=time.time();row=None;app=None
    gate_count=cfg['collection'].get('gate_attempts',2)
    require(0<gate_count<=len(plan), 'First takeover gate must be inside the finite attempt budget')
    try:
        app,env,player,lease=environment(root,evaluation=True,camera_content_validation=True)
        # Evaluation construction avoids the old speed wrapper; the bridge owns
        # the one expert conversion. Every wait still consumes collection time.
        lease.deadline=min(s['deadline'],s['collection_deadline'])
        for index,spec in enumerate(plan):
            row=None
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
                path=Path(result['dataset_capture']['path']);takeover=result['dagger']
                require(takeover['provenance_passed'] and takeover['position_executor_continuous'], 'Takeover provenance failed')
                row['exposure']=dict(episodes=1,history_ticks=result['steps'],supervised_ticks=0,
                    supervised_anchors=0,supervised_window_targets=0)
                # The strict training reader intentionally rejects empty masks.
                # Early student success/failure and switch-flush termination are
                # valid attempts, but have no eligible expert supervision.
                if takeover['expert_valid_ticks']>0:
                    entry=dict(path=str(path),episode=spec['identity'],length=result['steps'],family=spec['family'],split='train',
                        source_sha256=sha(path),trajectory_kind='expert_takeover_v1')
                    record,_,_=dagger_record(entry,include_geometry=False)
                    exposure=supervision_population({spec['identity']:record});row['exposure']=exposure
                    require(exposure['supervised_ticks']==takeover['expert_valid_ticks'], 'Recorded expert denominator differs')
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
        summary=collection_summary(results,cfg['collection'])
        summary.update(wall_s=time.time()-started,complete=True)
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


def prepare_frontend(root, group):
    import torch
    from visuotactile.narrow_data import build_dagger_aggregate
    from visuotactile.anchor_sampling import AnchorSampler, supervision_population
    cfg=read(root/'configuration.json');parent=Path(read(root/'registration.json')['predecessor'])
    job=root/'round_01'/group;pair=read(job/'initial_pair.json');source=Path(pair['data_root']).parent
    out=job/'dp';out.mkdir();(out/'training').mkdir()
    base=read(source/'configuration.json');session=read(source/'session.json')
    entries=read(job/'correction_entries.json')
    manifest=read(source/'manifest.json')
    manifest=dict(manifest,dagger_round=1,previous_manifest_sha256=sha(source/'manifest.json'),
        episodes=manifest['episodes']+[dict(id=i,**e,sha256=e['source_sha256']) for i,e in entries.items()])
    write(out/'manifest.json',manifest)
    data=build_dagger_aggregate(source/'state_data',entries,out/'state_data',pair['supervision'],
        manifest_sha256=sha(out/'manifest.json'),include_geometry=group=='QG01')
    raw=torch.load(out/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    records={i:raw['records'][i] for i in data['splits']['train']};exposure=supervision_population(records)
    count=len(records);mean_batch=count/math.ceil(count/64);policy=cfg['frontend']
    target=math.ceil(policy['maximum_anchor_epochs']*exposure['supervised_anchors']/mean_batch)
    minimum=math.ceil(policy['minimum_anchor_epochs']*exposure['supervised_anchors']/mean_batch)
    interval=max(1,math.ceil(policy['validation_anchor_epochs']*exposure['supervised_anchors']/mean_batch))
    modality=pair['modality'];profile={}
    if group=='QG01':profile=read(parent/'configuration/configuration.json')['vision_pair']['geometry_profiles']['QG00']
    base['vision_pair']=dict(base['vision_pair'],groups=[group],modalities={group:modality},warm_starts={},
        continuation_from={group:dict(path=pair['frontend'],sha256=pair['frontend_sha256'])})
    base['dagger']=cfg;base['resources']=cfg['resources']
    session.update(version='expert_takeover_aggregate_v1',vision_pair=base['vision_pair'],dagger_round=1,
        source_head=read(root/'session.json')['source_head'],resource_policy='full_gpu_v1',resources=cfg['resources'],
        resource_root=str(out),deadline=read(root/'session.json')['deadline'],training_deadline=read(root/'session.json')['deadline'],
        cache_deadline=read(root/'session.json')['deadline'],reserve_bytes=cfg['resources']['min_disk_gib']*1024**3,
        owned_worker_roots=[str(root)],dino_reuse_index=str(source/'cache_index.json'))
    write(out/'configuration.json',base);write(out/'session.json',session);write(out/'evaluation_plan.json',read(root/'evaluation_plan.json'))
    write(out/'COMPLETE.json',dict(dataset_prepared=True,dagger_round=1,exposure=exposure,test22_used=False))
    continuation=dict(path=pair['frontend'],sha256=pair['frontend_sha256'],teacher_supervision=pair['supervision'],
        statistics_sha256=pair['statistics_sha256'],normalization_identity=pair['normalization_identity'],round_index=1,source_group=group)
    runtime=dict(resource_policy='full_gpu_v1',external_monitor_root=str(out),modality=modality,random_seed=cfg['training_seed'],
        training_deadline=session['deadline'],continuation_from=continuation,microbatch=4,
        validation_interval=interval,early_stop_minimum_update=minimum,
        comparison_updates=sorted({min(target,interval*k) for k in range(1,5)}),deterministic_validation_loss=True,
        incremental_schedule=dict(maximum_updates=target,minimum_updates=minimum,warmup_updates=100,
            lr_peak=policy['learning_rate'],lr_floor=policy['learning_rate']/10,validation_interval=interval))
    if profile:runtime['hand_geometry']=profile
    write(out/'training/session.json',runtime)
    budget=dict(total_updates=target,anchor_population=exposure['supervised_anchors'],mean_batch=mean_batch,
        target_anchor_epochs=policy['maximum_anchor_epochs'],early_stop_minimum_update=minimum,
        validation_interval=interval,comparison_updates=runtime['comparison_updates'],exposure=exposure)
    write(out/'budget.json',budget);write(out/'training/budget.json',budget);share_monitor(root,out)
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
    new_ids=[i for i in data['splits']['train'] if data['entries'][i].get('trajectory_kind')=='expert_takeover_v1']
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
        gradients=gradients,save_restore_exact=exact,restore_checks=restore_checks,disposable_updates=1,formal_updates=0,test22_used=False))


def prepare_backend(root, group):
    from tools.tight_finish import residual_prepare
    job=root/'round_01'/group;front=job/'dp';pair=read(job/'initial_pair.json');cfg=read(root/'configuration.json')
    configuration=read(front/'configuration.json')
    parent=Path(read(root/'registration.json')['predecessor'])
    stage='command_residual' if pair['supervision']=='u_expert' else 'configuration_residual'
    configuration['residual_pair']=copy.deepcopy(read(parent/stage/'configuration.json')['residual_pair'])
    v=configuration['residual_pair'];selected=front/'training/best.pt'
    data=read(front/'state_data/data.json');normalizer=job/'command_statistics';normalizer.mkdir()
    shutil.copyfile(pair['residual_statistics_path'],normalizer/'statistics.pt')
    write(normalizer/'data.json',dict(data,supervision='u_expert',normalization_identity='u_expert_train166_affine_v1'))
    v.update(source_root=str(front),groups=[group],dagger_round=1,
        bases={group:dict(group=pair['modality'],modality=pair['modality'],checkpoint=str(selected),sha256=sha(selected))},
        residual_statistics_path=str(normalizer/'statistics.pt'),fixed_residual_scales=pair['scales'],
        continuation_from=dict(path=pair['backend'],sha256=pair['backend_sha256'],teacher_supervision='u_expert',
            statistics_sha256=pair['residual_statistics_sha256'],normalization_identity='u_expert_train166_affine_v1',round_index=1,source_group=group),
        maximum_control_tick_epochs=cfg['backend']['maximum_control_tick_epochs'],
        minimum_control_tick_epochs=cfg['backend']['minimum_control_tick_epochs'],
        comparison_control_tick_epochs=[10,20,30],validation_epoch_interval=5,learning_rate=cfg['backend']['learning_rate'],
        stop_rule='minimum10 supervised tick traversals then fixed validation plateau; maximum30 or finite measured wall budget',
        cache_definition='current paired frozen DP; complete actual prefixes, separate previous_nominal, expert supervision mask',
        training_hours=1.5,maximum_hours=4,cache_hours=1.5,offline_hours=.5,evaluation_hours=1)
    out=job/'backend';residual_prepare(out,configuration=configuration)
    share_monitor(root,out)
    return out


def phase_deadlines(root, directory, **values):
    global_session=read(root/'session.json');limit=global_session.get('compute_deadline',global_session['deadline']);s=read(directory/'session.json')
    s.update({k:min(limit,time.time()+seconds) for k,seconds in values.items()});s['deadline']=limit
    write(directory/'session.json',s)
    for g in s.get('residual_pair',{}).get('groups',[]):
        if (directory/g/'session.json').exists():write(directory/g/'session.json',dict(s,resource_root=str(directory),modality=g))
    return s


@supervised_owner
def run(root):
    """One round; each complete group updates its frontend and backend."""
    cfg=read(root/'configuration.json');require(not (root/'launch.json').exists(),'DAgger already launched')
    parent=Path(read(root/'registration.json')['predecessor']);now=time.time();source=git('rev-parse','HEAD')
    session=read(parent/'configuration/session.json')
    session.update(version='dagger_two_stage_v1',source_head=source,started=now,deadline=now+cfg['maximum_hours']*3600,
        resource_policy='full_gpu_v1',resource_target_percent=None,resources=cfg['resources'],reserve_bytes=cfg['resources']['min_disk_gib']*1024**3,
        compute_deadline=now+cfg['maximum_hours']*3600-600,report_reserve_seconds=600,
        owned_worker_roots=[str(root)]+[str(root/'round_01'/g/p) for g in cfg['groups'] for p in ('','dp','dp/probe','dp/training','backend','backend/'+g)],dagger=cfg)
    write(root/'session.json',session);write(root/'launch.json',dict(pid=os.getpid(),started=now,deadline=session['deadline'],source_head=source))
    files=git('ls-files','*.py','configs/tight_state.json','visuotactile/run_cpu.sh').splitlines()
    write(root/'source_freeze.json',{p:sha(CODE/p) for p in files})
    states={g:'PENDING' for g in cfg['groups']};stages=Stages(root);monitor=None
    def status(stage):write(root/'status.json',dict(state='RUNNING',stage=stage,groups=states,source_head=source,utc=time.time(),deadline=session['deadline']))
    try:
        for group in states:
            job=root/'round_01'/group;job.mkdir(parents=True)
            try:pair=selected_pair(root,group)
            except Exception as error:
                states[group]='INVALID_INITIALIZATION';write(job/'failure.json',dict(stage='initialization',error=repr(error)));continue
            write(job/'initial_pair.json',pair);write(job/'configuration.json',cfg);write(job/'collection_plan.json',read(root/'collection_plan.json'))
            write(job/'session.json',dict(session,source_root=pair['data_root'],collection_deadline=min(session['deadline'],time.time()+cfg['collection']['maximum_hours']*3600)))
            share_monitor(root,job)
        with (root/'monitor.log').open('a') as stream:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(15):
            if (root/'shared_resources.json').exists():break
            time.sleep(1)
        require((root/'shared_resources.json').exists(),'New-stage monitor not ready')
        for group in states:
            if states[group]!='PENDING':continue
            job=root/'round_01'/group;states[group]='COLLECTING';status('successful_takeover_collection')
            try:
                phase_deadlines(root,job,collection_deadline=cfg['collection']['maximum_hours']*3600)
                stages.execute(group+'_collect',['-m','tools.tight_finish','dagger_collect','--root',str(job)],cfg['collection']['maximum_hours']*3600)
                if not read(job/'collection_finished.json')['sufficient']:
                    states[group]='INSUFFICIENT_DP_CORRECTIONS';continue
                states[group]='COLLECTED'
            except Exception as error:
                states[group]='FAILED';write(job/'failure.json',dict(stage='collection',error=repr(error),traceback=traceback.format_exc()))
        for group in states:
            if states[group]!='COLLECTED':continue
            job=root/'round_01'/group
            try:
                states[group]='FRONTEND';status('aggregate_then_frontend')
                front=prepare_frontend(root,group)
                stages.execute(group+'_features',['-m','tools.tight_finish','dagger_cache_visual','--root',str(front)],3600)
                stages.execute(group+'_input_check',['-m','tools.tight_finish','dagger_check','--root',str(front)],900)
                runtime=read(front/'training/session.json');budget=read(front/'budget.json');probe=front/'probe';probe.mkdir()
                probe_runtime=dict(runtime,disposable_gate=True,probe_limit_updates=8)
                if runtime.get('hand_geometry'):probe_runtime['geometry_probe_gain_sequence']=[0.,0.,.25,.25,1.,1.,1.,1.]
                write(probe/'session.json',probe_runtime);write(probe/'budget.json',budget)
                stages.execute(group+'_probe',['-m','visuotactile.anchor_continue','fresh-train','--job','state','--root',str(probe),
                    '--data',str(front/'state_data'),'--target',str(budget['total_updates']),'--output',str(probe/'launch')],900)
                rows=[json.loads(line) for line in (probe/'training.jsonl').read_text().splitlines()]
                timed=rows[4:] if runtime.get('hand_geometry') else rows[2:]
                average=sum(r['seconds_per_update'] for r in timed)/len(timed)
                allocation=min(cfg['frontend']['maximum_hours']*3600,average*budget['total_updates']*1.5+1800)
                write(front/'throughput_probe.json',dict(disposable=True,updates=len(rows),seconds_per_update=average,
                    estimated_training_seconds=average*budget['total_updates'],allocated_seconds=allocation,target=budget['total_updates'],
                    per_update_measurements=rows,geometry_off_mean_seconds=sum(r['seconds_per_update'] for r in rows[:2])/2,
                    geometry_on_mean_seconds=average if runtime.get('hand_geometry') else None,
                    geometry_scope='QG disposable gain0/0/.25/.25/1/1/1/1; unchanged formal LR horizon and 2 configurations/update' if runtime.get('hand_geometry') else 'no added hand geometry'))
                runtime['training_deadline']=min(session['compute_deadline'],time.time()+allocation);write(front/'training/session.json',runtime)
                stages.execute(group+'_frontend',['-m','visuotactile.anchor_continue','fresh-train','--job','state','--root',str(front/'training'),
                    '--data',str(front/'state_data'),'--target',str(budget['total_updates']),'--output',str(front/'training/launch')],allocation)
                done=read(front/'training/finished.json');require(done['updates']>0 and (front/'training/best.pt').exists(),'No trained new frontend')
                states[group]='BACKEND';status('refresh_new_frontend_cache_then_backend')
                back=prepare_backend(root,group);phase_deadlines(root,back,cache_deadline=3600,training_deadline=5400)
                stages.execute(group+'_predictions',['-m','tools.tight_finish','residual_cache','--root',str(back)],3600)
                phase_deadlines(root,back,training_deadline=900)
                stages.execute(group+'_backend_probe',['-m','visuotactile.anchor_continue','residual-probe','--job','state','--root',str(back/group),'--output',str(back/group/'training_probe/launch')],900)
                measured=read(back/group/'training_probe/finished.json');population=read(back/group/'nominal/scales.json')['control_tick_population']
                estimate=measured['training_wall_s']/measured['supervised_ticks']*population*cfg['backend']['maximum_control_tick_epochs']
                allocation=min(cfg['backend']['maximum_hours']*3600,estimate*1.5+900)
                write(back/'throughput_probe.json',dict(measurement=measured,estimated_training_seconds=estimate,allocated_seconds=allocation))
                phase_deadlines(root,back,training_deadline=allocation)
                stages.execute(group+'_backend',['-m','visuotactile.anchor_continue','residual-train','--job','state','--root',str(back/group),'--output',str(back/group/'training/launch')],allocation)
                done=read(back/group/'training/finished.json');require(done['updates']>0,'Backend had no formal updates')
                states[group]='EVALUATING';status('pure_student_evaluation')
                phase_deadlines(root,back,offline_deadline=1800,evaluation_deadline=3600)
                stages.execute(group+'_offline',['-m','visuotactile.anchor_continue','residual-offline','--job','state','--root',str(back/group),'--output',str(back/group/'offline_launch')],1800)
                phase_deadlines(root,back,evaluation_deadline=3600)
                stages.execute(group+'_evaluation',['-m','tools.tight_finish','residual_evaluate','--root',str(back)],3600)
                require(read(back/'evaluation_complete.json')['complete'],'Pure-student evaluation receipt missing')
                states[group]='COMPLETE'
            except Exception as error:
                write(job/'failure.json',dict(stage=states[group],error=repr(error),traceback=traceback.format_exc()));states[group]='FAILED'
            status('finite_round')
    finally:
        gone=stages.close();(root/'STOP_MONITOR').touch()
        if monitor is not None:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        gone=gone and (monitor is None or monitor.poll() is not None)
        terminal='COMPLETE' if gone and all(v=='COMPLETE' for v in states.values()) else 'PARTIAL'
        write(root/'status.json',dict(state=terminal,stage='finite_round_resolved',groups=states,owned_workers_exited=gone,
            source_head=source,utc=time.time(),test22_used=False,no_next_round=True))
        try:
            from tools.tight_diagnostics import dagger_publish
            if gone:cleanup_obsolete(root)
            dagger_publish(root)
            if gone:publish_commit(root)
        except Exception as error:write(root/'publication_error.json',dict(error=repr(error),traceback=traceback.format_exc()))


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
    require(git('rev-parse','HEAD')==session['source_head'],'Successor source moved during this round')
    frozen=read(root/'source_freeze.json')
    require(all(sha(CODE/p)==value for p,value in frozen.items()),'Successor runtime changed')
    require(not git('diff','--cached','--name-only'),'Existing staged changes are not owned by this publisher')
    prefix='research_docs/evidence/'+root.name+'/'
    allowed={'research_docs/STATE_DP_RESULTS.md','research_docs/CLEANUP_AUDIT.md'}
    changed=set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    require(all(p in allowed or p.startswith(prefix) for p in changed),'Unreviewed unrelated working-tree change; preserve for review')
    git('fetch','origin','feature/tight-state-foundation')
    require(git('rev-parse','FETCH_HEAD')==session['source_head'],'Remote advanced; do not overwrite other work')
    git('add','--','research_docs/STATE_DP_RESULTS.md','research_docs/CLEANUP_AUDIT.md',prefix)
    staged=git('diff','--cached','--name-only').splitlines()
    require(all(p in allowed or p.startswith(prefix) for p in staged),'Staged publication scope differs')
    if staged:git('commit','-m','Publish finite two-stage takeover DAgger results')
    git('push','origin','HEAD:refs/heads/feature/tight-state-foundation')
    write(root/'public_git.json',dict(pushed=True,head=git('rev-parse','HEAD'),target='feature/tight-state-foundation',utc=time.time()))
