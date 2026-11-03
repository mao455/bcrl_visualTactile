"""Bounded resumable owner for pilots, train320, validation, frozen formal test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .compact import sha256
from .native_v2_repair import RepairRunner
from .training_resources import atomic_json


def initialize(root, preflight):
    oldroot = Path('outputs/visuotactile/native_dp_v2/state_history_20261102_02').resolve()
    old = json.loads((oldroot/'session.json').read_text())
    root.mkdir(parents=True, exist_ok=False); started = time.time()
    source = Path(old['source_root']); data = json.loads((source/'data.json').read_text())
    if sha256(source/'data.json') != old['data_sha256'] or sha256(old['expert_checkpoint']) != old['expert_sha256']:
        raise ValueError('Previously audited data/expert identity changed')
    files = list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks').rglob('*.py'))
    session = dict(old, protocol_version=5, started_epoch=started, deadline=started+24*3600,
        training_deadline=started+21*3600, resource_policy='shared_recent_v1',
        predecessor=str(oldroot), preflight=str(preflight.resolve()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256={str(p): sha256(p) for p in files},
        pilot_updates=300, bc_updates=10000, dp_updates=19000,
        bc_milestones=[1000, 3000, 6000, 10000], dp_milestones=[1000, 3000, 6000, 10000, 19000],
        early_stop=dict(minimum_updates=6000, relative_improvement=.01, stale_milestones=2),
        automatic_expansion=True, automatic_retry='one legal checkpoint recovery per training stage; never rerun completed trajectories',
        kit_isolation='one fresh Kit per controlled trajectory; stale articulation scene-query cache after reset',
        test_calls=2, maximum_controlled_trajectories=174,
        formal_specs=[dict(identity=i, cohort='test', seed=983101+n) for n, i in enumerate(sorted(data['splits']['test'])[:20])],
        initialization='same preregistered 4 fresh + train4/validation4; sorted first20 test IDs, no outcome filtering',
        admission='passed unchanged numeric audit; legal expert>=10/12 and saved>=6/8; two exact command replays',
        training_scope='pilots4 then full train320/validation40; test40 only after frozen EMA selection')
    atomic_json(root/'session.json', session)
    atomic_json(root/'gpu_schedule.json', dict(policy='shared_recent_v1', fixed_utilization_veto=False,
        free_vram_mib=6144, available_ram_gib=12, worker_count=1, external_throughput_observable=False,
        recent_window_seconds=[60, 300], pacing='external present .25; recent1m>85:1; 1m>92 and5m>85:2'))
    for name in ('preservation', 'parity'):
        relevant = ('visuotactile/state_history.py', 'visuotactile/environment_rng.py', 'visuotactile/state_history_audit.py',
            'visuotactile/native_data.py', 'visuotactile/models/native_dp.py')
        for path in relevant:
            if old['implementation_sha256'][path] != sha256(path):
                raise ValueError('Prior numerical audit is stale: '+path)
        shutil.copytree(oldroot/name, root/name)
        atomic_json(root/(name+'_committed.json'), dict(sha256=sha256(root/name/'result.json'),
            reused_from=str(oldroot/name), immutable_numeric_sources=True))
    for name in ('freeze',):
        if not (preflight/name/'result.json').exists():
            raise ValueError('Preflight stage incomplete: '+name)
        shutil.copytree(preflight/name, root/name)
        atomic_json(root/(name+'_committed.json'), dict(sha256=sha256(root/name/'result.json'), reused_from=str(preflight/name)))
    atomic_json(root/'status.json', dict(state='registered', new_updates=0))


def should_stop(history, minimum=6000):
    if len(history) < 3 or history[-1]['update'] < minimum:
        return False
    earlier = min(r['validation']['selection_mse'] for r in history[:-2])
    return min(r['validation']['selection_mse'] for r in history[-2:]) >= .99*earlier


def image_success_gate(cohort, validation):
    rows=[row for row in cohort['trials'] if row['cohort']=='validation']
    registered=[s['identity'] for s in validation]
    identities=[row['identity'] for row in rows]
    successes={r['identity'] for r in rows if r.get('legal') and r.get('success')}
    passed=(cohort.get('software_complete') is True and len(registered)>=8
        and len(registered)==len(set(registered)) and sorted(identities)==sorted(registered)
        and all(r.get('legal') and r.get('software_complete') for r in rows) and len(successes)>=2)
    return dict(passed=passed,distinct_successes=sorted(successes),validation_completed=len(rows),
        validation_registered=len(registered))


def initialize_position(root):
    from .position_control import contract,image_contract
    previous=Path('outputs/visuotactile/native_dp_v2/state_full_20261102_01').resolve()
    old=json.loads((previous/'session.json').read_text())
    if (root/'session.json').exists():raise ValueError('Never reset a registered position session')
    audit=json.loads((root/'position_audit/result.json').read_text())
    if not audit['passed']:raise ValueError('Direct recorded target audit required')
    data=json.loads((Path(old['source_root'])/'data.json').read_text())
    began=time.time()
    specs=[s for s in json.loads((previous/'freeze/benchmark.json').read_text())['specs'] if s['cohort']=='train']
    session=dict(old,position_study=True,started_epoch=began,deadline=began+24*3600,
        training_deadline=began+21*3600,gate_deadline=began+3*3600,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        implementation_sha256={str(p):sha256(p) for p in list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks').rglob('*.py'))},
        previous_root=str(previous),fresh_candidates=[982501,982502],fresh_count=2,saved_specs=specs,
        validation_candidates=[dict(identity=i,cohort='validation',seed=984101+n) for n,i in enumerate(sorted(data['splits']['validation']))],
        validation_count=8,kit_retries_per_stage=2,kit_retries_total=12,
        profiles={'native_h32':dict(contract=contract(32,False),updates=1000,optimizer_horizon=19000),
            'position_h32':dict(contract=contract(32,True),updates=19000,optimizer_horizon=19000),
            **{profile:dict(contract=image_contract(group),group=group,updates=3000,
                optimizer_horizon=3000,milestones=[300,1000,3000]) for profile,group in
                [('image_official_h32','I-OfficialDP'),('image_history_h32','I-HistoryDP')]}},
        state_milestones=[1000,3000,6000,10000,19000],maximum_controlled_trajectories=160,
        kit_gate=dict(startup_probes=4,experts=2,position_replays=2,maximum_pre_scene_retries=2,
            startup_prefix='kit_isolated',shutdown='isolated_process_exit'),
        image_gate=dict(validation_legal=8,distinct_native_successes=2,frozen_before_outcomes=True),
        continuation=dict(after_1000_arm_rad=.25,after_1000_hand_rad=.25,nonzero_history_gradient=True),
        learning_expert_min_successes=10,learning_expert_validation_min=6,
        bc_training_authorized=False,runtime_amendments=[],training_runtime_compatibility=[])
    atomic_json(root/'session.json',session)
    atomic_json(root/'status.json',dict(state='registered',position_audit_steps=audit['steps']))
    atomic_json(root/'image_status.json',dict(state='not_started',reason='State validation success Gate pending'))


def startup_failure_evidence(root, name):
    """Classify known native failures only before scene or trajectory artifacts."""
    root = Path(root); output = root/name; log = root/(name+'.log')
    if not output.is_dir() or not log.is_file():
        return None
    files = {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()}
    contents = log.read_text(errors='replace')
    allowed = {'runtime_controls.json','runtime_before.json','runtime_ready.json','lifecycle.json','app_ready.json'}
    if 'runtime_controls.json' not in files or not files <= allowed:
        return None
    if (output/'lifecycle.json').exists() and json.loads((output/'lifecycle.json').read_text()).get('state')!='launching':
        return None
    if '[INFO]: Completed setting up the environment' in contents or 'Traceback (most recent call last)' in contents:
        return None
    stack = all(signature in contents for signature in ('FileThread', 'std::system_error', 'abort'))
    exit_confirmed = False
    segv_confirmed = False
    for path in (root/(name+'_exit.json'), root/'blocked.json'):
        if path.exists():
            value = json.loads(path.read_text())
            reason = value.get('error', value.get('reason', ''))
            exit_confirmed |= f'{name} failed: exit=-6' in reason
            segv_confirmed |= f'{name} failed: exit=-11' in reason
    truncated = exit_confirmed and all(signature in contents for signature in (
        "terminate called after throwing an instance of 'std::system_error'", 'what():  Invalid argument'))
    filethread_segv = segv_confirmed and all(signature in contents for signature in (
        '[Fatal] [carb.crashreporter-breakpad.plugin]',
        'libomniclient.so!omniClientFreeContent', 'omniclient::provider_file::FileThread'))
    if not (stack or truncated or filethread_segv):
        return None
    classification = ('pre-scene FileThread/omniClientFreeContent SIGSEGV' if filethread_segv
        else 'pre-scene FileThread abort' if stack
        else 'pre-scene system_error/Invalid argument abort; native stack missing')
    return dict(native_stack_available=stack or filethread_segv, confirmed_sigabrt=exit_confirmed,
        confirmed_sigsegv=segv_confirmed, classification=classification)


def recover_preaction_start(root, name, deadline):
    """Retry one classified pre-scene failure, with a session-wide bound."""
    root = Path(root); output = root/name; log = root/(name+'.log')
    recoveries = root/'startup_recoveries'
    if time.time() >= deadline:
        return False
    session=json.loads((root/'session.json').read_text()) if (root/'session.json').exists() else {}
    per_stage=session.get('kit_retries_per_stage',1); total=session.get('kit_retries_total',3)
    prior=[p for p in recoveries.glob('*/recovery.json') if json.loads(p.read_text()).get('stage')==name]
    if len(prior)>=per_stage or len(list(recoveries.glob('*/recovery.json'))) >= total:
        return False
    evidence = startup_failure_evidence(root, name)
    if evidence is None:
        return False
    archive = recoveries/(name if not prior else name+f'_retry_{len(prior)+1}'); archive.mkdir(parents=True)
    output.rename(archive/'attempt')
    log.rename(archive/'worker.log')
    launch = root/(name+'_launch.json')
    if launch.exists():
        launch.rename(archive/'launch.json')
    atomic_json(archive/'recovery.json', dict(utc=time.time(), stage=name, deadline=deadline,
        reason=evidence['classification'], evidence=evidence,
        original_log_sha256=sha256(archive/'worker.log'), same_seed_and_job=True,
        new_controlled_trajectories=0, maximum_per_stage=per_stage, maximum_per_session=total))
    if session.get('position_study'):
        time.sleep(min(3*3**len(prior),30))
    return True


def initialize_motion(root):
    """Register one bounded continuation; never restart any predecessor session."""
    from .position_control import contract
    from .state_motion import SAMPLERS, SEEDS, MOVEMENT_RAD
    import torch
    import inspect
    import diffusers
    from diffusers import DDIMScheduler
    previous=Path('outputs/visuotactile/native_dp_v2/position_20261103_01').resolve()
    old=json.loads((previous/'session.json').read_text())
    checkpoint=previous/'position_h32_train/best.pt'
    expected='5ecc930ffd52af166e00904b0515159193d8527d9f5726908bf57967c445a416'
    if sha256(checkpoint)!=expected:raise ValueError('Original best identity differs from reviewed result')
    if root.exists():raise ValueError('Motion session directory already exists; no deadline reset')
    data=json.loads((Path(old['source_root'])/'data.json').read_text())
    if sha256(Path(old['source_root'])/'data.json')!=old['data_sha256']:raise ValueError('Data identity drift')
    cache=previous/'position_h32_numeric_trainval_cache.pt'
    cache_sha=sha256(cache);began=time.time();root.mkdir(parents=True)
    specs=json.loads((previous/'freeze/benchmark.json').read_text())['specs']
    specs=[s for s in specs if s['cohort'] in ('train','validation')]
    training_ids=data['splits']['train']
    single=[s['identity'] for s in specs if s['cohort']=='train'][:1]
    common=dict(updates=19000,optimizer_horizon=19000,numeric_cache=str(cache),numeric_cache_sha256=cache_sha,
        sampling=SAMPLERS['A0'])
    session=dict(old,motion_study=True,previous_root=str(previous),started_epoch=began,
        deadline=began+24*3600,training_deadline=began+21*3600,gate_deadline=began+3*3600,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        implementation_sha256={str(p):sha256(p) for p in list(Path('visuotactile').rglob('*.py'))+list(Path('isaacsimenvs/tasks').rglob('*.py'))},
        kit_retries_per_stage=2,kit_retries_total=2,maximum_controlled_trajectories=160,
        motion_checkpoint=str(checkpoint),motion_checkpoint_sha256=expected,
        motion_specs=specs,motion_validation16=sorted(data['splits']['validation'])[:16],
        motion_validation40=sorted(data['splits']['validation']),motion_seeds=list(SEEDS),
        motion_samplers=SAMPLERS,motion_threshold_rms_rad=MOVEMENT_RAD,
        motion_primary='episode/seed paired moving Arm7 prefix12 RMSE radians; >=3% improvement, Arm/Hand overall <=102% baseline, direction no worse by .02, cycle p95<=.2s',
        profiles={'single_fit':dict(common,contract=contract(32),updates=3000,optimizer_horizon=3000,
            train_ids=single,validation_ids=sorted(data['splits']['validation'])[:8],selection_scope='train_fit',
            initialization=dict(path=str(checkpoint),sha256=expected)),
            'h24':dict(common,contract=contract(24)),
            'temporal_h32':dict(common,contract=contract(32),temporal_rule='min(.1,.05*base/temporal), first3 train segments before updates, then frozen')},
        motion_continuation='single-fit Arm<=.005rad, Hand<=.03rad and moving amplitude .8..1.2 required; E/F max19k each, stages1k3k6k10k19k',
        test_calls=1,automatic_expansion=False,bc_training_authorized=False,
        resource_target_percent=50.,runtime_amendments=[],training_runtime_compatibility=[])
    dependency=inspect.getfile(DDIMScheduler)
    session['dependency_sha256']={dependency:sha256(dependency)}
    session['diffusers_version']=diffusers.__version__
    atomic_json(root/'session.json',session)
    for name in ('position_audit','freeze'):
        (root/name).symlink_to(previous/name,target_is_directory=True)
    # Immutable train320 statistics; do not refit on the one-trajectory subset.
    (root/'position_train_statistics.pt').symlink_to(previous/'position_train_statistics.pt')
    preserved={str(previous/p):sha256(previous/p) for p in
        ('position_h32_train/best.pt','position_train_statistics.pt','position_audit/result.json',
         'expert_learning/result.json','position_replay/result.json','freeze/benchmark.json')}
    atomic_json(root/'preserved.json',preserved)
    # Recreate original shuffled segment sampling exactly, without training.
    records=torch.load(cache,map_location='cpu',weights_only=False)['records']
    schedule=[(i,start) for i in training_ids for start in range(0,len(records[i]['time']),96)]
    generator=torch.Generator().manual_seed(20261102);order=[];cursor=0;counts={k:0 for k in schedule}
    for _ in range(19000*2):
        if cursor==len(order):order=torch.randperm(len(schedule),generator=generator).tolist();cursor=0
        counts[schedule[order[cursor]]]+=1;cursor+=1
    atomic_json(root/'sampling_audit.json',dict(updates=19000,segments_per_update=2,tbptt_ticks=96,
        complete_prefix_recomputed=True,shuffle_without_replacement=True,
        rows=[dict(identity=i,start_tick=start,visits=n,plan_anchors=list(range(start,min(start+96,len(records[i]['time'])),12)))
              for (i,start),n in counts.items()],policy_unchanged=True))
    atomic_json(root/'status.json',dict(state='registered',new_updates=0))


def motion_sampler_candidates(result, limit=2):
    """Pre-registered diagnostic shortlist; test results can never enter it."""
    def metric(name,key):return result['summary'][name][key]['mean']
    baseline=result['summary']['A0'];eligible=[]
    for name in result['summary']:
        if name=='A0':continue
        latency=result['planning_cycle_latency_s'][name]['p95']
        if (latency is not None and latency<=.2
                and metric(name,'moving_arm12_rmse_rad')<=.97*metric('A0','moving_arm12_rmse_rad')
                and metric(name,'arm12_rmse_rad')<=1.02*metric('A0','arm12_rmse_rad')
                and metric(name,'hand12_rmse_rad')<=1.02*metric('A0','hand12_rmse_rad')
                and metric(name,'moving12_direction_cosine')>=metric('A0','moving12_direction_cosine')-.02):
            eligible.append(name)
    return sorted(eligible,key=lambda n:metric(n,'moving_arm12_rmse_rad'))[:limit]


class Runner(RepairRunner):
    def check(self):
        super().check()
        if self.session.get('position_study') and not (self.root/'foundation.json').exists() and time.time()>=self.session['gate_deadline']:
            raise TimeoutError('Original position Gate deadline expired')

    def stage(self, name, module='visuotactile.state_benchmark', extra=None):
        output = self.root/name; marker = self.root/(name+'_committed.json')
        if self.session.get('position_study') and not (self.root/'foundation.json').exists() and time.time()>=self.session['gate_deadline']:
            raise TimeoutError('Original position Gate deadline expired')
        for path, digest in self.session['implementation_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Registered source drift: '+path)
        if output.exists() and not marker.exists():
            if (output/'result.json').exists() and (self.root/(name+'_launch.json')).exists():
                result = json.loads((output/'result.json').read_text())
                closed = (output/'closed.json').exists()
                if result.get('software_complete') and (module!='visuotactile.state_benchmark' or not self.session.get('position_study') or closed):
                    atomic_json(marker, dict(sha256=sha256(output/'result.json'), recovered_completion=True))
                else:
                    raise RuntimeError('Results saved but process shutdown incomplete: '+name)
            elif module == 'visuotactile.state_full_training':
                extra = extra or []; kind = extra[extra.index('--kind')+1]
                profile = extra[extra.index('--profile')+1] if '--profile' in extra else kind
                directory = self.root/(profile+('_pilot_train' if '--pilot' in extra else '_train'))
                if not (directory/'last.pt').exists() or (self.root/(name+'_recovery.json')).exists():
                    raise RuntimeError('No legal checkpoint or recovery bound reached: '+name)
                output.rename(self.root/(name+'_interrupted'))
                atomic_json(self.root/(name+'_recovery.json'), dict(utc=time.time(),
                    checkpoint_sha256=sha256(directory/'last.pt'), deadline_unchanged=True))
            elif module == 'visuotactile.state_benchmark' and recover_preaction_start(
                    self.root, name, self.session['deadline']):
                pass
            else:
                # A resumed owner has no live worker status for this preserved attempt.
                # Identify it explicitly so learning deferral inspects this exact stage.
                atomic_json(self.root/'status.json', dict(state='incomplete_attempt', stage=name,
                    pid=os.getpid(), utc=time.time()))
                raise RuntimeError('Incomplete simulation attempt preserved, no blind repeat: '+name)
        try:
            result=super().stage(name, module, extra)
            if self.session.get('position_study') and module=='visuotactile.state_benchmark':
                self.verify_worker_exit(name)
            return result
        except RuntimeError as error:
            if module == 'visuotactile.state_benchmark':
                atomic_json(self.root/(name+'_exit.json'), dict(error=str(error), utc=time.time()))
            if module == 'visuotactile.state_benchmark' and recover_preaction_start(
                    self.root, name, self.session['deadline']):
                return self.stage(name, module, extra)
            if self.session.get('motion_study'):
                # A worker exit may represent NaN/interface/data failures; only
                # explicitly classified Kit initialization gets an automatic retry.
                raise
            if module != 'visuotactile.state_full_training' or (self.root/(name+'_recovery.json')).exists():
                raise
            return self.stage(name, module, extra)

    def verify_worker_exit(self,name):
        output=self.root/name
        if (output/'closed.json').exists():return
        request=json.loads((output/'process_exit_requested.json').read_text())
        exited=json.loads((self.root/(name+'_process_exit.json')).read_text())
        if exited['code']!=0 or not exited['reaped'] or exited['pid']!=request['pid'] or request['result_sha256']!=sha256(output/'result.json'):
            raise RuntimeError('Durable worker exit evidence invalid: '+name)
        for attempt in range(5):
            gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).split()
            if not Path('/proc',str(exited['pid'])).exists() and str(exited['pid']) not in gpu_pids:
                atomic_json(output/'closed.json',dict(utc=time.time(),pid=exited['pid'],
                    method='parent verified isolated process exit',sdk_graceful_shutdown=False,
                    process_absent=True,gpu_process_absent=True))
                return
            time.sleep(1)
        raise RuntimeError('Worker resources not released: '+name)

    def evaluate(self, name, mode, checkpoint=None, specs=None, continue_errors=False, inference=None, takeover_steps=None):
        if specs is None:
            specs = json.loads((self.root/'freeze/benchmark.json').read_text())['specs']
        output = self.root/name; output.mkdir(exist_ok=True)
        trials = [];missing=[]
        for ordinal, spec in enumerate(specs):
            child = f'{name}_single_{ordinal:02d}'
            job = self.root/(child+'_job.json')
            values = dict(specs=[spec], video=ordinal == 0)
            if inference is not None: values['inference']=inference
            if takeover_steps is not None: values['takeover_steps']=takeover_steps
            if checkpoint is not None:
                values.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
            if job.exists() and json.loads(job.read_text()) != values:
                raise ValueError('Frozen evaluation job changed')
            atomic_json(job, values)
            if self.session.get('position_study') and not (self.root/child).exists():
                count=sum(1 for p in self.root.glob('*_single_*/*/started.json'))
                if count>=self.session['maximum_controlled_trajectories']:
                    raise RuntimeError('Registered trajectory bound reached')
            try:
                result = self.stage(child, extra=['--mode', mode, '--job', str(job)])
            except RuntimeError as error:
                if not continue_errors:raise
                evidence=startup_failure_evidence(self.root,child)
                committed=self.root/child/'result.json'
                saved=json.loads(committed.read_text()) if committed.exists() else {}
                row=dict(spec=spec,stage=child,error=str(error),native_outcome=None,
                    classification='pre_scene_native_fault' if evidence else 'worker_incomplete',evidence=evidence)
                if saved.get('software_complete') and len(saved.get('trials',[]))==1:
                    trials+=saved['trials']
                    row.update(native_outcome=saved['trials'][0].get('success'),classification='result_saved_process_cleanup_failed')
                    destination=output/spec['identity']
                    if not destination.exists():destination.symlink_to((self.root/child/spec['identity']).resolve(),target_is_directory=True)
                missing.append(row)
                atomic_json(output/'progress.json',dict(completed=len(trials),trials=trials,missing=missing))
                continue
            if len(result['trials']) != 1:
                raise ValueError('Singleton trajectory count mismatch')
            trials += result['trials']
            destination = output/spec['identity']
            if not destination.exists():
                destination.symlink_to((self.root/child/spec['identity']).resolve(), target_is_directory=True)
            atomic_json(output/'progress.json', dict(completed=len(trials), trials=trials,missing=missing))
        result = dict(software_complete=not missing, mode=mode, trials=trials,missing=missing,registered=len(specs),fresh_kit_per_trial=True)
        atomic_json(output/'result.json', result)
        return result

    def evaluate_learning(self, name, mode, checkpoint, specs):
        """A launcher abort cannot invalidate already passed control/learning."""
        if getattr(self,'session',{}).get('position_study'):
            return self.evaluate(name,mode,checkpoint,specs,continue_errors=True)
        try:
            result = self.evaluate(name, mode, checkpoint, specs)
            (self.root/(name+'_deferred.json')).unlink(missing_ok=True)
            return result
        except RuntimeError:
            status = json.loads((self.root/'status.json').read_text())
            child = status.get('stage', '')
            evidence = startup_failure_evidence(self.root, child)
            if not child.startswith(name+'_single_') or evidence is None:
                raise
            if not json.loads((self.root/'foundation.json').read_text())['passed']:
                raise
            atomic_json(self.root/(name+'_deferred.json'), dict(utc=time.time(), failed_stage=child,
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), specs=specs,
                evidence=evidence, reason='Evaluation startup unavailable; validated numerical training can continue',
                native_outcome=None, counted_as_success=False))
            return None

    def position_gates(self):
        """Front-load actual App lifecycle and actuator replay before optimization."""
        if (self.root/'foundation.json').exists():
            from .state_history_training import require_foundation
            require_foundation(self.root)
            return
        if time.time() >= self.session['gate_deadline']:
            raise TimeoutError('Original position Gate deadline expired')
        prefix = self.session['kit_gate'].get('startup_prefix','kit_startup')
        for index in range(self.session['kit_gate']['startup_probes']):
            name=f'{prefix}_{index:02d}'
            self.stage(name,extra=['--mode','startup'])
            if not (self.root/name/'closed.json').exists():
                raise RuntimeError('Kit did not complete graceful lifecycle: '+name)
        self.stage('freeze',extra=['--mode','freeze'])
        specs=json.loads((self.root/'freeze/benchmark.json').read_text())['specs']
        fresh=[s for s in specs if s['cohort']=='fresh']
        expert=self.evaluate('expert','expert',specs=fresh)
        replay=self.evaluate('position_replay','position_replay',specs=fresh)
        rows=replay['trials']
        checks=dict(data=json.loads((self.root/'position_audit/result.json').read_text())['passed'],
            experts=len(expert['trials'])==2 and all(t['legal'] and t['success'] for t in expert['trials']),
            replay=len(rows)==2 and all(t['success'] and t['submitted_target_max_error']==0 and
                t['actuator_target_max_error']==0 and all(t['bytewise_fields'].values()) for t in rows),
            kit_retry_count=len(list((self.root/'startup_recoveries').glob('*/recovery.json'))) <= self.session['kit_gate']['maximum_pre_scene_retries'])
        paths=['position_audit/result.json','freeze/result.json','expert/result.json','position_replay/result.json']
        paths += [f'{prefix}_{i:02d}/result.json' for i in range(self.session['kit_gate']['startup_probes'])]
        paths += [f'{prefix}_{i:02d}/closed.json' for i in range(self.session['kit_gate']['startup_probes'])]
        atomic_json(self.root/'foundation.json',dict(passed=all(checks.values()),checks=checks,
            evidence_sha256={p:sha256(self.root/p) for p in paths},position_replay_required=True))
        if not all(checks.values()):raise RuntimeError('Position/control/Kit foundation failed: '+str(checks))
        atomic_json(self.root/'status.json',dict(state='position_gates_passed',utc=time.time(),new_updates=0))

    def position_training(self):
        """One staged V2 pipeline, full split, no BC and no test-based decisions."""
        from .state_history_training import require_foundation
        require_foundation(self.root)
        specs=json.loads((self.root/'freeze/benchmark.json').read_text())['specs']
        train=[s for s in specs if s['cohort']=='train']
        validation=[s for s in specs if s['cohort']=='validation']
        # Actual online controls on every learning initialization, not geometry alone.
        if not (self.root/'expert_learning/result.json').exists():
            self.evaluate('expert_learning','expert',specs=train+validation)
        expert=json.loads((self.root/'expert_learning/result.json').read_text())
        expert_ok=(expert['software_complete'] and len(expert['trials'])==len(train+validation)
            and all(t['legal'] for t in expert['trials'])
            and sum(t['success'] for t in expert['trials'])>=self.session['learning_expert_min_successes']
            and sum(t['success'] for t in expert['trials'] if t['cohort']=='validation')>=self.session['learning_expert_validation_min'])
        atomic_json(self.root/'learning_environment.json',dict(passed=expert_ok,
            expert_successes=sum(t.get('success',False) for t in expert['trials']),required=len(train+validation)))
        if not expert_ok:raise RuntimeError('Registered learning initialization expert control failed')
        for profile in ('native_h32','position_h32'):
            study=self.session['profiles'][profile]
            milestones=[1000] if profile=='native_h32' else self.session['state_milestones']
            directory=self.root/(profile+'_train'); stop_reason=None
            for update in milestones:
                result=self.stage(f'{profile}_train_{update}','visuotactile.state_full_training',
                    ['--kind','dp','--profile',profile,'--target',str(update)])
                checkpoint=directory/f'milestone_{update}.pt'
                loop=self.evaluate_learning(f'{profile}_loop_{update}','dp',checkpoint,train[:2]+validation[:2])
                metrics=result['history'][-1]['validation']
                if profile=='position_h32' and update==1000:
                    thresholds=self.session['continuation']
                    passed=metrics['arm_rmse']<=thresholds['after_1000_arm_rad'] and metrics['hand_rmse']<=thresholds['after_1000_hand_rad'] and result['history_gradient_nonzero']
                    atomic_json(self.root/'position_learning_gate.json',dict(passed=passed,validation=metrics,
                        threshold=thresholds,scope='Offline continuation evidence only; no task success claim'))
                    if not passed:stop_reason='small_budget_position_fit_failed';break
                if profile=='position_h32':
                    # Eight IDs are frozen before any outcome. Same checkpoint per cohort.
                    cohort=self.evaluate_learning(f'{profile}_validation_{update}','dp',checkpoint,validation)
                    if self.register_image_success(cohort,validation,checkpoint,f'{profile}_validation_{update}'):
                        stop_reason='preregistered_validation_success_gate';break
                    if should_stop(result['history']):stop_reason='validation_plateau';break
            # Freeze lowest validation error; never switch by test outcome.
            best=__import__('torch').load(directory/'best.pt',map_location='cpu',weights_only=False)
            selection=dict(checkpoint_sha256=sha256(directory/'best.pt'),updates=best['updates'],
                criterion='minimum validation40 first12 balanced MSE',stop_reason=stop_reason,contract=best['contract'])
            del best
            atomic_json(self.root/(profile+'_selection.json'),selection)
            selected=self.evaluate_learning(profile+'_selected','dp',directory/'best.pt',train+validation)
            if profile=='position_h32':
                self.register_image_success(selected,validation,directory/'best.pt',profile+'_selected')
            if selected['software_complete']:
                self.stage(profile+'_offline_test','visuotactile.state_full_training',
                    ['--kind','dp','--profile',profile,'--target',str(selection['updates']),'--final-test'])
            else:
                atomic_json(self.root/(profile+'_test_withheld.json'),dict(reason='Selected closed-loop cohort incomplete'))
        if (self.root/'image_gate.json').exists():
            self.image_training(train,validation)
        else:
            atomic_json(self.root/'image_status.json',dict(state='not_started',reason='Fewer than two complete native successes on a complete preregistered eight-ID validation cohort'))
        atomic_json(self.root/'status.json',dict(state='position_study_finished',utc=time.time(),
            research_success=(self.root/'image_gate.json').exists(),image=json.loads((self.root/'image_status.json').read_text())))

    def register_image_success(self,cohort,validation,checkpoint,stage):
        from .state_history_training import require_foundation
        decision=image_success_gate(cohort,validation)
        if decision['passed']:
            require_foundation(self.root)
            atomic_json(self.root/'image_gate.json',dict(**decision,validation_stage=stage,
                validation_sha256=sha256(self.root/stage/'result.json'),checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),registered_validation=validation,
                foundation_sha256=sha256(self.root/'foundation.json')))
        return decision['passed']

    def image_training(self,train,validation):
        """Sequential existing visual networks, only after auditable State success."""
        for profile in ('image_official_h32','image_history_h32'):
            study=self.session['profiles'][profile]
            atomic_json(self.root/'image_status.json',dict(state='running',profile=profile,utc=time.time()))
            directory=self.root/(profile+'_train')
            for update in study['milestones']:
                result=self.stage(f'{profile}_train_{update}','visuotactile.state_full_training',
                    ['--kind','dp','--profile',profile,'--target',str(update)])
                self.evaluate_learning(f'{profile}_loop_{update}','dp',directory/f'milestone_{update}.pt',train[:2]+validation[:2])
                if should_stop(result['history'],minimum=1000):break
            best=__import__('torch').load(directory/'best.pt',map_location='cpu',weights_only=False)
            selection=dict(checkpoint_sha256=sha256(directory/'best.pt'),updates=best['updates'],criterion='minimum validation40 MSE')
            del best
            atomic_json(self.root/(profile+'_selection.json'),selection)
            selected=self.evaluate_learning(profile+'_selected','dp',directory/'best.pt',train+validation)
            if selected['software_complete']:
                self.stage(profile+'_offline_test','visuotactile.state_full_training',
                    ['--kind','dp','--profile',profile,'--target',str(selection['updates']),'--final-test'])
        atomic_json(self.root/'image_status.json',dict(state='finished',utc=time.time()))

    def motion_analysis(self,name,checkpoint,identities,samplers,split='validation',timing=True,overlap_steps=0):
        job=self.root/(name+'_job.json')
        values=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),identities=identities,
            samplers=samplers,seeds=self.session['motion_seeds'],split=split,timing=timing,overlap_steps=overlap_steps)
        if job.exists() and json.loads(job.read_text())!=values:raise ValueError('Frozen motion job drift')
        atomic_json(job,values)
        return self.stage(name,'visuotactile.state_motion',['--job',str(job)])

    def run_motion(self):
        """Reuse isolated workers, leases, checkpoints and the fixed stage owner."""
        from .state_motion import SAMPLERS
        try:
            (self.root/'STOP_MONITOR').unlink(missing_ok=True)
            with (self.root/'monitor.log').open('a') as log:
                self.monitor=subprocess.Popen(['bash','visuotactile/run_cpu.sh','-m','visuotactile.state_resources',
                    '--root',str(self.root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            for path,digest in json.loads((self.root/'preserved.json').read_text()).items():
                if sha256(path)!=digest:raise ValueError('Protected predecessor changed: '+path)
            self.stage('motion_smoke',extra=['--mode','smoke'])
            if not (self.root/'foundation.json').exists():
                old=Path(self.session['previous_root'])
                expert=json.loads((old/'expert_learning/result.json').read_text())
                replay=json.loads((old/'position_replay/result.json').read_text())
                if not (all(t['success'] and t['legal'] for t in expert['trials'])
                        and all(t['success'] and all(t['bytewise_fields'].values()) for t in replay['trials'])):
                    raise ValueError('Preserved control acceptance is invalid')
                evidence=['motion_smoke/result.json','motion_smoke/closed.json','preserved.json',
                    'position_audit/result.json','freeze/benchmark.json']
                atomic_json(self.root/'foundation.json',dict(passed=True,source='previous audited control plus current real one-step smoke',
                    evidence_sha256={p:sha256(self.root/p) for p in evidence}))
            checkpoint=Path(self.session['motion_checkpoint'])
            screen=self.motion_analysis('sampling16',checkpoint,self.session['motion_validation16'],SAMPLERS)
            shortlist=motion_sampler_candidates(screen)
            full=self.motion_analysis('sampling40',checkpoint,self.session['motion_validation40'],
                {n:SAMPLERS[n] for n in ['A0']+shortlist})
            selected=(motion_sampler_candidates(full,1) or ['A0'])[0]
            atomic_json(self.root/'sampling_selection.json',dict(selected=selected,shortlist=shortlist,
                selected_on='validation40 paired moving Arm7 prefix12 + overall errors + direction + p95 latency',test_used=False))
            specs=self.session['motion_specs'];train=[s for s in specs if s['cohort']=='train'];val=[s for s in specs if s['cohort']=='validation']
            baseline=dict(SAMPLERS['A0'],overlap_steps=0,sampler_seed=7091)
            candidate=dict(SAMPLERS[selected],overlap_steps=0,sampler_seed=7091)
            groups={}
            groups['Original']=dict(stage='C0',checkpoint=str(checkpoint),inference=baseline,
                offline_stage='sampling40',offline_name='A0',
                result=self.evaluate('C0','dp',checkpoint,specs,inference=baseline))
            if selected!='A0':
                groups['Sampling adjusted']=dict(stage='C1',checkpoint=str(checkpoint),inference=candidate,
                    offline_stage='sampling40',offline_name=selected,
                    result=self.evaluate('C1','dp',checkpoint,specs,inference=candidate))
                groups['Sampling adjusted']['deployable']=all(t['planning_latency_s']['p95']<=.2
                    for t in groups['Sampling adjusted']['result']['trials'])
                if not groups['Sampling adjusted']['deployable']:
                    candidate=baseline
                    selected='A0'
            base_group=groups.get('Sampling adjusted',groups['Original'])
            if not base_group.get('deployable',True):base_group=groups['Original']
            # Both teacher forcing and on-policy evidence stay separately reported.
            teacher=full['summary'][selected]
            ratios=[t.get('plan_consistency',{}).get('arm_boundary_jump_rms_rad',0)/
                    max(t.get('plan_consistency',{}).get('arm_internal_jump_rms_rad',0),1e-12)
                    for t in base_group['result']['trials']]
            discontinuity=(teacher['arm_boundary_jump_rms_rad']['mean']>
                1.5*teacher['arm_internal_jump_rms_rad']['mean'] or (ratios and sum(ratios)/len(ratios)>1.5))
            atomic_json(self.root/'overlap_decision.json',dict(run=bool(discontinuity),
                threshold='boundary/internal RMS ratio >1.5 in teacher-forced or on-policy condition',on_policy_ratios=ratios))
            if discontinuity:
                transition=dict(candidate,overlap_steps=4)
                groups['Overlap transition']=dict(stage='C2',checkpoint=str(checkpoint),inference=transition,
                    offline_stage='overlap_offline',offline_name=selected,
                    result=self.evaluate('C2','dp',checkpoint,specs,inference=transition))
                self.motion_analysis('overlap_offline',checkpoint,self.session['motion_validation40'],
                    {selected:SAMPLERS[selected]},overlap_steps=4)
            def successes(group):
                return {t['identity'] for t in group['result']['trials'] if t['cohort']=='validation' and t.get('success') and t.get('legal')}
            stable=False
            for label,group in list(groups.items()):
                if not group.get('deployable',True):continue
                if len(successes(group))>=2:
                    repeated=self.evaluate(group['stage']+'_repeat','dp',Path(group['checkpoint']),val,
                        inference=dict(group['inference'],sampler_seed=7092))
                    repeated_ids={t['identity'] for t in repeated['trials'] if t.get('success') and t.get('legal')}
                    if len(successes(group)&repeated_ids)>=2:
                        stable=True
                        data=json.loads((Path(self.session['source_root'])/'data.json').read_text())
                        extra=[i for i in sorted(data['splits']['validation']) if i not in {s['identity'] for s in val}][:8]
                        expanded=[dict(identity=i,cohort='validation',seed=985101+n) for n,i in enumerate(extra)]
                        self.evaluate(group['stage']+'_expanded','dp',Path(group['checkpoint']),expanded,inference=group['inference'])
                        atomic_json(self.root/'repeated_success.json',dict(group=label,identities=sorted(successes(group)&repeated_ids),
                            limitation='Paired observable initial states and isolated RNG; internal PhysX state not claimed exact'))
                        break
            if not stable:
                single=self.session['profiles']['single_fit'];fit=None
                for update in (1000,3000):
                    fit=self.stage('single_fit_'+str(update),'visuotactile.state_full_training',
                        ['--kind','dp','--profile','single_fit','--target',str(update)])
                    metrics=fit['history'][-1]['train']
                    fitted=(metrics['arm_rmse']<=.005 and metrics['hand_rmse']<=.03
                        and .8<=metrics.get('moving12_amplitude_ratio',0)<=1.2)
                    atomic_json(self.root/'single_fit_gate.json',dict(passed=fitted,updates=update,train=metrics,
                        thresholds=dict(arm_rmse_rad=.005,hand_rmse_rad=.03,amplitude_ratio=[.8,1.2]),
                        finite_budget_not_proof_of_architectural_impossibility=True))
                    if fitted:break
                if fitted:
                    single_ckpt=self.root/'single_fit_train'/f'milestone_{update}.pt'
                    single_spec=[s for s in train if s['identity']==single['train_ids'][0]]
                    one=self.evaluate('single_fit_loop','dp',single_ckpt,single_spec,inference=baseline)
                    if not all(t['success'] for t in one['trials']):
                        for ticks in (12,24):
                            self.evaluate('single_takeover_'+str(ticks),'dp',single_ckpt,single_spec,
                                inference=baseline,takeover_steps=ticks)
                    for profile,label in [('h24','H24'),('temporal_h32','Temporal-loss')]:
                        directory=self.root/(profile+'_train')
                        for update in (1000,3000,6000,10000,19000):
                            result=self.stage(profile+'_'+str(update),'visuotactile.state_full_training',
                                ['--kind','dp','--profile',profile,'--target',str(update)])
                            self.evaluate(profile+'_loop_'+str(update),'dp',directory/f'milestone_{update}.pt',
                                train[:2]+val[:2],inference=baseline)
                            history=result['history']
                            if len(history)>=3 and update>=6000:
                                old=min(r['validation']['moving_arm12_rmse_rad'] for r in history[:-2])
                                stale=all(r['validation']['moving_arm12_rmse_rad']>=.99*old for r in history[-2:])
                                overfit=(history[-1]['train']['selection_mse']<.9*history[-3]['train']['selection_mse']
                                    or history[-1]['validation']['arm_rmse']>1.1*history[-3]['validation']['arm_rmse'])
                                if stale and overfit:
                                    atomic_json(self.root/(profile+'_early_stop.json'),dict(updates=update,reason='Two motion plateaus with overfit or action degradation'))
                                    break
                        chosen=directory/'best.pt'
                        self.motion_analysis(profile+'_offline',chosen,self.session['motion_validation40'],{'A0':SAMPLERS['A0']})
                        groups[label]=dict(stage=profile+'_selected',checkpoint=str(chosen),inference=baseline,
                            offline_stage=profile+'_offline',offline_name='A0',
                            result=self.evaluate(profile+'_selected','dp',chosen,specs,inference=baseline))
                else:
                    atomic_json(self.root/'training_withheld.json',dict(reason='Single-trajectory fitting did not pass the fixed offline gate within 3000 updates; inspect fit/data before train320 E/F'))
            atomic_json(self.root/'motion_groups.json',groups)
            # Complete validation successes first, then the preregistered motion metric.
            def rank(item):
                label,g=item
                if not g.get('deployable',True):return (float('inf'),float('inf'))
                name=g['offline_name']
                source=json.loads((self.root/g['offline_stage']/'result.json').read_text())
                return (-len(successes(g)),source['summary'][name]['moving_arm12_rmse_rad']['mean'])
            label,chosen=min(groups.items(),key=rank)
            sampler={k:v for k,v in chosen['inference'].items() if k in ('inference_steps','clip_sample')}
            selection=dict(group=label,checkpoint=chosen['checkpoint'],checkpoint_sha256=sha256(chosen['checkpoint']),
                sampler=sampler,inference=chosen['inference'],criterion='complete validation successes first, paired moving arm error second',test_used=False)
            atomic_json(self.root/'motion_selection.json',selection)
            data=json.loads((Path(self.session['source_root'])/'data.json').read_text())
            if label!='Original':
                self.motion_analysis('final_test',Path(chosen['checkpoint']),sorted(data['splits']['test']),{'selected':sampler},
                    split='test',timing=False,overlap_steps=chosen['inference']['overlap_steps'])
            else:
                atomic_json(self.root/'test_reused.json',dict(reason='Original configuration already has frozen predecessor test40; no redundant test call',
                    source=str(Path(self.session['previous_root'])/'position_h32_offline_test/result.json')))
            atomic_json(self.root/'status.json',dict(state='motion_study_finished',utc=time.time(),
                selected=label,repeated_complete_success=stable,training_groups_run=[k for k in groups if k in ('H24','Temporal-loss')]))
        except BaseException as error:
            import traceback
            atomic_json(self.root/'blocked.json',dict(reason=repr(error),utc=time.time(),traceback=traceback.format_exc()))
            atomic_json(self.root/'status.json',dict(state='blocked',reason=repr(error),utc=time.time()))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                os.killpg(self.worker.pid,signal.SIGTERM)
                try:self.worker.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(self.worker.pid,signal.SIGKILL);self.worker.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch();self.monitor.wait(timeout=20)
            atomic_json(self.root/'owner_exit.json',dict(utc=time.time(),owned_children_exited=True))
            subprocess.run(['bash','visuotactile/run_cpu.sh','-m','visuotactile.state_motion',
                '--root',str(self.root),'--output',str(self.root/'report'),'--report'],check=False)

    def run_position(self, gates_only=False):
        try:
            (self.root/'STOP_MONITOR').unlink(missing_ok=True)
            (self.root/'owner_exit.json').unlink(missing_ok=True)
            with (self.root/'monitor.log').open('a') as log:
                self.monitor=subprocess.Popen(['bash','visuotactile/run_cpu.sh','-m','visuotactile.state_resources','--root',str(self.root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.position_gates()
            if not gates_only:
                self.position_training()
        except BaseException as error:
            atomic_json(self.root/'blocked.json',dict(reason=repr(error),utc=time.time()))
            atomic_json(self.root/'status.json',dict(state='blocked',reason=repr(error),utc=time.time()))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                os.killpg(self.worker.pid,signal.SIGTERM)
                try:self.worker.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(self.worker.pid,signal.SIGKILL);self.worker.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch();self.monitor.wait(timeout=20)
            atomic_json(self.root/'owner_exit.json',dict(utc=time.time(),owned_children_exited=True))
            if not gates_only:
                subprocess.run(['bash','visuotactile/run_cpu.sh','tools/state_history_finish.py',
                    '--root',str(self.root),'--output','research_docs/evidence/position_20261103/results'],check=False)

    def run(self):
        try:
            (self.root/'STOP_MONITOR').unlink(missing_ok=True)
            (self.root/'owner_exit.json').unlink(missing_ok=True)
            with (self.root/'monitor.log').open('a') as log:
                self.monitor = subprocess.Popen(['bash', 'visuotactile/run_cpu.sh', '-m', 'visuotactile.state_resources',
                    '--root', str(self.root)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            numeric = self.stage('parity', 'visuotactile.state_history_audit', [])
            self.stage('freeze', extra=['--mode', 'freeze'])
            benchmark = json.loads((self.root/'freeze/benchmark.json').read_text())['specs']
            expert = self.evaluate('expert', 'expert', specs=benchmark)
            replay = self.evaluate('replay', 'replay', specs=benchmark[:2])
            saved_specs = [s for s in benchmark if s['cohort'] != 'fresh']
            short_specs = sum(([s for s in saved_specs if s['cohort'] == c][:2] for c in ('train', 'validation')), [])
            passed = (numeric['passed'] and len(expert['trials']) == 12 and all(r['legal'] for r in expert['trials'])
                and sum(r['success'] for r in expert['trials']) >= 10
                and sum(r['success'] for r in expert['trials'] if r['cohort'] != 'fresh') >= 6
                and len(replay['trials']) == 2 and all(r.get('bytewise_fields') and all(r['bytewise_fields'].values()) for r in replay['trials']))
            names = ['preservation/result.json', 'parity/result.json', 'freeze/benchmark.json', 'expert/result.json', 'replay/result.json']
            atomic_json(self.root/'foundation.json', dict(passed=bool(passed), evidence_sha256={n: sha256(self.root/n) for n in names},
                historical_open_loop_is_training_veto=False, expert_successes=sum(r['success'] for r in expert['trials'])))
            if not passed:
                raise RuntimeError('Current legal expert/control foundation failed')
            for kind in ('bc', 'dp'):
                result = self.stage(kind+'_pilot', 'visuotactile.state_full_training',
                    ['--kind', kind, '--target', str(self.session['pilot_updates']), '--pilot'])
                if not result['history_gradient_nonzero'] or result['last25_loss'] >= result['first25_loss']:
                    raise RuntimeError(kind+' pilot did not demonstrate valid learning/GRU gradient')
            for kind in ('bc', 'dp'):
                self.evaluate_learning(kind+'_pilot_loop', kind, self.root/(kind+'_pilot_train/best.pt'), short_specs)
            for kind in ('bc', 'dp'):
                for target in self.session[kind+'_milestones']:
                    if time.time() >= self.session['training_deadline']:
                        raise TimeoutError('Training budget expired; reserve final three hours for existing weights')
                    result = self.stage(f'{kind}_train_{target}', 'visuotactile.state_full_training',
                        ['--kind', kind, '--target', str(target)])
                    self.evaluate_learning(f'{kind}_loop_{target}', kind, self.root/(kind+'_train')/f'milestone_{target}.pt', short_specs)
                    if should_stop(result['history']):
                        atomic_json(self.root/(kind+'_early_stop.json'), dict(updates=target,
                            reason='Two milestones without >=1% validation improvement; no test used'))
                        break
                import torch
                checkpoint = self.root/(kind+'_train/best.pt')
                chosen = torch.load(checkpoint, map_location='cpu', weights_only=False)
                selection = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    selected_updates=chosen['updates'], criterion='minimum validation40 teacher-forced prefix12 equal arm/hand MSE',
                    test_used_for_selection=False, configuration_frozen=True)
                del chosen
                atomic_json(self.root/(kind+'_selection.json'), selection)
                self.evaluate_learning(kind, kind, checkpoint, benchmark)
            deferred = [p.name for p in self.root.glob('*_deferred.json')]
            if deferred:
                atomic_json(self.root/'status.json', dict(state='training_complete_evaluation_incomplete',
                    utc=time.time(), deferred=deferred, formal_test_started=False,
                    reason='Preserved startup failures need resolution; no success inferred'))
                return
            self.evaluate('old', 'old', Path(self.session['old_state_checkpoint']), benchmark)
            atomic_json(self.root/'formal_selection_freeze.json', dict(utc=time.time(),
                models={k: json.loads((self.root/(k+'_selection.json')).read_text()) for k in ('bc', 'dp')},
                specs=self.session['formal_specs'], historical_test_usage_disclosed=True))
            for kind in ('bc', 'dp'):
                self.stage(kind+'_test_offline', 'visuotactile.state_full_training',
                    ['--kind', kind, '--target', str(self.session[kind+'_updates']), '--final-test'])
            for mode in ('expert', 'old', 'bc', 'dp'):
                checkpoint = (None if mode == 'expert' else Path(self.session['old_state_checkpoint']) if mode == 'old'
                    else self.root/(mode+'_train/best.pt'))
                self.evaluate(mode+'_formal', mode, checkpoint, self.session['formal_specs'])
            atomic_json(self.root/'status.json', dict(state='complete', utc=time.time(),
                research_success='Read native trial outcomes; training completion is not task success'))
        except BaseException as error:
            atomic_json(self.root/'blocked.json', dict(reason=repr(error), utc=time.time()))
            atomic_json(self.root/'status.json', dict(state='blocked', reason=repr(error), utc=time.time()))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                os.killpg(self.worker.pid, signal.SIGTERM)
                try:
                    self.worker.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(self.worker.pid, signal.SIGKILL); self.worker.wait()
            if self.monitor is not None:
                (self.root/'STOP_MONITOR').touch(); self.monitor.wait(timeout=20)
            atomic_json(self.root/'owner_exit.json', dict(utc=time.time(), owned_children_exited=True))
            subprocess.run(['bash', 'visuotactile/run_cpu.sh', 'tools/state_history_report.py', '--root', str(self.root),
                '--output', str(self.root/'report')], check=False)
            subprocess.run(['bash', 'visuotactile/run_cpu.sh', 'tools/state_history_finish.py', '--root', str(self.root),
                '--output', str(Path('research_docs/evidence/state_shared_20261102/results'))], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['initialize', 'run','position-init','position-gates','motion-init'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--preflight', type=Path)
    a = p.parse_args()
    if a.command == 'motion-init':
        initialize_motion(a.root.resolve())
    elif a.command == 'position-init':
        initialize_position(a.root.resolve())
    elif a.command == 'initialize':
        initialize(a.root.resolve(), a.preflight.resolve())
    else:
        with (a.root/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner=Runner(a.root.resolve())
            if runner.session.get('motion_study'):runner.run_motion()
            elif runner.session.get('position_study'):runner.run_position(gates_only=a.command=='position-gates')
            else:runner.run()
