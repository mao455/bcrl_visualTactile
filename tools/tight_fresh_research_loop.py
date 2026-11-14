"""A registered, finite collection/training/evaluation pipeline, not a research agent.

Only prepare creates a registration. Run never retries an existing identity, changes
the original session clock, or selects a winning policy. A completed candidate is
the next DAgger source by the registered rule; fallback references remain protected.
All child stages use the existing process-identity-aware collect/cycle owner.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from tools import tight_fresh_collect_cycle_owner as owner
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

VERSION = 'finite_fresh_research_loop_v1'
INPUTS = ('partition_manifest', 'frozen_families', 'collection_backend_json', 'evaluation_backend_json',
          'training_sampling_json', 'fallback_manifest')
OPTIONAL_INPUTS = {'frozen_visual_io_json'}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, low, label, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high),
            'Invalid explicit integer: '+label)


def example_configuration():
    """Editable, complete JSON; all seeds/paths must be deliberately assigned."""
    return dict(
        inputs={key: '/absolute/path/to/'+key+'.json' for key in INPUTS},
        bounds=dict(max_seconds=43200, collection_max_seconds=7200,
            minimum_cycle_remaining_seconds=7200, minimum_round_remaining_seconds=10800,
            minimum_accepted=8, minimum_free_bytes=110*1024**3, startup_seconds=300),
        collection=dict(lanes=16, vision_microbatch=8, writer_mib=512,
            verification_workers=2, verification_max_pending=16, capture_audit_memo=True,
            schedules=['fixed80']*6+['held_root60', 'contact0']),
        cycle=dict(epochs=20, rear_epochs=15, development_count_per_role=8,
            development_seed=28500000, latest_repeat=2, evaluation_lanes=16,
            vision_microbatch=8, full_trajectory_batch=True, parallel_frontends=True,
            parallel_backends=True, payload_audit_memo=True,continue_rear=True,reuse_source_visual_index=True,
            offline_anchor_batch=32, backend_offline_anchor_batch=32),
        rounds=[dict(index=index, registry_id=f'new_session_round{index:02d}',
            seeds=list(range(29000000+i*32, 29000000+(i+1)*32)), event_seed=29100000+i*1000)
            for i,index in enumerate((10,11,12))])


def validate_configuration(value):
    template=example_configuration()
    require(isinstance(value,dict) and set(value)==set(template), 'Explicit complete loop configuration required')
    for key in ('inputs','bounds','collection','cycle'):
        expected=set(template[key]);actual=set(value[key]) if isinstance(value[key],dict) else set()
        require(isinstance(value[key],dict) and (expected<=actual<=expected|OPTIONAL_INPUTS if key=='inputs' else actual==expected),
            'Unknown/missing '+key+' configuration')
    b,c,t=value['bounds'],value['collection'],value['cycle']
    for key,lo,hi in (('max_seconds',3600,43200),('collection_max_seconds',61,7200),
            ('minimum_cycle_remaining_seconds',3600,None),('minimum_round_remaining_seconds',3600,None),
            ('minimum_accepted',8,None),('minimum_free_bytes',1,None),('startup_seconds',1,600)):
        integer(b[key],lo,key,hi)
    require(b['minimum_round_remaining_seconds'] >= b['minimum_cycle_remaining_seconds']+120
        and b['minimum_round_remaining_seconds'] < b['max_seconds'], 'Round admission must reserve collection and cycle time')
    for key,lo in (('lanes',1),('vision_microbatch',1),('writer_mib',32),
                   ('verification_workers',0),('verification_max_pending',1)):
        integer(c[key],lo,key)
    require(c['verification_max_pending']>=max(1,c['verification_workers']), 'Verification pending bound below workers')
    require(type(c['capture_audit_memo']) is bool, 'Explicit capture memo boolean required')
    from visuotactile.training_expansion import validate_schedule
    require(isinstance(c['schedules'],list) and c['schedules'], 'Explicit nonempty schedule cycle required')
    for schedule in c['schedules']:validate_schedule(schedule)
    for key in ('epochs','rear_epochs','development_count_per_role','evaluation_lanes','vision_microbatch',
                'offline_anchor_batch','backend_offline_anchor_batch'):
        integer(t[key],1,key)
    integer(t['development_seed'],0,'development_seed')
    require(t['latest_repeat'] in (1,2,4) and type(t['latest_repeat']) is int, 'Unknown latest repeat budget')
    for key in ('full_trajectory_batch','parallel_frontends','parallel_backends','payload_audit_memo',
                'continue_rear','reuse_source_visual_index'):
        require(type(t[key]) is bool,'Explicit cycle boolean required: '+key)
    require(t['full_trajectory_batch'], 'This loop identity requires the explicit B2 training contract')
    require(t['evaluation_lanes']>1,'This loop uses the registered batched development path')
    rounds=value['rounds'];require(isinstance(rounds,list) and 1<=len(rounds)<=3,'Finite maximum of three rounds required')
    indices=[];registries=[];seeds=[]
    for row in rounds:
        require(isinstance(row,dict) and set(row)=={'index','registry_id','seeds','event_seed'},'Explicit round specification required')
        integer(row['index'],1,'round index');integer(row['event_seed'],0,'event seed')
        require(isinstance(row['registry_id'],str) and re.fullmatch(r'[A-Za-z0-9_-]+',row['registry_id']), 'Safe registry identity required')
        require(isinstance(row['seeds'],list) and len(row['seeds'])*2>=b['minimum_accepted'],'Too few registered opportunities')
        for seed in row['seeds']:integer(seed,0,'initial seed')
        indices.append(row['index']);registries.append(row['registry_id']);seeds.extend(row['seeds'])
    require(indices==sorted(set(indices)) and len(set(registries))==len(registries)
        and len(set(seeds))==len(seeds),'Round identities and new training seeds must be distinct')
    return copy.deepcopy(value)


def round_paths(session, ledger, row):
    index=row['index']
    return dict(collection=session/f'new_family_{index:02d}', output=session/f'round_{index:02d}',
        source_view=session/f'collection_source_{index:02d}', ledger=ledger/f'round_{index:02d}')


def bindings(paths):
    return {key:dict(path=str(Path(path).resolve(strict=True)),sha256=sha256(path)) for key,path in paths.items()}


def check_bindings(values):
    require(all(sha256(row['path'])==row['sha256'] for row in values.values()), 'Registered input bytes changed')


def prepare(session_root, output, worker_tree, configuration):
    session_root=Path(session_root).resolve(strict=True);output=Path(output).absolute()
    tree=Path(worker_tree).resolve(strict=True);config=validate_configuration(configuration)
    require(not output.exists() and not output.is_symlink(),'Preserve existing loop ledger')
    output=output.resolve();require(output!=session_root and output.is_relative_to(session_root),'Loop ledger must be a new session child')
    session=read(session_root/'session.json')
    require(session.get('resource_policy')=='full_gpu_v1' and session.get('control_hz')==60
        and session.get('planning_hz')==5 and session.get('visual_stride')==1,'Native continuous-vision full-card session required')
    require(type(session.get('deadline')) in (int,float) and math.isfinite(session['deadline'])
        and session['deadline']>time.time(),'Original finite session deadline must be active')
    require(read(session_root/'gate.json').get('passed') is True
        and (session_root/'shared_resources.json').is_file(),'Prepared original session gate and telemetry required')
    require(not (session_root/'STOP').exists(),'Original session STOP is set')
    initial=bindings({role:session_root/role/'state_data/data.json' for role in ('A','B')})
    for role in ('A','B'):
        require((session_root/role/'training/best.pt').is_file()
            and (session_root/role/'backend'/role/'training/best.pt').is_file(),'Initial full A/B pair links required')
    if config['cycle']['reuse_source_visual_index']:
        require((session_root/'A/cache_index.json').is_file(),'Initial source visual index link required')
    inputs=bindings(config['inputs']);config['inputs']={key:row['path'] for key,row in inputs.items()}
    from visuotactile.fresh_visit_sampling import validate_profile
    validate_profile(read(config['inputs']['training_sampling_json']))
    if 'frozen_visual_io_json' in config['inputs']:
        from visuotactile.frozen_visual_io import validate_profile as validate_visual_io
        require(validate_visual_io(read(config['inputs']['frozen_visual_io_json'])) is not None,'Explicit frozen visual I/O profile required')
    fallback=read(config['inputs']['fallback_manifest'])
    require(isinstance(fallback,dict) and set(fallback)=={'A','B'} and all(isinstance(fallback[k],dict)
        and {'root','frontend_sha256','backend_sha256'}<=set(fallback[k]) for k in fallback),'Explicit protected fallback pair references required')
    from visuotactile.execution_backend import normalize_backend
    for key in ('collection_backend_json','evaluation_backend_json'):
        normalize_backend(read(config['inputs'][key]))
    require((tree/'.venv_isaacsim/bin/python').is_file(),'Frozen worker Python missing')
    sources=owner.source_files(tree)
    require('tools/tight_fresh_research_loop.py' in sources,'Frozen worker lacks this loop implementation')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
    for row in config['rounds']:
        p=round_paths(session_root,output,row)
        for key in ('collection','output','source_view','ledger'):
            require(not p[key].exists() and not p[key].is_symlink(),'Preserve existing round identity: '+str(p[key]))
        require(not p['output'].with_name(p['output'].name+'_stages').exists(),'Preserve existing cycle stages')
        require(not any(output==p[k] or output.is_relative_to(p[k]) or p[k].is_relative_to(output)
            for k in ('collection','output','source_view')),'Loop ledger and stage outputs must be disjoint')
    registration=dict(version=VERSION,session_root=str(session_root),worker_tree=str(tree),worker_commit=head,
        deadline=session['deadline'],session_sha256=sha256(session_root/'session.json'),source_sha256=sources,
        configuration=config,input_bindings=inputs,initial_data_bindings=initial,protected_fallbacks=fallback,
        prepared_utc=time.time(),source_rule='Initial source is session_root; thereafter the last completed full candidate pair and its complete training pool',
        adoption='No policy adoption or scientific method selection; fixed preregistered DAgger continuation only',
        resume='No automatic retries or re-entry; preserve every failed/partial directory',
        development='Same registered visible starts each round; no hidden-state or independent-generalization claim')
    output.mkdir(parents=True)
    with (output/'registration.json').open('x') as stream:
        json.dump(registration,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    return registration


def validate(registration, *, fresh=True):
    path=Path(registration).resolve(strict=True);r=read(path)
    require(r.get('version')==VERSION,'Unknown research loop registration')
    validate_configuration(r['configuration'])
    session=Path(r['session_root']).resolve(strict=True);tree=Path(r['worker_tree']).resolve(strict=True)
    require(path.parent!=session and path.parent.is_relative_to(session),'Original session ledger changed')
    require(sha256(session/'session.json')==r['session_sha256'] and read(session/'session.json')['deadline']==r['deadline'],
            'Original session/deadline changed')
    check_bindings(r['input_bindings']);check_bindings(r['initial_data_bindings'])
    require({key:row['path'] for key,row in r['input_bindings'].items()}==r['configuration']['inputs'],'Input path binding differs')
    require(owner.source_files(tree)==r['source_sha256'] and subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==r['worker_commit'],'Frozen worker source changed')
    if fresh:
        require(not any((path.parent/n).exists() for n in ('owner_started.json','owner_finished.json')),'No automatic loop restart')
        for row in r['configuration']['rounds']:
            p=round_paths(session,path.parent,row)
            require(not any(v.exists() or v.is_symlink() for v in p.values())
                and not p['output'].with_name(p['output'].name+'_stages').exists(),'Preserve existing round outputs')
    return dict(registration=r,path=path,ledger=path.parent,session=session,tree=tree,sha256=sha256(path))


def commands(context, row, paths, source, previous):
    r=context['registration'];cfg=r['configuration'];c=cfg['collection'];t=cfg['cycle'];b=cfg['bounds'];i=cfg['inputs']
    python=str(context['tree']/'.venv_isaacsim/bin/python')
    def command(module, args):return [python,'-m',module,*map(str,args)]
    source_cmd=None
    if previous is not None:
        source_cmd=command('tools.tight_fresh_collection_source',['--session-root',context['session'],
            '--data-source',previous,'--output',source,'--role-source','A='+str(previous),
            '--role-source','B='+str(previous),'--role-evaluation','A='+str(previous/'development'),
            '--role-evaluation','B='+str(previous/'development')])
    expansion=command('tools.tight_fresh_expansion',['--source',source,'--session-root',context['session'],
        '--output',paths['collection'],'--registry-id',row['registry_id'],'--initial-seeds',*row['seeds'],
        '--partition-manifest',i['partition_manifest'],'--frozen-families',i['frozen_families'],
        '--schedule-cycle',*c['schedules'],'--lanes',c['lanes'],'--event-seed',row['event_seed'],
        '--round',row['index'],'--backend-json',i['collection_backend_json'],'--writer-mib',c['writer_mib'],
        '--vision-microbatch',c['vision_microbatch'],'--verification-workers',c['verification_workers'],
        '--verification-max-pending',c['verification_max_pending']]+(['--capture-audit-memo'] if c['capture_audit_memo'] else []))
    collection=command('tools.tight_batch_owner',['--output',paths['collection'],'--worker-tree',context['tree'],
        '--startup-seconds',b['startup_seconds']])
    cycle=command('tools.tight_fresh_cycle',['--root',source,'--collection',paths['collection'],'--output',paths['output'],
        '--epochs',t['epochs'],'--rear-epochs',t['rear_epochs'],'--count',t['development_count_per_role'],
        '--seed',t['development_seed'],'--latest-repeat',t['latest_repeat'],'--evaluation-lanes',t['evaluation_lanes'],
        '--vision-microbatch',t['vision_microbatch'],'--backend-json',i['evaluation_backend_json'],
        '--training-sampling-json',i['training_sampling_json'],'--offline-anchor-batch',t['offline_anchor_batch'],
        '--backend-offline-anchor-batch',t['backend_offline_anchor_batch']]+
        ['--'+key.replace('_','-') for key in ('full_trajectory_batch','parallel_frontends','parallel_backends','payload_audit_memo','continue_rear') if t[key]]+
        (['--reuse-index',source/'A/cache_index.json'] if t['reuse_source_visual_index'] else [])+
        (['--frozen-visual-io-json',i['frozen_visual_io_json']] if 'frozen_visual_io_json' in i else []))
    return dict(source_prepare=source_cmd,collection_prepare=expansion,collection=collection,cycle=cycle)


def register_round(context, row, paths, source, cmds, work_deadline):
    r=context['registration'];b=r['configuration']['bounds'];ledger=paths['ledger']/'collect_cycle';ledger.mkdir()
    effective=min(b['collection_max_seconds'],int(work_deadline-time.time()-b['minimum_cycle_remaining_seconds']-owner.CLEANUP_SECONDS))
    require(effective>owner.CLEANUP_SECONDS,'Insufficient loop time for collection and the registered full cycle')
    value=dict(version=owner.VERSION,session_root=str(context['session']),data_source=str(source),
        collection=str(paths['collection']),output=str(paths['output']),worker_tree=str(context['tree']),
        worker_commit=r['worker_commit'],deadline=r['deadline'],session_sha256=r['session_sha256'],
        collection_plan_sha256=sha256(paths['collection']/'plan.json'),
        collection_command=cmds['collection'],cycle_command=cmds['cycle'],
        **{key:b[key] for key in ('minimum_cycle_remaining_seconds','minimum_accepted')},
        collection_max_seconds=effective,requested_collection_max_seconds=b['collection_max_seconds'],
        loop_work_deadline=work_deadline,
        bound_scope='Collection may be shorter to preserve the registered cycle reserve within the loop limit; original session/deadline unchanged')
    if r['configuration']['cycle']['reuse_source_visual_index']:
        value['source_visual_index_binding']=bindings(dict(visual=source/'A/cache_index.json'))['visual']
    path=ledger/'registration.json'
    with path.open('x') as stream:json.dump(value,stream,indent=2)
    # The inner owner performs its full validation once at its actual startup.
    return path


def summarize_round(collection, candidate, output):
    """Only completed numeric and JSON; never RGB, checkpoints or source rehashes."""
    from tools.tight_fresh_stage_audit import audit_numeric
    collection,candidate,output=map(Path,(collection,candidate,output))
    require(not output.exists(),'Preserve an existing round summary')
    require(read(collection/'owner_finished.json').get('software_complete') is True
        and read(candidate.with_name(candidate.name+'_stages')/'finished.json').get('software_complete') is True,
        'Only completed collection/cycle may receive a round summary')
    plan=read(collection/'plan.json');indexed=read(collection/'results.json');entries=read(collection/'correction_entries.json')
    require(len(indexed)==len(plan['cases']) and {v['case'] for v in indexed}=={v['name'] for v in plan['cases']},'Incomplete collection index')
    rows=[];cases={case['name']:case for case in plan['cases']}
    for item in indexed:
        result=item['result'];dagger=result.get('dagger',{});phase={}
        if result.get('legal'):
            _,phase=audit_numeric(collection/item['case']/'numeric.h5',result)
        accepted=result.get('accepted_for_training') is True
        require(accepted==(item['case'] in entries),'Accepted summary must use committed manifest admission')
        authority=phase.get('authority',{})
        rows.append(dict(case=item['case'],role=item['role'],family=result['family'],legal=result.get('legal') is True,
            schedule=cases[item['case']]['expansion_schedule'],causal_trigger=cases[item['case']].get('dagger',{}).get('trigger') is not None,
            accepted=accepted,takeover=dagger.get('expert_ticks',0)>0,
            expert_recovery=dagger.get('expert_takeover_success') is True,
            valid_expert_ticks=dagger.get('expert_valid_ticks',0),
            student_autonomous_success=dagger.get('student_autonomous_success') is True,
            student_native_lift=phase.get('student_native_lift'),
            student_window_reference_available=phase.get('student_settled_reference_available'),
            student_held=phase.get('student_first_contact_height_window_end_tick') is not None if phase else None,
            student_old_stable=phase.get('first_student_stable') is not None if phase else None,
            student_native_near=authority.get('student_near_goal_first_tick') is not None if phase else None,
            student_native_subgoal=authority.get('student_max_subgoals',0)>0 if phase else None))
    def counts(selected):
        legal=[v for v in selected if v['legal']];known=[v for v in legal if v['student_window_reference_available']]
        return dict(attempts=len(selected),legal=len(legal),illegal=len(selected)-len(legal),
            accepted=sum(v['accepted'] for v in legal),takeovers=sum(v['takeover'] for v in legal),
            causal_takeovers=sum(v['takeover'] for v in legal if v['causal_trigger']),
            fixed_takeovers=sum(v['takeover'] for v in legal if not v['causal_trigger']),
            expert_recoveries=sum(v['expert_recovery'] for v in legal),
            valid_expert_ticks=sum(v['valid_expert_ticks'] for v in legal),
            accepted_valid_expert_ticks=sum(v['valid_expert_ticks'] for v in legal if v['accepted']),
            attempted_families=len({v['family'] for v in selected}),accepted_families=len({v['family'] for v in selected if v['accepted']}),
            student_native_lift=sum(bool(v['student_native_lift']) for v in legal),
            student_window_available=len(known),student_held=sum(v['student_held'] for v in known),
            student_old_stable=sum(v['student_old_stable'] for v in known),
            student_native_near=sum(v['student_native_near'] for v in legal),
            student_native_subgoal=sum(v['student_native_subgoal'] for v in legal),
            student_autonomous_success=sum(v['student_autonomous_success'] for v in legal))
    development=candidate/'development';devrows=read(development/'results.json');devplan=read(development/'plan.json')
    require(read(development/'result.json').get('software_complete') is True
        and len(devrows)==len(devplan['cases']),'Incomplete development cohort')
    dev={}
    for role in ('A','B'):
        results=[row['result'] for row in devrows if row['role']==role];legal=[v for v in results if v.get('legal') is True]
        require(all(not v.get('dagger') for v in results),'Development must remain wholly autonomous')
        windows=[development_windows(development/row['case']/'numeric.h5') for row in devrows
            if row['role']==role and row['result'].get('legal') is True]
        known=[v for v in windows if v['reference_available']]
        dev[role]=dict(attempts=len(results),legal=len(legal),illegal=len(results)-len(legal),
            successes=sum(bool(v['success']) for v in legal),native_lift=sum(bool(v['native_lift']) for v in legal),
            native_subgoal=sum(v['native_subgoals']>0 for v in legal),native_alignment=sum(bool(v['native_alignment']) for v in legal),
            window_reference_available=len(known),held=sum(v['held'] for v in known),old_stable=sum(v['old_stable'] for v in known),
            native_near=sum(v['native_near'] for v in windows),
            contact_summaries=[dict(identity=v['identity'],contact=v.get('contact')) for v in legal],
            success_denominator='legal; all illegal opportunities retained separately')
    value=dict(version=VERSION,collection=dict(total=counts(rows),by_role={role:counts([v for v in rows if v['role']==role]) for role in ('A','B')},cases=rows),
        development=dev,files=bindings(dict(collection_plan=collection/'plan.json',collection_results=collection/'results.json',
            collection_entries=collection/'correction_entries.json',development_plan=development/'plan.json',development_results=development/'results.json')),
        rgb_opened=0,weights_opened=0,policy_adopted=False,
        interpretation='Expert recovery is data yield, never student success. Held/stable require a student-only settled reference. Reused development starts are tracking, not independent generalization.')
    atomic_json(output,value);return value


def development_windows(path):
    """The same settled reference and 12 completed-transition windows as collection."""
    import h5py
    import numpy as np
    from visuotactile.takeover_trigger import stable_windows,contact_height_windows
    with h5py.File(path,'r') as stream:
        transition=stream['episode/transition'];after=transition['diagnostics_after']
        pose=after['object_pose_w'][:];palm=after['palm_body_pose_w'][:];force=transition['object_contact_force_w'][:]
        near=bool(transition['native_success_diagnostic/near_goal'][:].any())
    available=len(pose)>=48
    held=stable=False
    if available:
        height=float(np.median(pose[24:48,2]));stable,_=stable_windows(pose,palm,force,height)
        held=contact_height_windows(pose,force,height);stable[:47]=False;held[:47]=False
        stable,held=bool(stable.any()),bool(held.any())
    return dict(reference_available=available,held=held,old_stable=stable,native_near=near)


def run(registration):
    context=validate(registration);r=context['registration'];cfg=r['configuration'];ledger=context['ledger'];session=context['session']
    require(sha256(__file__)==r['source_sha256']['tools/tight_fresh_research_loop.py'],'Executed loop code differs from frozen registration')
    began=time.time();deadline=min(r['deadline'],began+cfg['bounds']['max_seconds'])-owner.CLEANUP_SECONDS
    stop_paths=(session/'STOP',ledger/'STOP');stages=[];rounds=[];reason='owner_interrupted';error=None;previous=None;handlers={}
    with (ledger/'owner_started.json').open('x') as stream:
        json.dump(dict(version=VERSION,started=began,identity=owner.process_identity(os.getpid()),
            registration_sha256=context['sha256'],original_deadline=r['deadline'],work_deadline=deadline,
            cleanup_reserve_seconds=owner.CLEANUP_SECONDS,stop_paths=list(map(str,stop_paths))),stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    def check():
        require(sha256(context['path'])==context['sha256'],'Loop registration changed')
        require(sha256(session/'session.json')==r['session_sha256'],'Original session changed')
        check_bindings(r['input_bindings'])
        if any(path.exists() for path in stop_paths):raise InterruptedError('requested_stop')
        if time.time()>=deadline:raise TimeoutError('original_or_loop_deadline')
    def interrupted(signum,frame):raise SystemExit(128+signum)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        for row in cfg['rounds']:
            check();b=cfg['bounds'];free=shutil.disk_usage(session).free
            required=max(b['minimum_free_bytes'],read(session/'session.json').get('reserve_bytes',0))
            admission=dict(round=row['index'],checked_utc=time.time(),free_bytes=free,required_free_bytes=required,
                remaining_work_seconds=deadline-time.time(),minimum_round_remaining_seconds=b['minimum_round_remaining_seconds'])
            admission['admitted']=free>=required and admission['remaining_work_seconds']>=b['minimum_round_remaining_seconds']
            admission['reason']='insufficient_space' if free<required else 'insufficient_remaining_time' if not admission['admitted'] else 'admitted'
            atomic_json(ledger/f'round_{row["index"]:02d}_admission.json',admission)
            if not admission['admitted']:reason=admission['reason'];break
            paths=round_paths(session,ledger,row);paths['ledger'].mkdir()
            source=session if previous is None else paths['source_view'];cmds=commands(context,row,paths,source,previous)
            stage_context=dict(ledger=paths['ledger'],paths=dict(worker_tree=context['tree'],collection=paths['collection'],output=paths['output']))
            stops=(*stop_paths,paths['ledger']/'STOP',paths['collection']/'STOP',paths['output']/'STOP')
            current=dict(index=row['index'],source=str(source),previous_candidate=str(previous) if previous else None,
                collection=str(paths['collection']),candidate=str(paths['output']),software_complete=False,stages=[])
            rounds.append(current)
            def stage(name,command):
                check();atomic_json(ledger/'owner_status.json',dict(stage=name,round=row['index'],completed_rounds=sum(v['software_complete'] for v in rounds),work_deadline=deadline))
                receipt=owner.run_stage(name,command,stage_context,deadline,stops)
                stages.append(receipt);current['stages'].append(dict(name=name,complete=receipt['complete'],receipt=str(paths['ledger']/(name+'_finished.json'))))
                if not receipt['complete']:current['failed_stage']=receipt
                atomic_json(paths['ledger']/'round_status.json',current)
                return receipt['complete']
            if cmds['source_prepare'] is not None:
                if not stage('source_prepare',cmds['source_prepare']):reason='source_prepare_incomplete';break
                require(read(source/'prepared.json').get('complete') is True,'Source view preparation lacks completion')
                current['source_selection']=bindings(dict(receipt=source/'source_selection.json'))
            if not stage('collection_prepare',cmds['collection_prepare']):reason='collection_prepare_incomplete';break
            check();inner=register_round(context,row,paths,source,cmds,deadline)
            command=[str(context['tree']/'.venv_isaacsim/bin/python'),'-m','tools.tight_fresh_collect_cycle_owner','run','--registration',str(inner)]
            if not stage('collect_cycle',command):
                reason='collect_cycle_incomplete'
                completed=inner.parent/'owner_finished.json'
                if completed.exists():current['inner_outcome']=read(completed);reason=current['inner_outcome'].get('reason',reason)
                break
            outcome=read(inner.parent/'owner_finished.json');require(outcome.get('software_complete') is True,'Inner owner did not complete collection plus cycle')
            current['completion']=bindings(dict(owner=inner.parent/'owner_finished.json',cycle=paths['output'].with_name(paths['output'].name+'_stages')/'finished.json'))
            summary=paths['ledger']/'summary.json'
            if not stage('summary',[str(context['tree']/'.venv_isaacsim/bin/python'),'-m','tools.tight_fresh_research_loop',
                    'summarize','--collection',str(paths['collection']),'--candidate',str(paths['output']),'--output',str(summary)]):
                reason='summary_incomplete';break
            current.update(software_complete=True,summary=dict(path=str(summary),sha256=sha256(summary)),
                use_as_next_dagger_source=True,policy_adopted=False)
            atomic_json(paths['ledger']/'round_finished.json',current);atomic_json(ledger/'rounds.json',rounds)
            previous=paths['output']
        else:reason='registered_rounds_complete'
    except InterruptedError as caught:reason='requested_stop';error=repr(caught)
    except TimeoutError as caught:reason='original_or_loop_deadline';error=repr(caught)
    except BaseException as caught:reason='owner_exception';error=repr(caught)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if reason!='registered_rounds_complete':
            if any(path.exists() for path in stop_paths):reason='requested_stop'
            elif time.time()>=deadline:reason='original_or_loop_deadline'
        complete=reason=='registered_rounds_complete'
        result=dict(version=VERSION,software_complete=complete,reason=reason,error=error,started=began,ended=time.time(),
            seconds=time.time()-began,original_deadline=r['deadline'],work_deadline=deadline,registration_sha256=context['sha256'],
            registered_rounds=len(cfg['rounds']),completed_rounds=sum(v['software_complete'] for v in rounds),
            rounds=rounds,stages=stages,policy_adopted=False,protected_fallbacks=r['protected_fallbacks'])
        atomic_json(ledger/'rounds.json',rounds);atomic_json(ledger/'owner_finished.json',result)
        atomic_json(ledger/'owner_status.json',dict(stage='complete' if complete else 'stopped',reason=reason,
            software_complete=complete,completed_rounds=result['completed_rounds'],error=error))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('example-config')
    prepare_cli=sub.add_parser('prepare')
    for key in ('session-root','output','worker-tree','configuration'):prepare_cli.add_argument('--'+key,type=Path,required=True)
    for mode in ('validate','run'):
        sub.add_parser(mode).add_argument('--registration',type=Path,required=True)
    summary=sub.add_parser('summarize')
    for key in ('collection','candidate','output'):summary.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.mode=='example-config':value=example_configuration()
    elif args.mode=='prepare':value=prepare(args.session_root,args.output,args.worker_tree,read(args.configuration))
    elif args.mode=='validate':
        context=validate(args.registration);value=dict(valid=True,registration_sha256=context['sha256'],deadline=context['registration']['deadline'])
    elif args.mode=='summarize':value=summarize_round(args.collection,args.candidate,args.output)
    else:value=run(args.registration)
    print(json.dumps(value,indent=2),flush=True)
    if args.mode=='run' and not value['software_complete']:raise SystemExit(1)


if __name__=='__main__':main()
