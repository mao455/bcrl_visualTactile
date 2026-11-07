"""Finite remaining stages of the current run, with evidence-based transitions."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE,RUN,read,write,supervised_owner,sha,kill_owned,proc
from tools.tight_state import capture,digest,beat,check_protected
from visuotactile.tight_gate import paired_admission,validate_formal_manifest,require,scheme_reliability,accepted_foundation,physical_motion_errors


def cap_motion_admission(row, reference, rules):
    """Supplement unchanged per-record admission with real motion and no detour."""
    import h5py
    import numpy as np
    from tools.narrow_repair_audit import path_speed_metrics,path_speed_comparison
    from visuotactile.expert_speed import cap_hand_target,progress_target
    import torch
    if not row.get('path') or not reference.get('path'):
        return dict(passed=False,errors=['missing_physical_capture'])
    path=Path(row['path']).parents[1]/'numeric.h5'
    other=Path(reference['path']).parents[1]/'numeric.h5'
    metrics=path_speed_metrics(path);comparison=path_speed_comparison(metrics,path_speed_metrics(other))
    errors=physical_motion_errors(comparison,rules)
    with h5py.File(path) as f:
        g=f['episode'];prev=torch.from_numpy(g['transition/target_before_rad'][:])
        native=torch.from_numpy(g['transition/speed_native_target_rad'][:]);scheme=row['speed_scheme']
        expected=cap_hand_target(prev,progress_target(prev,native,scheme['arm'],scheme['hand']),scheme['hand_target_cap_rad_s'])
        actual=torch.from_numpy(g['transition/target_after_rad'][:])
        error=float((expected-actual).abs().max())
        maximum=float((actual[:,7:]-prev[:,7:]).abs().max()*60)
        requested=float((native[:,7:]-prev[:,7:]).abs().max()*60)
        before=g['transition/diagnostics_before'];after=g['transition/diagnostics_after']
        actual_hand_rms=float(np.sqrt(np.mean(before['joint_velocity_rad_s'][:,7:]**2)))
        sub=after['successes'][:].reshape(len(actual),-1).max(-1)
        transitions={str(k):int(np.flatnonzero(sub>=k)[0]+1) if (sub>=k).any() else None for k in (1,2)}
    if error>1e-6:errors.append('cap_applied_target_formula_mismatch')
    return dict(passed=not errors,errors=errors,metrics=metrics,comparison=comparison,
        target_formula_max_error_rad=error,maximum_hand_target_speed_rad_s=maximum,
        maximum_native_hand_target_request_rad_s=requested,actual_hand_rms_rad_s=actual_hand_rms,
        native_subgoal_completed_ticks=transitions)


def hand_cap_pilot(root):
    from tools.tight_state import environment
    s=read(root/'session.json');plan=read(root/'hand_cap_plan.json')
    require(not (root/'hand_cap_results.json').exists(),'No cap pilot restart/replacement')
    for path,expected in plan['protected_references'].items():require(sha(path)==expected,'Reference changed: '+path)
    app,env,player,lease=environment(root);lease.deadline=plan['deadline'];rows=[]
    for pair in plan['pairs']:
        ref=pair['reference'];spec=dict(pair['spec'],identity=pair['pair_id']+'_cap')
        row=capture(root,env,player,lease,spec,None,speed_scheme=plan['speed_scheme'])
        row['admission']=paired_admission(row,ref,s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=ref.get(key):row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        row['physical_motion']=cap_motion_admission(row,ref,plan['physical_rules'])
        row['admission']['errors'].extend(row['physical_motion']['errors'])
        row['admission']['passed']=not row['admission']['errors']
        rows.append(row);write(root/'hand_cap_results.json',rows)
        require(row['classification'] not in ('sensor_failure','software_or_sensor_failure'),'Software/sensor fault; no replacement')
    coverage={family:dict(attempts=len(rr),successes=sum(r['success'] for r in rr),admitted=sum(r['admission']['passed'] for r in rr))
        for family in plan['families'] for rr in [[r for r in rows if r['family']==family]]}
    successes=sum(r['success'] for r in rows);admitted=sum(r['admission']['passed'] for r in rows)
    rules=plan['pilot_improvement']
    passed=(len(rows)==rules['complete_attempts'] and successes>=rules['minimum_task_successes'] and
        admitted>=rules['minimum_admitted'] and all(x['admitted']>=rules['minimum_family_admitted'] for x in coverage.values()))
    write(root/'hand_cap_summary.json',dict(complete=True,passed=passed,attempts=len(rows),successes=successes,admitted=admitted,
        coverage=coverage,scope='Pilot improvement gate only; independent reliability required before production',
        parameters=plan['speed_scheme'],automatic_parameter_search=False))


def validate_speed(root):
    """One preregistered fixed scheme, distinct post-reset event seeds, no search."""
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');plan=read(root/'speed_validation_plan.json')
    require(digest(s['domain'])==plan['domain_sha256'],'Registered domain changed')
    require(s['speed_scheme']==plan['speed_scheme'],'Registered scheme changed')
    require(not (root/'validation_results.json').exists(),'No blind validation restart')
    app,env,player,lease=environment(root);lease.deadline=plan['deadline']
    refs=[];slow=[];halt=None
    for pair in plan['pairs']:
        spec=dict(pair,split='development',cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(s['domain'],pair['seed']))
        spec['identity']=pair['pair_id']+'_reference';reference=capture(root,env,player,lease,spec,1.)
        refs.append(reference)
        spec['identity']=pair['pair_id']+'_slow'
        row=capture(root,env,player,lease,spec,None,speed_scheme=plan['speed_scheme'])
        row['admission']=paired_admission(row,reference,s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=reference.get(key):
                row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        if 'physical_rules' in plan:
            row['physical_motion']=cap_motion_admission(row,reference,plan['physical_rules'])
            row['admission']['errors'].extend(row['physical_motion']['errors'])
            row['admission']['passed']=not row['admission']['errors']
        slow.append(row);gate=scheme_reliability(slow,plan)
        gate.update(reference_attempts=len(refs),reference_successes=sum(r['success'] for r in refs))
        write(root/'validation_results.json',dict(references=refs,slow=slow,gate=gate))
        if row['classification'] in ('sensor_failure','software_or_sensor_failure') or reference['classification'] in ('sensor_failure','software_or_sensor_failure'):
            halt='Sensor/software fault; no replacement attempt';break
        if gate['preregistered_threshold_unreachable'] or sum(not r['success'] for r in refs)>plan['slow_attempts']-plan['reliability']['minimum_reference_successes']:
            halt='Preregistered reliability/yield/coverage threshold is mathematically unreachable';break
    gate=scheme_reliability(slow,plan)
    reference_ok=len(refs)==plan['slow_attempts'] and sum(r['success'] for r in refs)>=plan['reliability']['minimum_reference_successes']
    gate.update(reference_reliability_passed=reference_ok,reference_attempts=len(refs),reference_successes=sum(r['success'] for r in refs),
        passed=gate['passed'] and reference_ok,stop_reason=halt,remaining_unstarted_pairs=plan['slow_attempts']-len(slow))
    write(root/'scheme_reliability.json',gate)
    write(root/'selected_development_gate.json',dict(passed=gate['passed'],alpha=None,speed_scheme=plan['speed_scheme'],
        references=refs,slow=slow,reliability=gate,reason=halt or 'Completed registered fixed-scheme validation'))


def refine(root):
    """The one registered narrower development domain, no final-test feedback."""
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');plan=read(root/'development_refinement_plan.json')
    require(s.get('development_revision')==2 and s['domain']==plan['domain'] and s['development']['seeds']==plan['seeds'],
        'Refinement configuration must be applied before any Kit start')
    app,env,player,lease=environment(root);lease.deadline=s['development_deadline']
    refs=[];slow=[]
    for j,seed in enumerate(s['development']['seeds']):
        family=f'development_refined_{j:02d}'
        spec=dict(identity=family+'_native',family=family,seed=seed,split='development',cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed))
        refs.append(capture(root,env,player,lease,spec,1.))
        spec['identity']=family+'_a055'
        row=capture(root,env,player,lease,spec,.55)
        row['admission']=paired_admission(row,refs[-1],s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=refs[-1].get(key):row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        slow.append(row)
        write(root/'refined_development_results.json',dict(references=refs,slow=slow))
    write(root/'selected_development_gate.json',dict(passed=all(r['admission']['passed'] for r in slow),alpha=.55,
        references=refs,slow=slow,protocol='One geometry narrowing using development evidence; original12 attempts remain in denominator'))
    actuator(root,env,player,lease,slow,.55)


def actuator(root,env,player,lease,slow,alpha):
    s=read(root/'session.json')
    # One processed-target H24 execution gate with alpha deliberately still .55
    # in the wrapper. Task success of this open-loop replay is not an admission
    # requirement; only exact issued/applied/two-physics-substep targets are.
    import h5py
    from visuotactile.state_benchmark import trial
    source=next((r for r in slow if r.get('numeric_valid')),None)
    require(source is not None,'No valid numeric source for actuator gate')
    path=Path(source['path']);original=read(path.parents[1]/'started.json')['spec']
    with h5py.File(path) as f:targets=next(iter(f['episodes'].values()))['transition/target_after_rad'][:]
    spec=dict(original,identity='processed_target_oracle',role='actuator_gate')
    with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(spec,started=time.time(),classification='started',alpha_reapplication_allowed=False))+'\n')
    result=trial(env,lease,root/'processed_target_oracle',spec,{},player,s['expert_checkpoint'],replay=targets,position=True,replay_horizon=24)
    passed=result.get('legal') and result.get('software_complete') and result.get('actuator_target_max_error',1)<1e-6 and result.get('submitted_target_max_error',1)<1e-6
    write(root/'actuator_gate.json',dict(passed=bool(passed),result=result,retained_expert_alpha=alpha,
        replay_success_required=False,source_sha256=source['source_sha256'],source_length=len(targets)))
    with (root/'completed_attempts.jsonl').open('a') as f:f.write(json.dumps(dict(result,classification='actuator_gate',admission_passed=bool(passed)))+'\n')
    require(passed,'Actual actuator target mapping failed')


def actuator_worker(root):
    from tools.tight_state import environment
    gate=read(root/'selected_development_gate.json')
    app,env,player,lease=environment(root)
    alpha=gate['alpha'];env.speed_config=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all')
    actuator(root,env,player,lease,gate['slow'],alpha)


def collect(root):
    import numpy as np
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration,build_cache
    s=read(root/'session.json');cfg=read(root/'configuration.json');gate=read(root/'selected_development_gate.json')
    require(gate['passed'],'Development did not pass');c=s['collection'];alpha=gate['alpha']
    scheme=gate.get('speed_scheme',dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all'))
    require(scheme==s.get('speed_scheme',scheme),'Validated speed scheme differs from production configuration')
    if scheme.get('hand_target_cap_rad_s') is not None:
        require('physical_motion_gate' in s,'Extract physical admission into standalone current configuration before production')
    families=[];order=np.random.default_rng(c['split_seed']).permutation(c['families'])
    splits={int(i):'train' if j<c['train_families'] else 'validation' if j<c['train_families']+c['validation_families'] else 'test' for j,i in enumerate(order)}
    for i in range(c['families']):
        seed=c['seed_start']+i;family=f'formal_{i:03d}'
        families.append(dict(family=family,seed=seed,cohort='fresh',split=splits[i],collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(s['domain'],seed)))
    write(root/'frozen_families.json',families)
    jobs=[]
    for f in families:jobs.append(dict(f,identity=f['family']+'_reference',alpha=1.,speed_scheme=dict(kind='closed_loop',arm=1.,hand=1.,phase='all')))
    for j in range(c['variations_per_family']):
        for f in families:
            # Same fixed-pose family/configuration; independent native joint,
            # observation/delay/wrench RNG draws. No alteration of DR settings.
            jobs.append(dict(f,identity=f['family']+f'_slow_{j}',seed=f['seed']+j*100000,
                event_seed=c['event_seed_start']+j*c['families']+int(f['family'].split('_')[-1]),alpha=alpha,speed_scheme=scheme))
    require(len(jobs)<=c['maximum_attempts'],'Predetermined attempt cap')
    write(root/'frozen_collection_plan.json',dict(jobs=jobs,configuration_sha256=s['configuration_sha256'],alpha=alpha,speed_scheme=scheme,
        reference_scope='Same fixed-pose family and native reset/DR configuration; variation seeds change joint/noise/event draws, not an exact hidden-state pair'))
    app,env,player,lease=environment(root);refs={};entries=[];results=[];consecutive=0
    budget_stop=None
    for index,job in enumerate(jobs):
        import shutil
        if shutil.disk_usage(root).free<s['reserve_bytes']+5*1024**3:
            candidate=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=entries)
            validate_formal_manifest(candidate,cfg)
            budget_stop=dict(reason='Soft count target yields to fixed disk reserve plus5GiB training/checkpoint allowance',
                unstarted=len(jobs)-index,completed=len(results),accepted=len(entries));break
        row=capture(root,env,player,lease,job,job['alpha'],speed_scheme=job['speed_scheme']);results.append(row)
        if job['alpha']==1.:refs[job['family']]=row
        else:
            ref=refs[job['family']];admission=paired_admission(row,ref,s['development']);row['admission']=admission
            if scheme.get('hand_target_cap_rad_s') is not None:
                row['physical_motion']=cap_motion_admission(row,ref,s['physical_motion_gate'])
                admission['physical_motion_comparison']=row['physical_motion'].get('comparison',{})
                admission['errors'].extend(row['physical_motion']['errors']);admission['passed']=not admission['errors']
            if admission['passed']:
                entries.append(dict(id=job['identity'],family=job['family'],split=job['split'],alpha=alpha,speed_scheme=scheme,
                    path=row['path'],sha256=row['source_sha256'],steps=row['steps'],size=Path(row['path']).stat().st_size,
                    state_valid=True,multi_view_valid=True,tactile_valid=True,admission=admission,
                    reference_path=ref['path'],reference_sha256=ref['source_sha256'],visual_acceptance_sha256=row['visual_acceptance_sha256']))
            elif row.get('numeric_valid') and row.get('visual_passed') and row.get('software_complete'):
                # Classified, complete rejected NEW captures keep full numeric
                # traces, initial/config evidence, sensor hashes and attempt row.
                # Never remove protected400/275 or unknown partial/sensor faults.
                path=Path(row['path']);require(root/'episodes' in path.parents,'New own capture only')
                require(sha(path)==row['source_sha256'],'Rejected payload changed before cleanup')
                numeric=path.parents[1]/'numeric.h5';require(numeric.is_file(),'Numeric failure evidence required')
                st=path.stat();path.unlink()
                row['payload_retention']=dict(mode='classified_rejected_rgbd_removed_numeric_and_small_evidence_retained',
                    numeric_path=str(numeric),numeric_sha256=sha(numeric),deleted_source_sha256=row['source_sha256'],
                    released_bytes=st.st_blocks*512 if st.st_nlink==1 else 0)
                with (root/'collection_cleanup.jsonl').open('a') as stream:stream.write(json.dumps(dict(identity=row['identity'],**row['payload_retention']))+'\n')
        consecutive=0 if row['success'] else consecutive+1
        write(root/'formal_results.json',results)
        write(root/'manifest.pending.json',dict(version='tight_slow_multimodal_v1',collection_finished=False,
            development_passed=True,frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=entries))
        if row.get('classification') in ('sensor_failure','software_or_sensor_failure'):raise RuntimeError('Sensor/software failure; no promotion')
        if consecutive>=c['maximum_consecutive_task_failures']:raise RuntimeError('Preregistered consecutive task-failure stop')
        completed_slow=[r for r in results if r['alpha']!=1.]
        if c.get('reliability_window') and len(completed_slow) and len(completed_slow)%c['reliability_window']==0:
            task_rate=sum(r['success'] for r in completed_slow)/len(completed_slow)
            yield_rate=sum(r.get('admission',{}).get('passed',False) for r in completed_slow)/len(completed_slow)
            require(task_rate>=c['minimum_production_task_rate'] and yield_rate>=c['minimum_production_yield'],
                'Preregistered cumulative production reliability/yield stop')
    manifest=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,
        frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=entries,all_attempts=len(results),configuration_sha256=s['configuration_sha256'],
        original400_included=False,original275_included=False,quarantined=False,budget_stop=budget_stop)
    validate_formal_manifest(manifest,cfg)
    # Recheck every accepted payload and both modalities at freeze, never trust
    # flags copied from the old legacy camera path.
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.tight_gate import numeric_admission
    for e in entries:
        require(sha(e['path'])==e['sha256'] and sha(e['reference_path'])==e['reference_sha256'],'Frozen pair source changed')
        require(numeric_admission(e['path'])[3]['success'],'Native complete success missing')
        require(inspect_capture(e['path'])['passed'],'RGB-D content failed at freeze')
    write(root/'manifest.json',manifest);data=build_cache(root)
    write(root/'DATA_READY.json',dict(passed=True,episodes=len(entries),families=len({e['family'] for e in entries}),manifest_sha256=sha(root/'manifest.json')))
    write(root/'COMPLETE.json',read(root/'DATA_READY.json'))


def baseline_checkpoint(saved, input_builder):
    """Explicit evaluation-only I/O intervention; preserve source weights/state."""
    from visuotactile.position_control import contract
    require(saved.get('contract') == contract(24, True), 'Baseline requires frozen H24 command-target StateDP')
    require(saved.get('kind') == 'dp' and saved.get('updates', 0) > 0, 'Trained StateDP required')
    require(input_builder in ('legacy_packet_inverse_v1', 'canonical_joint_state_v1'), 'Unknown baseline input intervention')
    result = dict(saved)
    result['configuration'] = dict(saved.get('configuration', {}), input_builder=input_builder)
    return result


def baseline_evaluate(root):
    """Paired old275 diagnostic, separate from formal slow-data admission."""
    import torch
    from visuotactile.state_benchmark import trial
    from tools.tight_state import environment
    s=read(root/'session.json');plan=read(root/'baseline_plan.json')
    require(plan['dataset_scope']=='original275_state_valid_mixed_speed', 'Explicit historical baseline scope required')
    require(not (root/'baseline_results.json').exists(), 'No baseline restart/replacement')
    for path,expected in plan['protected_inputs'].items():require(sha(path)==expected, 'Baseline source drift: '+path)
    saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    require(len(plan['input_builders']) in (1,2) and len(set(plan['input_builders']))==len(plan['input_builders']),'One independent variant or two distinct paired variants required')
    variants={key:baseline_checkpoint(saved,key) for key in plan['input_builders']}
    app,env,player,lease=environment(root,True);rows=[];pairs=[]
    for index,pair in enumerate(plan['pairs']):
        current=[]
        order=plan['input_builders'][::1 if index%2==0 else -1]
        for builder in order:
            identity=pair['pair_id']+'_'+('legacy' if builder.startswith('legacy') else 'canonical')
            spec=dict(pair['spec'],identity=identity)
            with (root/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(identity=identity,pair_id=pair['pair_id'],input_builder=builder,started=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
            output=root/'baseline_evaluation'/identity
            result=trial(env,lease,output,spec,{},player,s['expert_checkpoint'],saved=variants[builder],position=True,inference=plan['inference'])
            result.update(pair_id=pair['pair_id'],input_builder=builder,source_student_sha256=plan['protected_inputs'][plan['checkpoint']],
                evaluation_only_input_override=True,dataset_scope=plan['dataset_scope'])
            if result['legal']:
                require(result['actuator_target_max_error']<1e-6 and result['submitted_target_max_error']<1e-6,'Baseline actuator mismatch')
            rows.append(result);current.append(result);write(root/'baseline_results.json',rows)
            require(result['software_complete'],'Baseline software fault; no replacement')
            beat(root,stage='baseline_evaluation',identity=identity,completed=len(rows))
        initial=[root/'baseline_evaluation'/x['identity']/'initial_observable.json' for x in current]
        pairs.append(dict(pair_id=pair['pair_id'],split=pair['spec']['split'],initial_observable_exact=sha(initial[0])==sha(initial[1]) if len(initial)==2 else None,
            members=[x['identity'] for x in current],successes={x['input_builder']:x['success'] for x in current},
            scope='Independent frozen canonical rollout' if len(current)==1 else 'Same native reset/config/event and diffusion seeds; observable equality checked, hidden PhysX state not asserted'))
        write(root/'baseline_pairs.json',pairs)
    summary={builder:{split:dict(attempts=len(rr),legal=sum(r['legal'] for r in rr),successes=sum(r['success'] for r in rr),
        initial_rejections=sum(not r['legal'] for r in rr),software_failures=sum(not r['software_complete'] for r in rr))
        for split in ('train','unseen') for rr in [[r for r in rows if r['input_builder']==builder and r['split']==split]]} for builder in plan['input_builders']}
    for path,expected in plan['protected_inputs'].items():require(sha(path)==expected,'Baseline source changed during evaluation')
    write(root/'baseline_summary.json',dict(complete=True,cohorts=summary,physical_attempts=len(rows),pairs=len(pairs),
        paired_comparisons=sum(len(p['members'])==2 for p in pairs),
        observable_exact_pairs=sum(p['initial_observable_exact'] is True for p in pairs),optimizer_updates=0,
        dataset_scope=plan['dataset_scope'],formal_slow_data_admitted=False,final_test_for_selection=False))


def evaluate(root):
    import torch
    from visuotactile.state_benchmark import trial
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');app,env,player,lease=environment(root,True)
    with lease.block():saved=torch.load(root/'training/best.pt',map_location='cpu',weights_only=False)
    families=read(root/'frozen_families.json');plan=[];e=s['evaluation']
    for split,number in [('train',e['train_families']),('validation',e['validation_families'])]:
        plan.extend(dict(f,identity='evaluation_'+f['family']) for f in families if f['split']==split)
        # Finite first families by identity, never selected by expert/student success.
        plan=[p for p in plan if p['split']!=split]+[p for p in plan if p['split']==split][:number]
    for j in range(e['new_initial_attempts']):
        seed=e['new_seed_start']+j
        plan.append(dict(identity=f'new_{j:03d}',family=f'new_{j:03d}',split='new',seed=seed,cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed)))
    write(root/'frozen_evaluation_plan.json',plan);results=[]
    for spec in plan:
        previous=[r for r in results if r['split']=='new']
        if spec['split']=='new' and len(previous)>=e['initial_probe'] and not any(r['success'] for r in previous):break
        if spec['split']=='new' and len(previous)>=e['zero_success_stop_at'] and not any(r['success'] for r in previous[:e['zero_success_stop_at']]):break
        result=trial(env,lease,root/'evaluation'/spec['identity'],spec,{},player,s['expert_checkpoint'],saved=saved,position=True,
            inference=dict(inference_steps=16,clip_sample=False,overlap_steps=0,sampler_seed=e['sampler_seed']))
        if result['legal']:
            require(result['actuator_target_max_error']<1e-6 and result['submitted_target_max_error']<1e-6,'Actual actuator mismatch')
        results.append(result);write(root/'evaluation_results.json',results)
        beat(root,stage='evaluation',identity=spec['identity'])
    cohorts={k:dict(attempts=len(rr),legal=sum(r['legal'] for r in rr),successes=sum(r['success'] for r in rr),
        initial_rejections=sum(not r['legal'] for r in rr)) for k in ('train','validation','new') for rr in [[r for r in results if r['split']==k]]}
    write(root/'evaluation_summary.json',dict(complete=True,cohorts=cohorts,checkpoint_sha256=sha(root/'training/best.pt'),
        final_test_for_selection=False,new_denominator_not_padded=True))


@supervised_owner
def pipeline(root,refinement=False,speed_validation=False,baseline=False,mechanism=False):
    import fcntl
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=read(root/'session.json');monitor=None;child=None
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        def worker(stage,args,deadline):
            nonlocal child
            if (root/'execution_source_freeze.json').exists():
                for path,expected in read(root/'execution_source_freeze.json').items():require(sha(path)==expected,'Execution source drift: '+path)
            path=root/'stages'/stage;path.mkdir(parents=True,exist_ok=False)
            write(root/'status.json',dict(state='running',stage=stage,utc=time.time()))
            with (path/'worker.log').open('w') as log:child=subprocess.Popen(RUN+args,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'active_owner.json',dict(owner=proc(os.getpid()),worker=proc(child.pid),monitor=proc(monitor.pid)))
            while child.poll() is None:
                if monitor.poll() is not None:raise RuntimeError('Monitor stopped')
                if time.time()>=deadline or (root/'STOP').exists():raise TimeoutError('Fixed deadline/STOP')
                time.sleep(2)
            write(path/'exit.json',dict(returncode=child.returncode,utc=time.time()))
            if child.returncode:raise RuntimeError(stage+' failed; no blind retry')
        try:
            with (root/'monitor.log').open('a') as log:monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            inherited=(root/'accepted_foundation.json').exists()
            if inherited:accepted_foundation(root)
            if mechanism:
                require(inherited,'Reuse accepted foundation with affected regressions')
                worker('hand_cap_pilot',['-m','tools.tight_finish','hand_cap_pilot','--root',str(root)],read(root/'hand_cap_plan.json')['deadline'])
                # A negative mechanism result does not cancel independent B.
                worker('baseline_evaluation',['-m','tools.tight_finish','baseline_evaluate','--root',str(root)],read(root/'baseline_plan.json')['deadline'])
                if not read(root/'hand_cap_summary.json')['passed']:
                    write(root/'status.json',dict(state='complete',stage='bounded_mechanism_and_baseline_evaluated',
                        mechanism=read(root/'hand_cap_summary.json'),baseline=read(root/'baseline_summary.json'),
                        formal_updates=0,formal_slow_episodes=0,production_blocked=True))
                    return
                worker('independent_cap_validation',['-m','tools.tight_finish','validate_speed','--root',str(root)],s['development_deadline'])
                if not read(root/'scheme_reliability.json')['passed']:
                    write(root/'status.json',dict(state='complete',stage='independent_validation_rejected',
                        mechanism=read(root/'scheme_reliability.json'),baseline=read(root/'baseline_summary.json'),
                        formal_updates=0,formal_slow_episodes=0,production_blocked=True))
                    return
            if baseline:
                require(inherited,'Baseline must explicitly retain accepted engineering evidence')
                worker('baseline_evaluation',['-m','tools.tight_finish','baseline_evaluate','--root',str(root)],s['deadline'])
                write(root/'status.json',dict(state='complete',stage='baseline_evaluated',formal_updates=0,formal_slow_episodes=0,
                    baseline=read(root/'baseline_summary.json')))
                return
            elif not inherited and not (root/'gpu_parity.json').exists():
                worker('foundation_fixture',['-m','tools.tight_audit','fixture','--root',str(root)],s['development_deadline'])
                worker('foundation_gpu',['-m','tools.tight_audit','gpu','--root',str(root),'--data',str(root/'diagnostic_data')],s['development_deadline'])
            if not inherited:require(read(root/'gpu_parity.json')['passed'],'Foundation full trajectory GPU gate must pass')
            if speed_validation:
                worker('speed_validation',['-m','tools.tight_finish','validate_speed','--root',str(root)],s['development_deadline'])
            if refinement:
                require(s.get('development_revision')==2,'Refinement revision missing; refuse startup')
                worker('refined_development_v2',['-m','tools.tight_finish','refine','--root',str(root)],s['development_deadline'])
            gate=read(root/'selected_development_gate.json')
            if not gate['passed']:raise ValueError('Fixed-scheme gate failed: '+json.dumps(gate.get('reliability',gate.get('reason'))))
            if not inherited and not (root/'actuator_gate.json').exists():
                worker('actuator_gate',['-m','tools.tight_finish','actuator_worker','--root',str(root)],s['development_deadline'])
            if not inherited:require(read(root/'actuator_gate.json')['passed'],'Actual processed-target execution gate must pass')
            worker('collection',['-m','tools.tight_finish','collect','--root',str(root)],s['collection_deadline'])
            worker('formal_cpu_audit',['-m','tools.tight_audit','cpu','--root',str(root),'--data',str(root/'state_data')],s['deadline'])
            if not inherited:worker('formal_gpu_audit',['-m','tools.tight_audit','gpu','--root',str(root),'--data',str(root/'state_data')],s['deadline'])
            data=read(root/'state_data/data.json');ids=data['splits']['train'];anchors=sum(math.ceil(data['entries'][i]['length']/12) for i in ids)
            batch=len(ids)/math.ceil(len(ids)/64);t=s['training'];train=root/'training';train.mkdir()
            total=math.ceil(t['maximum_anchor_epochs']*anchors/batch);boundary=math.ceil(t['initial_anchor_epochs']*anchors/batch)
            write(train/'budget.json',dict(anchors=anchors,mean_batch=batch,total_updates=total,initial_updates=boundary,
                selection='validation only',sufficient_training_not_inferred_from_budget=True))
            runtime=dict(resource_policy=s['resource_policy'],resource_target_percent=s['resource_target_percent'],external_monitor_root=str(root),
                training_deadline=min(s['deadline']-3600,time.time()+s['training_hours']*3600),random_seed=t['random_seed'],
                validation_interval=t['validation_interval'],early_stop_minimum_update=math.ceil(t['minimum_anchor_epochs']*anchors/batch),
                deterministic_validation_loss=True)
            stage=0
            while True:
                stage+=1;runtime['stage_stop_update']=boundary;write(train/'session.json',runtime)
                worker('training_'+str(stage),['-m','visuotactile.anchor_continue','fresh-train','--job','state','--data',str(root/'state_data'),
                    '--root',str(train),'--target',str(total),'--output',str(train/f'launch_{stage}')],runtime['training_deadline'])
                end=read(train/'finished.json');rows=[json.loads(x) for x in (train/'validation.jsonl').read_text().splitlines()]
                recent=rows[-t['plateau_validations']:];gain=(recent[0]['metrics']['selection_mse']-min(r['metrics']['selection_mse'] for r in recent))/max(recent[0]['metrics']['selection_mse'],1e-12)
                improving=len(recent)<t['plateau_validations'] or gain>=t['plateau_relative_improvement']
                if end['updates']<boundary or not improving or boundary==total:break
                boundary=min(total,boundary+math.ceil(t['extension_anchor_epochs']*anchors/batch))
            write(train/'sufficiency.json',dict(updates=end['updates'],stop_reason=end['stop_reason'],recent_validation_gain=gain,
                still_improving=improving,sufficient=not improving and end['updates']>=runtime['early_stop_minimum_update'],
                final_test_used=False,scope='Finite preregistered exposure and validation trend; no claim of global convergence'))
            worker('evaluation',['-m','tools.tight_finish','evaluate','--root',str(root)],s['deadline'])
            write(root/'status.json',dict(state='complete',stage='evaluated',formal_updates=end['updates'],data=read(root/'DATA_READY.json'),evaluation=read(root/'evaluation_summary.json')))
        except BaseException as error:
            write(root/'finish_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            previous=read(root/'status.json');write(root/'status.json',dict(previous,state='blocked',reason=repr(error)))
            raise
        finally:
            if child is not None:kill_owned(child)
            (root/'STOP_MONITOR').touch()
            if monitor is not None:
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
            check_protected(root)
            status=read(root/'status.json');status['owned_workers_exited']=True;write(root/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','refine','collect','evaluate','actuator_worker','validate_speed','baseline_evaluate','hand_cap_pilot']);p.add_argument('--root',type=Path,required=True);p.add_argument('--refinement',action='store_true');p.add_argument('--speed-validation',action='store_true');p.add_argument('--baseline',action='store_true');p.add_argument('--mechanism',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='run':pipeline(root,a.refinement,a.speed_validation,a.baseline,a.mechanism)
    else:
        code=0
        try:globals()[a.mode](root)
        except BaseException:
            write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc()));code=1
        sys.stdout.flush();sys.stderr.flush();os._exit(code)
