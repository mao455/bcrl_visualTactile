"""Independent CPU post-hoc review. Never edits frozen code or rollout evidence."""
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

# Helpers are imported from the explicitly supplied, fully hash-verified frozen tree.
# Run as a standalone file, so a development tools package cannot shadow that tree.
import argparse
import hashlib


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_helpers(tree):
    # Reject an imported development implementation instead of silently mixing it
    # with historical frozen helpers. Direct script invocation starts clean.
    for package in ('tools', 'visuotactile'):
        module = sys.modules.get(package)
        if module is not None and list(getattr(module, '__path__', [])) != [str(tree / package)]:
            raise ValueError('Run this tool as a standalone file with no foreign research packages imported')
    sys.path.insert(0, str(tree))
    global phase_case, contact_motion, causal_contact_height, role_report, completed_pair
    from tools.tight_fresh_phases import case as phase_case
    from tools.tight_fresh_visual_closeout import contact_motion
    from tools.tight_fresh_contact_height_coverage import causal_contact_height
    from tools.tight_fresh_learning_report import role_report
    from tools.tight_fresh_rollouts import completed_pair


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def check_sources(plan,tree):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==plan['source_commit'],'Frozen commit differs')
    for name,digest in plan['source_sha256'].items():
        require(sha256(tree/name)==digest,'Frozen file changed: '+str(tree/name))
    return dict(tree=str(tree),commit=plan['source_commit'],verified_source_files=len(plan['source_sha256']))


def held_orientation(directory):
    with h5py.File(directory/'numeric.h5','r') as f:
        t=f['episode/transition'];p=t['diagnostics_after/object_pose_w'][:]
        goal=t['diagnostics_before/goal_pose_w'][:];forces=t['object_contact_force_w'][:]
        mask,height=causal_contact_height(p,forces)
        indices=np.flatnonzero(mask & (t['diagnostics_before/successes'][:]==0))
        cosine=np.abs((p[:,3:]*goal[:,3:]).sum(-1))/(np.linalg.norm(p[:,3:],axis=-1)*np.linalg.norm(goal[:,3:],axis=-1))
        angle=2*np.arccos(np.clip(cosine,0.,1.));keypoint=t['native_keypoints_max_dist'][:]
        tolerance=t['native_success_diagnostic/insertion_tolerance_m'][:]
        def snapshot(i):
            return dict(completed_tick=int(i+1),root_angle_rad=float(angle[i]),
                root_distance_m=float(np.linalg.norm(p[i,:3]-goal[i,:3])),
                native_keypoint_distance_m=float(keypoint[i]),native_tolerance_m=float(tolerance[i]))
        if not len(indices):return dict(eligible_contact_height_windows=0)
        first=int(indices[0]);best=int(indices[np.argmin(angle[indices])])
        return dict(eligible_contact_height_windows=len(indices),first=snapshot(first),
            smallest_root_angle=snapshot(best),angle_decrease_from_first_window_rad=float(angle[first]-angle[best]),
            interpretation='Both snapshots have full causal 12-tick contact-height windows and the same first goal; quaternion angles ignore object symmetry.')


def pair_root(pair,role):
    """The evaluated pair may belong to a different training root for each role."""
    front=Path(pair['frontend']).resolve(strict=True)
    require(front.name=='best.pt' and front.parent.name=='training' and front.parent.parent.name==role,
        'Frontend is not an explicitly selected role checkpoint')
    root=front.parents[2]
    require(Path(pair['backend']).resolve(strict=True)==root/role/'backend'/role/'training/best.pt',
        'Frontend/backend training roots differ')
    return root


def execution_identity(plan):
    batch=plan.get('batch')
    if batch is not None:
        require(batch.get('version')=='whole_wave_complete_student_v1' and batch.get('neural_batch') is True,
            'Unsupported batch execution identity')
        require(type(batch['lanes']) is int and batch['lanes']>0,'Invalid registered lane count')
    return dict(mode='whole_wave' if batch else 'serial',lanes=batch['lanes'] if batch else 1,
        vision_microbatch=batch.get('vision_microbatch',2) if batch else 2,
        vision_tail_profile=batch.get('vision_tail_profile') if batch else None,
        inactive_render_profile=batch.get('inactive_render_profile') if batch else None,
        batch_seed=batch['seed'] if batch else None,
        inference_scheduler_index_mode=plan.get('inference_scheduler_index_mode','legacy'),
        execution_backend=plan.get('execution_backend',dict(version='tight_execution_backend_v1',
            physics_device='cpu',camera_backend='standard',kit_tasking_threads=2,physics_threads=2)))


def validate_development_plan(plan):
    require(plan['kind']=='autonomous' and plan['test_used'] is False and plan['cases'],
        'Complete autonomous development cohort required')
    require(set(plan['pairs'])=={'A','B'},'Both registered model roles required')
    names=[c['name'] for c in plan['cases']]
    require(len(names)==len(set(names)) and all(c['role'] in ('A','B') and
        c['spec']['split']=='development' and not c.get('dagger') for c in plan['cases']),
        'Duplicate or non-development opportunities')
    families={role:[c['spec']['family'] for c in plan['cases'] if c['role']==role] for role in ('A','B')}
    require(families['A'] and len(set(families['A']))==len(families['A']) and
        sorted(families['A'])==sorted(families['B']),'A/B development families differ or repeat')

def compare_plans(old,new):
    for plan in (old,new):validate_development_plan(plan)
    require([(c['name'],c['role'],c['spec']) for c in old['cases']]==
        [(c['name'],c['role'],c['spec']) for c in new['cases']],'Development opportunities changed')
    require(execution_identity(old)==execution_identity(new),'Development execution backend/batch contract changed')
    return dict(opportunities_per_role={r:sum(c['role']==r for c in old['cases']) for r in ('A','B')},
        case_specs_identical=True,execution_identity=execution_identity(old),exact_hidden_state_pairing=False)


def wave_assignments(plan):
    assignments={}
    if 'batch' in plan:
        wave=0
        for role in dict.fromkeys(case['role'] for case in plan['cases']):
            selected=[case for case in plan['cases'] if case['role']==role]
            for start in range(0,len(selected),plan['batch']['lanes']):
                for lane,case in enumerate(selected[start:start+plan['batch']['lanes']]):
                    assignments[case['name']]=(wave,lane)
                wave+=1
    return assignments


def compare_role_plans(old,new,role):
    """Compare one role without inventing a plan from two historical runs."""
    require(role in ('A','B'),'Unknown source role')
    for plan in (old,new):validate_development_plan(plan)
    selected=lambda plan:[(c['name'],c['role'],c['spec']) for c in plan['cases'] if c['role']==role]
    require(selected(old)==selected(new),'Development opportunities changed for '+role)
    require(execution_identity(old)==execution_identity(new),'Development execution backend/batch contract changed for '+role)
    before,after=wave_assignments(old),wave_assignments(new)
    require(all(before.get(name)==after.get(name) for name,_,_ in selected(old)),
        'Development wave/lane assignment changed for '+role)
    return dict(role=role,opportunities=len(selected(old)),case_specs_identical=True,
        wave_lane_assignments_identical=True,execution_identity=execution_identity(old),exact_hidden_state_pairing=False)


def source_manifest_inputs(path):
    """Read only original resource/plan/tree bindings; never synthesize a plan."""
    path=Path(path).resolve(strict=True);manifest=read(path)
    require(set(manifest)=={'version','roles'} and manifest['version']=='per_role_development_sources_v1',
        'Unsupported source manifest')
    require(set(manifest['roles'])=={'A','B'},'Source manifest requires exactly A and B')
    inputs={}
    for role,entry in manifest['roles'].items():
        require(set(entry)=={'root','development','tree'},'Source role requires root/development/tree only')
        require(all(isinstance(value,str) and Path(value).is_absolute() for value in entry.values()),
            'Source manifest paths must be absolute')
        resolved={key:Path(value).resolve(strict=True) for key,value in entry.items()}
        require(all(value.is_dir() for value in resolved.values()),'Source manifest paths must be directories')
        plan=read(resolved['development']/'plan.json');validate_development_plan(plan)
        require(Path(plan['resource_root']).resolve(strict=True)==resolved['root'],
            'Source root differs from executed plan resource_root')
        require(resolved['development']!=resolved['root'] and resolved['development'].is_relative_to(resolved['root']),
            'Source development must remain within its executed resource root')
        check_sources(plan,resolved['tree'])
        inputs[role]=dict(resolved,plan=plan)
    return inputs,dict(path=str(path),sha256=sha256(path),version=manifest['version'],
        roles={role:{key:str(value[key]) for key in ('root','development','tree')} for role,value in inputs.items()},
        scope='Independent references to executed source plans. No combined plan or new rollout was created.')


def verify_execution_receipt(directory,plan):
    identity=execution_identity(plan)
    if 'batch' not in plan:return identity
    observed=read(directory/'execution_backend_observed.json');initial=read(directory/'initialization.json')
    require(observed['requested']['configuration']==identity['execution_backend'] and
        observed['physics_device']==identity['execution_backend']['physics_device'] and
        observed['camera_backend']==identity['execution_backend']['camera_backend'] and
        observed['num_envs']==observed['requested']['num_envs']==initial['lanes']==identity['lanes'],
        'Observed execution backend differs from plan')
    require(initial['execution_backend']==identity['execution_backend'] and
        initial['vision_microbatch']==identity['vision_microbatch'] and
        initial.get('vision_tail_profile')==identity['vision_tail_profile'] and
        initial.get('inactive_render_profile')==identity['inactive_render_profile'],'Observed visual batching differs')
    require(abs(observed['physics_dt']-1/120)<1e-8 and abs(observed['control_dt']-1/60)<1e-8,
        'Observed native clocks differ')
    return dict(identity,observed_sha256=sha256(directory/'execution_backend_observed.json'))


def audit_numeric_clock(path,steps):
    with h5py.File(path/'numeric.h5','r') as f:
        t=f['episode/transition'];obs=f['episode/obs']
        targets=t['target_after_rad'][:];previous=obs['actuator_target_rad'][:]
        require(targets.shape==previous.shape==(steps,29) and np.isfinite(targets).all(),
            'Nonfinite or noncanonical 29D targets')
        require(np.array_equal(targets[:-1],previous[1:]),'Actual target/history continuity differs')
        times=obs['time_s'][:];camera=obs['camera_time_s'][:];frames=obs['camera_frame'][:]
        require(times.shape==(steps,) and camera.shape==frames.shape==(steps,3) and
            np.isfinite(times).all() and np.isfinite(camera).all() and
            np.allclose(camera,times[:,None],atol=1e-8,rtol=0) and
            np.allclose(np.diff(times),1/60,atol=1e-8,rtol=0),'Real observation clock differs')
        require(obs['camera_valid'][:].shape==(steps,3) and obs['camera_valid'][:].all() and
            (np.diff(frames,axis=0)>0).all(),'Invalid or reused real camera frames')
        require(np.allclose(t['time_before_s'][:],times,rtol=0,atol=1e-8) and
            np.allclose(t['time_after_s'][:]-times,1/60,rtol=0,atol=1e-8),'Native transition clock differs')
    return dict(control_ticks=steps,real_camera_views=3,finite_absolute_target_dimensions=29,
        target_history_continuous=True,camera_clock_verified=True)


def invalid_initialization_evidence(path,result):
    require(result.get('software_complete') is True and result.get('legal') is False and
        result.get('not_started') is True and result.get('success') is False and
        result.get('steps') in (None,0) and result.get('residual_ticks') in (None,0) and
        result.get('dagger') is None and not result.get('censored'),
        'Illegal initialization contains an executed or incomplete task outcome')
    collision=read(path/'initial_collision.json')
    require(collision.get('legal') is False and collision['version']==result['collision_gate'],
        'Illegal initialization lacks matching collision rejection')
    numeric=path/'numeric.h5'
    if numeric.exists():
        with h5py.File(numeric,'r') as stream:
            require('episode/transition/step_id' not in stream or len(stream['episode/transition/step_id'])==0,
                'Rejected initialization contains actual control transitions')
    return dict(classification='invalid_initialization',task_outcome=None,raw_success_field=result['success'],
        not_started=True,reason=result.get('reason'),
        collision={key:collision.get(key) for key in ('version','legal','minimum_gap_m','margin_m','limitation')},
        evidence_sha256={name:sha256(path/name) for name in
            ('result.json','started.json','initial_collision.json','initial_observable.json')},
        interpretation='Registered opportunity rejected before control; not an autonomous task failure or replacement opportunity.')


def autonomy_totals(cases,*,batch=False,explicit=False):
    chosen=[c for c in cases if c['legal']]
    successes=sum(c['success'] for c in chosen)
    result=dict(opportunities=len(cases),successes=successes,
        native_lift_cases=sum(c['native_lift'] for c in chosen),
        old_stable_window_cases=sum(c['settled_height_stable_grasp'] for c in chosen),
        contact_height12_cases=sum(c['contact_motion']['first_contact_height_window_end_tick'] is not None for c in chosen),
        native_alignment_cases=sum(c['native_alignment_first_tick'] is not None for c in chosen),
        native_near_goal_cases=sum(c['native_near_goal_first_tick'] is not None for c in chosen),
        first_subgoal_cases=sum(c['native_subgoals']>0 for c in chosen),
        termination_reasons=dict(Counter(k for c in chosen for k,v in c['reasons'].items() if v)),
        total_control_ticks=sum(c['steps'] for c in chosen),
        **({'overlapping_lane_lifetime_seconds':sum(c['wall_seconds'] for c in chosen)} if batch else
           {'rollout_wall_seconds':sum(c['wall_seconds'] for c in chosen)}))
    if explicit:
        result.update(legal_initializations=len(chosen),invalid_initializations=len(cases)-len(chosen),
            autonomous_success_rate_among_legal=successes/len(chosen) if chosen else None,
            raw_successes_per_registered_opportunity=dict(numerator=successes,denominator=len(cases),
                ratio=successes/len(cases) if cases else None,
                scope='Raw outcome yield including rejected starts in the opportunity denominator; not legal-start task failure rate.'),
            task_outcome_denominator='Legal completed autonomous executions only; invalid starts excluded.')
    return result


def matched_legal_comparison(source,candidate,*,roles=('A','B')):
    left={r['case']:r for r in source['cases']};right={r['case']:r for r in candidate['cases']}
    require(set(left)==set(right),'Completed source/candidate opportunity identities differ')
    by_role={}
    for role in roles:
        names=sorted(k for k,row in left.items() if row['role']==role)
        require(all(right[k]['role']==role for k in names),'Source/candidate role identity differs')
        matched=[k for k in names if left[k]['legal'] and right[k]['legal']]
        by_role[role]=dict(registered_opportunities=len(names),matched_legal_opportunities=len(matched),
            matched_legal_cases=matched,source_only_legal_cases=[k for k in names if left[k]['legal'] and not right[k]['legal']],
            candidate_only_legal_cases=[k for k in names if right[k]['legal'] and not left[k]['legal']],
            both_invalid_cases=[k for k in names if not left[k]['legal'] and not right[k]['legal']],
            source=autonomy_totals([left[k] for k in matched],batch=source['execution']['mode']=='whole_wave',explicit=True),
            candidate=autonomy_totals([right[k] for k in matched],batch=candidate['execution']['mode']=='whole_wave',explicit=True))
    return dict(by_role=by_role,scope='Intersection of actual legal initializations under the same registered case specs; '
        'admission differences remain in all-opportunity evidence. No hidden-state equality or new held-out generalization claim.')


def audit_group(label,root,tree,development=None,*,allow_illegal=False,roles=('A','B')):
    directory=(root/'development' if development is None else development).resolve()
    plan=read(directory/'plan.json');finished=read(directory/'result.json')
    source_receipt=check_sources(plan,tree);rows=read(directory/'results.json')
    count=len(plan['cases'])
    require(finished['software_complete'] and not finished.get('worker_error') and finished['cases']==count,'Incomplete worker')
    require(read(directory/'run_started.json')['plan_sha256']==sha256(directory/'plan.json'),'Startup plan hash differs')
    require(plan['kind']=='autonomous' and plan['test_used'] is False and len(rows)==count,'Wrong cohort')
    require(roles and len(set(roles))==len(roles) and all(role in ('A','B') for role in roles),'Invalid audit roles')
    pairs={role:plan['pairs'][role] for role in roles}
    for role,pair in pairs.items():
        require(completed_pair(pair_root(pair,role),role)==pair,'Completed pair differs from executed pair')
    for pair in pairs.values():
        for key in ('frontend','backend'):
            require(sha256(pair[key])==pair[key+'_sha256'],'Model changed')
    execution=verify_execution_receipt(directory,plan)
    indexed={r['case']:r for r in rows};require(len(indexed)==count,'Duplicate case')
    require(set(indexed)=={case['name'] for case in plan['cases']},'Completed opportunity index differs')
    assignments=wave_assignments(plan)
    cases=[]
    for planned in plan['cases']:
        if planned['role'] not in roles:continue
        name=planned['name'];role=planned['role'];path=directory/name;row=indexed[name]
        r=read(path/'result.json');started=read(path/'started.json');pair=pairs[role]
        require(row['result']==r and row['role']==role,'Index/case mismatch')
        require(started['spec']==planned['spec'],'Started identity mismatch')
        if 'batch' in plan:
            meta=read(path/'metadata.json');batch=r['batch_execution'];wave,lane=assignments[name]
            require(started['batch_execution']==meta['batch_execution']==batch and
                (batch['wave'],batch['lane'])==(wave,lane) and batch['lanes']==plan['batch']['lanes'] and
                batch['vision_microbatch']==execution['vision_microbatch'] and
                batch.get('vision_tail_profile')==execution['vision_tail_profile'] and
                batch['physics_seed']==plan['batch']['seed']+wave and
                batch['event_seed']==plan['batch']['seed']+100000+wave,'Per-lane execution binding differs')
            require(meta['policy_checkpoint_sha256']==pair['backend_sha256'] and
                meta['execution_backend']==execution['execution_backend'],
                'Per-lane policy/backend binding differs')
            if r['legal']:require(r['complete_student_policy']==pair,'Executed complete student policy differs')
        else:require(started['checkpoint']==pair['backend'],'Started checkpoint differs')
        require(all(r[k]==v for k,v in planned['spec'].items()),'Executed reset differs')
        if not r['legal'] and allow_illegal:
            cases.append(dict(group=label,role=role,case=name,legal=False,
                planned_pair_sha256={key:pair[key] for key in ('frontend_sha256','backend_sha256')},
                initial_rejection=invalid_initialization_evidence(path,r)))
            continue
        require(r['software_complete'] and r['legal'] and not r.get('censored') and r.get('dagger') is None,'Not complete autonomous execution')
        require(r['checkpoint_sha256']==pair['backend_sha256'] and r['residual_version']==pair['residual_version'],'Model contract mismatch')
        require(r['residual_ticks']==r['steps'] and r['simulation_step_hz']==60 and r['physics_hz']==120 and r['plan_hz']==5,'Clock mismatch')
        require(r['action_control_mode']=='absolute_joint_position_target_rad' and not r['expert_shadow_computed'],'Control mismatch')
        for stage in ('dino_rgb','learned_visual_and_history','residual_forward_compose'):
            require(r['native_inference']['summary'][stage]['wall_s']['count']==r['steps'],'Dropped real-frame student stage')
        require(r['native_inference']['summary']['ddim_plan']['wall_s']['count']==(r['steps']+11)//12,'Planning clock mismatch')
        phase=phase_case(path);require(phase['authority']['expert_ticks']==0 and phase['steps']==r['steps'],'Expert action or numeric length mismatch')
        require(phase['native_lift']==r['native_lift'] and phase['native_subgoals']==r['native_subgoals'],'Native numeric/result mismatch')
        clocks=audit_numeric_clock(path,r['steps'])
        cases.append(dict(group=label,role=role,**phase,contact_motion=contact_motion(path,phase),
            held_orientation=held_orientation(path),clock_audit=clocks,wall_seconds=row['timing']['total_seconds']))
    totals={role:autonomy_totals([c for c in cases if c['role']==role],batch='batch' in plan,explicit=allow_illegal)
        for role in roles}
    return dict(root=str(root),development=str(directory),execution=execution,
        plan_sha256=sha256(directory/'plan.json'),results_sha256=sha256(directory/'results.json'),
        completion_sha256=sha256(directory/'result.json'),source_verification=source_receipt,pairs=pairs,totals=totals,cases=cases,
        initialization=read(directory/'initialization.json'))


def compact_learning(root, roles):
    """Separate completed-run exposure from the validation-selected frontend."""
    from tools.tight_fresh_learning_report import exposure_rows, learning_sources
    result = {}
    for role, report in roles.items():
        directory = root / role
        data = read(directory / 'state_data/data.json')
        corrections, current, sources = learning_sources(data)
        receipts = [json.loads(line) for line in (directory / 'training/learning.jsonl').read_text().splitlines()]
        selected = receipts[:report['selected_updates']]
        counts, exposure = exposure_rows(selected, set(data['splits']['train']), corrections, source_groups=sources)
        epoch = next(row for row in report['history'] if row['epoch'] == report['selected_epoch'])
        require(dict(counts) == epoch['sampling_exposure'], 'Selected exposure differs from epoch receipt')
        compact = lambda value: {key: item for key, item in value.items() if key != 'episodes'}
        result[role] = dict(
            frontend_sha256=report['checkpoint_sha256'],
            backend_sha256=report['backend_selected']['checkpoint_sha256'],
            source_populations={key: len(ids) for key, ids in report['source_populations'].items()},
            completed_frontend_epochs=report['latest_epoch'], completed_frontend_updates=report['latest_updates'],
            completed_frontend_exposure=report['actual_exposure_by_correction_age'],
            selected_frontend_epoch=report['selected_epoch'], selected_frontend_updates=report['selected_updates'],
            selected_frontend_exposure=exposure,
            selected_current_episode_visits={identity: counts[identity] for identity in sorted(current)},
            validation_initial=compact(report['initial_metrics']['validation']),
            validation_selected=compact(report['selected_validation']),
            validation_relative_change=report['selected_validation_relative_change'],
            correction_initial=compact(report['initial_metrics']['correction']),
            correction_last_epoch=compact(report['last_epoch_correction_metrics']),
            correction_last_epoch_is_selected=report['correction_metrics_match_selected'],
            selected_backend_epoch=report['backend_selected_metrics']['epoch'],
            selected_backend_validation=report['backend_selected_metrics']['validation'],
            completed_backend_exposure=report['backend_actual_exposure_by_correction_age'],
            backend_cache=report['backend_cache'],
            test_used=False,
            limits='Exposure reports actual supervised slots/draws, not independent states. '
                'Last-epoch correction fitting describes the deployed frontend only when its epoch equals the selected epoch.')
    return result


def frontend_checkpoint_receipt(pair,role):
    """Read actual selected and last frontend identities on CPU, without a session-root assumption."""
    import torch
    root=pair_root(pair,role);training=root/role/'training'
    finished=read(training/'finished.json')
    require(finished.get('complete') is True and finished['checkpoint_sha256']==pair['frontend_sha256'],
        'Selected frontend lacks matching completion receipt')
    checkpoints={}
    for name in ('best','last'):
        path=training/(name+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        checkpoints[name]=dict(epoch=saved['epoch'],updates=saved['updates'],sha256=sha256(path),
            continuation=saved['configuration'].get('continuation'))
        del saved
    require(checkpoints['best']['sha256']==pair['frontend_sha256'] and
        checkpoints['last']['epoch']==finished['epochs'] and checkpoints['last']['updates']==finished['updates'],
        'Frontend selected/last completion binding differs')
    return dict(training_root=str(root),selected=checkpoints['best'],last=checkpoints['last'],
        completed_target_slots=finished['target_ticks'],completed_anchors=finished['anchors_seen'],
        finished_sha256=sha256(training/'finished.json'),
        budget_scope='This continuation stage only; source roles may have different prior rounds/data/updates. Not a matched cumulative-budget comparison.')


def initial_observation_comparison(source,candidate,cases):
    result={}
    for case in cases:
        name=case['name'];paths=[directory/name/'initial_observable.json' for directory in (source,candidate)]
        before,after=map(read,paths)
        require(set(before)==set(after),'Saved observable initial-state fields differ')
        fields={}
        for key in before:
            a,b=np.asarray(before[key]),np.asarray(after[key])
            require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                'Invalid saved observable initial-state geometry')
            fields[key]=dict(equal=bool(np.array_equal(a,b)),max_absolute_difference=float(np.abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.)
        result[name]=dict(all_fields_equal=all(row['equal'] for row in fields.values()),fields=fields,
            source_sha256=sha256(paths[0]),candidate_sha256=sha256(paths[1]))
    return dict(cases=result,all_cases_equal=all(row['all_fields_equal'] for row in result.values()),
        scope='Saved observable state only; equality does not establish identical hidden PhysX, renderer, contact solver or RNG state.')


def stable_arithmetic_identical(trees):
    sources=[]
    for tree in trees:
        source=(tree/'visuotactile/takeover_trigger.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='stable_windows')
        sources.append(ast.get_source_segment(source,node))
    require(all(value==sources[0] for value in sources),'Old stable arithmetic differs')


def write_learning(output,root,candidate_pairs,source_pairs,tree,*,explicit):
    roles={r:role_report(root,r) for r in ('A','B')}
    bindings={r:frontend_checkpoint_receipt(candidate_pairs[r],r) for r in ('A','B')}
    for role,report in roles.items():
        require(report['checkpoint_sha256']==candidate_pairs[role]['frontend_sha256'] and
            report['backend_selected']['checkpoint_sha256']==candidate_pairs[role]['backend_sha256'] and
            bindings[role]['training_root']==str(root),'Learning report is not the evaluated candidate pair')
        require((report['selected_epoch'],report['selected_updates'])==
            (bindings[role]['selected']['epoch'],bindings[role]['selected']['updates']) and
            (report['latest_epoch'],report['latest_updates'])==
            (bindings[role]['last']['epoch'],bindings[role]['last']['updates']),
            'Selected/last metrics differ from actual checkpoints')
    learning_name='learning_candidate.json' if explicit else 'learning_round06.json'
    atomic_json(output/learning_name,dict(roles=roles,checkpoint_bindings=bindings,
        helper_root=str(tree),helper_sha256=sha256(tree/'tools/tight_fresh_learning_report.py'),test_used=False))
    atomic_json(output/'learning_summary.json',dict(roles=compact_learning(root,roles),
        source_frontend_checkpoints={r:frontend_checkpoint_receipt(source_pairs[r],r) for r in ('A','B')},
        candidate_frontend_checkpoints=bindings,
        full_report_sha256=sha256(output/learning_name),source_sha256=sha256(__file__),test_used=False))


def run_role_sources(source_manifest,candidate_root,candidate_tree,output,*,candidate_development=None,preflight=False):
    inputs,manifest_receipt=source_manifest_inputs(source_manifest)
    helper_tree=inputs['A']['tree'] if preflight else candidate_tree.resolve(strict=True)
    trees=[inputs[role]['tree'] for role in ('A','B')]+[helper_tree]
    stable_arithmetic_identical(trees)
    if not preflight:
        candidate_root=candidate_root.resolve(strict=True)
        directory=(candidate_root/'development' if candidate_development is None else candidate_development).resolve(strict=True)
        new=read(directory/'plan.json');check_sources(new,helper_tree)
        require(Path(new['resource_root']).resolve(strict=True)==candidate_root and directory!=candidate_root and
            directory.is_relative_to(candidate_root),'Candidate development/resource root differs from executed plan')
        population={role:compare_role_plans(inputs[role]['plan'],new,role) for role in ('A','B')}
    load_helpers(helper_tree)
    sources={role:audit_group('source_'+role,value['root'],value['tree'],value['development'],
        allow_illegal=True,roles=(role,)) for role,value in inputs.items()}
    common=dict(source_manifest=manifest_receipt,source_sha256=sha256(__file__),helper_root=str(helper_tree),
        helper_sha256={name:sha256(helper_tree/name) for name in
            ('tools/tight_fresh_phases.py','tools/tight_fresh_visual_closeout.py',
             'tools/tight_fresh_contact_height_coverage.py','tools/tight_fresh_learning_report.py','visuotactile/takeover_trigger.py')},
        old_stable_function_source_identical=True,simulator_starts=0,optimizer_updates=0,test_used=False)
    output.mkdir(parents=True,exist_ok=True)
    if preflight:
        atomic_json(output/'source_preflight.json',dict(version='per_role_source_preflight_v1',complete=True,
            **common,source_by_role=sources,candidate_audited=False,
            scope='Completed source evidence only; no candidate plan, result, or comparison was created.'))
        print(json.dumps({role:value['totals'] for role,value in sources.items()},indent=2),flush=True)
        return
    candidate=audit_group('candidate',candidate_root,helper_tree,directory,allow_illegal=True)
    initial={};matching={}
    for role,value in inputs.items():
        cases=[case for case in value['plan']['cases'] if case['role']==role]
        initial[role]=initial_observation_comparison(value['development'],directory,cases)
        candidate_role=dict(candidate,cases=[case for case in candidate['cases'] if case['role']==role])
        matching[role]=matched_legal_comparison(sources[role],candidate_role,roles=(role,))['by_role'][role]
    atomic_json(output/'autonomy.json',dict(version='per_role_development_cpu_review_v1',complete=True,
        **common,population_by_role=population,initial_observation_comparison_by_role=initial,
        source_by_role=sources,candidate=candidate,matched_legal_comparison=dict(by_role=matching,
            scope='Per-role intersection of actual legal starts from independent executed plans; all rejected opportunities remain in source/candidate evidence.'),
        limits='Sources are role-specific historical executions, not a jointly run mixed source pair. '
            'These reused development starts are not new held-out data or exact hidden-state pairs. '
            'Contact-height and native lift/success remain separate. Source budgets and prior data may differ.'))
    print(json.dumps(dict(source={role:value['totals'][role] for role,value in sources.items()},
        candidate=candidate['totals']),indent=2),flush=True)
    write_learning(output,candidate_root,new['pairs'],{role:value['plan']['pairs'][role] for role,value in inputs.items()},
        helper_tree,explicit=True)
    print('Finished per-role CPU autonomy and learning audits',flush=True)


def run(source_root=None, candidate_root=None, source_tree=None, candidate_tree=None, output=None, *,
        source_development=None,candidate_development=None,source_manifest=None,source_preflight=False):
    require(output is not None,'Review output is required')
    OUT = output.resolve()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Preserve previous review evidence")
    if source_manifest is not None:
        require(all(value is None for value in (source_root,source_tree,source_development)),
            'Source manifest cannot be combined with single-source arguments')
        if source_preflight:
            require(all(value is None for value in (candidate_root,candidate_tree,candidate_development)),
                'Source preflight does not accept candidate arguments')
        else:require(candidate_root is not None and candidate_tree is not None,'Candidate root/tree required')
        return run_role_sources(source_manifest,candidate_root,candidate_tree,OUT,
            candidate_development=candidate_development,preflight=source_preflight)
    require(not source_preflight,'Source preflight requires a source manifest')
    require(all(value is not None for value in (source_root,source_tree,candidate_root,candidate_tree)),
        'Single-source comparison requires source/candidate root/tree')
    FROZEN, OLD_FROZEN = candidate_tree.resolve(), source_tree.resolve()
    explicit=source_development is not None or candidate_development is not None
    labels=('source','candidate') if explicit else ('source_round05','candidate_round06')
    ROOTS = dict(zip(labels,(source_root.resolve(),candidate_root.resolve())))
    TREES = dict(zip(labels,(OLD_FROZEN,FROZEN)))
    directories={labels[0]:(source_root/'development' if source_development is None else source_development).resolve(),
        labels[1]:(candidate_root/'development' if candidate_development is None else candidate_development).resolve()}
    for label, root in ROOTS.items():
        check_sources(read(directories[label]/'plan.json'), TREES[label])
    old,new=(read(directory/'plan.json') for directory in directories.values())
    population=compare_plans(old,new)
    load_helpers(FROZEN)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'autonomy.json').exists():raise FileExistsError('Preserve prior CPU analysis')
    # Old source files differ by additions, but original stability arithmetic is exact.
    stable_arithmetic_identical((OLD_FROZEN,FROZEN))
    groups={label:audit_group(label,root,TREES[label],directories[label],allow_illegal=explicit) for label,root in ROOTS.items()}
    initial=initial_observation_comparison(*directories.values(),old['cases'])
    matching=matched_legal_comparison(*groups.values()) if explicit else None
    helpers=['tools/tight_fresh_phases.py','tools/tight_fresh_visual_closeout.py',
        'tools/tight_fresh_contact_height_coverage.py','tools/tight_fresh_learning_report.py','visuotactile/takeover_trigger.py']
    atomic_json(OUT/'autonomy.json',dict(version='explicit_development_cpu_review_v2' if explicit else 'round06_independent_cpu_review_v1',complete=True,
        source_sha256=sha256(__file__),helper_root=str(FROZEN),helper_sha256={n:sha256(FROZEN/n) for n in helpers},
        old_stable_function_source_identical=True,population=population,initial_observation_comparison=initial,
        groups=groups,**({'matched_legal_comparison':matching} if explicit else {}),
        simulator_starts=0,optimizer_updates=0,test_used=False,
        limits='Registered development starts, not a held-out test estimate or exact hidden-state pairs. Contact-height permits palm rotation and is not native lift/success. Missing windows do not prove dropping. Angular and position minima are shown at their actual same-tick snapshots. Source A/B histories may have unequal budgets.'))
    print(json.dumps({label:g['totals'] for label,g in groups.items()},indent=2),flush=True)
    write_learning(OUT,ROOTS[labels[1]],new['pairs'],old['pairs'],FROZEN,explicit=explicit)
    print('Finished CPU autonomy and learning audits',flush=True)


def parser():
    value = argparse.ArgumentParser(__doc__)
    value.add_argument('--source-root', type=Path)
    value.add_argument('--source-manifest', type=Path, help='per_role_development_sources_v1 JSON with A/B root, development and tree; exclusive with single-source arguments')
    value.add_argument('--source-preflight',action='store_true',help='Audit only the two actual source roles; no candidate required or created')
    value.add_argument('--candidate-root', type=Path)
    value.add_argument('--source-development', type=Path, help='Explicit executed source development directory; defaults to source-root/development')
    value.add_argument('--candidate-development', type=Path, help='Explicit executed candidate development directory; defaults to candidate-root/development')
    value.add_argument('--source-tree', type=Path)
    value.add_argument('--candidate-tree', type=Path)
    value.add_argument('--output', type=Path, required=True)
    return value


if __name__ == '__main__':
    arguments = parser().parse_args()
    run(**vars(arguments))
