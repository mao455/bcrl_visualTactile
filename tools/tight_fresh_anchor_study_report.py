"""CPU evidence binding for the matched output-anchor study; never promotes models."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.tight_fresh_backend_study_report import (EXPOSURE_KEYS, check_onset_binding,
    read, rear_report)
from tools.tight_fresh_phases import case as phase_case
from tools.tight_fresh_rollouts import completed_pair
from tools.tight_fresh_study_onset import aggregate, aggregate_command_motion, correction_cohorts
from visuotactile.compact import sha256
from visuotactile.residual_output import (MATCHED_FINAL_HEAD_RESET,
    REACHABLE_CENTERED_OUTPUT, REACHABLE_PREVIOUS_TARGET_OUTPUT)
from visuotactile.training_resources import atomic_json

ARMS = ('control', 'previous')
STAGE_SEMANTICS = dict(
    lift='拿起: native lifted_object flag; elevation alone does not establish a secure grasp.',
    hold='拿稳: existing12tick two-finger/elevation/palm-relative-motion diagnostic; failure does not prove a drop.',
    turn='转正: report raw object-root/goal-root quaternion angle at the same recorded tick, including after first stable hold. No new success threshold; shape symmetries are not removed.',
    near='对孔接近: native near_goal keypoint condition; root distance alone is not this condition.',
    preinsert='对孔子目标: first native subgoal event. Historical alignment_cases names native _is_success after its near-goal counter, not an independent orientation criterion.',
    insert='插入: original native final completion/max_successes; neither offline error nor an intermediate flag establishes completion.',
    pose='All historical center_distance fields are root-reference-origin distances, not collision-AABB centers. New phase snapshots use the goal before the transition, because diagnostics_after may already contain the next goal at a native subgoal event.',
    order='Milestone counts are independent marginals, not a guaranteed nested funnel. Post-hold turning snapshots explicitly require a prior stable window. Ticks are1-based completed transitions.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_equal_exposure(control, previous):
    require(bool(control) and len(control)==len(previous), 'Optimizer window counts differ or are empty')
    for update, (a,b) in enumerate(zip(control,previous),1):
        require(a['update']==update and all(a[k]==b[k] for k in EXPOSURE_KEYS),
            'Actual optimizer schedule or exposure differs')
        require(a['loss_weight_mass']==a['supervised_ticks'] and b['loss_weight_mass']==b['supervised_ticks'],
            'Anchor comparison must retain unweighted supervision')
    return dict(updates=len(control), supervised_ticks=sum(r['supervised_ticks'] for r in control),
        history_ticks=sum(r['history_ticks'] for r in control),
        contact_ticks=sum(r['contact_ticks'] for r in control),
        fixed_window_denominator=control[0]['fixed_denominator'], loss_weight=1.,
        interpretation='history_ticks is optimizer-window history only, excluding whole chunks with zero legal supervision and excluding repeated prefix recomputation; the unweighted receipts do not separately label recovery-onset counts')


def audit_history_accounting(masks, saved, receipts):
    """Replay only the registered shuffle/masks, without model or optimizer calls."""
    from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT
    config=saved['configuration'];ids=saved['train_ids'];epochs=config['maximum_control_tick_epochs']
    require(config['tick_window_contract']==TICK_WINDOW_CONTRACT
        and config['history_contract']=={'version':'current_weight_prefix_v1'} and not config.get('sampling_strata')
        and len(ids)==len(set(ids)) and set(ids)==set(masks)
        and all(mask.ndim==1 and mask.dtype==torch.bool for mask in masks.values()),
        'History accounting requires exact full-population unweighted128/32 schedule')
    require(saved['episode_counts']=={identity:epochs for identity in ids}, 'Incomplete episode exposure for history accounting')
    rng=torch.Generator().manual_seed(config['initialization_seed']);expected=[];summaries=[]
    for epoch in range(1,epochs+1):
        order=[ids[index] for index in torch.randperm(len(ids),generator=rng).tolist()]
        full=zero=zero_chunks=window_history=replay_history=replay_calls=0
        for offset in range(0,len(order),config['microbatch']):
            group=[masks[identity] for identity in order[offset:offset+config['microbatch']]]
            length=max(map(len,group));pending_h=pending_s=pending_c=prefix_ticks=0;dirty=False
            for begin in range(0,length,32):
                end=min(begin+32,length);history=sum(len(mask[begin:end]) for mask in group)
                supervised=sum(int(mask[begin:end].sum()) for mask in group)
                full+=history
                if supervised:
                    if dirty:
                        replay_history+=prefix_ticks;replay_calls+=1;dirty=False
                    window_history+=history;pending_h+=history;pending_s+=supervised;pending_c+=1
                else:
                    zero+=history;zero_chunks+=1
                if pending_s and (pending_s>=128 or end==length):
                    expected.append(dict(update=len(expected)+1,epoch=epoch,history_ticks=pending_h,
                        supervised_ticks=pending_s,window_chunks=pending_c,fixed_denominator=128,group_tail=pending_s<128))
                    pending_h=pending_s=pending_c=0;dirty=True
                prefix_ticks+=history
        summaries.append(dict(epoch=epoch,chronological_history_ticks=full,optimizer_window_history_ticks=window_history,
            zero_supervision_chunk_history_ticks=zero,zero_supervision_chunks=zero_chunks,
            derived_prefix_recompute_history_ticks=replay_history,derived_prefix_recompute_calls=replay_calls))
    require(len(expected)==len(receipts) and all(all(row[key]==actual[key] for key in row)
        for row,actual in zip(expected,receipts)), 'Actual optimizer receipts differ from replayed shuffle/masks')
    require(order==saved['episode_order'] and torch.equal(rng.get_state(),saved['schedule_rng']),
        'Actual final shuffle order/RNG differs from replayed schedule')
    full=sum(row['chronological_history_ticks'] for row in summaries)
    window=sum(row['optimizer_window_history_ticks'] for row in summaries)
    zero=sum(row['zero_supervision_chunk_history_ticks'] for row in summaries)
    supervised=sum(int(mask.sum())*epochs for mask in masks.values())
    require(full==sum(len(masks[identity])*saved['episode_counts'][identity] for identity in ids)==saved['control_ticks']
        and full==window+zero and supervised==saved['supervised_ticks']==sum(row['supervised_ticks'] for row in receipts),
        'Full chronological/zero-supervision/window accounting does not reconcile')
    return dict(passed=True,epochs=summaries,chronological_history_ticks=full,optimizer_window_history_ticks=window,
        zero_supervision_chunk_history_ticks=zero,supervised_ticks=supervised,optimizer_windows=len(expected),
        prefix_recompute_scope='Derived from the source loop and replayed exact schedule, separately from all exposure counters; not a measured runtime receipt and excludes crash/restart work.',
        derived_prefix_recompute_history_ticks=sum(row['derived_prefix_recompute_history_ticks'] for row in summaries))


def load_training_masks(study, role, saved):
    directory=study/'control'/role;index_path=directory/'backend'/role/'nominal/index.json';index=read(index_path)
    require(sha256(index_path)==saved['configuration']['cache_index_sha256'], 'Training cache index differs')
    require(read(directory/'state_data/data.json')['splits']['train']==saved['train_ids'], 'Actual training IDs differ')
    masks={}
    for identity in saved['train_ids']:
        path=index_path.parent/(identity+'.pt')
        require(sha256(path)==index['episodes'][identity]['sha256'], 'Training mask cache changed')
        record=torch.load(path,map_location='cpu',weights_only=False)
        require(record['identity']==identity, 'Training mask identity changed')
        masks[identity]=record['supervision_mask'].clone()
    return masks


def check_configuration(study, role, checkpoints, registration):
    normalized = []
    for arm in ARMS:
        saved = checkpoints[arm]; config = saved['configuration']; audit = config['continuation_audit']
        expected = REACHABLE_CENTERED_OUTPUT if arm=='control' else REACHABLE_PREVIOUS_TARGET_OUTPUT
        require(config['continuation_from']['sha256']==registration['source_pairs'][role]['backend_sha256']
            and config['base_sha256']==registration['source_pairs'][role]['frontend_sha256'],
            'Continuation source or frozen frontend differs')
        require(config.get('output_contract')==expected and saved['residual_output_version']==expected['version']
            and config.get('continuation_head_reset')==MATCHED_FINAL_HEAD_RESET
            and audit.get('continuation_head_reset')==MATCHED_FINAL_HEAD_RESET
            and audit.get('reset_parameter_names')==['policy.2.weight','policy.2.bias']
            and audit.get('output_contract_before')==REACHABLE_CENTERED_OUTPUT
            and audit.get('output_contract_after')==expected
            and audit.get('output_contract_changed')==(arm=='previous')
            and audit.get('requires_training_before_deployment') is True,
            'Matched final-head reset or output contract differs')
        require(audit['optimizer_state_restored'] is False and audit['new_optimizer_updates']==0
            and audit['statistics_and_delta_buffers_unchanged'] is True
            and audit['command_normalizer_unchanged'] is True and audit['scale_adaptation'] is None
            and not config.get('fresh_recovery_weighting') and not config.get('recovery_prefix')
            and not config.get('sampling_strata'), 'Fresh optimizer, unchanged buffers or equal supervision differs')
        require(saved['updates']>0 and not saved.get('disposable',False), 'A reset head requires actual formal training')
        states = saved['optimizer']['state']
        require(bool(states) and all(int(state['step'])==saved['updates'] for state in states.values()),
            'Actual Adam step counters differ from recorded updates')
        value = json.loads(json.dumps(config).replace(str(study/arm), '<candidate>'))
        value['output_contract'] = REACHABLE_CENTERED_OUTPUT
        value['continuation_audit']['output_contract_after'] = REACHABLE_CENTERED_OUTPUT
        value['continuation_audit']['output_contract_changed'] = False
        normalized.append(value)
    require(normalized[0]==normalized[1], 'Configurations differ beyond output anchor and candidate paths')
    require(checkpoints['control']['optimizer']['param_groups']==checkpoints['previous']['optimizer']['param_groups'],
        'Optimizer hyperparameters or parameter groups differ')
    return dict(source_backend_sha256=registration['source_pairs'][role]['backend_sha256'],
        common_final_head_reset=MATCHED_FINAL_HEAD_RESET,
        configurations_match_except_anchor_and_paths=True, fresh_optimizers=True,
        actual_optimizer_step_counters_match=True,
        interpretation='Both inherit the same role-specific nominal-trained trunk and reset the final Linear; inherited trunk may favor nominal coordinates')


def audit_reconstructed_initialization(study, role, registration):
    """Recreate each frozen loader's declared start and compare every parameter."""
    from visuotactile.anchor_continue import load_residual_continuation
    from visuotactile.models.native_dp import TargetResidual
    source_path=Path(registration['source_pairs'][role]['backend'])
    require(sha256(source_path)==registration['source_pairs'][role]['backend_sha256'], 'Source rear bytes changed')
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    reference=None; counts={}
    for arm in ARMS:
        config=read(study/arm/role/'backend'/role/'training/configuration.json')
        stats_path=Path(config['residual_statistics_path'])
        require(sha256(stats_path)==config['residual_statistics_sha256'], 'Command statistics changed')
        stats=torch.load(stats_path,map_location='cpu',weights_only=False)
        model=TargetResidual(stats, config['fixed_residual_scales'],
            execution_adapter=config['execution_adapter'],loss_contract=config['loss_contract'],
            history_contract=config['history_contract'],output_contract=config['output_contract'])
        audit=load_residual_continuation(model,source,config['residual_statistics_sha256'],head_reset=MATCHED_FINAL_HEAD_RESET)
        require(audit==config['continuation_audit'], 'Reconstructed continuation audit differs from actual training record')
        state=model.state_dict()
        for name,value in state.items():
            if name in ('policy.2.weight','policy.2.bias'):
                require(bool((value==0).all()), 'Matched final Linear was not zeroed')
            else:
                require(torch.equal(value,source['model'][name]), 'Inherited parameter or buffer differs: '+name)
        if reference is not None:
            require(state.keys()==reference.keys() and all(torch.equal(value,reference[name]) for name,value in state.items()),
                'Reconstructed initial model tensors differ between comparison arms')
        reference={name:value.clone() for name,value in state.items()}
        counts[arm]=dict(state_tensors=len(state),zeroed_final_head_parameters=sum(state[name].numel() for name in ('policy.2.weight','policy.2.bias')))
        del model
    return dict(passed=True, arms=counts, source_sha256=sha256(source_path),
        scope='CPU reconstruction from pinned source/config and the recorded loader contract; actual runtime audit is also checked. No optimizer or simulation.')


def validation_population(report):
    return [(r['epoch'],r['update'],{split:{phase:metrics['ticks'] for phase,metrics in r[split].items()}
        for split in ('train','validation')}) for r in report['history']]


def check_registered_opportunities(plan, registration):
    from tools.tight_fresh_anchor_study import development_cases
    expected=development_cases(registration['session_binding']['domain'],
        registration['count_per_role'],registration['development_seed'])
    require(plan['cases']==expected, 'Autonomous opportunities/order differ from registered seed/domain/count')
    names=[row['name'] for row in expected]
    require(len(set(names))==len(names), 'Registered opportunity identities repeat')


def phase_snapshots(directory, phases):
    """Same-tick diagnostics on the actual goal active before its success reset."""
    from visuotactile.takeover_trigger import stable_windows
    path=directory/'numeric.h5'
    require(sha256(path)==phases['numeric_sha256'], 'Phase input bytes changed')
    with h5py.File(path) as stream:
        t=stream['episode/transition'];after=t['diagnostics_after'];before=t['diagnostics_before']
        pose=after['object_pose_w'][:];palm=after['palm_body_pose_w'][:];goal=before['goal_pose_w'][:]
        force=t['object_contact_force_w'][:];n=len(pose)
        require(n==phases['steps'] and n>=48, 'Complete native trajectory and settled reference required')
        before_count=before['successes'][:].reshape(-1);after_count=after['successes'][:].reshape(-1)
        require(np.array_equal(after_count[:-1],before_count[1:]), 'Subgoal timeline is discontinuous')
        stable,contact=stable_windows(pose,palm,force,float(np.median(pose[24:48,2])))
        distance=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1)
        angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
        require(np.isfinite(distance).all() and np.isfinite(angle).all(), 'Nonfinite actual root-pose diagnostics')
        near=t['native_success_diagnostic/near_goal'][:].astype(bool);event=t['native_alignment_condition'][:].astype(bool)
        keypoint=t['native_keypoints_max_dist'][:];found=np.flatnonzero(stable)
        first_goal=before_count==0
        after_hold=first_goal & (np.arange(n)>=found[0]) if len(found) else np.zeros(n,bool)
        def snapshot(index):
            if index is None:return None
            return dict(completed_tick=int(index)+1,goal_index_before=int(before_count[index]),
                subgoals_after=int(after_count[index]),goal_root_distance_m=float(distance[index]),
                root_quaternion_angle_rad=float(angle[index]),native_keypoint_max_distance_m=float(keypoint[index]),
                native_near_goal=bool(near[index]),native_subgoal_event=bool(event[index]),
                stable_window=bool(stable[index]),two_contact_fingers_both_substeps=bool((contact[index].sum(-1)>=2).all()))
        def first(mask):
            indices=np.flatnonzero(mask);return snapshot(indices[0]) if len(indices) else None
        def best(metric,mask):
            indices=np.flatnonzero(mask);return snapshot(indices[np.argmin(metric[indices])]) if len(indices) else None
        result=dict(first_native_lift=first(after['lifted_object'][:].astype(bool)),
            first_stable_hold=first(stable),first_native_near_goal=first(near),
            first_native_subgoal=first(after_count>before_count),
            first_goal_best_orientation=best(angle,first_goal),
            first_goal_after_hold_best_orientation=best(angle,after_hold),
            first_goal_after_hold_closest_root=best(distance,after_hold),final=snapshot(n-1))
    require(sha256(path)==phases['numeric_sha256'], 'Numeric record changed during phase audit')
    return result


def summarize_autonomous_cases(chosen, seconds):
    legal=[row for row in chosen if row['legal']]
    return dict(opportunities=len(chosen),legal=len(legal),invalid_initializations=len(chosen)-len(legal),
        successes=sum(r['success'] for r in legal),native_lift_cases=sum(r['native_lift'] for r in legal),
        settled_stable_cases=sum(r['settled_height_stable_grasp'] for r in legal),
        alignment_cases=sum(r['native_alignment_first_tick'] is not None for r in legal),
        near_hole_cases=sum(r['native_near_goal_first_tick'] is not None for r in legal),
        first_subgoal_cases=sum(r['native_first_subgoal_tick'] is not None for r in legal),
        steps=sum(r['steps'] for r in legal),seconds=seconds,
        control_ticks_per_wall_second=sum(r['steps'] for r in legal)/seconds if seconds else None)


def source_autonomous(registration, ordered_specs):
    """Previously completed source cohort, kept separate from the new16 trials."""
    from tools.tight_fresh_followup_session import evaluated_pairs
    binding=registration['source_evidence'];root=Path(binding['root']);directory=root/'development'
    require(sha256(directory/'plan.json')==binding['plan_sha256']
        and sha256(directory/'results.json')==binding['results_sha256'], 'Registered source evaluation changed')
    require(evaluated_pairs(root,directory,registration['source_pairs'])==binding['audit'],
        'Registered source evaluation audit changed')
    plan=read(directory/'plan.json');rows=read(directory/'results.json');cases=[]
    for role in ('A','B'):
        require([r['spec'] for r in plan['cases'] if r['role']==role]==ordered_specs,
            'Source and candidate reset opportunities differ')
    for registered in plan['cases']:
        directory_case=directory/registered['name'];phases=phase_case(directory_case)
        require(phases['authority']['expert_ticks']==0, 'Source reference used expert authority')
        cases.append(dict(arm='source',role=registered['role'],**phases,
            phase_snapshots=phase_snapshots(directory_case,phases)))
    return dict(cases=cases,totals={role:summarize_autonomous_cases([r for r in cases if r['role']==role],
        sum(r['timing']['total_seconds'] for r in rows if r['role']==role)) for role in ('A','B')},
        root=str(root),source_pairs=registration['source_pairs'],
        plan_sha256=binding['plan_sha256'],results_sha256=binding['results_sha256'],
        scope='Previously completed source evaluation; same ordered reset specifications, separate timing/cohort. Not part of the new interleaved comparison and not exact hidden-state pairs.')


def autonomous(study, registration):
    directory=study/'development';plan=read(directory/'plan.json')
    finished=read(directory/'result.json')
    require(finished['software_complete'] and not finished.get('worker_error')
        and read(directory/'run_started.json')['plan_sha256']==sha256(directory/'plan.json'),
        'Autonomous cohort incomplete or launch plan differs')
    rows=read(directory/'results.json'); registered=plan['cases']
    check_registered_opportunities(plan,registration)
    require(finished['cases']==len(rows)==len(registered) and len({r['case'] for r in rows})==len(rows)
        and {r['case'] for r in rows}=={c['name'] for c in registered}, 'Autonomous opportunity denominator differs')
    pairs={arm+'_'+role:completed_pair(study/arm,role) for arm in ARMS for role in ('A','B')}
    require(plan['kind']=='autonomous' and plan.get('test_used') is False and plan['pairs']==pairs
        and plan.get('study')==registration['version'], 'Interleaved autonomous plan identity differs')
    cohorts={key:[] for key in pairs};cases=[]
    for expected in registered:
        key=expected['role'];arm=expected['study_arm'];role=expected['student_role'];name=expected['name']
        require(key==arm+'_'+role and key in pairs, 'Unexpected role/arm identity')
        cohorts[key].append(expected['spec'])
        row=next(r for r in rows if r['case']==name)
        result=read(directory/name/'result.json');started=read(directory/name/'started.json')
        require(row['role']==key and row['result']==result and result['software_complete']
            and not result.get('worker_error') and not result.get('censored') and result.get('dagger') is None
            and started['spec']==expected['spec'] and started['checkpoint']==pairs[key]['backend'],
            'Recorded autonomous case identity differs')
        require(all(result[k]==v for k,v in expected['spec'].items()), 'Actual autonomous reset specification differs')
        if not result.get('legal'):
            cases.append(dict(case=name,arm=arm,role=role,legal=False,success=False,
                result_sha256=sha256(directory/name/'result.json')))
            continue
        require(result['checkpoint_sha256']==pairs[key]['backend_sha256'] and not result['expert_shadow_computed']
            and result['residual_version']==pairs[key]['residual_version']
            and result['residual_ticks']==result['steps']
            and result['action_control_mode']=='absolute_joint_position_target_rad',
            'Autonomous controller or expert computation differs')
        for component in ('dino_rgb','learned_visual_and_history','residual_forward_compose'):
            require(result['native_inference']['summary'][component]['wall_s']['count']==result['steps'],
                'Missing native-frame visual/history/backend computation')
        phases=phase_case(directory/name)
        require(phases['steps']==result['steps'] and phases['authority']['expert_ticks']==0
            and result['simulation_step_hz']==60 and result['physics_hz']==120 and result['plan_hz']==5,
            'Autonomous authority or native rates differ')
        cases.append(dict(arm=arm,role=role,**phases,phase_snapshots=phase_snapshots(directory/name,phases)))
    reference=next(iter(cohorts.values()))
    require(len(reference)==registration['count_per_role'] and all(v==reference for v in cohorts.values()),
        'Four cells do not use the same ordered registered opportunities')
    totals={}
    for arm in ARMS:
        totals[arm]={}
        for role in ('A','B'):
            chosen=[r for r in cases if r['arm']==arm and r['role']==role]
            seconds=sum(r['timing']['total_seconds'] for r in rows if r['role']==arm+'_'+role)
            totals[arm][role]=summarize_autonomous_cases(chosen,seconds)
    return dict(cases=cases,totals=totals,plan_sha256=sha256(directory/'plan.json'),
        results_sha256=sha256(directory/'results.json'),interleaved=True,
        ordered_specs=reference,
        exact_hidden_state_replay_claim=False)


def run(study,output,onset_suffix='_anchor_onset_cpu'):
    if output.exists():raise FileExistsError('Preserve earlier analysis')
    torch.set_num_threads(2)
    registration=read(study/'registration.json')
    require(registration['version']=='matched_rear_output_anchor_study_v1'
        and registration['arms']==list(ARMS) and registration['loss_weight']==1.
        and registration['matched_head_reset']==MATCHED_FINAL_HEAD_RESET, 'Unknown anchor study registration')
    require(read(study/'finished.json')['software_complete'], 'Study incomplete')
    for path,digest in registration['protected'].items():
        require(sha256(path)==digest, 'Protected frozen dependency changed: '+path)
    roles={}
    for role in ('A','B'):
        reports={};losses={};checkpoints={}
        for arm in ARMS:
            reports[arm],losses[arm],checkpoints[arm]=rear_report(study/arm,role,registration['source_pairs'][role]['frontend_sha256'])
        configuration=check_configuration(study,role,checkpoints,registration)
        initial=audit_reconstructed_initialization(study,role,registration)
        a,b=checkpoints['control'],checkpoints['previous']
        for key in ('train_ids','validation_ids','episode_counts','episode_order'):
            require(a[key]==b[key], 'Actual schedule differs: '+key)
        require(torch.equal(a['schedule_rng'],b['schedule_rng']), 'Actual shuffle RNG differs')
        for key in ('data_sha256','cache_index_sha256'):
            require(reports['control'][key]==reports['previous'][key], 'Comparison inputs differ: '+key)
        exposure=check_equal_exposure(losses['control'],losses['previous'])
        masks=load_training_masks(study,role,checkpoints['control'])
        accounting={arm:audit_history_accounting(masks,checkpoints[arm],losses[arm]) for arm in ARMS}
        require(accounting['control']==accounting['previous'], 'Matched chronological histories differ')
        for arm in ARMS:
            report=reports[arm];saved=checkpoints[arm]
            require(all(n==registration['epochs'] for n in saved['episode_counts'].values())
                and report['finished']['supervised_ticks']==exposure['supervised_ticks']
                and report['finished']['control_ticks']==accounting[arm]['chronological_history_ticks']
                and exposure['history_ticks']==accounting[arm]['optimizer_window_history_ticks'], 'Actual cumulative exposure differs')
            best=torch.load(study/arm/role/'backend'/role/'training/best.pt',map_location='cpu',weights_only=False)
            require(best['updates']>0 and best['updates']==report['selected']['update'], 'Selected reset head has not actually trained')
            report['selected_actual_episode_counts']=best['episode_counts']
            report['selected_optimizer_updates']=best['updates']
            report['selected_supervised_ticks']=best['supervised_ticks']
        require(validation_population(reports['control'])==validation_population(reports['previous']), 'Validation schedule or denominators differ')
        roles[role]=dict(arms=reports,matched_configuration=configuration,reconstructed_initialization=initial,
            matched_actual_exposure=exposure,matched_history_accounting=accounting['control'])
    outcomes=autonomous(study,registration)
    reference_outcomes=source_autonomous(registration,outcomes['ordered_specs'])
    paths={arm:study/(arm+onset_suffix+'.json') for arm in ('source',*ARMS)}
    onset={arm:read(path) for arm,path in paths.items()}
    for role in ('A','B'):
        source_data=read(Path(registration['source_root'])/role/'state_data/data.json')
        expected_cohorts=correction_cohorts(source_data,registration['version'])
        for arm in ('source',*ARMS):
            expected=registration['source_pairs'][role]['backend_sha256'] if arm=='source' else roles[role]['arms'][arm]['selected']['checkpoint_sha256']
            current=check_onset_binding(onset[arm],onset['source'],role,arm,sha256(study/'registration.json'),expected)
            require(onset[arm].get('command_motion_contract')==onset['source'].get('command_motion_contract')
                and onset[arm].get('command_motion_contract',{}).get('version')=='actual_pre_action_target_delta_v1',
                'Missing or different motion/hold-baseline diagnostic')
            require(current.get('fitting_population')==expected_cohorts
                and sorted(r['identity'] for r in current['cases'])==expected_cohorts['all_corrections']
                and sorted(r['identity'] for r in current['cases'] if r['current_correction'])==expected_cohorts['current_corrections'],
                'Recovery fitting records differ from actual training population')
            for cohort,identities in expected_cohorts.items():
                chosen=[r for r in current['cases'] if r['identity'] in identities]
                summary=current['summaries'][cohort]
                require(summary['episodes']==len(chosen)==len(identities), 'Recovery fitting population differs')
                require(summary['bins']=={name:aggregate([r['bins'][name] for r in chosen]) for name in ('first12','next12','remaining')}
                    and summary['command_motion_bins']=={name:aggregate_command_motion([r['command_motion_bins'][name] for r in chosen])
                        for name in ('first12','next12','remaining')}, 'Recovery physical errors or command motion aggregation differs')
        roles[role]['onset_fitting']={arm:onset[arm]['roles'][role]['summaries'] for arm in onset}
    require(not torch.cuda.is_initialized(), 'CPU report initialized CUDA')
    atomic_json(output,dict(complete=True,version='matched_anchor_study_report_v1',
        report_schema_revision=2,stage_semantics=STAGE_SEMANTICS,
        arm_semantics=dict(source=dict(label='Source ('+Path(registration['source_evidence']['root']).name+')',
            root=registration['source_evidence']['root'],meaning='Previously selected complete pair from the named source round, before this head reset/retraining.'),
            control=dict(label='Nominal anchor',meaning='New matched-head-reset/retrained control around the bounded DP target.'),
            previous=dict(label='Previous-target anchor',meaning='New matched-head-reset/retrained candidate around the actual pre-action actuator target; previous names a target coordinate, not an older model.')),
        registration_sha256=sha256(study/'registration.json'),collector_source_commit=registration['source_commit'],
        report_source_sha256=sha256(__file__),roles=roles,autonomous=outcomes,source_autonomous=reference_outcomes,
        onset_reports_sha256={arm:sha256(path) for arm,path in paths.items()},device='cpu',test_used=False,
        limits='Matched source trunk/buffers and zeroed final Linear, frozen DP/data/cache, equal actual unweighted updates. '
        'The inherited trunk was trained with nominal coordinates. Fixed validation6 selects trained checkpoints. '
        'Motion and error metrics use the registered learned recovery histories and do not replace autonomous behavior. '
        'Interleaved autonomous cells share registered opportunities, not exact hidden simulator state. No automatic promotion.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--onset-suffix',default='_anchor_onset_cpu')
    args=parser.parse_args()
    run(args.study.resolve(),args.output.resolve(),args.onset_suffix)
