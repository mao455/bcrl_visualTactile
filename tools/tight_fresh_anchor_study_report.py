"""CPU evidence binding for the matched output-anchor study; never promotes models."""
import argparse
import json
from pathlib import Path

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
        interpretation='Actual matched optimizer windows; the unweighted trainer does not label recovery-onset counts in its window receipts')


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


def autonomous(study, registration):
    directory=study/'development';plan=read(directory/'plan.json')
    require(read(directory/'result.json')['software_complete'], 'Autonomous cohort incomplete')
    rows=read(directory/'results.json'); registered=plan['cases']
    require(len(rows)==len(registered) and len({r['case'] for r in rows})==len(rows)
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
            and started['spec']==expected['spec'] and started['checkpoint']==pairs[key]['backend'],
            'Recorded autonomous case identity differs')
        require(all(result[k]==v for k,v in expected['spec'].items()), 'Actual autonomous reset specification differs')
        if not result.get('legal'):
            cases.append(dict(case=name,arm=arm,role=role,legal=False,success=False,
                result_sha256=sha256(directory/name/'result.json')))
            continue
        require(result['checkpoint_sha256']==pairs[key]['backend_sha256'] and not result['expert_shadow_computed']
            and result['residual_version']==pairs[key]['residual_version'], 'Autonomous controller or expert computation differs')
        for component in ('dino_rgb','learned_visual_and_history','residual_forward_compose'):
            require(result['native_inference']['summary'][component]['wall_s']['count']==result['steps'],
                'Missing native-frame visual/history/backend computation')
        phases=phase_case(directory/name)
        require(phases['steps']==result['steps'] and phases['authority']['expert_ticks']==0
            and result['simulation_step_hz']==60 and result['physics_hz']==120 and result['plan_hz']==5,
            'Autonomous authority or native rates differ')
        cases.append(dict(arm=arm,role=role,**phases))
    reference=next(iter(cohorts.values()))
    require(len(reference)==registration['count_per_role'] and all(v==reference for v in cohorts.values()),
        'Four cells do not use the same ordered registered opportunities')
    totals={}
    for arm in ARMS:
        totals[arm]={}
        for role in ('A','B'):
            chosen=[r for r in cases if r['arm']==arm and r['role']==role]
            legal=[r for r in chosen if r['legal']]
            seconds=sum(r['timing']['total_seconds'] for r in rows if r['role']==arm+'_'+role)
            totals[arm][role]=dict(opportunities=len(chosen),legal=len(legal),invalid_initializations=len(chosen)-len(legal),
                successes=sum(r['success'] for r in legal),native_lift_cases=sum(r['native_lift'] for r in legal),
                settled_stable_cases=sum(r['settled_height_stable_grasp'] for r in legal),
                alignment_cases=sum(r['native_alignment_first_tick'] is not None for r in legal),
                near_hole_cases=sum(r['native_near_goal_first_tick'] is not None for r in legal),
                first_subgoal_cases=sum(r['native_first_subgoal_tick'] is not None for r in legal),
                steps=sum(r['steps'] for r in legal),seconds=seconds,
                control_ticks_per_wall_second=sum(r['steps'] for r in legal)/seconds if seconds else None)
    return dict(cases=cases,totals=totals,plan_sha256=sha256(directory/'plan.json'),
        results_sha256=sha256(directory/'results.json'),interleaved=True,
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
        for arm in ARMS:
            report=reports[arm];saved=checkpoints[arm]
            require(all(n==registration['epochs'] for n in saved['episode_counts'].values())
                and report['finished']['supervised_ticks']==exposure['supervised_ticks']
                and report['finished']['control_ticks']==exposure['history_ticks'], 'Actual cumulative exposure differs')
            best=torch.load(study/arm/role/'backend'/role/'training/best.pt',map_location='cpu',weights_only=False)
            require(best['updates']>0 and best['updates']==report['selected']['update'], 'Selected reset head has not actually trained')
            report['selected_actual_episode_counts']=best['episode_counts']
            report['selected_optimizer_updates']=best['updates']
            report['selected_supervised_ticks']=best['supervised_ticks']
        require(validation_population(reports['control'])==validation_population(reports['previous']), 'Validation schedule or denominators differ')
        roles[role]=dict(arms=reports,matched_configuration=configuration,reconstructed_initialization=initial,
            matched_actual_exposure=exposure)
    outcomes=autonomous(study,registration)
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
        registration_sha256=sha256(study/'registration.json'),collector_source_commit=registration['source_commit'],
        report_source_sha256=sha256(__file__),roles=roles,autonomous=outcomes,
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
