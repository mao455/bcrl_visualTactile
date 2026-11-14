"""Read-only four-cell evidence report, run outside the frozen execution tree."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

import torch

from tools.tight_fresh_anchor_study_report import (STAGE_SEMANTICS, audit_history_accounting,
    check_equal_exposure, phase_snapshots, source_autonomous, summarize_autonomous_cases,
    validation_population)
from tools.tight_fresh_backend_study_report import rear_report
from tools.tight_fresh_learning_report import exposure_rows, learning_sources, role_report
from tools.tight_fresh_phases import case as phase_case
from tools.tight_fresh_rollouts import completed_pair
from tools.tight_fresh_visual_aux_study import ARMS, ROUTES, VERSION as STUDY_VERSION, development_cases
from visuotactile.compact import sha256
from visuotactile.position_control import visual_auxiliary_image_contract
from visuotactile.training_resources import atomic_json
from visuotactile.visual_aux_training import prepare_pose_contexts, validate_exposure

VERSION = 'matched_visual_pose_aux_report_v1'
ACTION_WEIGHTS = dict(position=.3342396121814294, rotation=.029946394754032265)


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-7)


def check_metric_population(action, pose, ids, labels):
    require([r['identity'] for r in action['episodes']] == ids
        and [r['identity'] for r in pose['episodes']] == ids,
        'Fixed action/pose metric episode population differs')
    require(all(finite(action[k]) and action[k] >= 0 for k in ('selection', 'arm_rmse_rad', 'hand_rmse_rad')),
        'Nonfinite action metric')
    for row in pose['episodes']:
        expected = labels[row['identity']]
        require(row['valid_ticks'] == expected['valid_perception_ticks']
            and row['invalid_perception_ticks'] == expected['invalid_perception_ticks'],
            'Pose metric valid/invalid population differs')
        for key in ('position_vector_rmse_mm', 'orientation6_mse'):
            require(row[key] is None if not row['valid_ticks'] else finite(row[key]) and row[key] >= 0,
                'Pose metric finite/missing semantics differ')
        require(row['valid_orientation_predictions'] + row['degenerate_orientation_predictions'] == row['valid_ticks'],
            'Degenerate rotation predictions were omitted')
        require(row['orientation_mean_degrees'] is None if not row['valid_orientation_predictions'] else
            finite(row['orientation_mean_degrees']) and 0<=row['orientation_mean_degrees']<=180.00001,
            'Orientation metric finite/missing semantics differ')
    require(pose['valid_ticks'] == sum(r['valid_ticks'] for r in pose['episodes'])
        and pose['invalid_perception_ticks'] == sum(r['invalid_perception_ticks'] for r in pose['episodes'])
        and pose['degenerate_orientation_predictions'] == sum(r['degenerate_orientation_predictions'] for r in pose['episodes']),
        'Pooled pose metric population differs')
    oriented=sum(r['valid_orientation_predictions'] for r in pose['episodes'])
    require(pose['orientation_mean_degrees'] is None if not oriented else close(pose['orientation_mean_degrees'],
        sum(r['orientation_mean_degrees']*r['valid_orientation_predictions'] for r in pose['episodes']
            if r['valid_orientation_predictions'])/oriented), 'Pooled orientation metric denominator differs')
    valid = pose['valid_ticks']
    if valid:
        require(close(pose['position_vector_rmse_mm'], math.sqrt(sum(
            r['position_vector_rmse_mm']**2*r['valid_ticks'] for r in pose['episodes'] if r['valid_ticks'])/valid))
            and close(pose['orientation6_mse'], sum(r['orientation6_mse']*r['valid_ticks']
                for r in pose['episodes'] if r['valid_ticks'])/valid), 'Pooled pose errors differ from actual tick denominators')


def objective_history(learning, config, episodes_per_epoch):
    """Keep action and perception components comparable; never compare total loss."""
    weights = dict(ACTION_WEIGHTS)
    if config.get('hand_geometry'):
        weights['hand_geometry'] = config['hand_geometry']['weight']
    weight = config['visual_auxiliary']['loss_weight']; groups = {}
    for row in learning:
        parts = row['parts']; expected = {'base', *weights, 'auxiliary_position', 'auxiliary_orientation6', 'auxiliary_weighted'}
        require(set(parts) == expected and all(finite(v) and v >= 0 for v in parts.values()),
            'Actual finite action/pose component receipt required')
        action = parts['base'] + sum(weights[k]*parts[k] for k in weights)
        require(close(parts['auxiliary_weighted'], weight*(parts['auxiliary_position']+parts['auxiliary_orientation6']))
            and close(row['loss'], action+parts['auxiliary_weighted']), 'Actual total does not reconcile with action plus pose loss')
        groups.setdefault(row['epoch'], []).append(dict(action_objective=action, **parts,
            total_within_cell_only=row['loss']))
    result = []
    for epoch, rows in sorted(groups.items()):
        require(len(rows) == episodes_per_epoch, 'Incomplete equal-episode epoch loss')
        result.append(dict(epoch=epoch, optimizer_episodes=len(rows),
            **{key:sum(r[key] for r in rows)/len(rows) for key in rows[0]}))
    return dict(epochs=result, action_weights=weights, auxiliary_weight=weight,
        aggregation='Unweighted mean of actual full-episode optimizer losses; separate from fixed-validation DDIM metrics.',
        total_loss_comparable_across_routes=False,
        interpretation='Use action components/action metrics and pose components separately. Auxiliary learning alone does not prove autonomous improvement or that the source lacked pose information.')


def check_frontend_receipts(best, last, history, learning, finished, config, data, labels, expected_epochs):
    ids = data['splits']['train']; n = len(ids)
    require(n > 0 and len(set(ids)) == n and expected_epochs > 0, 'Explicit unique training population required')
    require(finished.get('complete') is True and last['epoch'] == finished['epochs'] == expected_epochs
        and last['updates'] == finished['updates'] == len(learning) == expected_epochs*n,
        'Frontend completion differs from actual update count')
    require(last['history'] == history and [r['epoch'] for r in history] == list(range(1,expected_epochs+1)),
        'Durable checkpoint and unique validation history differ')
    selected = min(history, key=lambda r:r['validation']['selection'])
    require(best['epoch'] == selected['epoch'] and best['updates'] == selected['updates']
        and best['history'] == history[:best['epoch']], 'Selected frontend differs from fixed action validation minimum')
    for saved in (best, last):
        require(saved['configuration'] == config and saved['train_ids'] == ids
            and saved['validation_ids'] == data['splits']['validation'], 'Checkpoint configuration/population differs')
        validate_exposure(saved['auxiliary_exposure'], labels, ids, data['normalization_train_ids'],
            completed_epochs=saved['epoch'], updates=saved['updates'])
        require(bool(saved['optimizer']['state']) and all(int(v['step']) == saved['updates']
            for v in saved['optimizer']['state'].values()), 'Actual frontend Adam step counters differ')
    require(last['auxiliary_exposure'] == finished['auxiliary_exposure'], 'Completed auxiliary exposure differs')
    for epoch in range(1, expected_epochs+1):
        rows = learning[(epoch-1)*n:epoch*n]
        require(Counter(r['episode'] for r in rows) == Counter(ids)
            and all(r['epoch'] == epoch and r['episode_index'] == i+1 for i,r in enumerate(rows)),
            'Actual epoch shuffle duplicates/omits episodes')
        for row in rows:
            proof = labels[row['episode']]
            require(row['visual_frames'] == proof['ticks']
                and row['auxiliary_exposure'] == proof['perception_exposure'], 'Actual visual/phase learning exposure differs')
        require(history[epoch-1]['updates'] == epoch*n, 'Validation update counter differs')
        for key in ('anchors_seen', 'target_ticks'):
            require(history[epoch-1][key] == rows[-1][key], 'Validation and learning action exposure differ')
    for key in ('anchors_seen', 'target_ticks'):
        require(last[key] == finished[key] == learning[-1][key], 'Durable action exposure differs')
    corrections, _, groups = learning_sources(data)
    counts, by_source = exposure_rows(learning, set(ids), corrections, source_groups=groups)
    return dict(actual_episode_counts=dict(counts), actual_action_exposure_by_source=by_source,
        selected_auxiliary_exposure=best['auxiliary_exposure'], last_auxiliary_exposure=last['auxiliary_exposure'])


def recovery_evidence(training):
    result = []
    for path in sorted(training.glob('recovery.*.json')):
        row = read(path); abandoned = Path(row['abandoned_path'])
        require(sha256(abandoned) == row['abandoned_sha256']
            and row['discarded_updates_in_restored_model'] == 0, 'Discarded-work recovery evidence differs')
        result.append(dict(receipt_sha256=sha256(path), **row))
    return dict(recoveries=result, excluded_from_model_exposure=True,
        known_discarded_optimizer_updates=sum(len(r['known_discarded_update_ids']) for r in result),
        scope='Preserved real crash-tail work, separate from best/last checkpoint learning doses. Partial rows may leave the exact attempted count unknown.')


def action_change(before,after):
    result={}
    for split in ('train','validation'):
        result[split]={}
        for key in ('selection','arm_rmse_rad','hand_rmse_rad'):
            a,b=before[split][key],after[split][key]
            require(finite(a) and finite(b) and a>=0 and b>=0,'Finite action before/after required')
            result[split][key]=dict(before=a,after=b,absolute_change=b-a,relative_change=b/a-1 if a else None)
    return result


def frontend_cell(study, arm, role, registration, data, labels):
    directory = study/arm/role; training = directory/'training'
    config = read(training/'configuration.json'); spec = config['visual_auxiliary']; route = ROUTES[arm]
    require(config['version'] == 'fresh30_continuous_episode_aux_v1' and spec['route'] == route
        and spec['head_seed'] == registration['head_seed'] and spec['loss_weight'] == registration['loss_weight']
        and spec['calibration'] == registration['calibration']
        and config['source_sha256'] == registration['source_sha256']
        and config['learning_rate'] == registration['frontend_learning_rate'], 'Frontend study configuration differs')
    source = registration['source_pairs'][role]
    require(config['continuation']['sha256'] == source['frontend_sha256']
        and config['continuation']['previous_backend_sha256'] == source['backend_sha256']
        and config['data_sha256'] == sha256(directory/'state_data/data.json')
        and config['statistics_sha256'] == sha256(directory/'state_data/statistics.pt')
        and config['cache_index_sha256'] == sha256(directory/'cache_index.json'), 'Frontend source/data/statistics binding differs')
    label_path = training/'visual_auxiliary_labels.json'; label_proof = read(label_path)
    require(config['auxiliary_label_report_sha256'] == sha256(label_path)
        and label_proof['spec'] == spec and label_proof['episodes'] == labels,
        'Actual raw label/camera/authority report differs from training')
    best = torch.load(training/'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(training/'last.pt', map_location='cpu', weights_only=False)
    expected = visual_auxiliary_image_contract('q_reached' if role == 'A' else 'u_expert', route)
    require(best['contract'] == last['contract'] == expected, 'Frontend route or supervision contract differs')
    history = jsonl(training/'validation.jsonl'); learning = jsonl(training/'learning.jsonl'); finished = read(training/'finished.json')
    require(finished['checkpoint_sha256'] == sha256(training/'best.pt'), 'Selected frontend hash differs')
    exposure = check_frontend_receipts(best,last,history,learning,finished,config,data,labels,registration['frontend_epochs'])
    initial = read(training/'initial_metrics.json')
    for entry in [initial, *history]:
        for split, ids in [('train',data['splits']['train'][:2]), ('validation',data['splits']['validation'])]:
            check_metric_population(entry[split],entry['visual_auxiliary'][split],ids,labels)
    selected = history[best['epoch']-1]
    metrics = lambda row:dict(action={k:row[k] for k in ('train','validation')},
        pose={k:row['visual_auxiliary'][k] for k in ('train','validation')})
    report = dict(route=route, source=source, data_sha256=config['data_sha256'], statistics_sha256=config['statistics_sha256'],
        configuration_sha256=sha256(training/'configuration.json'), raw_label_report_sha256=sha256(label_path),
        learning_receipts_sha256=sha256(training/'learning.jsonl'), validation_receipts_sha256=sha256(training/'validation.jsonl'),
        selected=dict(checkpoint_sha256=sha256(training/'best.pt'),epoch=best['epoch'],updates=best['updates'],metrics=metrics(selected)),
        last=dict(checkpoint_sha256=sha256(training/'last.pt'),epoch=last['epoch'],updates=last['updates'],metrics=metrics(history[-1])),
        source_initial=dict(checkpoint_sha256=source['frontend_sha256'],metrics=metrics(initial),
            pose_scope='Same selected v6 source with the common untrained random auxiliary head. Readout error is not a test of whether the original policy contained pose information.'),
        metric_history=[dict(epoch=row['epoch'],updates=row['updates'],**metrics(row)) for row in history],
        action_change_from_source=dict(selected=action_change(initial,selected),last=action_change(initial,history[-1])),
        actual_training=objective_history(learning,config,len(data['splits']['train'])), exposure=exposure,
        recovery=recovery_evidence(training), finished=finished,
        metric_cohorts=dict(train=data['splits']['train'][:2],validation=data['splits']['validation']),
        training_metric_scope='The train action/pose metric uses fixed train2, not the full aggregate84. Actual_training uses all recorded full-episode updates.',
        selection='Minimum fixed validation action selection; pose metrics do not select best.pt')
    return report, learning, last


def matched_frontends(study, role, cells, receipts, saved):
    a,b = [saved[arm] for arm in ARMS]
    keys = ('epoch','episode','episode_index','updates','visual_frames','anchors_seen','target_ticks','auxiliary_exposure')
    require(len(receipts['control']) == len(receipts['auxiliary']) and all(all(x[k] == y[k] for k in keys)
        for x,y in zip(receipts['control'],receipts['auxiliary'])), 'Actual frontend learning sequence/exposure differs')
    for key in ('schedule_rng','noise_rng'):
        require(torch.equal(a[key], b[key]), 'Actual frontend shuffle/diffusion RNG differs')
    require(a['optimizer']['param_groups'] == b['optimizer']['param_groups'], 'Frontend Adam hyperparameters differ')
    normalized = []
    for arm in ARMS:
        value = copy.deepcopy(saved[arm]['configuration'])
        value['visual_auxiliary']['route'] = '<route>'
        value['visual_auxiliary'].pop('spec_sha256')
        value.pop('auxiliary_label_report_sha256')
        normalized.append(json.dumps(value,sort_keys=True).replace(str(study/arm),'<arm>'))
    require(normalized[0] == normalized[1], 'Frontend configurations differ beyond declared gradient route')
    require(cells['control']['source_initial']['metrics'] == cells['auxiliary']['source_initial']['metrics'],
        'Same-source initial action/pose metrics differ')
    return dict(passed=True,matched_updates=len(receipts['control']),matched_episode_order=True,
        matched_action_and_perception_exposure=True,matched_shuffle_and_diffusion_rng=True,
        configuration_difference='Only detached versus shared-vision pose gradient route and identity hashes')


def backend_cell(study, arm, role, registration, frontend):
    root=study/arm; directory=root/role/'backend'/role; training=directory/'training'
    report, losses, last=rear_report(root,role,frontend['selected']['checkpoint_sha256'])
    best=torch.load(training/'best.pt',map_location='cpu',weights_only=False)
    cfg=last['configuration']; epochs=registration['backend_epochs']; source=registration['source_pairs'][role]
    selected_frontend=root/role/'training/best.pt'; selected_hash=frontend['selected']['checkpoint_sha256']
    require(cfg['base_sha256']==cfg['bases'][role]['sha256']==selected_hash
        and Path(cfg['base_checkpoint']).resolve()==selected_frontend.resolve()
        and Path(cfg['bases'][role]['checkpoint']).resolve()==selected_frontend.resolve()
        and sha256(selected_frontend)==selected_hash, 'Backend declared frontend path/hash differs before comparison normalization')
    from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
    require(cfg['source_sha256']==registration['source_sha256'] and cfg['output_contract']==REACHABLE_CENTERED_OUTPUT
        and cfg['continuation_from']['sha256']==source['backend_sha256']
        and cfg['maximum_control_tick_epochs']==cfg['minimum_control_tick_epochs']==epochs
        and all(count==epochs for count in last['episode_counts'].values()), 'Backend source/budget/output differs')
    require(best['updates']==report['selected']['update'] and best['configuration']==cfg
        and last['contract']==best['contract']==visual_auxiliary_image_contract('q_reached' if role=='A' else 'u_expert',ROUTES[arm]),
        'Selected/last complete pair identity differs')
    index=read(directory/'nominal/index.json')
    require(index['base_sha256']==frontend['selected']['checkpoint_sha256']
        and cfg['cache_index_sha256']==sha256(directory/'nominal/index.json'), 'Learned backend plans/features were not refreshed for selected DP')
    ordinary=role_report(root,role)
    report.update(last_checkpoint_sha256=sha256(training/'last.pt'),selected_checkpoint_sha256=sha256(training/'best.pt'),
        selected_actual_episode_counts=best['episode_counts'],selected_supervised_ticks=best['supervised_ticks'],
        actual_exposure_by_source=ordinary['backend_actual_exposure_by_source'],
        actual_exposure_by_correction_age=ordinary['backend_actual_exposure_by_correction_age'],
        cache=ordinary['backend_cache'],offline=ordinary['backend_offline'],actual_training=backend_loss_history(losses))
    return report, losses, last


def backend_loss_history(rows):
    grouped={}
    for row in rows:
        parts=row['loss_components']
        require(set(parts)=={'exec_fit','fy_fit','label_roundoff_max_rad','regularizer'}
            and all(finite(v) and v>=0 for v in parts.values())
            and row['fixed_denominator']==128 and row['loss_weight_mass']==row['supervised_ticks'],
            'Expected unweighted128-window backend loss components')
        grouped.setdefault(row['epoch'],[]).append(row)
    result=[]
    for epoch,items in sorted(grouped.items()):
        ticks=sum(r['supervised_ticks'] for r in items)
        require(ticks>0,'Positive backend learning population required')
        result.append(dict(epoch=epoch,updates=len(items),supervised_ticks=ticks,
            optimizer_objective_mean=sum(r['loss'] for r in items)/len(items),
            legal_tick_components={k:128*sum(r['loss_components'][k] for r in items)/ticks
                for k in ('exec_fit','fy_fit','regularizer')},
            label_roundoff_max_rad=max(r['loss_components']['label_roundoff_max_rad'] for r in items)))
    return dict(epochs=result,scope='Actual optimizer receipts. Component sums multiplied by fixed128 then divided by legal ticks; not fixed train2/validation metrics.')


def cached_training_masks(study,arm,role,saved,numeric):
    path=study/arm/role/'backend'/role/'nominal/index.json';index=read(path)
    require(sha256(path)==saved['configuration']['cache_index_sha256'],'Actual backend cache index changed')
    masks={}
    for identity in saved['train_ids']+saved['validation_ids']:
        shard=path.parent/(identity+'.pt')
        require(sha256(shard)==index['episodes'][identity]['sha256'],'Actual learned plan/feature shard changed')
        record=torch.load(shard,map_location='cpu',weights_only=False);raw=numeric[identity]
        expected=raw.get('supervision_mask',torch.ones(len(raw['time']),dtype=torch.bool))
        require(record['identity']==identity and torch.equal(record['time'],raw['time'])
            and torch.equal(record['supervision_mask'],expected), 'Cached action authority/clock differs from actual recorded source')
        if 'control_segment' in raw:
            require(torch.equal(record['control_segment'],raw['control_segment']),'Cached authority segment differs')
        if identity in saved['train_ids']:masks[identity]=record['supervision_mask'].clone()
    return masks


def matched_backends(study,role,reports,losses,saved,masks):
    exposure=check_equal_exposure(losses['control'],losses['auxiliary']);normalized=[]
    for arm in ARMS:
        checkpoint=saved[arm];cfg=copy.deepcopy(checkpoint['configuration'])
        cfg['base_sha256']='<selected_frontend>';cfg['bases'][role]['sha256']='<selected_frontend>'
        cfg['cache_index_sha256']='<refreshed_learned_cache>'
        normalized.append(json.dumps(cfg,sort_keys=True).replace(str(study/arm),'<arm>'))
    require(normalized[0]==normalized[1], 'Backend configuration differs beyond updated frontend/cache identities')
    for key in ('train_ids','validation_ids','episode_counts','episode_order'):
        require(saved['control'][key]==saved['auxiliary'][key], 'Backend actual schedule/population differs: '+key)
    require(torch.equal(saved['control']['schedule_rng'],saved['auxiliary']['schedule_rng'])
        and saved['control']['optimizer']['param_groups']==saved['auxiliary']['optimizer']['param_groups'],
        'Backend shuffle/Adam settings differ')
    require(validation_population(reports['control'])==validation_population(reports['auxiliary']),
        'Backend fixed metric tick populations differ')
    require(masks['control'].keys()==masks['auxiliary'].keys()
        and all(torch.equal(mask,masks['auxiliary'][key]) for key,mask in masks['control'].items()),
        'Actual regenerated cache supervision masks differ')
    accounting={arm:audit_history_accounting(masks[arm],saved[arm],losses[arm]) for arm in ARMS}
    require(accounting['control']==accounting['auxiliary'], 'Actual chronological backend history differs')
    return dict(passed=True,optimizer_exposure=exposure,history_accounting=accounting['control'])


def timing_report(rows):
    """Aggregate only sums/counts, keeping nested timers and startup separate."""
    total=sum(r['timing']['total_seconds'] for r in rows)
    wait=sum(r['timing']['resource_wait_seconds'] for r in rows)
    require(finite(total) and total>0 and finite(wait) and 0<=wait<=total, 'Invalid actual rollout timing')
    steps=sum(r['result'].get('steps',0) for r in rows if r['result'].get('legal'))
    nested={};wall={};inference={}
    for row in rows:
        require(finite(row['timing']['total_seconds']) and row['timing']['total_seconds']>0
            and finite(row['timing']['resource_wait_seconds'])
            and 0<=row['timing']['resource_wait_seconds']<=row['timing']['total_seconds'], 'Invalid individual rollout timing')
        for key,value in row['timing'].get('nested_environment_components',{}).items():
            require(finite(value) and value>=0,'Invalid environment timing');nested[key]=nested.get(key,0.)+value
        for key,value in row['result'].get('wall_profile',{}).items():
            if key.endswith('_s'):
                require(finite(value) and value>=0,'Invalid wall timing');wall[key]=wall.get(key,0.)+value
        for name,scopes in row['result'].get('native_inference',{}).get('summary',{}).items():
            dest=inference.setdefault(name,{})
            for scope,value in scopes.items():
                require(type(value['count']) is int and value['count']>=0 and finite(value['mean']) and value['mean']>=0,
                    'Invalid component timing population')
                part=dest.setdefault(scope,dict(count=0,total_seconds=0.))
                part['count']+=value['count'];part['total_seconds']+=value['count']*value['mean']
    for scopes in inference.values():
        for value in scopes.values():
            value['mean_seconds']=value['total_seconds']/value['count'] if value['count'] else None
    return dict(control_ticks=steps,rollout_wall_seconds=total,resource_wait_seconds=wait,
        ticks_per_wall_second=steps/total,nested_environment_components=nested,wall_profile_sums=wall,
        native_inference=inference,
        scope='Rollout wall includes setup, recording and waits; Kit startup is separate. Nested timers overlap and must not be summed. CUDA stream intervals include submission gaps and are not pure kernel time.')


def autonomous(study,registration):
    directory=study/'development';plan=read(directory/'plan.json');finished=read(directory/'result.json')
    expected=development_cases(registration['session_binding']['domain'],registration['count_per_role'],registration['development_seed'])
    rows=read(directory/'results.json');pairs={arm+'_'+role:completed_pair(study/arm,role) for arm in ARMS for role in ('A','B')}
    require(finished.get('software_complete') is True and not finished.get('worker_error')
        and read(directory/'run_started.json')['plan_sha256']==sha256(directory/'plan.json')
        and plan['cases']==expected and plan['pairs']==pairs and plan['kind']=='autonomous'
        and plan.get('test_used') is False and plan['study']==STUDY_VERSION
        and plan['source_sha256']==registration['source_sha256'], 'Actual registered autonomous plan differs')
    require(finished['cases']==len(rows)==len(expected) and len({r['case'] for r in rows})==len(rows)
        and {r['case'] for r in rows}=={r['name'] for r in expected}, 'Autonomous opportunity denominator differs')
    cases=[]
    for item in expected:
        name=item['name'];key=item['role'];pair=pairs[key]
        row=next(r for r in rows if r['case']==name);result=read(directory/name/'result.json');started=read(directory/name/'started.json')
        require(row['role']==key and row['result']==result and result.get('software_complete') is True
            and not result.get('worker_error') and not result.get('censored') and result.get('dagger') is None
            and started['spec']==item['spec'] and started['checkpoint']==pair['backend']
            and all(result[k]==v for k,v in item['spec'].items()), 'Actual autonomous case identity differs')
        require(read(directory/name/'runtime_profile.json')==row['timing'],'Saved timing receipt differs')
        if not result.get('legal'):
            cases.append(dict(case=name,arm=item['study_arm'],role=item['student_role'],legal=False,success=False,
                result_sha256=sha256(directory/name/'result.json')));continue
        require(result['checkpoint_sha256']==pair['backend_sha256'] and not result['expert_shadow_computed']
            and result['residual_ticks']==result['steps'] and result['residual_version']==pair['residual_version']
            and result['action_control_mode']=='absolute_joint_position_target_rad'
            and result['simulation_step_hz']==60 and result['physics_hz']==120 and result['plan_hz']==5,
            'Full student execution/control identity differs')
        for component in ('dino_rgb','learned_visual_and_history','residual_forward_compose'):
            require(result['native_inference']['summary'][component]['wall_s']['count']==result['steps'],
                'Every control frame must use real visual/history/backend computation')
        require(result['native_inference']['summary']['ddim_plan']['wall_s']['count']==(result['steps']+11)//12,
            'Actual DP native planning clock differs')
        phases=phase_case(directory/name)
        require(phases['steps']==result['steps'] and phases['authority']['expert_ticks']==0, 'Autonomous trajectory used expert control')
        cases.append(dict(arm=item['study_arm'],role=item['student_role'],**phases,phase_snapshots=phase_snapshots(directory/name,phases)))
    totals={};timings={}
    for arm in ARMS:
        totals[arm]={};timings[arm]={}
        for role in ('A','B'):
            selected=[r for r in rows if r['role']==arm+'_'+role]
            timings[arm][role]=timing_report(selected)
            totals[arm][role]=summarize_autonomous_cases([r for r in cases if r['arm']==arm and r['role']==role],
                timings[arm][role]['rollout_wall_seconds'])
    specs=[r['spec'] for r in expected if r['role']=='control_A']
    source=source_autonomous(registration,specs)
    source_rows=read(Path(registration['source_evidence']['root'])/'development/results.json')
    source['timing']={role:timing_report([r for r in source_rows if r['role']==role]) for role in ('A','B')}
    return dict(cases=cases,totals=totals,timing=timings,source=source,
        initialization=read(directory/'initialization.json'),plan_sha256=sha256(directory/'plan.json'),
        results_sha256=sha256(directory/'results.json'),ordered_specs=specs,
        exact_hidden_state_replay_claim=False,interleaved=True)


def stage_inventory(study):
    paths={}
    for arm in ARMS:
        for role in ('A','B'):
            paths[arm+'_'+role+'_frontend']=study/arm/role/'training/finished.json'
            paths[arm+'_'+role+'_backend']=study/arm/role/'backend'/role/'training/finished.json'
    paths['development']=study/'development/result.json'
    paths['owner']=study/'finished.json'
    return {key:dict(path=str(path),present=path.exists(),sha256=sha256(path) if path.exists() else None)
        for key,path in paths.items()}


def partial_metrics(study,registration):
    """Only complete-epoch checkpoint curves, never an active uncommitted tail."""
    result={arm:{} for arm in ARMS};source=Path(registration['source_root'])
    for role in ('A','B'):
        if not any((study/arm/role/'training/last.pt').exists() for arm in ARMS):continue
        data=read(source/role/'state_data/data.json')
        numeric=torch.load(source/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
        ids=data['splits']['train']+data['splits']['validation']
        _,labels=prepare_pose_contexts(data,{i:numeric[i] for i in ids})
        for arm in ARMS:
            training=study/arm/role/'training';path=training/'last.pt'
            if not path.exists():continue
            digest=sha256(path);saved=torch.load(path,map_location='cpu',weights_only=False)
            require(digest==sha256(path),'Live durable checkpoint changed during report; retry with a new report identity')
            config=read(training/'configuration.json');initial=read(training/'initial_metrics.json')
            require(saved['configuration']==config and config['source_sha256']==registration['source_sha256']
                and config['continuation']['sha256']==registration['source_pairs'][role]['frontend_sha256']
                and config['data_sha256']==sha256(source/role/'state_data/data.json')
                and config['statistics_sha256']==sha256(source/role/'state_data/statistics.pt')
                and config['visual_auxiliary']['route']==ROUTES[arm]
                and saved['contract']==visual_auxiliary_image_contract('q_reached' if role=='A' else 'u_expert',ROUTES[arm]),
                'Partial metrics source/data/route checkpoint binding differs')
            require(saved['train_ids']==data['splits']['train'] and saved['validation_ids']==data['splits']['validation']
                and 1<=saved['epoch']<=registration['frontend_epochs']
                and saved['updates']==saved['epoch']*len(data['splits']['train'])
                and [r['epoch'] for r in saved['history']]==list(range(1,saved['epoch']+1)),
                'Partial durable epoch/population differs')
            for row in [initial,*saved['history']]:
                for split,cohort in [('train',data['splits']['train'][:2]),('validation',data['splits']['validation'])]:
                    check_metric_population(row[split],row['visual_auxiliary'][split],cohort,labels)
            metric=lambda row:dict(action={k:row[k] for k in ('train','validation')},pose=row['visual_auxiliary'])
            result[arm][role]=dict(durable_checkpoint_sha256=digest,complete_frontend=(training/'finished.json').exists(),
                durable_epochs=saved['epoch'],expected_epochs=registration['frontend_epochs'],
                source_initial={'metrics':metric(initial)},
                metric_history=[dict(epoch=row['epoch'],updates=row['updates'],**metric(row)) for row in saved['history']],
                scope='Only actual completed-epoch checkpoint metrics. Active unsaved work is omitted; no selected-pair or autonomy claim.')
    return result


def frontend_audit(study,registration):
    """Completed frontends only; a complete policy pair and autonomy are still pending."""
    source_root=Path(registration['source_root']);roles={}
    for role in ('A','B'):
        require(all((study/arm/role/'training/finished.json').exists() for arm in ARMS),
            'All four frontends must complete for matched actual-learning audit')
        data=read(source_root/role/'state_data/data.json')
        require(data['splits']==registration['training_population'] and not data['splits'].get('test')
            and data['normalization_train_ids']==registration['normalization_train_ids'], 'Source fitting population differs')
        numeric=torch.load(source_root/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)
        require(numeric['data_sha256']==sha256(source_root/role/'state_data/data.json'),'Numeric source differs')
        ids=data['splits']['train']+data['splits']['validation']
        _,labels=prepare_pose_contexts(data,{i:numeric['records'][i] for i in ids})
        cells={};logs={};last={}
        for arm in ARMS:cells[arm],logs[arm],last[arm]=frontend_cell(study,arm,role,registration,data,labels)
        fairness=matched_frontends(study,role,cells,logs,last)
        pair=registration['source_pairs'][role]
        require(sha256(pair['frontend'])==pair['frontend_sha256'],'Selected source frontend changed')
        source=torch.load(pair['frontend'],map_location='cpu',weights_only=False)
        selected=next(row for row in source['history'] if row['epoch']==source['epoch'])
        for arm in ARMS:
            initial=cells[arm]['source_initial']['metrics']['action']
            require(initial['validation']==selected['validation'] and initial['train']==selected['train'],
                'Current common source action measurement differs from actual selected source')
        roles[role]=dict(frontend=cells,matched_frontend=fairness,
            source=dict(checkpoint_sha256=pair['frontend_sha256'],epoch=source['epoch'],updates=source['updates'],
                action={key:selected[key] for key in ('train','validation')}),
            within_role_action_comparison_only=True)
    return roles


def plot_report(value,path):
    path=Path(path)
    if path.exists() or path.is_symlink():raise FileExistsError('Preserve earlier plots')
    behavior=path.with_name(path.stem+'_behavior'+path.suffix)
    if value['complete'] and (behavior.exists() or behavior.is_symlink()):raise FileExistsError('Preserve earlier behavior plot')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,5,figsize=(20,7),squeeze=False)
    palette={'control':'#176b87','auxiliary':'#b33b55'}
    for row,role in enumerate(('A','B')):
        baseline_drawn=False
        for arm in ARMS:
            cell=(value['roles'][role]['frontend'][arm] if value['complete'] else value.get('frontend_metrics',{}).get(arm,{}).get(role))
            if cell is None:continue
            history=cell['metric_history'];initial=cell['source_initial']['metrics'];epochs=[h['epoch'] for h in history]
            for split,style in [('train','--'),('validation','-')]:
                label=arm+' '+('train2' if split=='train' else 'val6')
                for col,key in enumerate(('selection','arm_rmse_rad','hand_rmse_rad')):
                    axes[row,col].plot(epochs,[h['action'][split][key] for h in history],style,color=palette[arm],label=label)
                    if not baseline_drawn:axes[row,col].axhline(initial['action'][split][key],color='#777777',linestyle=style,
                        alpha=.65,label='source '+('train2' if split=='train' else 'val6'))
                for col,key in [(3,'position_vector_rmse_mm'),(4,'orientation_mean_degrees')]:
                    points=[(h['epoch'],h['pose'][split][key]) for h in history if h['pose'][split][key] is not None]
                    if points:axes[row,col].plot(*zip(*points),style,color=palette[arm],label=label)
            baseline_drawn=True
        for col,title in enumerate(('Action normalized MSE','Action arm RMSE (rad)','Action hand RMSE (rad)','Pose position RMSE (mm)','Pose mean orientation (deg)')):
            axis=axes[row,col];axis.set_title(role+' · '+title);axis.set_xlabel('Completed epoch');axis.grid(alpha=.2)
            if axis.lines:axis.legend(fontsize=7)
            else:axis.text(.5,.5,'Pending actual checkpoint',ha='center',transform=axis.transAxes)
    figure.suptitle(('Complete' if value['complete'] else 'INCOMPLETE — actual saved epochs only')+
        ' · Fixed train2/val6; no auxiliary total-loss comparison\nPose readout learning alone does not establish policy vision or autonomous improvement',fontsize=11)
    figure.tight_layout(rect=(0,0,1,.92));path.parent.mkdir(parents=True,exist_ok=True);figure.savefig(path,dpi=150);plt.close(figure)
    if not value['complete']:return
    figure,axes=plt.subplots(2,4,figsize=(17,7),squeeze=False);outcomes=value['autonomous']
    for row,role in enumerate(('A','B')):
        labels=['source','control','auxiliary'];totals=[outcomes['source']['totals'][role]]+[outcomes['totals'][arm][role] for arm in ARMS]
        keys=['native_lift_cases','settled_stable_cases','near_hole_cases','first_subgoal_cases','successes']
        for offset,(name,total) in enumerate(zip(labels,totals)):
            axes[row,0].bar([i+(offset-1)*.25 for i in range(len(keys))],[total[k] for k in keys],width=.25,
                label=name+' n='+str(total['opportunities']))
        axes[row,0].set_xticks(range(len(keys)),['Lift','Hold','Near','Subgoal','Insert']);axes[row,0].set_ylabel('Cases / registered opportunities');axes[row,0].legend(fontsize=7)
        axes[row,1].bar(labels,[total['control_ticks_per_wall_second'] for total in totals]);axes[row,1].set_ylabel('Control ticks / wall second')
        for name in labels:
            cases=(outcomes['source']['cases'] if name=='source' else [r for r in outcomes['cases'] if r['arm']==name])
            cases=[r for r in cases if r['role']==role and r['legal']]
            for col,key in [(2,'root_quaternion_angle_rad'),(3,'native_keypoint_max_distance_m')]:
                points=[r['phase_snapshots']['first_goal_after_hold_best_orientation'] for r in cases]
                ys=[point[key] for point in points if point is not None]
                if ys:axes[row,col].scatter([labels.index(name)]*len(ys),ys,label=name,alpha=.8)
                else:axes[row,col].text(labels.index(name),.03,'No hold',ha='center',transform=axes[row,col].get_xaxis_transform(),fontsize=7)
                axes[row,col].set_xlim(-.5,2.5);axes[row,col].set_xticks(range(3),labels)
        for col,title in enumerate(('Native phase marginals','Measured rollout throughput','Post-hold best angle (rad)','Same-tick keypoint distance (m)')):
            axes[row,col].set_title(role+' · '+title);axes[row,col].grid(axis='y',alpha=.2)
    figure.suptitle('Source is a separate earlier cohort; matched reset opportunities are not exact hidden-state replay\nPost-hold plots omit cases without a stable window; angle and keypoint are from the same recorded tick',fontsize=11)
    figure.tight_layout(rect=(0,0,1,.92));figure.savefig(behavior,dpi=150);plt.close(figure)


def run(study,output,*,execution_source,allow_incomplete=False,plot=None,frontends_only=False):
    study=Path(study).resolve();output=Path(output).absolute();execution_source=Path(execution_source).resolve()
    if output.exists() or output.is_symlink():raise FileExistsError('Preserve earlier reports')
    torch.set_num_threads(2);registration=read(study/'registration.json')
    require(registration['version']==STUDY_VERSION and registration['arms']==list(ARMS)
        and registration['routes']==ROUTES and registration['test_used'] is False,
        'Unknown four-cell study registration')
    require(read(study/'prepared.json')['registration_sha256']==sha256(study/'registration.json'),'Registration changed after prepare')
    for name,digest in registration['source_sha256'].items():
        path=Path(name);path=path if path.is_absolute() else execution_source/path
        require(sha256(path)==digest,'Frozen execution source changed: '+str(path))
    project=Path(__file__).resolve().parents[1]
    dependencies=['tools/tight_fresh_anchor_study_report.py','tools/tight_fresh_backend_study_report.py',
        'tools/tight_fresh_learning_report.py','tools/tight_fresh_phases.py','tools/tight_fresh_rollouts.py',
        'tools/tight_fresh_visual_aux_study.py','visuotactile/visual_aux_training.py','visuotactile/visual_pose_aux.py']
    report_dependencies={name:sha256(project/name) for name in dependencies}
    for name,digest in report_dependencies.items():
        require(registration['source_sha256'].get(name)==digest,'Current reporting helper differs from frozen execution version: '+name)
    for path,digest in registration['protected'].items():
        require(sha256(path)==digest,'Protected study dependency changed: '+path)
    inventory=stage_inventory(study)
    base=dict(version=VERSION,registration_sha256=sha256(study/'registration.json'),
        execution_source=str(execution_source),execution_commit=registration['source_commit'],
        report_source_sha256=sha256(__file__),device='cpu',test_used=False,stages=inventory,
        report_dependencies_sha256=report_dependencies,stage_semantics=STAGE_SEMANTICS)
    if frontends_only:
        roles=frontend_audit(study,registration)
        value=dict(base,complete=False,frontend_audit_complete=True,roles=roles,
            frontend_metrics={arm:{role:roles[role]['frontend'][arm] for role in ('A','B')} for arm in ARMS},
            scope='Four completed frontends: actual matched exposure and source/best/last offline metrics only; no complete-pair or autonomy claim.',
            pending=[key for key,item in inventory.items() if not item['present']])
        require(not torch.cuda.is_initialized(),'Read-only frontend report unexpectedly initialized CUDA')
        atomic_json(output,value)
        if plot is not None:plot_report(value,plot)
        return
    if not all(r['present'] for r in inventory.values()):
        require(allow_incomplete,'Study incomplete; an explicit partial report cannot claim completion')
        value=dict(base,complete=False,pending=[k for k,v in inventory.items() if not v['present']],
            frontend_metrics=partial_metrics(study,registration),status=read(study/'status.json'),
            interpretation='Stage presence and actual durable-epoch curves only; no complete-study or autonomous improvement claim.')
        require(not torch.cuda.is_initialized(),'Read-only partial report unexpectedly initialized CUDA')
        atomic_json(output,value)
        if plot is not None:plot_report(value,plot)
        return
    require(read(study/'finished.json').get('software_complete') is True,'Study owner did not complete')
    roles={};source_root=Path(registration['source_root'])
    for role in ('A','B'):
        data=read(source_root/role/'state_data/data.json')
        require(data['splits']==registration['training_population'] and not data['splits'].get('test')
            and data['normalization_train_ids']==registration['normalization_train_ids'], 'Source fitting population differs')
        numeric=torch.load(source_root/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)
        require(numeric['data_sha256']==sha256(source_root/role/'state_data/data.json'),'Numeric source differs')
        ids=data['splits']['train']+data['splits']['validation'];records={i:numeric['records'][i] for i in ids}
        _,labels=prepare_pose_contexts(data,records)
        front={};front_logs={};front_last={};rear={};rear_logs={};rear_last={}
        for arm in ARMS:
            front[arm],front_logs[arm],front_last[arm]=frontend_cell(study,arm,role,registration,data,labels)
            rear[arm],rear_logs[arm],rear_last[arm]=backend_cell(study,arm,role,registration,front[arm])
        fairness=matched_frontends(study,role,front,front_logs,front_last)
        masks={arm:cached_training_masks(study,arm,role,rear_last[arm],records) for arm in ARMS}
        rear_fairness=matched_backends(study,role,rear,rear_logs,rear_last,masks)
        source=role_report(source_root,role)
        for arm in ARMS:
            initial=front[arm]['source_initial']['metrics']['action']
            require(initial['validation']==source['selected_validation'] and initial['train']==source['selected_train'],
                'Current common source action measurement differs from actual selected source')
        roles[role]=dict(frontend=front,backend=rear,matched_frontend=fairness,matched_backend=rear_fairness,
            source=source,within_role_action_comparison_only=True)
    outcomes=autonomous(study,registration)
    require(not torch.cuda.is_initialized(),'Read-only report unexpectedly initialized CUDA')
    value=dict(base,complete=True,roles=roles,autonomous=outcomes,
        limits='Action A/B supervision differs; compare source/control/auxiliary within each role. '
        'Both new cells train an identical pose head; only its vision-gradient route differs. '
        'Pose decoding gains alone do not prove task gains or a missing source representation. '
        'Total action-plus-pose loss is not a cross-route improvement metric. '
        'Use selected complete DP+backend autonomous outcomes; last-checkpoint training dose is reported separately. '
        'Native phase counts are independent marginals; same reset opportunities do not establish identical simulator hidden states.')
    atomic_json(output,value)
    if plot is not None:plot_report(value,plot)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execution-source',type=Path,required=True)
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--plot',type=Path)
    parser.add_argument('--frontends-only',action='store_true')
    args=parser.parse_args()
    run(args.study,args.output,execution_source=args.execution_source,allow_incomplete=args.allow_incomplete,
        plot=args.plot,frontends_only=args.frontends_only)
