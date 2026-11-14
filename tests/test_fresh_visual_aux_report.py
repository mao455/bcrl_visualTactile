import copy
import json
from pathlib import Path

import pytest
import torch

from tools import tight_fresh_visual_aux_report as report
from visuotactile.visual_aux_training import empty_exposure,add_exposure


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_action_and_pose_objective_components_are_separate_and_reconcile():
    config=dict(visual_auxiliary={'loss_weight':.02},hand_geometry={'weight':.5})
    parts=dict(base=.2,position=.3,rotation=.4,hand_geometry=.5,
        auxiliary_position=10.,auxiliary_orientation6=2.,auxiliary_weighted=.24)
    action=.2+report.ACTION_WEIGHTS['position']*.3+report.ACTION_WEIGHTS['rotation']*.4+.5*.5
    rows=[dict(epoch=1,parts=parts,loss=action+.24)]
    actual=report.objective_history(rows,config,1)
    assert actual['epochs'][0]['action_objective']==pytest.approx(action)
    assert actual['epochs'][0]['base']==.2 and actual['epochs'][0]['auxiliary_weighted']==.24
    assert actual['total_loss_comparable_across_routes'] is False
    bad=copy.deepcopy(rows);bad[0]['loss']-=.24
    with pytest.raises(ValueError,match='reconcile'):report.objective_history(bad,config,1)
    bad=copy.deepcopy(rows);bad[0]['parts']['extra_unknown']=0
    with pytest.raises(ValueError,match='component'):report.objective_history(bad,config,1)
    with pytest.raises(ValueError,match='equal-episode'):report.objective_history(rows,config,2)


def metric_fixture():
    ids=['a','b']; labels={i:dict(valid_perception_ticks=n,invalid_perception_ticks=1) for i,n in zip(ids,[1,3])}
    action=dict(selection=.1,arm_rmse_rad=.2,hand_rmse_rad=.3,episodes=[{'identity':i} for i in ids])
    rows=[dict(identity=i,valid_ticks=n,invalid_perception_ticks=1,position_vector_rmse_mm=x,
        orientation6_mse=.5,valid_orientation_predictions=n-1,degenerate_orientation_predictions=1,
        orientation_mean_degrees=60. if n>1 else None)
        for i,n,x in zip(ids,[1,3],[1.,3.])]
    pose=dict(episodes=rows,valid_ticks=4,invalid_perception_ticks=2,degenerate_orientation_predictions=2,
        position_vector_rmse_mm=7**.5,orientation6_mse=.5,orientation_mean_degrees=60.)
    return action,pose,ids,labels


def test_pose_and_action_validation_use_exact_fixed_population_and_tick_weights():
    action,pose,ids,labels=metric_fixture()
    report.check_metric_population(action,pose,ids,labels)
    bad=copy.deepcopy(pose);bad['position_vector_rmse_mm']=2.
    with pytest.raises(ValueError,match='tick denominators'):report.check_metric_population(action,bad,ids,labels)
    bad=copy.deepcopy(pose);bad['episodes'][0]['degenerate_orientation_predictions']=0
    with pytest.raises(ValueError,match='Degenerate'):report.check_metric_population(action,bad,ids,labels)
    with pytest.raises(ValueError,match='population'):report.check_metric_population(action,pose,list(reversed(ids)),labels)


def receipts_fixture():
    ids=['original','correction'];data=dict(splits=dict(train=ids,validation=['validation']),
        normalization_train_ids=['original'],aggregation={'new_training_ids':['correction']})
    labels={i:dict(ticks=4,valid_perception_ticks=4,perception_exposure={
        'original' if i=='original' else 'legal_expert':dict(observed_ticks=4,perception_ticks=4,invalid_perception_ticks=0)}) for i in ids}
    config={'example':'bound'};learning=[];history=[];ledger=empty_exposure();snapshots=[]
    for epoch in [1,2]:
        for index,i in enumerate(ids):
            update=len(learning)+1
            add_exposure(ledger,i,labels[i],data['normalization_train_ids'])
            learning.append(dict(epoch=epoch,episode=i,episode_index=index+1,updates=update,
                visual_frames=4,anchors_seen=update,target_ticks=4*update,
                auxiliary_exposure=labels[i]['perception_exposure']))
        history.append(dict(epoch=epoch,updates=2*epoch,validation={'selection':float(epoch)},
            anchors_seen=2*epoch,target_ticks=8*epoch))
        snapshots.append(dict(epoch=epoch,updates=2*epoch,configuration=config,train_ids=ids,
            validation_ids=['validation'],history=copy.deepcopy(history),auxiliary_exposure=copy.deepcopy(ledger),
            optimizer={'state':{0:dict(step=torch.tensor(2*epoch))}},anchors_seen=2*epoch,target_ticks=8*epoch))
    best,last=snapshots
    finished=dict(complete=True,epochs=2,updates=4,auxiliary_exposure=copy.deepcopy(ledger),anchors_seen=4,target_ticks=16)
    return best,last,history,learning,finished,config,data,labels,2


def test_best_and_last_exposure_are_separate_and_actual_adam_receipts_bind():
    args=receipts_fixture();value=report.check_frontend_receipts(*args)
    assert value['selected_auxiliary_exposure']['episode_updates']==2
    assert value['last_auxiliary_exposure']['episode_updates']==4
    assert value['actual_episode_counts']=={'original':2,'correction':2}
    assert value['actual_action_exposure_by_source']['current_correction']['visual_frames']==8
    changed=copy.deepcopy(args);changed[1]['optimizer']['state'][0]['step']=torch.tensor(3)
    with pytest.raises(ValueError,match='Adam'):report.check_frontend_receipts(*changed)


def test_source_action_change_does_not_use_pose_or_global_training_loss():
    before={s:dict(selection=.1,arm_rmse_rad=.02,hand_rmse_rad=.3) for s in ('train','validation')}
    after=copy.deepcopy(before);after['validation']['selection']=.12
    before.update(loss=100.,visual_auxiliary={'loss':100.})
    after.update(loss=1.,visual_auxiliary={'loss':1.})
    value=report.action_change(before,after)
    assert value['validation']['selection']['relative_change']==pytest.approx(.2)
    assert value['train']['selection']['relative_change']==0


@pytest.mark.parametrize('kind',['duplicated_episode','wrong_phase','noncontiguous_update','wrong_best','wrong_visual_frames','wrong_history'])
def test_frontend_receipt_corruption_fails_instead_of_becoming_training_progress(kind):
    args=list(copy.deepcopy(receipts_fixture()))
    if kind=='duplicated_episode':args[3][1]['episode']='original'
    elif kind=='wrong_phase':args[3][0]['auxiliary_exposure']={}
    elif kind=='noncontiguous_update':args[3][1]['updates']=3
    elif kind=='wrong_best':args[0]['epoch']=2
    elif kind=='wrong_visual_frames':args[3][0]['visual_frames']=3
    else:args[2][0]['epoch']=2
    with pytest.raises(ValueError):report.check_frontend_receipts(*args)


def test_matched_frontend_ignores_only_route_hashes_and_rejects_changed_exposure(tmp_path):
    saved={};cells={};logs={}
    for arm in report.ARMS:
        saved[arm]=dict(configuration=dict(visual_auxiliary=dict(route=report.ROUTES[arm],spec_sha256=arm,
            loss_weight=.02),auxiliary_label_report_sha256=arm,source_root=str(tmp_path/arm),learning_rate=3e-5),
            schedule_rng=torch.tensor([1]),noise_rng=torch.tensor([2]),optimizer={'param_groups':[{'lr':3e-5}]})
        cells[arm]=dict(source_initial={'metrics':{'pose':1,'action':2}})
        logs[arm]=[dict(epoch=1,episode='same',episode_index=1,updates=1,visual_frames=4,anchors_seen=1,
            target_ticks=4,auxiliary_exposure={'phase':4})]
    assert report.matched_frontends(tmp_path,'A',cells,logs,saved)['passed']
    saved['auxiliary']['configuration']['learning_rate']=1e-4
    with pytest.raises(ValueError,match='configuration'):report.matched_frontends(tmp_path,'A',cells,logs,saved)
    saved['auxiliary']['configuration']['learning_rate']=3e-5;logs['auxiliary'][0]['target_ticks']=5
    with pytest.raises(ValueError,match='exposure'):report.matched_frontends(tmp_path,'A',cells,logs,saved)


def test_runtime_uses_actual_count_weighted_means_keeps_nested_timers_separate():
    def row(seconds,count,mean):
        return dict(timing=dict(total_seconds=seconds,resource_wait_seconds=1.,nested_environment_components={'render_s':2.}),
            result=dict(legal=True,steps=count,wall_profile={'environment_step_render_s':3.},
                native_inference={'summary':{'dino_rgb':{'wall_s':dict(count=count,mean=mean)}}}))
    timing=report.timing_report([row(10.,1,.1),row(20.,3,.3)])
    assert timing['ticks_per_wall_second']==pytest.approx(4/30)
    assert timing['native_inference']['dino_rgb']['wall_s']['mean_seconds']==pytest.approx(.25)
    assert timing['wall_profile_sums']['environment_step_render_s']==6.
    assert timing['nested_environment_components']['render_s']==4.
    assert timing['rollout_wall_seconds']==30.
    broken=row(1.,1,.1);broken['timing']['resource_wait_seconds']=2.
    with pytest.raises(ValueError,match='timing'):report.timing_report([broken])


def test_backend_components_use_legal_tick_denominators_and_keep_roundoff_max():
    rows=[dict(epoch=1,supervised_ticks=n,loss_weight_mass=n,fixed_denominator=128,loss=n/128,
        loss_components=dict(exec_fit=2*n/128,fy_fit=n/128,regularizer=0.,label_roundoff_max_rad=e))
        for n,e in [(128,1e-8),(32,2e-8)]]
    value=report.backend_loss_history(rows)['epochs'][0]
    assert value['legal_tick_components']==dict(exec_fit=2.,fy_fit=1.,regularizer=0.)
    assert value['optimizer_objective_mean']==.625 and value['label_roundoff_max_rad']==2e-8
    rows[0]['loss_weight_mass']=256
    with pytest.raises(ValueError,match='unweighted128'):report.backend_loss_history(rows)


def test_cached_training_masks_bind_actual_feature_shard_bytes_and_authority(tmp_path):
    directory=tmp_path/'control/A/backend/A/nominal';directory.mkdir(parents=True)
    raw=dict(time=torch.arange(3,dtype=torch.float64)/60,supervision_mask=torch.tensor([False,True,True]),
        control_segment=torch.tensor([0,1,1]))
    shard=dict(identity='c',**raw);path=directory/'c.pt';torch.save(shard,path)
    index={'episodes':{'c':{'sha256':report.sha256(path)}}};write(directory/'index.json',index)
    saved=dict(train_ids=['c'],validation_ids=[],configuration={'cache_index_sha256':report.sha256(directory/'index.json')})
    value=report.cached_training_masks(tmp_path,'control','A',saved,{'c':raw})
    assert torch.equal(value['c'],raw['supervision_mask'])
    shard['supervision_mask']=torch.ones(3,dtype=torch.bool);torch.save(shard,path)
    with pytest.raises(ValueError,match='shard changed'):
        report.cached_training_masks(tmp_path,'control','A',saved,{'c':raw})
    index['episodes']['c']['sha256']=report.sha256(path);write(directory/'index.json',index)
    saved['configuration']['cache_index_sha256']=report.sha256(directory/'index.json')
    with pytest.raises(ValueError,match='authority/clock'):
        report.cached_training_masks(tmp_path,'control','A',saved,{'c':raw})


def test_plot_partial_uses_only_actual_points_without_fake_zero_cells(tmp_path,monkeypatch):
    import matplotlib.axes
    plotted=[];original=matplotlib.axes.Axes.plot
    def record(self,*args,**kwargs):
        plotted.append(args)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes,'plot',record)
    metric=dict(action={s:dict(selection=.1,arm_rmse_rad=.02,hand_rmse_rad=.03) for s in ('train','validation')},
        pose={s:dict(position_vector_rmse_mm=40.,orientation_mean_degrees=60.) for s in ('train','validation')})
    value=dict(complete=False,frontend_metrics={'control':{'A':dict(source_initial={'metrics':metric},
        metric_history=[dict(epoch=1,**metric)])},'auxiliary':{}})
    output=tmp_path/'partial.png';report.plot_report(value,output)
    assert output.stat().st_size>1000 and len(plotted)==10
    assert all(args[0]==[1] or args[0]==(1,) for args in plotted)
    with pytest.raises(FileExistsError):report.plot_report(value,output)


def test_plot_complete_has_behavior_denominators_and_no_fake_posthold_points(tmp_path,monkeypatch):
    import matplotlib.figure
    figures=[];original=matplotlib.figure.Figure.savefig
    def capture(self,*args,**kwargs):
        figures.append(self)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',capture)
    metric=dict(action={s:dict(selection=.1,arm_rmse_rad=.02,hand_rmse_rad=.03) for s in ('train','validation')},
        pose={s:dict(position_vector_rmse_mm=40.,orientation_mean_degrees=60.) for s in ('train','validation')})
    cell=dict(source_initial={'metrics':metric},metric_history=[dict(epoch=1,**metric)])
    totals=dict(opportunities=4,native_lift_cases=1,settled_stable_cases=1,near_hole_cases=0,
        first_subgoal_cases=0,successes=0,control_ticks_per_wall_second=10.)
    cases=[dict(arm=arm,role=role,legal=True,phase_snapshots={'first_goal_after_hold_best_orientation':None})
        for arm in report.ARMS for role in ('A','B')]
    source_cases=[dict(arm='source',role=role,legal=True,phase_snapshots={'first_goal_after_hold_best_orientation':
        dict(root_quaternion_angle_rad=1.,native_keypoint_max_distance_m=.05)}) for role in ('A','B')]
    value=dict(complete=True,roles={r:{'frontend':{a:cell for a in report.ARMS}} for r in ('A','B')},
        autonomous=dict(cases=cases,totals={a:{r:totals for r in ('A','B')} for a in report.ARMS},
            source=dict(cases=source_cases,totals={r:totals for r in ('A','B')})))
    path=tmp_path/'full.png';report.plot_report(value,path)
    assert path.exists() and (tmp_path/'full_behavior.png').exists()
    assert len(figures[0].axes)==10 and len(figures[1].axes)==8
    assert all(len(figures[1].axes[i].collections)==1 for i in (2,3,6,7))
    assert all('n=4' in item.get_text() for item in figures[1].axes[0].get_legend().get_texts())


def test_recovery_records_keep_actual_discarded_work_outside_model_dose(tmp_path):
    archive=tmp_path/'learning.abandoned.example.jsonl';archive.write_text('{"updates":3}\n')
    write(tmp_path/'recovery.example.json',dict(abandoned_path=str(archive),abandoned_sha256=report.sha256(archive),
        discarded_updates_in_restored_model=0,known_discarded_update_ids=[3]))
    value=report.recovery_evidence(tmp_path)
    assert value['known_discarded_optimizer_updates']==1 and value['excluded_from_model_exposure']
    archive.write_text('changed')
    with pytest.raises(ValueError,match='Discarded'):report.recovery_evidence(tmp_path)


def test_partial_report_never_claims_training_or_autonomy_and_checks_frozen_source(tmp_path):
    study=tmp_path/'study';source=tmp_path/'execution';source.mkdir();code=source/'example.py';code.write_text('actual source\n')
    project=Path(report.__file__).resolve().parents[1]
    dependencies=['tools/tight_fresh_anchor_study_report.py','tools/tight_fresh_backend_study_report.py',
        'tools/tight_fresh_learning_report.py','tools/tight_fresh_phases.py','tools/tight_fresh_rollouts.py',
        'tools/tight_fresh_visual_aux_study.py','visuotactile/visual_aux_training.py','visuotactile/visual_pose_aux.py']
    for name in dependencies:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((project/name).read_bytes())
    registration=dict(version=report.STUDY_VERSION,arms=list(report.ARMS),routes=report.ROUTES,test_used=False,
        source_sha256={'example.py':report.sha256(code),**{n:report.sha256(project/n) for n in dependencies}},
        protected={},source_commit='frozen',source_root=str(tmp_path/'source'))
    write(study/'registration.json',registration)
    write(study/'prepared.json',{'registration_sha256':report.sha256(study/'registration.json')})
    write(study/'status.json',{'stage':'control_A_frontend'})
    output=tmp_path/'partial.json'
    with pytest.raises(ValueError,match='incomplete'):report.run(study,output,execution_source=source)
    assert not output.exists()
    report.run(study,output,execution_source=source,allow_incomplete=True)
    value=json.loads(output.read_text())
    assert value['complete'] is False and 'roles' not in value and 'autonomous' not in value
    assert len(value['pending'])==10
    with pytest.raises(FileExistsError):report.run(study,output,execution_source=source,allow_incomplete=True)
    code.write_text('changed\n')
    with pytest.raises(ValueError,match='Frozen execution'):report.run(study,tmp_path/'other.json',execution_source=source,allow_incomplete=True)


def test_autonomous_counts_invalid_opportunities_and_checks_registered_order(tmp_path,monkeypatch):
    from visuotactile import narrow_data
    monkeypatch.setattr(narrow_data,'direct_configuration',lambda domain,seed:dict(seed=seed))
    pairs={arm+'_'+role:{'backend':str(tmp_path/arm/role/'best.pt')} for arm in report.ARMS for role in ('A','B')}
    monkeypatch.setattr(report,'completed_pair',lambda root,role:pairs[root.name+'_'+role])
    monkeypatch.setattr(report,'source_autonomous',lambda *args:dict(scope='separate source cohort'))
    source=tmp_path/'source';write(source/'development/results.json',[
        dict(role=role,timing={'total_seconds':1.,'resource_wait_seconds':0.},result={'legal':False}) for role in ('A','B')])
    registration=dict(count_per_role=1,development_seed=17,session_binding={'domain':{}},
        source_evidence={'root':str(source)},source_sha256={'frozen':'hash'})
    expected=report.development_cases({},1,17);rows=[];directory=tmp_path/'development'
    for item in expected:
        result=dict(software_complete=True,legal=False,**item['spec'])
        timing=dict(total_seconds=1.,resource_wait_seconds=0.)
        rows.append(dict(case=item['name'],role=item['role'],result=result,timing=timing))
        write(directory/item['name']/'result.json',result)
        write(directory/item['name']/'started.json',dict(spec=item['spec'],checkpoint=pairs[item['role']]['backend']))
        write(directory/item['name']/'runtime_profile.json',timing)
    plan=dict(cases=expected,pairs=pairs,kind='autonomous',test_used=False,study=report.STUDY_VERSION,source_sha256=registration['source_sha256'])
    write(directory/'plan.json',plan);write(directory/'result.json',dict(software_complete=True,cases=4));write(directory/'results.json',rows)
    write(directory/'run_started.json',dict(plan_sha256=report.sha256(directory/'plan.json')));write(directory/'initialization.json',{'seconds':2.})
    result=report.autonomous(tmp_path,registration)
    assert result['totals']['auxiliary']['B']['opportunities']==1
    assert result['totals']['auxiliary']['B']['invalid_initializations']==1
    assert result['totals']['auxiliary']['B']['legal']==0
    plan['cases']=list(reversed(expected));write(directory/'plan.json',plan)
    write(directory/'run_started.json',dict(plan_sha256=report.sha256(directory/'plan.json')))
    with pytest.raises(ValueError,match='registered autonomous plan'):report.autonomous(tmp_path,registration)
