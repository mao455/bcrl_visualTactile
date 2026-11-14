import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_held_near_goal_trigger import observation, stacked
from test_selected_frontend_fit import artifacts, write
from test_batch_collection_review import numeric as write_numeric
from tools.tight_fresh_selected_fit import validate_inputs
from tools import tight_fresh_stage_fit as fit


def masks():
    values=stacked([observation(tick,distance=.05 if tick>=60 else .2) for tick in range(80)])
    before,after=values['diagnostics_before'],values['diagnostics_after']
    valid=np.arange(80)>=42
    stage,height=fit.stage_masks(after['object_pose_w'],after['palm_body_pose_w'],values['object_contact_force_w'],
        before['goal_pose_w'],after['goal_pose_w'],before['successes'],after['successes'],
        values['native_keypoints_max_dist'],valid)
    stage['native_alignment']=np.zeros(80,bool);stage['legal_expert']=valid
    return stage,height


def test_stage_masks_count_real_expert_suffix_and_overlap_without_supervising_prefix():
    stage,height=masks()
    assert height==.5
    assert not any(value[:42].any() for value in stage.values())
    assert stage['holding'].sum()==21
    assert stage['near_hole60'].sum()==20
    assert stage['hand_rotation'].sum()==21
    assert stage['grasp_before_first_hold'].sum()==17
    assert not stage['after_first_native_subgoal'].any()


def test_current_goal_change_excludes_near_state_but_not_contact():
    values=stacked([observation(tick,distance=.05,goal_changed=tick==60) for tick in range(80)])
    before,after=values['diagnostics_before'],values['diagnostics_after']
    stage,_=fit.stage_masks(after['object_pose_w'],after['palm_body_pose_w'],values['object_contact_force_w'],
        before['goal_pose_w'],after['goal_pose_w'],before['successes'],after['successes'],
        values['native_keypoints_max_dist'],np.arange(80)>=42)
    assert stage['holding'][60] and not stage['near_hole60'][60]


def test_reduction_uses_commit_slots_and_rejects_invalid_prefix():
    stage,_=masks();matrix=np.stack([stage[key] for key in (*fit.STAGES,'legal_expert')])
    ticks=np.arange(60,72);errors=np.ones((12,29));errors[:,7:]*=2
    result=fit.finalized(fit.error_totals(errors,ticks,matrix))
    assert result['holding']['evaluated_commit_slots']==12
    assert result['holding']['arm_rmse_rad']==1 and result['holding']['hand_rmse_rad']==2
    assert result['after_first_native_subgoal']['arm_rmse_rad'] is None
    with pytest.raises(ValueError,match='student or invalid'):
        fit.error_totals(errors,np.arange(30,42),matrix)


def test_train_projection_copies_only_selected_tensors_and_visual_links(tmp_path):
    class Poison:
        def values(self):pytest.fail('Held-out tensor payload was touched')
    data=dict(splits=dict(train=['new'],validation=['v'],test=['t']),
        entries={key:dict(family=key) for key in ['new','v','t']})
    source=torch.arange(12).reshape(3,4)
    container=dict(records=dict(new=dict(robot=source),v=Poison(),t=Poison()))
    visual=tmp_path/'visual.pt';visual.write_bytes(b'frozen train shard')
    projected=fit.project_training_records(tmp_path,data,container,['new'],{'new':dict(path=str(visual))})
    root=Path(projected['root']);copied=torch.load(root/'numeric.pt',weights_only=False)
    assert set(copied['records'])=={'new'} and torch.equal(copied['records']['new']['robot'],source)
    assert fit.read(root/'data.json')['splits']==dict(train=['new'],validation=[],test=[])
    assert (root/'perception/new.pt').resolve()==visual
    assert not (root/'perception/v.pt').exists()
    with pytest.raises(ValueError,match='Only training'):
        fit.project_training_records(tmp_path,data,container,['v'],{})


def test_source_checkpoint_is_exact_continuation_best_not_last(artifacts,tmp_path):
    root,frozen,selected=artifacts
    receipt=validate_inputs(root,'B',frozen)
    source=tmp_path/'source_training';source.mkdir()
    old=copy.deepcopy(selected);old.update(epoch=2,updates=204)
    torch.save(old,source/'best.pt');torch.save(dict(old,epoch=99),source/'last.pt')
    write(source/'finished.json',dict(checkpoint_sha256=fit.sha256(source/'best.pt')))
    receipt['configuration']['continuation'].update(path=str(source/'best.pt'),sha256=fit.sha256(source/'best.pt'))
    bound,_=fit.bind_source(receipt,frozen,selected)
    assert bound['epoch']==2 and bound['path'].endswith('best.pt')
    receipt['configuration']['continuation'].update(path=str(source/'last.pt'),sha256=fit.sha256(source/'last.pt'))
    with pytest.raises(ValueError,match='actual selected source best'):fit.bind_source(receipt,frozen,selected)


def test_each_frontend_reencodes_entire_student_prefix_and_legal_history():
    stage,_=masks();matrix=np.stack([stage[key] for key in (*fit.STAGES,'legal_expert')])
    record=dict(time=torch.arange(80),robot=torch.arange(80).float()[:,None],joint_limits=None)
    calls=[]
    def encode(model,actual):
        assert actual is record and len(actual['time'])==80
        assert actual['robot'][0]==0 and actual['robot'][40]==40
        calls.append(model.scale)
        return actual['robot'].sum().reshape(1,1)*model.scale,torch.tensor([60])
    trainer=SimpleNamespace(supervised_conditions=encode,
        predict_on_native_clock=lambda model,c,a:c[:,:,None].expand(1,24,29),
        bounded_targets=lambda p,limits:p,
        labels=lambda r,a,h:(torch.zeros(1,h,29),torch.ones(1,h,dtype=torch.bool)))
    first=fit.frontend_episode(trainer,SimpleNamespace(scale=1),record,matrix)
    second=fit.frontend_episode(trainer,SimpleNamespace(scale=2),record,matrix)
    assert calls==[1,2]
    assert first['legal_expert']['evaluated_commit_slots']==second['legal_expert']['evaluated_commit_slots']==12
    assert second['holding']['arm_squared_sum']==4*first['holding']['arm_squared_sum']


def test_complete_cpu_admission_binds_real_stage_masks_selected_exposure_and_train_projection(artifacts,tmp_path):
    import h5py
    root,frozen,selected=artifacts;role=root/'B';collection=tmp_path/'collection'
    case=collection/'new1';case.mkdir(parents=True)
    result=write_numeric(case,40)
    result['accepted_for_training']=True;result['dataset_capture']['sha256']='source-new1'
    data=fit.read(role/'state_data/data.json');entry=data['entries']['new1']
    entry.update(length=80,takeover_tick=40,episode='new1',path='/unused/raw_train')
    write(role/'state_data/data.json',data)
    with h5py.File(case/'numeric.h5') as stream:
        transition=stream['episode/transition']
        record=dict(time=torch.from_numpy(transition['time_before_s'][:]),
            supervision_mask=torch.from_numpy(transition['dagger/expert_label_valid'][:]),
            control_segment=torch.from_numpy(transition['dagger/segment_id'][:]),
            u_expert=torch.from_numpy(transition['dagger/u_expert'][:]),
            q_reached=torch.from_numpy(transition['dagger/q_reached'][:]),
            command=torch.zeros(80,29),joint_limits=torch.zeros(29,2),robot=torch.zeros(80,58))
    torch.save(dict(data_sha256=fit.sha256(role/'state_data/data.json'),records={'new1':record}),role/'state_data/numeric.pt')
    cache=fit.read(role/'cache_index.json');cache['entries']['new1']['frames']=80
    cache['data_sha256']=fit.sha256(role/'state_data/data.json');write(role/'cache_index.json',cache)
    source=tmp_path/'source_training';source.mkdir()
    old=copy.deepcopy(selected);old.update(epoch=2,updates=204)
    torch.save(old,source/'best.pt');write(source/'finished.json',dict(checkpoint_sha256=fit.sha256(source/'best.pt')))
    continuation=fit.read(role/'continuation.json')
    continuation.update(path=str(source/'best.pt'),sha256=fit.sha256(source/'best.pt'))
    write(role/'continuation.json',continuation)
    config=selected['configuration'];config.update(continuation=continuation,
        data_sha256=fit.sha256(role/'state_data/data.json'),cache_index_sha256=fit.sha256(role/'cache_index.json'))
    write(role/'training/configuration.json',config)
    torch.save(selected,role/'training/best.pt')
    write(role/'training/finished.json',dict(checkpoint_sha256=fit.sha256(role/'training/best.pt')))
    write(collection/'result.json',dict(software_complete=True))
    write(collection/'correction_entries.json',dict(new1=entry))
    write(collection/'results.json',[dict(case='new1',result=result)])
    assert not torch.cuda.is_initialized()
    proof=fit.prepare(root,'B',collection,frozen,frozen,tmp_path/'prepared')
    assert not torch.cuda.is_initialized()
    assert proof['selected']['epoch']==3 and proof['source']['epoch']==2
    exposure=proof['episodes']['new1']
    assert exposure['selected_full_history_visits']==6 and exposure['legal_expert_actions']==38
    assert exposure['actual_training_anchor_target_slots_by_stage'] is None
    assert exposure['selected_legal_action_visible_count']['holding']==6*21
    assert proof['train_projection']['identities']==['new1']


def test_compare_requires_identical_stage_population_and_bound_checkpoints(tmp_path):
    registration=tmp_path/'registration.json'
    plan=dict(source=dict(sha256='old',epoch=5),selected=dict(sha256='new',epoch=3),
        episodes={},scope='frontend only')
    write(registration,plan)
    for arm in ('source','selected'):
        write(tmp_path/arm/'result.json',dict(complete=True,registration_sha256=fit.sha256(registration),
            checkpoint_sha256=plan[arm]['sha256'],stages={key:dict(evaluated_commit_slots=2,
                arm_rmse_rad=.2 if arm=='source' else .1,hand_rmse_rad=.4 if arm=='source' else .3)
                for key in (*fit.STAGES,'legal_expert')}))
    result=fit.compare(registration)
    assert result['selected_epoch']==3 and not result['rear_evaluated']
    assert result['stages']['holding']['delta_rad']['arm_rmse_rad']==-.1
    wrong=fit.read(tmp_path/'selected/result.json');wrong['stages']['holding']['evaluated_commit_slots']=1
    write(tmp_path/'selected/result.json',wrong)
    with pytest.raises(ValueError,match='populations differ'):fit.compare(registration)
