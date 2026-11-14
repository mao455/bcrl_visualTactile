import json

import pytest
import torch

from tools.tight_fresh_batch_eval import (probe_options,execution_limit,lane_outcome,
    inference_counts,check_sensor_clock,prepare)


def outcome(**changes):
    values=dict(legal=True,steps=36,reasons=None,still_active=True,max_control_ticks=36,native_bound=1803)
    return lane_outcome(**dict(values,**changes))


def test_censored_diagnostic_is_not_timeout_success_or_training_data():
    result=outcome()
    assert result['success'] is None and result['censored']
    assert result['censor_reason']=='diagnostic_control_limit'
    assert not result['naturally_ended'] and not result['observed_native_completion']
    assert not result['autonomous_evaluation_eligible'] and not result['accepted_for_training']
    assert result['termination_classification']=='diagnostic_censored'


def test_natural_completion_inside_probe_remains_diagnostic_only():
    result=outcome(steps=20,reasons=dict(max_successes=True,timeout=False),still_active=False)
    assert result['observed_native_completion'] and result['naturally_ended']
    assert result['success'] is None and not result['censored']
    assert not result['autonomous_evaluation_eligible']


def test_default_complete_episode_semantics_and_missing_termination_guard():
    result=outcome(max_control_ticks=None,steps=220,still_active=False,reasons=dict(max_successes=True))
    assert result['success'] is True and result['autonomous_evaluation_eligible']
    assert not result['diagnostic_only'] and not result['censored']
    assert execution_limit(1803,None)==1803
    assert execution_limit(1803,36)==36
    with pytest.raises(ValueError,match='natural-termination'):
        outcome(max_control_ticks=None,steps=1803)
    with pytest.raises(ValueError,match='natural-termination'):
        outcome(max_control_ticks=2000,steps=1803)
    with pytest.raises(ValueError,match='every registered'):
        outcome(steps=35)
    invalid=outcome(legal=False,steps=0,still_active=False)
    assert invalid['termination_classification']=='invalid_initialization' and not invalid['censored']


@pytest.mark.parametrize('limit',[True,0,-1,3.5,'36'])
def test_invalid_diagnostic_limit_rejected_before_start(limit):
    with pytest.raises(ValueError):probe_options(limit)


def test_complete_real_tick_and_planning_counts_required():
    rows=[]
    for tick in range(36):
        rows.extend(dict(stage=stage,tick=tick) for stage in
            ('learned_visual_and_history','robot_history_input','residual_forward_compose'))
        if tick%12==0:rows.append(dict(stage='ddim_plan',tick=tick))
    counts=inference_counts(dict(rows=rows),36,36,strict=True)
    assert counts['plans']==3 and counts['history_updates']==36 and counts['backend_updates']==36
    with pytest.raises(ValueError,match='full real-frame'):
        inference_counts(dict(rows=rows[:-1]),36,36,strict=True)
    with pytest.raises(ValueError,match='full real-frame'):
        inference_counts(dict(rows=rows),36,35,strict=True)


def test_probe_rejects_reused_sensor_frames_wrong_time_or_invalid_content():
    packet=dict(image=torch.zeros(2,3,4,2,2),camera_valid=torch.ones(2,3,dtype=torch.bool),
        camera_time_s=torch.zeros(2,3,dtype=torch.float64),time_s=torch.zeros(2,dtype=torch.float64),
        camera_frame=torch.ones(2,3,dtype=torch.long))
    previous=check_sensor_clock(packet,[True,False],[None,None])
    assert previous[1] is None
    with pytest.raises(ValueError,match='repeated'):check_sensor_clock(packet,[True,False],previous)
    packet['camera_frame']+=1;packet['camera_time_s'][0,0]=.001
    with pytest.raises(ValueError,match='clock'):check_sensor_clock(packet,[True,False],previous)
    packet['camera_time_s'].zero_();packet['camera_valid'][0,1]=False
    with pytest.raises(ValueError,match='RGB-D'):check_sensor_clock(packet,[True,False],previous)


def test_prepare_default_compatibility_and_large_single_role_probe(tmp_path,monkeypatch):
    import tools.tight_fresh_batch_eval as module
    import visuotactile.narrow_data as data
    root=tmp_path/'source';root.mkdir();(root/'session.json').write_text(json.dumps(dict(domain={'unchanged':True})))
    calls=[]
    monkeypatch.setattr(module,'completed_pair',lambda root,role:calls.append(role) or {'role':role})
    monkeypatch.setattr(data,'direct_configuration',lambda domain,seed:dict(seed=seed))
    monkeypatch.setattr('subprocess.check_output',lambda *args,**kwargs:'frozen_commit\n')
    monkeypatch.chdir(tmp_path)
    prepare(root,tmp_path/'default',2,17)
    old=json.loads((tmp_path/'default/plan.json').read_text())
    assert old['kind']=='engineering_batch_development' and list(old['pairs'])==['A','B']
    assert 'max_control_ticks' not in old and 'execution_backend' not in old
    assert 'inference_scheduler_index_mode' not in old
    calls.clear()
    prepare(root,tmp_path/'probe',256,19,neural_batch=True,max_control_ticks=36,roles=['B'],
        execution_backend=dict(physics_device='cuda:0',camera_backend='tiled'))
    new=json.loads((tmp_path/'probe/plan.json').read_text())
    assert calls==['B'] and len(new['cases'])==256 and new['diagnostic_only']
    assert new['kind']=='engineering_batch_throughput_probe' and new['max_control_ticks']==36
    assert new['inference_scheduler_index_mode']=='host'
    assert new['execution_backend']['camera_backend']=='tiled'
    with pytest.raises(ValueError):probe_options(36,['A','A'])
    with pytest.raises(ValueError):probe_options(36,[])
