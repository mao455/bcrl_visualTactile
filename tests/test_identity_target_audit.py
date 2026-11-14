import numpy as np
import json
import h5py
import pytest
import torch

from tools import tight_fresh_identity_target_audit as audit
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW


def record(length=80,first=42):
    q=torch.arange(length+1).float()[:,None].expand(-1,29)*.001
    valid=torch.arange(length)>=first
    command=q[:-1]+.01
    return dict(robot=torch.cat((q[:-1],torch.zeros(length,149)),dim=1),
        time=torch.arange(length,dtype=torch.float64)/60,command=command,
        q_reached=torch.where(valid[:,None],q[1:],0.),u_expert=torch.where(valid[:,None],command,0.),
        supervision_mask=valid,control_segment=torch.where(valid,2,0),joint_limits=torch.tensor([[-5.,5.]]).expand(29,-1))


def test_actual_one_step_is_distinct_from_command_and_invalid_prefix_zeros_are_excluded():
    r=record();q0,q1,u,valid=audit.cached_arrays(r)
    result=audit.summarize(q0[valid],q1[valid],u[valid])
    for part in ('arm','hand'):
        assert result[part]['A_q_next_minus_q_before']['rmse_rad']==pytest.approx(.001,abs=1e-8)
        assert result[part]['B_u_minus_q_before']['rmse_rad']==pytest.approx(.01,abs=1e-8)
        assert result[part]['u_minus_q_next']['rmse_rad']==pytest.approx(.009,abs=1e-8)
        assert result[part]['increment_direction']['vector_positive_fraction']==1
    r['q_reached'][50,0]+=.001
    with pytest.raises(ValueError,match='next actual observation'):audit.cached_arrays(r)


def test_onset_anchor_is_retained_only_at_its_actual_legal_target_offsets():
    r=record();samples=audit.anchor_samples(r,r['supervision_mask'].numpy(),RECOVERY_ONSET_WINDOW)
    assert 36 not in samples['1'][0] and 36 not in samples['6'][0]
    assert 36 in samples['12'][0] and 36 in samples['24'][0]
    assert samples['12'][1][list(samples['12'][0]).index(36)]==47
    assert all((targets>=42).all() for _,targets in samples.values())
    summary=audit.collect_summary({'x':r},['x'],[r['supervision_mask'].numpy()],RECOVERY_ONSET_WINDOW)
    baseline=summary['native_5hz_anchor_hold_current']
    assert baseline['1']['arm']['A_q_target_minus_anchor_q']['rmse_rad']==pytest.approx(.001,abs=1e-8)
    assert baseline['24']['arm']['A_q_target_minus_anchor_q']['rmse_rad']==pytest.approx(.024,abs=1e-8)


def test_near_zero_direction_has_explicit_denominator_and_no_arbitrary_angle():
    actual=np.array([[0.,0.],[1e-8,-1e-8],[1.,0.],[-1.,0.]])
    command=np.array([[0.,0.],[1.,0.],[1.,0.],[1.,0.]])
    result=audit.directions(actual,command)
    assert result['actual_near_zero_ticks']==2 and result['command_near_zero_ticks']==1
    assert result['both_directions_resolvable']==2 and result['vector_positive_fraction']==.5
    assert result['vector_cosine_quantiles']['0.0']==-1 and result['vector_cosine_quantiles']['1.0']==1


def test_empty_stage_does_not_manufacture_zero_error():
    r=record();summary=audit.collect_summary({'x':r},['x'],[np.zeros(80,bool)],RECOVERY_ONSET_WINDOW)
    assert summary['legal_expert_ticks']==0
    assert summary['one_control_step']['arm']['A_q_next_minus_q_before']['rmse_rad'] is None
    assert summary['one_control_step']['arm']['increment_direction']['vector_cosine_quantiles'] is None


def test_command_or_clock_corruption_is_rejected():
    r=record();r['u_expert'][50,0]+=.01
    with pytest.raises(ValueError,match='issued target'):audit.cached_arrays(r)
    r=record();r['time'][50]+=.0001
    with pytest.raises(ValueError,match='interval'):audit.cached_arrays(r)


def test_phase_reader_confirms_actual_before_after_units_clock_and_joint_order(tmp_path):
    from test_batch_collection_review import numeric
    numeric(tmp_path,60)
    with h5py.File(tmp_path/'numeric.h5','r+') as stream:
        stream.attrs['metadata_json']=json.dumps(dict(physics_dt=1/120,control_dt=1/60,
            joint_order=[f'iiwa14_joint_{i}' for i in range(1,8)]+[f'hand{i}' for i in range(22)]))
        t=stream['episode/transition'];t['diagnostics_before/joint_position_rad']=np.zeros((80,29))
        r=dict(robot=torch.zeros(80,178),q_reached=torch.zeros(80,29),u_expert=torch.zeros(80,29),
            command=torch.zeros(80,29),time=torch.from_numpy(t['time_before_s'][:]),
            supervision_mask=torch.from_numpy(t['dagger/expert_label_valid'][:]))
    masks,proof=audit.phase_masks(tmp_path/'numeric.h5',r,dict(takeover_tick=60))
    assert masks['legal_expert'].sum()==18 and proof['metadata_control_dt_s']==1/60
    assert masks['held_near_hole60'].sum()==18
    with h5py.File(tmp_path/'numeric.h5','r+') as stream:
        metadata=json.loads(stream.attrs['metadata_json']);metadata['control_dt']=1/30
        stream.attrs['metadata_json']=json.dumps(metadata)
    with pytest.raises(ValueError,match='metadata timing'):audit.phase_masks(tmp_path/'numeric.h5',r,dict(takeover_tick=60))
