"""V2 admission, causal cache parity, tail gradients and recovery boundaries."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from visuotactile.dp_objectives import masked_objectives
from visuotactile.native_v2_gates import Gate,replay_gate,rollout_gate,visual_gate,plateau,require_gate
from visuotactile.native_v2_state import StateWindows,StateBC,condition_windows,BC_CONTRACT,load_state_policy
from visuotactile.native_v2_training import next_batch
from visuotactile.native_data import Episode
from visuotactile.models.native_dp import NativePolicy,encode_segment

SOURCE=Path('outputs/visuotactile/native_dp_3way/native_20261101_02')


def replay_rows(n=8,success=True):
    return [dict(identity=str(i),success=success,comparison={'errors':{k:{'first_exceeded':None}
        for k in ('command','delayed','delay_index','target','force','torque')}}) for i in range(n)]


def test_replay_software_completion_cannot_pass_failure():
    g=replay_gate({'infrastructure_passed':True},replay_rows(success=False),[str(i) for i in range(8)])
    assert g.infrastructure_passed and not g.acceptance_passed and g.closed_loop_success['successes']==0


def test_replay_distinct_identity_required():
    rows=replay_rows();rows[-1]['identity']='0'
    assert not replay_gate({'infrastructure_passed':True},rows,[str(i) for i in range(8)]).infrastructure_passed


def test_replay_native_control_mismatch_blocks():
    rows=replay_rows();rows[0]['comparison']['errors']['target']['first_exceeded']=0
    assert not replay_gate({'infrastructure_passed':True},rows,[str(i) for i in range(8)]).acceptance_passed


def test_all_eight_expert_successes_required():
    rows=replay_rows();rows[0]['success']=False
    assert not replay_gate({'infrastructure_passed':True},rows,[str(i) for i in range(8)]).acceptance_passed
    assert replay_gate({'infrastructure_passed':True},replay_rows(),[str(i) for i in range(8)]).acceptance_passed


def test_visual_locked_on_state_zero():
    good=Gate('G0',True,True,None,True,None);bad=Gate('G2',True,True,{'successes':0},False,'zero')
    assert not visual_gate(good,good,bad,good,[dict(identity='a',success=True),dict(identity='b',success=True)]).acceptance_passed


def test_rollout_requires_full_native_termination():
    rows=[dict(identity=str(i),success=True,software_complete=True,native_termination=False) for i in range(8)]
    assert not rollout_gate('G2',rows,[str(i) for i in range(8)]).acceptance_passed


def test_gate_missing_or_false_blocks_updates(tmp_path):
    (tmp_path/'gates').mkdir();(tmp_path/'gates/G0.json').write_text(json.dumps({'name':'G0','acceptance_passed':False}))
    with pytest.raises(RuntimeError):require_gate(tmp_path,'G0')


def test_plateau_requires_several_milestones_and_stage_stagnation():
    rows=[dict(updates=k,successes=0,stage_score=1) for k in (1000,3000,6000)]
    assert not plateau(rows[:2]);assert plateau(rows)
    rows[-1]['stage_score']=2;assert not plateau(rows)


def test_resume_sampler_no_duplicate_or_missing_anchors():
    seen=[];epoch=cursor=0
    while len(seen)<101:
        indices,epoch,cursor=next_batch(101,epoch,cursor,32);seen.extend(indices.tolist())
    assert sorted(seen)==list(range(101))
    a,e,c=next_batch(101,epoch,cursor,32);b,e2,c2=next_batch(101,0,101,32)
    assert torch.equal(a,b) and (e,c)==(e2,c2)==(1,32)


def test_tail_padding_has_no_loss_or_gradient():
    prediction=torch.randn(2,64,29,requires_grad=True);target=torch.randn_like(prediction);valid=torch.zeros(2,64,dtype=torch.bool);valid[:,:3]=True
    a=masked_objectives(prediction,target,valid)['B'];a.backward();grad=prediction.grad.clone()
    assert (grad[:,3:]==0).all()
    target[:,3:]=float('nan');b=masked_objectives(prediction,target,valid)['B']
    assert torch.equal(a,b)


@pytest.mark.skipif(not (SOURCE/'data.json').exists(),reason='Local original dataset unavailable')
@pytest.mark.parametrize('kind',['dp','bc'])
def test_real_causal_windows_match_complete_online_history(kind):
    torch.set_num_threads(2);torch.manual_seed(17)
    ids=json.loads((SOURCE/'data.json').read_text())['splits']['train'][:1]
    windows=StateWindows(SOURCE,ids);model=NativePolicy('S-NativeDP') if kind=='dp' else StateBC()
    for k,v in windows.stats.items():getattr(model,k).copy_(v)
    model.eval();episode=Episode(SOURCE,ids[0],'S-NativeDP')
    with torch.no_grad():
        expected,anchors,_=encode_segment(model,episode,0,len(episode))
        actual=condition_windows(model,windows.robot,windows.geometry)
    assert torch.allclose(actual,expected,atol=2e-6,rtol=1e-5),(actual-expected).abs().max()
    assert [tick for _,tick in windows.samples]==anchors.tolist()
    # A future action must not enter an earlier condition; strict t-1 feedback checked against HDF5 in full audit.
    assert not windows.robot[0].reshape(13,180)[:-1].any()
    assert windows.robot.shape[1]+windows.geometry.shape[1]==2601


def test_bc_gradient_and_checkpoint_identity():
    torch.set_num_threads(2);model=StateBC();condition=torch.randn(2,2601)
    loss=model.loss(condition,torch.randn(2,64,29),torch.ones(2,64,dtype=torch.bool));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saved=dict(contract=BC_CONTRACT,ema=model.state_dict(),model=model.state_dict())
    read=load_state_policy(saved)
    assert torch.equal(model.predict(condition),read.predict(condition))


def test_formal_gate_never_allows_test_selection_or_duplicate_cohort():
    from visuotactile.native_v2_gates import final_gate
    gate=Gate('G4',True,True,None,True,None)
    selected={k:{'sha256':k} for k in ('S-NativeDP','I-OfficialDP','I-HistoryDP')}
    ids=[f'{i:03d}' for i in range(40)]
    rows=[dict(group=k,identity=i,checkpoint_sha256=k,software_complete=True,native_termination=True,success=True)
          for k in selected for i in ids[:20]]
    result=final_gate(gate,selected,rows,ids,configuration_frozen=True,prior_test_uses=['old_run'])
    assert result.acceptance_passed and not result.closed_loop_success['virgin_test']
    assert not final_gate(gate,selected,rows,ids,configuration_frozen=False,prior_test_uses=[]).acceptance_passed
    rows[0]['identity']=rows[1]['identity']
    assert not final_gate(gate,selected,rows,ids,configuration_frozen=True,prior_test_uses=[]).infrastructure_passed


def test_resource_wait_and_pacing_never_hold_gpu_lock(tmp_path,monkeypatch):
    from visuotactile.native_v2_resources import LeaseV2
    import visuotactile.native_v2_resources as resource
    lease=LeaseV2(tmp_path,deadline=9999999999)
    lease.check=lambda *args:None;lease.memory=lambda cuda: {}
    row=dict(utc_epoch=1.,gpu_util_percent=0,rolling_60s_percent=0)
    attempts=iter([(False,row),(True,row)]);lease.admission=lambda:next(attempts)
    held=[False];sleeps=[]
    def flock(fd,operation):held[0]=operation!=resource.fcntl.LOCK_UN
    def sleep(duration):
        assert not held[0];sleeps.append(duration)
    monkeypatch.setattr(resource.fcntl,'flock',flock);monkeypatch.setattr(resource.time,'sleep',sleep)
    with lease.block(cuda=False):assert held[0]
    assert not held[0] and sleeps


def test_gpu_lock_released_after_computation_error(tmp_path,monkeypatch):
    from visuotactile.native_v2_resources import LeaseV2
    import visuotactile.native_v2_resources as resource
    lease=LeaseV2(tmp_path);lease.check=lambda *args:None;lease.memory=lambda cuda:{}
    lease.admission=lambda:(True,dict(utc_epoch=1.,gpu_util_percent=0,rolling_60s_percent=0))
    held=[False]
    monkeypatch.setattr(resource.fcntl,'flock',lambda fd,op:held.__setitem__(0,op!=resource.fcntl.LOCK_UN))
    with pytest.raises(ValueError):
        with lease.block(cuda=False):raise ValueError('test')
    assert not held[0]


def test_visual_repeat_success_cannot_be_borrowed_from_other_checkpoint():
    common=Gate('G0',True,True,None,True,None)
    state=Gate('G3',True,True,{'checkpoint_sha256':'new'},True,None)
    rows=[dict(identity=str(i),success=True,checkpoint_sha256='old') for i in range(2)]
    assert not visual_gate(common,common,state,state,rows).acceptance_passed
    for row in rows:row['checkpoint_sha256']='new'
    assert visual_gate(common,common,state,state,rows).acceptance_passed


def test_predecessor_wait_does_not_reset_absolute_24h_budget():
    from visuotactile.native_v2_pipeline import gpu_deadlines
    session={'session_deadline':24*3600,'config':{'diagnostic_window_hours':4,'training_window_hours':16}}
    diagnostic,absolute=gpu_deadlines(session,10*3600)
    assert diagnostic==14*3600 and absolute==24*3600
    diagnostic,absolute=gpu_deadlines(session,23*3600)
    assert diagnostic==absolute==24*3600


def test_configuration_cannot_silently_claim_different_action_contract():
    from visuotactile.native_v2_pipeline import validate_config
    config=json.loads(Path('configs/native_dp_v2.json').read_text());validate_config(config)
    config['state_training']['prediction_type']='epsilon'
    with pytest.raises(ValueError,match='prediction_type'):validate_config(config)
