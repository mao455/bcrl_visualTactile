"""Control, causal batching and reload regressions; fixtures are not demonstrations."""
import copy
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_perception_v2 import packet
from visuotactile.observations import nominal_keys
from visuotactile.history import reset_state
from visuotactile.models.nominal import PolicyConfig
from visuotactile.models.tactile import FingerTactileEncoder
from visuotactile.train_dp import _make_model, _condition, _save
from visuotactile.rollout import load_command_controller


def chunk(length=6):
    packets = [packet(t=i, views=3) for i in range(length)]
    obs = {k:torch.stack([p[k] for p in packets], 1) for k in nominal_keys(2)}
    obs['valid'] = torch.ones(2, length, dtype=torch.bool)
    names = ('contact_substeps_w_N','contact_substeps_local_N','finger_position_base_m',
             'finger_rotation_base','contact_time_s')
    return dict(obs=obs, tactile={k:torch.stack([p[k] for p in packets],1) for k in names},
                tactile_valid=torch.stack([p['contact_valid'] for p in packets],1))


def sliced(data, rows, times):
    return {k:({n:x[rows,times] for n,x in v.items()} if isinstance(v,dict) else v[rows,times])
            for k,v in data.items()}


def compare_state(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: compare_state(a[k],b[k])
    else: torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('variant',['b0','b1'])
def test_causal_packing_swap_padding_and_row_reset(variant):
    cfg=PolicyConfig(views=3,own_plan_mode='none')
    model=_make_model(variant,cfg).eval(); data=chunk()
    data['obs']['valid'][0,4:]=False
    data['tactile_valid'][0,4:]=False
    with torch.no_grad():
        full,state=_condition(model,variant,data,None)
        carried=None; parts=[]
        for lo,hi in ((0,2),(2,4),(4,6)):
            out,carried=_condition(model,variant,sliced(data,slice(None),slice(lo,hi)),carried)
            parts.append(out)
        torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-6,rtol=2e-5)
        compare_state(state,carried)
        for row,n in ((0,4),(1,6)):
            one,one_state=_condition(model,variant,sliced(data,slice(row,row+1),slice(0,n)),None)
            torch.testing.assert_close(full[row,:n],one[0],atol=2e-6,rtol=2e-5)
            def select(s): return {k:select(v) for k,v in s.items()} if isinstance(s,dict) else s[row:row+1]
            compare_state(select(state),one_state)
        swapped,_=_condition(model,variant,sliced(data,[1,0],slice(None)),None)
        torch.testing.assert_close(full,swapped.flip(0),atol=2e-6,rtol=2e-5)
        changed=copy.deepcopy(data)
        changed['obs']['image'][:,3:]=.9
        changed['tactile']['contact_substeps_w_N'][:,3:]*=15
        out,_=_condition(model,variant,changed,None)
        torch.testing.assert_close(full[:,:3],out[:,:3],atol=0,rtol=0)
        cleared=reset_state(state,torch.tensor([True,False]))
        def unchanged(s,c):
            if isinstance(s,dict):
                for k in s:unchanged(s[k],c[k])
            else:
                assert not c[0].any()
                torch.testing.assert_close(s[1],c[1],atol=0,rtol=0)
        unchanged(state,cleared)


def test_tactile_reference_with_initial_missing_and_padded_tail():
    data=chunk()['tactile']; names=dict(contact_substeps_w_N='force_w_N',contact_substeps_local_N='force_local_N',
        finger_position_base_m='position_base_m',finger_rotation_base='rotation_base',contact_time_s='time_s')
    tactile={names[k]:v.flatten(1,2) for k,v in data.items()}
    valid=torch.tensor([[False,False,True,True,True,True,False,False,False,False,False,False]]*2)
    model=FingerTactileEncoder()
    _,state=model(tactile,valid,return_state=True)
    torch.testing.assert_close(state['previous']['time_s'],tactile['time_s'][:,5],atol=0,rtol=0)
    _,tail=model({k:v[:,:2]*0 for k,v in tactile.items()},torch.zeros(2,2,dtype=torch.bool),state,return_state=True)
    compare_state(state,tail)


@pytest.mark.parametrize('variant',['b0','b1'])
def test_checkpoint_actual_controller_online_no_plan_rng_and_causal_state(tmp_path,variant):
    cfg=PolicyConfig(views=3,own_plan_mode='none');model=_make_model(variant,cfg).eval()
    opt=torch.optim.AdamW(model.parameters())
    args=SimpleNamespace(variant=variant,chunk_size=2,batch_size=2,seed=41,amp='off')
    _save(tmp_path/'model.pt',model,opt,cfg,args,'fixture',(['a'],['b'],['c']),1,[],
          torch.zeros(58),torch.ones(58),metadata={},optimizer_updates=1)
    before=torch.get_rng_state().clone(); controller=load_command_controller(tmp_path/'model.pt',2)
    torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)
    data=chunk(8)
    with torch.no_grad():
        cond,state=_condition(model,variant,data,None)
        for t in range(8):
            p=packet(t=t,views=3)
            p['prior_plan'].fill_(777);p['previous_nominal_action'].fill_(777)
            p['prior_plan_valid'].fill_(True);p['previous_nominal_valid'].fill_(True)
            action=controller.act(p)
            assert action.shape==(2,29) and action.abs().max()<=1
            assert not controller.last_packet['prior_plan_valid'].any()
            assert not controller.last_packet['previous_nominal_valid'].any()
            torch.testing.assert_close(controller.condition_rows(torch.tensor([True,True])),cond[:,t],atol=2e-6,rtol=2e-5)
            assert controller.replanned.all()==(t%6==0)
        compare_state(state,controller.memory)
    torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)


def test_activation_recomputation_keeps_features_and_parameter_gradients():
    data=chunk(2)['obs']
    plain=_make_model('b0',PolicyConfig(views=3)).encoder
    bounded=_make_model('b0',PolicyConfig(views=3,vision_checkpoint_images=2)).encoder
    bounded.load_state_dict(plain.state_dict())
    a=plain(data);b=bounded(data)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    a.square().sum().backward();b.square().sum().backward()
    for (name,p),(_,q) in zip(plain.named_parameters(),bounded.named_parameters(),strict=True):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=5e-4)


def test_future_tactile_rejected_and_b0_rejects_extra_fields():
    data=chunk(2)
    data['tactile']['contact_time_s']+=10
    with pytest.raises(ValueError,match='future'):
        _condition(_make_model('b1',PolicyConfig(views=3)),'b1',data,None)
    data['obs']['expert_action']=torch.zeros(2,2,29)
    with pytest.raises(ValueError,match='allowlist'):
        _condition(_make_model('b0',PolicyConfig(views=3)),'b0',data,None)


def test_resume_checkpoint_restores_optimizer_rng_and_next_update(tmp_path):
    cfg=PolicyConfig(views=3,own_plan_mode='none',vision_checkpoint_images=2)
    model=_make_model('b0',cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    args=SimpleNamespace(variant='b0',chunk_size=2,batch_size=2,seed=41,amp='off')
    data=chunk(2)
    # Unit fixture commands only; these never enter formal demonstration data.
    target=torch.linspace(-.5,.5,4*12*29).reshape(4,12,29)
    def step(m,o):
        o.zero_grad(set_to_none=True)
        cond,_=_condition(m,'b0',data,None)
        loss=m.loss_from_condition(cond.flatten(0,1),target,torch.ones(4,12,dtype=torch.bool))
        loss.backward();o.step()
        return loss.detach()
    step(model,opt)
    _save(tmp_path/'resume.pt',model,opt,cfg,args,'fixture',(['a'],[],[]),0,[],torch.zeros(58),torch.ones(58),
          metadata={},progress={'next_group':1},optimizer_updates=1)
    expected=step(model,opt)
    restored=_make_model('b0',cfg);other=torch.optim.AdamW(restored.parameters(),lr=1e-4)
    state=torch.load(tmp_path/'resume.pt',weights_only=False)
    restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    actual=step(restored,other)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for a,b in zip(model.parameters(),restored.parameters(),strict=True):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.parametrize('variant',['b0','b1'])
def test_sampled_metrics_replan_phase_and_episode_tail_masks(monkeypatch, variant):
    from visuotactile.command_metrics import sampled_command_metrics
    import visuotactile.train_dp as training
    # Deterministic metric oracle only, never a policy training example.
    batches=[]
    for start in (0,4):
        current=torch.arange(start,start+4)[None] < torch.tensor([5,8])[:,None]
        target_valid=(torch.arange(start,start+4)[None,:,None]+torch.arange(12)
                      < torch.tensor([5,8])[:,None,None])
        batches.append(dict(start=start,episode_start=start==0,obs={'valid':current},
            action_target=torch.zeros(2,4,12,29), action_valid=target_valid,
            tactile={k:torch.ones(2,4,2,5,3) for k in ('contact_substeps_w_N','contact_substeps_local_N')}))
    monkeypatch.setattr(training,'_batch_episode_chunks',lambda *a:iter(batches))
    monkeypatch.setattr(training,'_condition',lambda m,v,d,s:(torch.zeros(2,4,128),s))
    class Oracle:
        cfg=PolicyConfig(views=3)
        def eval(self):return self
        def predict_from_condition(self,cond,generator):return torch.ones(len(cond),12,29)
    result=sampled_command_metrics(Oracle(),None,[],variant,torch.device('cpu'),4,2)
    assert result['replans']==3  # t0 in both streams, t6 in the longer one
    assert result['horizon_valid_targets']==[3,3,2,2,2,1,1,1,0,0,0,0]
    assert result['arm_rmse']==result['hand_rmse']==1.
    assert result['zero_force_rmse']==(1. if variant=='b1' else None)
