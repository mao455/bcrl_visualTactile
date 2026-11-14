"""Formal metrics opt-in preserves legal slots, real prefix and optimizer/RNG dose."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import tight_fresh_train as fresh
from visuotactile.anchor_sampling import planning_anchors
from visuotactile.offline_planning import profile
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW
from test_fresh_parallel_backends import CycleFixture


class TinyPolicy(torch.nn.Module):
    horizon=24
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2))
        self.register_buffer('action_scale',torch.linspace(.5,2.,29))
        self.contract=dict(normalization_identity='fixed_training_affine')
        self.draws=[]
    def predict(self,condition,generator,*,inference_steps=16,clip_sample=False,scheduler_index_mode='host'):
        assert inference_steps==16 and not clip_sample and scheduler_index_mode=='host'
        generators=generator if isinstance(generator,list) else [generator]
        noise=torch.cat([torch.randn((1,24,29),generator=g) for g in generators])
        self.draws.extend(noise.detach().unbind())
        return noise+condition[:,:1,None]


def record():
    n=145
    return dict(time=torch.arange(n,dtype=torch.float64)/60,history=torch.arange(n).float()/n,
        command=torch.zeros(n,29),clean_sample=torch.zeros(n,29),
        supervision_mask=(torch.arange(n)>=82)&~((torch.arange(n)>=108)&(torch.arange(n)<120)),
        control_segment=(torch.arange(n)>=80).long()+(torch.arange(n)>=82).long()+
            (torch.arange(n)>=108).long()+(torch.arange(n)>=120).long(),
        target_window_contract=RECOVERY_ONSET_WINDOW,joint_limits=torch.tensor([-10.,10.]).repeat(29,1))


def conditions(model,rec,**kwargs):
    # The full prefix, including illegal student-label ticks, changes the state.
    memory=(rec['history']*model.weight).cumsum(0)
    anchors=planning_anchors(rec)
    return memory[anchors,None],anchors


@pytest.mark.parametrize('batch',[1,8,32])
def test_metrics_keep_labels_noise_and_full_prefix_with_actual_masks(monkeypatch,batch):
    monkeypatch.setattr(fresh,'supervised_conditions',conditions)
    rec=record();serial=TinyPolicy();candidate=TinyPolicy();before=torch.get_rng_state().clone()
    expected=fresh.metrics(serial,{'correction':rec},['correction'])
    actual=fresh.metrics(candidate,{'correction':rec},['correction'],offline_planning=profile(batch))
    for key in ('arm_rmse_rad','hand_rmse_rad','selection'):assert actual[key]==expected[key]
    assert torch.equal(torch.stack(serial.draws),torch.stack(candidate.draws))
    assert torch.equal(before,torch.get_rng_state())
    trace=actual['episodes'][0]['offline_planning']
    assert trace['skipped_noise_draws']>0 and trace['first_anchor']==72
    assert trace['full_history_frames']==len(rec['time']) and trace['predict_calls']<=trace['supervised_anchors']
    assert 'offline_planning' not in expected and 'offline_planning' not in expected['episodes'][0]
    altered=copy.deepcopy(rec);altered['history'][:72]+=.1
    changed=fresh.metrics(candidate,{'correction':altered},['correction'],offline_planning=profile(batch))
    assert changed['selection']!=actual['selection'], 'Cached learned history would ignore the changed real prefix'
    assert not rec['supervision_mask'][:82].any()


def test_metric_deadline_stops_before_history_or_prediction(monkeypatch):
    monkeypatch.setattr(fresh,'supervised_conditions',lambda *a:pytest.fail('Checked deadline must precede encoding'))
    def stop():raise TimeoutError('registered session')
    with pytest.raises(TimeoutError):fresh.metrics(TinyPolicy(),{'x':record()},['x'],offline_planning=profile(32),check=stop)


@pytest.mark.parametrize('parallel',[False,True])
@pytest.mark.parametrize('front,back',[(None,None),(32,None),(None,8),(32,8)])
def test_cycle_registers_and_dispatches_front_and_backend_independently(tmp_path,monkeypatch,parallel,front,back):
    fixture=CycleFixture(tmp_path,monkeypatch,frontend_parallel=parallel,backend_parallel=parallel)
    fixture.run(parallel_frontends=parallel,parallel_backends=parallel,
        offline_anchor_batch=front,backend_offline_anchor_batch=back)
    registration=json.loads((fixture.ledger/'registration.json').read_text())
    assert ('frontend_offline_planning' in registration)==(front is not None)
    assert ('backend_offline_planning' in registration)==(back is not None)
    if front is not None:assert registration['frontend_offline_planning']==profile(front)
    if back is not None:assert registration['backend_offline_planning']==profile(back)
    for name,command in fixture.calls:
        expected=front if name.startswith('frontend_') else back if name.startswith('backend_prepare_') else None
        assert ('--offline-anchor-batch' in command)==(expected is not None)
        if expected is not None:assert int(command[command.index('--offline-anchor-batch')+1])==expected


def test_actual_cpu_training_checkpoint_binds_optin_without_changing_updates_rng_or_parameters(tmp_path,monkeypatch):
    from visuotactile import state_resources,precision,visual_aux_training,anchor_continue
    from visuotactile.training_resources import atomic_json
    # Initialize optimizer support before substituting only requested generator devices.
    torch.optim.AdamW(TinyPolicy().parameters())
    real_generator=torch.Generator
    monkeypatch.setattr(torch,'Generator',lambda device=None:real_generator(device='cpu'))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**k:None)
    monkeypatch.setattr(precision,'state_fp32',lambda:None)
    checks=[]
    lease=SimpleNamespace(block=lambda:nullcontext(),check=lambda:checks.append(True))
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**k:lease)
    monkeypatch.setattr(visual_aux_training,'load_auxiliary_spec',lambda *a:None)
    monkeypatch.setattr(fresh,'supervised_conditions',conditions)
    monkeypatch.setattr(fresh,'new_model',lambda *a:TinyPolicy())
    monkeypatch.setattr(anchor_continue,'projection_batch',lambda *a:None)
    class Objective:
        def forward(self,model,condition,truth,valid,noise,**kwargs):
            draw=torch.randn((1,),generator=noise)
            loss=(condition.mean()+draw*.01).square().sum()
            return loss,dict(action=loss),None
    monkeypatch.setattr(fresh,'fresh_objective',lambda *a:(Objective(),None,None))
    data=dict(splits=dict(train=['x'],validation=['v']))
    records={'x':record(),'v':record()}
    monkeypatch.setattr(fresh,'inputs',lambda root,role:(root/role/'state_data',data,records,{}))
    monkeypatch.chdir(tmp_path)
    outputs=[]
    for enabled in (False,True):
        root=tmp_path/str(enabled);dr=root/'B/state_data';dr.mkdir(parents=True)
        atomic_json(root/'gate.json',dict(passed=True));atomic_json(root/'B/cache_index.json',dict(complete=True))
        atomic_json(root/'session.json',dict(deadline=9999999999,dino_sha256='frozen'))
        atomic_json(dr/'data.json',data);torch.save({},dr/'statistics.pt')
        fresh.train(root,'B',2,**(dict(offline_anchor_batch=32) if enabled else {}))
        saved=torch.load(root/'B/training/last.pt',map_location='cpu',weights_only=False)
        assert ('offline_planning' in saved['configuration'])==enabled
        if enabled:assert saved['configuration']['offline_planning']==profile(32)
        outputs.append(saved)
        if enabled:
            with pytest.raises(ValueError,match='source/config changed'):fresh.train(root,'B',2)
    from test_paired_episode_io import equal_tree
    for key in ('model','ema','optimizer','schedule_rng','noise_rng','updates','anchors_seen','target_ticks','ema_optimization_step'):
        equal_tree(outputs[0][key],outputs[1][key])
    assert outputs[1]['updates']==2 and checks


@pytest.mark.parametrize('option',['offline_anchor_batch','backend_offline_anchor_batch'])
def test_cycle_invalid_batch_rejected_before_registration(tmp_path,option):
    from tools.tight_fresh_cycle import run
    with pytest.raises(ValueError,match='offline anchor batch'):
        run(tmp_path,tmp_path,tmp_path/'out',1,1,1,1,**{option:True})
    assert not (tmp_path/'out_stages').exists()
