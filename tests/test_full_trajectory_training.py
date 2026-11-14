"""Actual CPU AdamW/EMA transactions and durable full-epoch recovery for B2."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from tools import tight_fresh_train as fresh
from visuotactile import trajectory_training as training
from visuotactile.training_receipts import iter_visits,learning_counters,BATCH_ROW_VERSION
from visuotactile.training_resources import atomic_json
from visuotactile.models.vendor.ema_model import EMAModel
from test_paired_episode_io import equal_tree


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(4,29);self.scale=torch.nn.Parameter(torch.tensor(.2))
    def forward(self,sample,timestep,global_cond):
        return sample*self.scale+self.linear(global_cond)[:,None]+timestep[:,None,None]/100


class TinyNative(torch.nn.Module):
    horizon=24;modality='G00';group='I-HistoryDP'
    def __init__(self):
        super().__init__()
        self.contract=dict(visual_stride=1,normalization_identity='fixed_toy_affine')
        self.vision=torch.nn.Linear(3,4)
        self.native_input=torch.nn.Linear(178,4)
        self.native_gru=torch.nn.GRUCell(4,4);self.visual_gru=torch.nn.GRUCell(4,4)
        self.robot_encoder=torch.nn.Linear(13*180,4)
        self.condition_projection=torch.nn.Linear(16,4);self.denoiser=Denoiser()
        self.register_buffer('action_scale',torch.ones(29));self.register_buffer('action_offset',torch.zeros(29))
        self.noise=DDPMScheduler(num_train_timesteps=100,prediction_type='sample',clip_sample=False)
    def normalized_robot(self,robot):return robot


class Objective:
    gain=1.;weights={'toy_geometry':.1};last_geometry_metrics=None
    def components(self,prediction,truth,valid,scale,offset,context):
        self.last_geometry_metrics=dict(selected=len(context))
        base=(prediction-truth*scale-offset)[valid].square().mean()
        return dict(base=base,toy_geometry=prediction[:,0,:7].square().mean())


def make_records():
    records={};episodes={}
    for j,(identity,n) in enumerate([('x',13),('y',25),('z',37),('v',13)]):
        generator=torch.Generator().manual_seed(101+j)
        time=torch.arange(n,dtype=torch.float64)/60
        records[identity]=dict(time=time,robot=torch.randn(n,178,generator=generator)*.01,
            command=torch.randn(n,29,generator=generator)*.01,
            clean_sample=torch.zeros(n,29),supervision_mask=torch.arange(n)>=3,
            control_segment=torch.zeros(n,dtype=torch.long))
        episodes[identity]=SimpleNamespace(visual_stride=1,record=dict(camera_time=time[:,None].expand(-1,3),
            camera_valid=torch.ones(n,3,dtype=torch.bool)),
            perception=dict(patches=torch.randn(n,3,generator=generator),depth=torch.zeros(n,1)))
    return records,episodes


def install_small_vision(monkeypatch):
    from visuotactile import trajectory_batch,anchor_continue
    # All real recurrent transitions remain; only image dimensions are shrunk.
    monkeypatch.setattr(trajectory_batch,'frame_features',
        lambda model,patches,depth,age,valid:(model.vision(patches),model.vision(patches)))
    monkeypatch.setattr(trajectory_batch,'causal_visual_readouts',lambda model,tokens,time,ct,cv,a:tokens[a])
    monkeypatch.setattr(anchor_continue,'projection_batch',lambda objective,selection,device:selection)


def test_adjacent_deck_keeps_duplicates_and_singleton_with_explicit_changed_budget():
    assert training.batches(['x','x','y'])==[[(1,'x'),(2,'x')],[(3,'y')]]
    assert training.budget(137,10)['planned_episode_visits']==1370
    assert training.budget(137,10)['planned_optimizer_updates']==690
    assert training.budget(137,10)['planned_ema_updates']==690
    assert training.validate_profile(None) is None
    for value in (True,{},dict(training.profile(),batch_size=4),dict(training.profile(),extra=True)):
        with pytest.raises(ValueError):training.validate_profile(value)


def test_actual_mean_gradient_one_adamw_ema_and_independent_duplicate_history(monkeypatch):
    from visuotactile.trajectory_batch import supervised_trajectories,trajectory_objective
    install_small_vision(monkeypatch);torch.manual_seed(42)
    records,episodes=make_records();candidate=TinyNative();reference=copy.deepcopy(candidate)
    optimizer=torch.optim.AdamW(candidate.parameters(),lr=1e-3)
    expected_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-3)
    averaged=copy.deepcopy(candidate);expected_averaged=copy.deepcopy(reference)
    ema=EMAModel(averaged,power=.75);expected_ema=EMAModel(expected_averaged,power=.75)
    actual_noise=torch.Generator().manual_seed(22);expected_noise=torch.Generator().manual_seed(22)
    # Same episode twice is two complete prefix reconstructions, never one cache reuse.
    ids=['x','x'];recs=[records[i] for i in ids];eps=[episodes[i] for i in ids]
    observed=[]
    hook=candidate.native_gru.register_forward_pre_hook(lambda m,args:observed.append(len(args[0])))
    result=training.execute_update(candidate,optimizer,ema,Objective(),ids,recs,eps,actual_noise)
    hook.remove()
    losses=[]
    for rec,ep in zip(recs,eps):
        value=supervised_trajectories(reference,[rec],[ep])
        loss,_,_=trajectory_objective(reference,Objective(),value,expected_noise,[value[0]['anchors'].tolist()])
        losses.append(loss)
    torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
    expected_optimizer.step();expected_ema.step(reference)
    assert observed==[2]*13
    assert ema.optimization_step==1
    assert {int(v['step']) for v in optimizer.state.values()}=={1}
    assert [v['loss_weight'] for v in result['visits']]==[.5,.5]
    assert [v['visual_frames'] for v in result['visits']]==[13,13]
    assert torch.equal(actual_noise.get_state(),expected_noise.get_state())
    for a,b in zip(candidate.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    for a,b in zip(averaged.parameters(),expected_averaged.parameters()):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    # A real singleton is a second update, without dropping or cloning its visit.
    result=training.execute_update(candidate,optimizer,ema,Objective(),['z'],[records['z']],[episodes['z']],actual_noise)
    assert ema.optimization_step==2 and result['visits'][0]['loss_weight']==1
    assert {int(v['step']) for v in optimizer.state.values()}=={2}


def setup_trainer(tmp_path,monkeypatch):
    from visuotactile import state_resources,precision,visual_aux_training
    install_small_vision(monkeypatch)
    torch.optim.AdamW(TinyNative().parameters())
    real_generator=torch.Generator
    monkeypatch.setattr(torch,'Generator',lambda device=None:real_generator(device='cpu'))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**k:None)
    monkeypatch.setattr(precision,'state_fp32',lambda:None)
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**k:SimpleNamespace(block=lambda:nullcontext(),check=lambda:None))
    monkeypatch.setattr(visual_aux_training,'load_auxiliary_spec',lambda *a:None)
    def model(*args):
        torch.manual_seed(333);return TinyNative()
    monkeypatch.setattr(fresh,'new_model',model)
    monkeypatch.setattr(fresh,'fresh_objective',lambda *a:(Objective(),None,None))
    monkeypatch.setattr(fresh,'metrics',lambda model,*a,**k:dict(selection=float(sum(p.square().sum() for p in model.parameters()))))
    records,episodes=make_records()
    data=dict(splits=dict(train=['x','y','z'],validation=['v']))
    monkeypatch.setattr(fresh,'inputs',lambda root,role:(root/role/'state_data',data,records,{}))
    monkeypatch.setattr(fresh,'Episode',lambda root,identity,group:copy.deepcopy(episodes[identity]))
    monkeypatch.chdir(tmp_path)
    def prepare(name):
        root=tmp_path/name;dr=root/'B/state_data';dr.mkdir(parents=True)
        atomic_json(root/'gate.json',dict(passed=True));atomic_json(root/'B/cache_index.json',dict(complete=True))
        atomic_json(root/'session.json',dict(deadline=9999999999,dino_sha256='frozen'))
        atomic_json(dr/'data.json',data);torch.save({},dr/'statistics.pt')
        return root
    return prepare


def test_formal_b2_train_restart_restores_optimizer_ema_rng_and_excludes_abandoned_work(tmp_path,monkeypatch):
    prepare=setup_trainer(tmp_path,monkeypatch);complete=prepare('complete');interrupted=prepare('interrupted')
    kwargs=dict(trajectory_batch_profile=training.profile())
    fresh.train(complete,'B',2,**kwargs)
    original=training.execute_update;calls=[]
    def crash(*a,**k):
        calls.append(True)
        if len(calls)==4:raise InterruptedError('CPU injected second-epoch singleton failure')
        return original(*a,**k)
    monkeypatch.setattr(training,'execute_update',crash)
    with pytest.raises(InterruptedError):fresh.train(interrupted,'B',2,**kwargs)
    out=interrupted/'B/training'
    durable=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    assert durable['updates']==2 and durable['visits_seen']==3
    assert len((out/'learning.jsonl').read_text().splitlines())==3
    assert not (out/'finished.json').exists()
    monkeypatch.setattr(training,'execute_update',original)
    fresh.train(interrupted,'B',2,**kwargs)
    saved=[]
    for root in (complete,interrupted):
        saved.append(torch.load(root/'B/training/last.pt',map_location='cpu',weights_only=False))
        final=json.loads((root/'B/training/finished.json').read_text())
        assert final['updates']==4 and final['ema_updates']==4 and final['visits_seen']==6
        assert final['planned_optimizer_updates']==4 and final['planned_episode_visits']==6
        rows=[json.loads(line) for line in (root/'B/training/learning.jsonl').read_text().splitlines()]
        assert [len(row['visits']) for row in rows]==[2,1,2,1]
        assert learning_counters(rows)['visits_seen']==6
        assert final['sampling_exposure']==dict(x=2,y=2,z=2)
        assert all(row['full_trajectory_batch']==training.profile() for row in rows)
    for key in ('model','ema','optimizer','schedule_rng','noise_rng','updates','anchors_seen','target_ticks',
                'ema_optimization_step','visits_seen','sampling_exposure','torch_rng'):
        equal_tree(saved[0][key],saved[1][key])
    archives=list(out.glob('learning.abandoned.*.jsonl'))
    assert len(archives)==1 and json.loads(archives[0].read_text())['updates']==3
    with pytest.raises(ValueError,match='source/config changed'):fresh.train(interrupted,'B',2)
    with pytest.raises(ValueError,match='source/config changed'):fresh.train(interrupted,'B',3,**kwargs)


@pytest.mark.parametrize('name',['paired_episode_io.json','B/episode_io.json'])
def test_incompatible_io_rejected_before_entering_trainer(tmp_path,monkeypatch,name):
    path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('{}')
    monkeypatch.setattr(fresh,'_train',lambda *a,**k:pytest.fail('Invalid B2 must not enter training'))
    with pytest.raises(ValueError):fresh.train(tmp_path,'B',2,trajectory_batch_profile=training.profile())


def test_registered_b2_cannot_silently_drop_explicit_method(tmp_path,monkeypatch):
    atomic_json(tmp_path/'full_trajectory_batch.json',dict(profile=training.profile(),roles={}))
    monkeypatch.setattr(fresh,'_train',lambda *a,**k:pytest.fail('Missing explicit method'))
    with pytest.raises(ValueError,match='registered B2'):fresh.train(tmp_path,'B',2)


@pytest.mark.parametrize('change',['swap_visits','deck_content','row_digest','missing_deck'])
def test_resume_binds_actual_visit_order_to_saved_epoch_deck(tmp_path,monkeypatch,change):
    prepare=setup_trainer(tmp_path,monkeypatch);root=prepare('source');kwargs=dict(trajectory_batch_profile=training.profile())
    fresh.train(root,'B',1,**kwargs)
    out=root/'B/training';path=out/'learning.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if change=='swap_visits':
        first,second=rows[0]['visits'];first['episode'],second['episode']=second['episode'],first['episode']
        # Same per-episode totals cannot legitimize a different update ordering.
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change=='row_digest':
        rows[0]['deck_sha256']='0'*64
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change=='deck_content':
        deck=json.loads((out/'deck_001.json').read_text());deck['order'].reverse()
        atomic_json(out/'deck_001.json',deck)
    else:(out/'deck_001.json').unlink()
    with pytest.raises(ValueError,match='deck'):fresh.train(root,'B',1,**kwargs)


def receipt_rows():
    visits=[dict(episode='x',anchors=2,target_ticks=14,visual_frames=13,loss_weight=.5),
        dict(episode='x',anchors=2,target_ticks=14,visual_frames=13,loss_weight=.5)]
    return [dict(version=BATCH_ROW_VERSION,full_trajectory_batch=training.profile(),updates=1,
        anchors_seen=4,target_ticks=28,visual_frames=26,visits_seen=2,visits=visits),
        dict(version=BATCH_ROW_VERSION,full_trajectory_batch=training.profile(),updates=2,
        anchors_seen=7,target_ticks=55,visual_frames=25,visits_seen=3,
        visits=[dict(episode='y',anchors=3,target_ticks=27,visual_frames=25,loss_weight=1.)])]


def test_receipts_preserve_duplicate_visits_and_use_optimizer_selection_limit():
    rows=receipt_rows()
    assert [v['episode'] for v in iter_visits(rows)]==['x','x','y']
    assert learning_counters(rows,limit=1)==dict(updates=1,anchors_seen=4,target_ticks=28,visits_seen=2)
    legacy=[dict(updates=1,episode='x',anchors_seen=4,target_ticks=28,visual_frames=13)]
    assert learning_counters(legacy)==dict(updates=1,anchors_seen=4,target_ticks=28,visits_seen=1)


@pytest.mark.parametrize('change',[
    lambda r:r[0].update(visits_seen=1),lambda r:r[1].update(updates=3),
    lambda r:r[1].update(visual_frames=True),lambda r:r[1]['visits'][0].update(loss_weight=.5),
    lambda r:r[0]['visits'][0].update(target_ticks=13),lambda r:r[0].update(version='unknown')])
def test_receipt_drift_is_not_silently_counted(change):
    rows=receipt_rows();change(rows)
    with pytest.raises(ValueError):list(iter_visits(rows))
