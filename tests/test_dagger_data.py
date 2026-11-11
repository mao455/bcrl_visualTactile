"""Actual-history/label boundaries for shared Core + takeover data loaders."""
import json

import h5py
import numpy as np
import pytest
import torch

from visuotactile.anchor_sampling import AnchorSampler, labels, mask_totals, supervision_population
from visuotactile.compact import sha256
from visuotactile.native_data import residual_batch, residual_supervision_mask, supervision_mask_sha256, visual_records
from visuotactile.narrow_data import build_dagger_aggregate, dagger_record


def takeover(tmp_path,n=40):
    path=tmp_path/'takeover.h5';time=np.arange(n,dtype=np.float64)/60
    target=np.arange(n*29,dtype=np.float32).reshape(n,29)/10000
    before=np.concatenate((np.zeros((1,29),np.float32),target[:-1]))
    q=np.arange(n*29,dtype=np.float32).reshape(n,29)/20000
    after=np.concatenate((q[1:],q[-1:]+.001))
    mask=np.arange(n)>=24;source=(np.arange(n)>=14).astype(np.int64)
    segment=np.where(np.arange(n)<14,0,np.where(mask,2,1)).astype(np.int64)
    with h5py.File(path,'w') as f:
        f.attrs['metadata_json']=json.dumps(dict(dataset_split='train',joint_limits_rad=[[[-2,2]]*29],physics_dt=1/120,control_dt=1/60))
        g=f.create_group('episodes/0');g.attrs['complete']=True
        fields={'obs/time_s':time,'obs/step_id':np.arange(n),'obs/actuator_target_rad':before,
            'obs/camera_time_s':np.repeat(time[:,None],3,1),'obs/camera_valid':np.ones((n,3),bool),
            'obs/contact_valid':np.arange(n)>0,'obs/contact_substeps_w_N':np.zeros((n,2,5,3),np.float32),
            'obs/contact_substeps_local_N':np.zeros((n,2,5,3),np.float32),
            'obs/contact_time_s':time[:,None]+np.array([-1/120,0]),
            'transition/target_after_rad':target,'transition/target_before_rad':before,
            'transition/time_after_s':time+1/60,'transition/time_before_s':time,
            'transition/contact_substeps_after_w_N':np.zeros((n,2,5,3),np.float32),
            'transition/diagnostics_before/joint_position_rad':q,
            'transition/diagnostics_before/joint_velocity_rad_s':np.ones((n,29),np.float32),
            'transition/diagnostics_after/joint_position_rad':after,
            'transition/actuator_substep_target_rad':np.repeat(target[:,None],2,1),
            'frames/post_index':np.r_[np.arange(1,n),-1],
            'terminated':np.arange(n)==n-1,'truncated':np.zeros(n,bool),
            'transition/dagger/expert_label_valid':mask,'transition/dagger/segment_id':segment,
            'transition/dagger/control_source':source,'transition/dagger/switch_clean':mask,
            'transition/dagger/u_expert':np.where(mask[:,None],target,0),
            'transition/dagger/q_reached':np.where(mask[:,None],after,0),
            'transition/dagger/previous_issued':before}
        for name,value in fields.items():g.create_dataset(name,data=value)
        # Unallocated image storage: this CPU test checks control/data semantics.
        g.create_dataset('frames/rgb_u8',shape=(n,3,3,252,448),dtype='u1',chunks=(1,3,3,252,448))
        g.create_dataset('frames/depth_m',shape=(n,3,1,252,448),dtype='f4',chunks=(1,3,1,252,448))
    return dict(path=str(path),episode='0',length=n,source_sha256=sha256(path),
        trajectory_kind='expert_takeover_v1',family='train_takeover',split='train')


def test_takeover_preserves_actual_execution_and_masks_counterfactual_q(tmp_path):
    entry=takeover(tmp_path);row,touch,context=dagger_record(entry,include_geometry=False)
    assert context is None and touch.shape==(40,2,32)
    assert torch.count_nonzero(row['command'][1:24])>0
    assert torch.count_nonzero(row['u_expert'][:24])==torch.count_nonzero(row['q_reached'][:24])==0
    assert torch.equal(row['robot'][1:,116:145],row['command'][:-1])
    assert torch.equal(row['robot'][1:,145:174],row['command'][:-1])
    assert not torch.equal(row['q_reached'][24:],row['u_expert'][24:])
    assert row['supervision_mask'].sum()==16
    # A false q label at a supervised actual expert transition must be rejected.
    with h5py.File(entry['path'],'r+') as f:f['episodes/0/transition/dagger/q_reached'][25,0]+=1
    entry['source_sha256']=sha256(entry['path'])
    with pytest.raises(ValueError,match='actual issued'):dagger_record(entry,include_geometry=False)


def test_h24_never_bridges_invalid_or_authority_boundary_and_sampler_is_supervised():
    n=73;row=dict(time=torch.arange(n),command=torch.randn(n,29),
        supervision_mask=torch.ones(n,dtype=torch.bool),control_segment=torch.zeros(n,dtype=torch.long))
    row['supervision_mask'][:24]=False;row['supervision_mask'][36]=False
    row['control_segment'][48:]=1
    _,mask=labels(row,torch.tensor([12,24,36,48,72]),24)
    assert mask.sum(1).tolist()==[0,12,0,24,1]
    assert mask_totals({'r':n},[('r',24),('r',48),('r',72)],12,records={'r':row})==(37,34,25)
    sampler=AnchorSampler({'r':n},torch.Generator().manual_seed(11),records={'r':row})
    assert {sampler.draw()[0][1] for _ in range(4)}=={24,48,60,72}
    counts=supervision_population({'r':row})
    assert counts['supervised_anchors']==4 and counts['supervised_ticks']==48 and counts['history_ticks']==73
    assert sampler.coverage()['supervised_tick_population']==48
    original=supervision_mask_sha256({'r':row})
    row['control_segment'][60:]=2
    assert supervision_mask_sha256({'r':row})!=original


def test_residual_history_mask_keeps_prefix_while_loss_ignores_it():
    n=30;row=dict(time=torch.arange(n),robot=torch.randn(n,178),visual=torch.randn(n,512),
        plans=torch.randn(n,24,29),plan_valid=torch.ones(n,24,dtype=torch.bool),
        tactile={'force_w_N':torch.randn(n*2,5,3)},tactile_valid=torch.ones(n*2,dtype=torch.bool),
        supervision_mask=torch.arange(n)>=24,control_segment=(torch.arange(n)>=24).long())
    inputs=residual_batch([row],0,36,'cpu');loss=residual_supervision_mask([row],0,36,'cpu')
    assert inputs['valid'].sum()==30 and loss.sum()==6
    assert inputs['valid'][0,:24].all() and not loss[0,:24].any()
    assert torch.equal(inputs['robot'][0,:24],row['robot'][:24])
    legacy={'robot':row['robot']}
    assert residual_supervision_mask([legacy],0,36,'cpu').sum()==30
    with pytest.raises(ValueError,match='clock'):
        residual_supervision_mask([dict(legacy,supervision_mask=row['supervision_mask'])],0,36,'cpu')


def test_aggregate_copies_frozen_affine_and_never_opens_test_source(tmp_path):
    entry=takeover(tmp_path);new,touch,_=dagger_record(entry,include_geometry=False)
    source=tmp_path/'source';source.mkdir();out=tmp_path/'aggregate'
    old={k:new[k].clone() for k in ('robot','command','time','joint_limits','q_reached')}
    data=dict(splits={'train':['old'],'validation':['val'],'test':['heldout']},
        entries={'old':dict(entry),'val':dict(entry),'heldout':{'path':'/MUST_NOT_OPEN_TEST'}},
        families={'train_takeover':'train'},supervision='q_reached',normalization_identity='q_reached_train166_affine_v1')
    (source/'data.json').write_text(json.dumps(data))
    torch.save(dict(contract={},data_sha256=sha256(source/'data.json'),records={'old':old,'val':old}),source/'numeric.pt')
    torch.save({'old':touch,'val':touch},source/'tactile.pt')
    torch.save({'action_scale':torch.rand(29)},source/'statistics.pt');torch.save({'fixed':torch.rand(30)},source/'tactile_statistics.pt')
    result=build_dagger_aggregate(source,{'new':entry},out,'configuration',include_geometry=False)
    assert result['splits']=={'train':['old','new'],'validation':['val'],'test':['heldout']}
    assert result['normalization_train_ids']==['old']
    assert sha256(source/'statistics.pt')==sha256(out/'statistics.pt')
    rows=visual_records(out,['old','val','new'],'G00')
    assert rows['old']['supervision_mask'].all() and rows['new']['supervision_mask'].sum()==16
    assert torch.equal(rows['new']['clean_sample'],new['q_reached'])
    assert torch.equal(rows['new']['expert_command'],new['u_expert'])
    assert torch.equal(rows['new']['command'],new['command'])
    assert 'actor' not in rows['new'] and 'geometry' not in rows['new'] and 'tactile' not in rows['new']
    with pytest.raises(ValueError,match='semantics'):build_dagger_aggregate(source,{},tmp_path/'wrong','command',include_geometry=False)


def test_real_command_ema_continuation_keeps_g10_tactile_and_normalizers():
    from pathlib import Path
    from visuotactile.position_control import image_contract
    from visuotactile.state_history import load_image_continuation
    source=Path('outputs/visuotactile/run/tight_vision_20261109_01/G10/training/best.pt')
    if not source.exists():pytest.skip('Historical selected EMA stays local')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    stats={key:saved['ema'][key] for key in
        ('robot_mean','robot_std','action_scale','action_offset','tactile_mean','tactile_std')}
    contract=image_contract('I-HistoryDP',24,'G10','u_expert')
    model,audit=load_image_continuation(saved,contract,stats,supervision='command')
    assert audit['legacy_command_contract_declared'] and not audit['affine_replaced']
    assert all(torch.equal(value,saved['ema'][key]) for key,value in model.state_dict().items())
    assert 'tactile_mean' in audit['affine_buffers_verified']
    del model
    wrong=dict(stats,action_scale=stats['action_scale']+1)
    with pytest.raises(ValueError,match='affine changed'):load_image_continuation(saved,contract,wrong)
    with pytest.raises(ValueError,match='already-trained configuration'):
        load_image_continuation(saved,image_contract('I-HistoryDP',24,'G10','q_reached'),stats)
    # A real selected initialization is deployable and usable as the next round's
    # exact EMA source; its local update count must not masquerade as random.
    candidate=dict(saved,updates=0,new_optimizer_updates=0,contract=contract,
        configuration=dict(continuation_from=dict(path=str(source),sha256=sha256(source)),
            continuation_audit=dict(source_updates=saved['updates'],source_checkpoint_updates=saved['updates'],
                source=str(source.resolve()),source_sha256=sha256(source))))
    model,audit=load_image_continuation(candidate,contract,stats)
    assert audit['source_checkpoint_updates']==0 and audit['source_updates']==saved['updates']
    assert torch.equal(model.state_dict()['action_scale'],saved['ema']['action_scale'])
    del model
    from visuotactile.models.native_dp import load_policy
    deployed=load_policy(candidate)
    assert torch.equal(deployed.state_dict()['action_scale'],saved['ema']['action_scale'])
    del deployed
    from visuotactile.native_controller import NativeController
    deployed=NativeController(candidate,{'joint_limits_rad':[[[-2.,2.]]*29]},device='cpu')
    assert deployed.model.contract==contract
    del deployed
    with pytest.raises(ValueError,match='Trained selected'):
        load_image_continuation(dict(candidate,configuration={}),contract,stats)
    with pytest.raises(ValueError,match='Update-zero policy'):
        load_policy(dict(candidate,configuration={}))
    with pytest.raises(ValueError,match='untrained native'):
        NativeController(dict(candidate,configuration={}),{},device='cpu')


def test_update_zero_requires_actual_hash_bound_trained_nondisposable_parent(tmp_path):
    from copy import deepcopy
    from visuotactile.state_history import trained_continuation_updates
    parent=tmp_path/'selected.pt';parent.write_bytes(b'opaque immutable trained parent')
    saved=dict(updates=0,new_optimizer_updates=0,configuration=dict(
        continuation_from=dict(path=str(parent),sha256=sha256(parent)),
        continuation_audit=dict(source_updates=19,source_checkpoint_updates=0)))
    assert trained_continuation_updates(saved)==19
    for key,value in [('source_updates',0),('source_checkpoint_updates',None),('source_sha256','f'*64),('source',str(tmp_path/'wrong.pt'))]:
        bad=deepcopy(saved);bad['configuration']['continuation_audit'][key]=value
        assert trained_continuation_updates(bad)==0
    assert trained_continuation_updates(dict(saved,disposable=True))==0
    assert trained_continuation_updates(dict(saved,new_optimizer_updates=1))==0
    parent.write_bytes(b'changed parent')
    assert trained_continuation_updates(saved)==0


def test_real_residual_update_zero_controller_keeps_selected_trained_pair():
    from pathlib import Path
    from visuotactile.models.native_dp import load_residual
    from visuotactile.native_controller import ResidualController
    source=Path('outputs/visuotactile/run/tight_dagger_20261111_01/round_01/G11/backend/G11/training/best.pt')
    if not source.exists():pytest.skip('Prior-round selected residual stays local')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    candidate=dict(saved,updates=0,new_optimizer_updates=0,configuration=dict(saved['configuration'],
        continuation_from=dict(path=str(source),sha256=sha256(source)),
        continuation_audit=dict(source_updates=saved['updates'],source_checkpoint_updates=saved['updates'])))
    model=load_residual(candidate)
    assert all(torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items())
    del model
    controller=ResidualController(candidate,{'joint_limits_rad':[[[-2.,2.]]*29]},device='cpu')
    assert controller.residual.version==saved['residual_version']
    del controller
    with pytest.raises(ValueError,match='Update-zero residual'):load_residual(dict(candidate,configuration={}))
    with pytest.raises(ValueError,match='Untrained'):ResidualController(dict(candidate,configuration={}),{},device='cpu')


def test_frozen_dino_reuse_never_constructs_gpu_model_and_checks_source(tmp_path,monkeypatch):
    from visuotactile.native_data import cache_visual_pair, CAMERAS, DINO_REVISION
    from visuotactile.training_resources import atomic_json
    root=tmp_path/'new';data=root/'state_data';data.mkdir(parents=True)
    checkpoint=tmp_path/'dino.pth';checkpoint.write_bytes(b'frozen-test-backbone')
    payload=tmp_path/'old_features.pt';torch.save({'features':torch.randn(2,3)},payload)
    old=tmp_path/'old_index.json'
    entry=dict(path=str(payload),sha256=sha256(payload),source_sha256='a'*64,frames=2)
    index=dict(entries={'core':entry},dino_sha256=sha256(checkpoint),dino_revision=DINO_REVISION,
        stride=12,cameras=CAMERAS,resolution=[448,252],patches=[384,18,32],depth=[2,18,32],
        trainable_projection_cached=False,random_augmentation=False,complete=True)
    atomic_json(old,index)
    atomic_json(data/'data.json',dict(splits={'train':['core'],'validation':[]},
        entries={'core':dict(path='/MUST_NOT_OPEN_SOURCE_IMAGE',length=20,source_sha256='a'*64)}))
    atomic_json(root/'session.json',dict(dino_checkpoint=str(checkpoint),dino_sha256=sha256(checkpoint),cache_deadline=1))
    def no_model(*a,**kw):raise AssertionError('Reused frozen features must not start DINO/GPU')
    monkeypatch.setattr('visuotactile.models.native_vision.NativeDinoSpatial',no_model)
    monkeypatch.setattr('visuotactile.state_resources.lease_for',lambda *a,**kw:None)
    cache_visual_pair(root,reuse_index=old)
    assert (data/'perception/core.pt').is_symlink()
    assert sha256(data/'perception/core.pt')==sha256(payload)
    assert json.loads((root/'cache_index.json').read_text())['entries']['core']['source_sha256']=='a'*64
    index['entries']['core']['source_sha256']='b'*64;atomic_json(old,index)
    with pytest.raises(ValueError,match='source identity'):cache_visual_pair(root,reuse_index=old)


def test_nominal_refresh_replays_student_prefix_and_keeps_previous_nominal_separate(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from visuotactile.native_data import cache_residual_episode
    class Base(torch.nn.Module):
        group='I-HistoryDP';modality='G00';contract={'supervision':'u_expert'}
        def __init__(self,gain):
            super().__init__();self.gain=torch.nn.Parameter(torch.tensor(gain),requires_grad=False);self.eval()
        def observe(self,robot,stamp,tick,frame,memory,**kwargs):
            assert (memory is None)==(tick==0)
            return SimpleNamespace(total=(0 if memory is None else memory.total)+robot[0]*self.gain)
        def condition(self,memory,stamp):return memory.total
        def predict(self,condition,generator):return torch.ones(1,24,29)*condition
    monkeypatch.setattr('visuotactile.native_data.Episode',lambda *a:SimpleNamespace(frame=lambda *a:None))
    monkeypatch.setattr('visuotactile.native_data.residual_observations',lambda entry,robot,time:dict(robot=robot.clone()))
    monkeypatch.setattr('visuotactile.native_data.residual_visual',lambda model,memory,stamp:torch.ones(1,512)*memory.total)
    (tmp_path/'data.json').write_text('{}');n=30
    row=dict(source_root=str(tmp_path),identity='correction',robot=torch.ones(n,178)*.01,
        time=torch.arange(n)/60,command=torch.ones(n,29)*.6,expert_command=torch.ones(n,29)*.7,
        joint_limits=torch.tensor([[-2.,2.]]*29),supervision_mask=torch.arange(n)>=24,
        control_segment=(torch.arange(n)>=24).long())
    binding=dict(base_checkpoint_sha256='a'*64,statistics_sha256='b'*64,
        data_sha256=sha256(tmp_path/'data.json'),residual_statistics_sha256='c'*64)
    lease=SimpleNamespace(block=nullcontext);entry={'trajectory_kind':'expert_takeover_v1'}
    one=cache_residual_episode(Base(1.),row,entry,lease,cache_identity=binding)
    two=cache_residual_episode(Base(2.),row,entry,lease,cache_identity=dict(binding,base_checkpoint_sha256='d'*64))
    assert torch.equal(one['robot'],row['robot']) and torch.equal(one['target'],row['expert_command'])
    assert float(one['base'][24,0])==pytest.approx(.25)
    assert not torch.equal(one['base'],two['base'])
    assert torch.equal(one['previous_nominal'][1:],one['base'][:-1])
    assert not torch.equal(one['previous_nominal'][1:],row['command'][:-1])
    assert not one['previous_nominal_valid'][0] and one['previous_nominal_valid'][1:].all()
    assert one['plan_valid'][11].sum()==13 and one['plan_valid'][12].sum()==24
    assert one['supervision_mask'][:24].sum()==0 and one['cache_identity']==binding
    with pytest.raises(ValueError,match='SHA bindings'):cache_residual_episode(Base(1.),row,entry,lease)
