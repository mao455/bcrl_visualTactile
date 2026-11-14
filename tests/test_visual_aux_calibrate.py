from types import SimpleNamespace
import copy
import json
import pytest
import torch

from tools.tight_visual_aux_calibrate import (base_loss,subset,paired_subset,weight_from_rows,
    source_binding,tensor_sha,gradient_norm,head_sha,check_label_binding)
from visuotactile.compact import sha256
from visuotactile.native_data import CAMERAS,DINO_REVISION
from visuotactile.position_control import continuous_image_contract
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW,with_target_window
from visuotactile.anchor_sampling import planning_anchors,labels
from visuotactile.visual_pose_aux import VisualPoseHead
from visuotactile.state_image_loss import GeometryObjective,PALM_LOSS_ID
from visuotactile.dp_objectives import masked_objectives


def test_subset_excludes_entire_other_families_and_validation():
    train=[f'e{i}' for i in range(30)]
    entries={i:dict(family=f'f{j:02d}') for j,i in enumerate(train)}
    entries.update(c0=dict(family='f00',trajectory_kind='expert_takeover_v1'),
        c25=dict(family='f25',trajectory_kind='expert_takeover_v1'),val=dict(family='validation'))
    data=dict(normalization_train_ids=train,splits=dict(train=train+['c0','c25'],validation=['val'],test=[]),entries=entries)
    families,ids=subset(data)
    assert families==[f'f{i:02d}' for i in range(6)]
    assert set(ids)=={f'e{i}' for i in range(6)}|{'c0'}
    assert ids==train[:6]+['c0']  # actual trainer order, not sorted episode IDs
    data['splits']['validation'].append('c0')
    with pytest.raises(ValueError,match='overlap'):subset(data)


def test_family_overlap_and_different_ab_payloads_fail_before_access():
    original=[f'e{i}' for i in range(30)]
    entries={i:dict(family=f'f{j:02d}') for j,i in enumerate(original)}
    entries['val']=dict(family='heldout')
    data=dict(normalization_train_ids=original,splits=dict(train=original,validation=['val'],test=[]),entries=entries)
    changed=copy.deepcopy(data);changed['entries']['val']['family']='f00'
    with pytest.raises(ValueError,match='family overlap'):subset(changed)
    changed=copy.deepcopy(data);changed['entries']['e0']['path']='different_source'
    with pytest.raises(ValueError,match='source population'):paired_subset(dict(A=data,B=changed))


def test_common_weight_uses_prespecified_median_norms_without_role_tuning():
    rows=[dict(role=r,identity=i,base_gradient_norm=b,pose_gradient_norm=p)
        for r in ('A','B') for i,b,p in [('a',2.,10.),('b',4.,20.)]]
    weight,details=weight_from_rows(rows,.1)
    assert weight==pytest.approx(.02)
    assert details==dict(base_gradient_median=3.,pose_gradient_median=15.)
    with pytest.raises(ValueError):weight_from_rows(rows,.2)
    rows[-1]['pose_gradient_norm']=0.
    with pytest.raises(ValueError,match='Nonzero'):weight_from_rows(rows,.1)


def test_common_weight_is_ratio_of_medians_and_rejects_duplicate_episode_rows():
    rows=[dict(role=r,identity=i,base_gradient_norm=b,pose_gradient_norm=p)
        for r in ('A','B') for i,b,p in [('x',1.,1.),('y',2.,10.),('z',20.,10.)]]
    assert weight_from_rows(rows,.1)[0]==pytest.approx(.02)
    with pytest.raises(ValueError,match='unique'):weight_from_rows(rows+[rows[0]],.1)


def test_base_loss_and_rng_exactly_match_fresh_objective_forward():
    # Reuse the real GeometryObjective.forward while replacing only FK parts;
    # the calibration deliberately measures the same base term, without FK.
    objective=GeometryObjective.__new__(GeometryObjective)
    objective.weights={};objective.gain=1.
    objective.components=lambda pred,truth,valid,scale,offset,ctx:dict(
        base=masked_objectives(pred,torch.where(valid[...,None],truth*scale+offset,0.),valid)['B'])
    model=SimpleNamespace(horizon=24,action_scale=torch.linspace(.5,2.,29),action_offset=torch.linspace(-1.,1.,29),
        noise=SimpleNamespace(config=SimpleNamespace(prediction_type='sample'),
            add_noise=lambda target,noise,ticks:target+noise*ticks[:,None,None]/100),
        denoiser=lambda noisy,ticks,global_cond:noisy*.2+global_cond[:,None,:])
    truth=torch.randn(2,24,29);valid=torch.ones(2,24,dtype=torch.bool);valid[0,:4]=False
    condition=torch.randn(2,29,requires_grad=True)
    a=torch.Generator().manual_seed(29);b=torch.Generator().manual_seed(29)
    actual=base_loss(model,condition,truth,valid,a)
    expected=objective.forward(model,condition,truth,valid,b)[1]['base']
    assert torch.equal(actual,expected) and torch.equal(a.get_state(),b.get_state())
    assert torch.equal(torch.autograd.grad(actual,condition,retain_graph=True)[0],torch.autograd.grad(expected,condition)[0])
    model.noise.config.prediction_type='epsilon'
    with pytest.raises(ValueError,match='H24 sample'):base_loss(model,condition,truth,valid,a)


def test_shared_gradient_norm_and_identical_head_preserve_parameters_and_rng():
    generator=torch.random.get_rng_state().clone()
    a,b=VisualPoseHead(seed=11),VisualPoseHead(seed=11)
    assert torch.equal(generator,torch.random.get_rng_state()) and head_sha(a)==head_sha(b)
    shared=torch.nn.Linear(3,512);before={k:v.clone() for k,v in shared.state_dict().items()}
    value=shared(torch.ones(2,3));loss=a(value,route='shared_vision').square().mean()
    norm=gradient_norm(loss,tuple(shared.parameters()),retain_graph=True)
    direct=torch.autograd.grad(loss,tuple(shared.parameters()))
    expected=torch.sqrt(sum(g.double().square().sum() for g in direct)).item()
    assert norm==pytest.approx(expected)
    assert all(p.grad is None for p in (*shared.parameters(),*a.parameters()))
    assert all(torch.equal(before[k],v) for k,v in shared.state_dict().items())
    detached=a(shared(torch.ones(2,3)),route='detached_readout').square().mean()
    with pytest.raises(ValueError,match='Nonzero finite'):gradient_norm(detached,tuple(shared.parameters()),retain_graph=False)


def source_fixture(tmp_path):
    role=tmp_path/'A';state=role/'state_data';training=role/'training'
    state.mkdir(parents=True);training.mkdir();(state/'perception').mkdir()
    write=lambda path,data:path.write_text(json.dumps(data))
    data=dict(splits=dict(train=['episode'],validation=[]),entries=dict(episode=dict(length=2,source_sha256='raw_source')))
    write(state/'data.json',data)
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    torch.save(stats,state/'statistics.pt')
    stamp=torch.arange(2,dtype=torch.float64)/60;ct=stamp[:,None].expand(-1,3).clone()
    torch.save(dict(data_sha256=sha256(state/'data.json'),records=dict(episode=dict(time=stamp))),state/'numeric.pt')
    shard=dict(camera_time=ct,camera_valid=torch.ones(2,3,dtype=torch.bool),
        patches=torch.zeros(2,3,384,18,32,dtype=torch.float16),depth=torch.zeros(2,3,2,18,32,dtype=torch.float16))
    torch.save(shard,state/'perception/episode.pt')
    index=dict(data_sha256=sha256(state/'data.json'),dino_revision=DINO_REVISION,dino_sha256='dino',stride=1,
        cameras=CAMERAS,resolution=[448,252],patches=[384,18,32],depth=[2,18,32],random_augmentation=False,
        trainable_projection_cached=False,entries=dict(episode=dict(sha256=sha256(state/'perception/episode.pt'),
            source_sha256='raw_source',frames=2,camera_time_sha256=tensor_sha(ct))))
    write(role/'cache_index.json',index)
    config=dict(data_sha256=sha256(state/'data.json'),statistics_sha256=sha256(state/'statistics.pt'),
        cache_index_sha256=sha256(role/'cache_index.json'),dino_sha256='dino',
        continuation=dict(target_window_contract=RECOVERY_ONSET_WINDOW))
    write(training/'configuration.json',config)
    checkpoint=dict(contract=continuous_image_contract('q_reached'),updates=10,kind='dp',group='I-HistoryDP',
        configuration=config,train_ids=['episode'],validation_ids=[],ema=stats)
    torch.save(checkpoint,training/'best.pt')
    write(training/'finished.json',dict(complete=True,checkpoint_sha256=sha256(training/'best.pt')))
    return data,checkpoint


def test_source_binding_retains_window_and_rejects_changed_cache_bytes(tmp_path):
    data,_=source_fixture(tmp_path)
    proof=source_binding(tmp_path,'A',data,['episode'])
    assert proof['target_window_contract']==RECOVERY_ONSET_WINDOW
    assert proof['files']['statistics']==sha256(tmp_path/'A/state_data/statistics.pt')
    assert proof['entries']['episode']['frames']==2
    with (tmp_path/'A/state_data/perception/episode.pt').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='shard bytes'):source_binding(tmp_path,'A',data,['episode'])


def test_source_binding_rejects_wrong_ema_normalizer_even_with_updated_checkpoint_hash(tmp_path):
    data,checkpoint=source_fixture(tmp_path);checkpoint['ema']['action_offset']=torch.ones(29)
    training=tmp_path/'A/training';torch.save(checkpoint,training/'best.pt')
    (training/'finished.json').write_text(json.dumps(dict(complete=True,checkpoint_sha256=sha256(training/'best.pt'))))
    with pytest.raises(ValueError,match='EMA normalization'):source_binding(tmp_path,'A',data,['episode'])


def test_correction_onset_window_adds_only_actual_legal_targets():
    mask=torch.arange(40)>=14;segment=(torch.arange(40)>=14).long()
    record=dict(time=torch.arange(40)/60,command=torch.zeros(40,29),clean_sample=torch.zeros(40,29),
        supervision_mask=mask,control_segment=segment)
    ordinary=planning_anchors(record)
    current=with_target_window({'x':record},RECOVERY_ONSET_WINDOW)['x']
    anchors=planning_anchors(current)
    assert 12 not in ordinary and 12 in anchors
    _,valid=labels(current,torch.tensor([12]),24)
    assert not valid[0,:2].any() and valid[0,2:].all()


def test_pose_diagnostics_do_not_change_action_mask_and_require_recorded_authority():
    record=dict(time=torch.arange(4,dtype=torch.float64)/60,
        supervision_mask=torch.tensor([False,False,True,True]),control_segment=torch.tensor([0,1,2,2]))
    proof=dict(label_sha256={'time_s':tensor_sha(record['time'])},
        input_tensor_sha256={'camera_time_s':'camera_hash'},authority_tensor_sha256=dict(
            expert_label_valid=tensor_sha(record['supervision_mask']),segment_id=tensor_sha(record['control_segment'])))
    check_label_binding(record,proof,{'camera_time_sha256':'camera_hash'})
    changed=dict(record,supervision_mask=torch.ones(4,dtype=torch.bool))
    with pytest.raises(ValueError,match='authority'):check_label_binding(changed,proof,{'camera_time_sha256':'camera_hash'})
    with pytest.raises(ValueError,match='clocks'):check_label_binding(record,proof,{'camera_time_sha256':'other'})


def test_calibration_reads_expired_source_but_uses_explicit_active_session_lease(tmp_path,monkeypatch):
    from tools import tight_visual_aux_calibrate as calibrate
    from visuotactile import precision,state_resources
    source=tmp_path/'source';active=tmp_path/'active';output=tmp_path/'calibration.json'
    source.mkdir();active.mkdir()
    old=dict(deadline=1,domain={'fixed':'narrow'},resource_root=str(source))
    current=dict(deadline=calibrate.time.time()+3600,domain=old['domain'],resource_root=str(active))
    for directory,value in ((source,old),(active,current)):
        (directory/'session.json').write_text(json.dumps(value))
        (directory/'shared_resources.json').write_text(json.dumps({'owner':str(directory)}))
    for role in ('A','B'):
        (source/role/'state_data').mkdir(parents=True);(source/role/'training').mkdir()
        (source/role/'state_data/data.json').write_text(json.dumps({'source_role':role}))
        (source/role/'training/best.pt').write_bytes(b'CPU-only fixture; never loaded')
    before={str(p):sha256(p) for p in source.rglob('*') if p.is_file()};seen=[]
    monkeypatch.setattr(calibrate,'paired_subset',lambda data:(['family'],['episode']))
    def bind(root,role,data,ids):
        assert root==source and data=={'source_role':role} and ids==['episode']
        seen.append(role)
        return dict(files={'statistics':'a'*64},target_window_contract=None)
    monkeypatch.setattr(calibrate,'source_binding',bind)
    monkeypatch.setattr(calibrate.subprocess,'check_output',lambda *args,**kwargs:'reviewed\n')
    monkeypatch.setattr(precision,'state_fp32',lambda:None)
    class ReachedLease(Exception):pass
    def lease(root,deadline):
        assert root==active and deadline==current['deadline']
        raise ReachedLease
    monkeypatch.setattr(state_resources,'lease_for',lease)
    with pytest.raises(ReachedLease):calibrate.run(source,output,71,active)
    assert seen==['A','B'] and not output.exists()
    started=calibrate.read(output.with_suffix('.started.json'))
    assert started['source_root']==str(source) and started['session_root']==str(active)
    assert started['deadline']==current['deadline']
    assert started['source_session_sha256']==sha256(source/'session.json')
    assert started['session_sha256']==sha256(active/'session.json')
    assert before=={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(TimeoutError,match='expired'):calibrate.session_binding(source)
    assert not torch.cuda.is_initialized()


def test_calibration_session_root_rejects_domain_or_telemetry_owner_change(tmp_path):
    from tools import tight_visual_aux_calibrate as calibrate
    source=tmp_path/'source';active=tmp_path/'active';other=tmp_path/'other'
    for directory in (source,active,other):
        directory.mkdir();(directory/'shared_resources.json').write_text('{}')
    (source/'session.json').write_text(json.dumps({'deadline':1,'domain':{'narrow':'original'}}))
    current=dict(deadline=calibrate.time.time()+3600,domain={'different':True},resource_root=str(active))
    (active/'session.json').write_text(json.dumps(current))
    with pytest.raises(ValueError,match='domains'):calibrate.session_binding(source,active)
    current.update(domain={'narrow':'original'},resource_root=str(other))
    (active/'session.json').write_text(json.dumps(current))
    with pytest.raises(ValueError,match='telemetry owner'):calibrate.session_binding(source,active)
