"""Full-history cache batching is opt-in and never supervises the student prefix."""
import copy
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from visuotactile import native_data
from visuotactile.offline_planning import profile
from tools import tight_residual_cache_gate as gate


class Base(torch.nn.Module):
    group='I-HistoryDP'
    modality='G00'
    horizon=24
    def __init__(self,supervision):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1),requires_grad=False)
        self.contract=dict(supervision=supervision,visual_stride=1);self.calls=[];self.eval()
    def observe(self,robot,stamp,tick,frame,memory,*,tactile):
        assert (memory is None)==(tick==0) and frame['tick']==tick and tactile is None
        total=(0 if memory is None else memory.total)+robot[0]*self.weight
        self.calls.append(('observe',tick,float(total),float(robot[116])))
        return SimpleNamespace(total=total)
    def condition(self,memory,stamp):return memory.total.reshape(1,1).repeat(1,3)
    def predict(self,condition,generator,*,inference_steps=16,clip_sample=False,scheduler_index_mode='legacy'):
        assert scheduler_index_mode=='host' and inference_steps==16 and clip_sample is False
        self.calls.append(('predict',len(condition)))
        if isinstance(generator,list):
            assert all(g is generator[0] for g in generator)
            noise=torch.cat([torch.randn((1,24,29),generator=g) for g in generator])
        else:noise=torch.randn((1,24,29),generator=generator)
        return noise*.01+condition[:,:1,None]


def cache_fixture(tmp_path,monkeypatch,supervision='q_reached',length=47):
    monkeypatch.setattr(native_data,'Episode',lambda *a:SimpleNamespace(frame=lambda tick,device:dict(tick=tick)))
    monkeypatch.setattr(native_data,'residual_observations',lambda e,r,t:dict(robot=r.clone(),
        tactile={'force':torch.arange(len(t)*4).reshape(-1,2)},gate=torch.arange(len(t))%2==0))
    monkeypatch.setattr(native_data,'residual_visual',lambda m,h,t:torch.arange(512)[None].float()+h.total)
    (tmp_path/'data.json').write_text('{}')
    row=dict(source_root=str(tmp_path),identity='recovery',robot=torch.arange(length*178).reshape(length,178).float()*.0001,
        time=torch.arange(length)/60,command=torch.ones(length,29)*.4,expert_command=torch.ones(length,29)*.6,
        joint_limits=torch.tensor([[-2.,2.]]*29),clean_sample=torch.ones(length,29)*.3,
        supervision_mask=torch.arange(length)>=15,control_segment=(torch.arange(length)>=15).long())
    binding=dict(base_checkpoint_sha256='a'*64,statistics_sha256='b'*64,
        data_sha256=native_data.sha256(tmp_path/'data.json'),residual_statistics_sha256='c'*64)
    entry=dict(trajectory_kind='expert_takeover_v1')
    return row,binding,entry,SimpleNamespace(block=nullcontext)


@pytest.mark.parametrize('batch',[1,8,32,128])
@pytest.mark.parametrize('supervision',['q_reached','u_expert'])
def test_complete_cache_exact_cpu_parity_and_all_prefix_grid_anchors(tmp_path,monkeypatch,batch,supervision):
    row,binding,entry,lease=cache_fixture(tmp_path,monkeypatch,supervision)
    serial,candidate=Base(supervision),Base(supervision);before=copy.deepcopy(row);rng=torch.get_rng_state().clone()
    common=dict(transfer_mode='episode',scheduler_index_mode='host',cache_identity=binding)
    expected=native_data.cache_residual_episode(serial,row,entry,lease,**common)
    progress=[]
    actual=native_data.cache_residual_episode(candidate,row,entry,lease,**common,
        offline_planning_profile=profile(batch),progress=lambda **kw:progress.append(kw))
    assert gate.compare_caches(expected,actual,profile(batch))['passed']
    assert all(torch.equal(expected[k],actual[k]) for k in gate.PLAN_KEYS)
    assert [v for v in serial.calls if v[0]=='observe']==[v for v in candidate.calls if v[0]=='observe']
    assert len([v for v in candidate.calls if v[0]=='observe'])==47
    assert sum(v[1] for v in candidate.calls if v[0]=='predict')==4 # 0,12,24,36 including prefix
    first_plan=next(i for i,v in enumerate(candidate.calls) if v[0]=='predict')
    assert all(v[0]=='observe' for v in candidate.calls[:first_plan]) and first_plan==47
    assert progress[-1]['plans']==4 and progress[-1]['phase']=='offline_planning_complete'
    assert torch.equal(actual['base'],actual['plans'][:,0])
    assert torch.equal(actual['previous_nominal'][1:],actual['base'][:-1])
    assert torch.equal(actual['plan_valid'],torch.arange(24)[None]+actual['plan_cursor'][:,None]<24)
    assert not actual['supervision_mask'][:15].any() and int(actual['supervision_mask'].sum())==32
    assert torch.equal(actual['target'],row['expert_command'])
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in before.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,row[k])
    assert all(p.grad is None for p in candidate.parameters())
    assert actual['q_reached'] is None if supervision=='u_expert' else torch.equal(actual['q_reached'],row['clean_sample'])


@pytest.mark.parametrize('change',[{'transfer_mode':'tick'},{'scheduler_index_mode':'legacy'},
    {'modality':'G10'},{'horizon':32},{'visual_stride':12}])
def test_candidate_rejects_unsupported_contract_before_replaying(tmp_path,monkeypatch,change):
    row,binding,entry,lease=cache_fixture(tmp_path,monkeypatch)
    base=Base('q_reached');options=dict(transfer_mode='episode',scheduler_index_mode='host')
    for k,v in change.items():
        if k in options:options[k]=v
        elif k=='visual_stride':base.contract[k]=v
        else:setattr(base,k,v)
    with pytest.raises(ValueError,match='episode/host'):
        native_data.cache_residual_episode(base,row,entry,lease,cache_identity=binding,
            offline_planning_profile=profile(8),**options)
    assert not base.calls


@pytest.mark.parametrize('key',['visual','robot','time','target','supervision_mask','plan_valid','plan_cursor','cache_identity'])
def test_gate_strict_nonplan_fields_and_binding(tmp_path,monkeypatch,key):
    row,binding,entry,lease=cache_fixture(tmp_path,monkeypatch)
    expected=native_data.cache_residual_episode(Base('q_reached'),row,entry,lease,
        transfer_mode='episode',scheduler_index_mode='host',cache_identity=binding)
    candidate=copy.deepcopy(expected);candidate[gate.PROFILE_KEY]=profile(32)
    if key=='cache_identity':candidate[key]['base_checkpoint_sha256']='z'*64
    elif candidate[key].dtype==torch.bool:candidate[key].flatten()[0]=~candidate[key].flatten()[0]
    else:candidate[key].flatten()[0]+=1
    assert not gate.compare_caches(expected,candidate,profile(32))['passed']


def test_gate_target_tolerance_is_fixed_and_all_slots_are_compared():
    reference=dict(plans=torch.zeros(3,24,29),base=torch.zeros(3,29),previous_nominal=torch.zeros(3,29),visual=torch.ones(3,512))
    candidate=copy.deepcopy(reference);candidate[gate.PROFILE_KEY]=profile(32)
    candidate['plans'][0,23,0]=1e-5
    assert gate.compare_caches(reference,candidate,profile(32))['passed']
    candidate['plans'][0,23,0]=2.001e-5
    assert not gate.compare_caches(reference,candidate,profile(32))['passed']
    candidate['plans'][0,23,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):gate.compare_caches(reference,candidate,profile(32))
    del candidate['visual']
    with pytest.raises(ValueError,match='population'):gate.compare_caches(reference,candidate,profile(32))


def test_orphan_profile_mismatch_rejected_before_source_reads(tmp_path):
    from tools.tight_finish import recover_residual_cache
    path=tmp_path/'cache.pt';torch.save({gate.PROFILE_KEY:profile(8)},path)
    for expected in (None,profile(32)):
        with pytest.raises(ValueError,match='planning profile'):
            recover_residual_cache(path,'x',{}, {}, {},offline_planning_profile=expected)


def test_registered_index_cannot_be_reinterpreted_as_serial(tmp_path,monkeypatch):
    from test_dagger_evaluation import cache_fixture as existing_fixture
    from tools import tight_finish
    existing_fixture(tmp_path,monkeypatch)
    path=tmp_path/'G11/nominal/index.json'
    value=json.loads(path.read_text()) if path.exists() else {}
    value[gate.PROFILE_KEY]=profile(32);path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='planning profile'):
        tight_finish.residual_cache(tmp_path)


@pytest.mark.parametrize('batch',[None,1,32])
def test_backend_prepare_explicit_profile_propagates_and_default_remains_absent(tmp_path,monkeypatch,batch):
    from tools import tight_fresh_backend as backend
    from visuotactile.compact import sha256
    root=tmp_path/'round';root.mkdir()
    train=[f't{i}' for i in range(30)];val=[f'v{i}' for i in range(6)]
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write(root/'session.json',dict(deadline=10000));write(root/'shared_resources.json',{})
    for role,supervision in [('A','q_reached'),('B','u_expert')]:
        rr=root/role
        data=dict(splits=dict(train=train,validation=val,test=[]),entries={},visual_stride=1,
            supervision=supervision,normalization_scope='selected train30 only',
            normalization_identity=supervision+'_fresh_train_only_affine_v1')
        write(rr/'state_data/data.json',data);write(rr/'cache_index.json',{})
        torch.save({'stats':role},rr/'state_data/statistics.pt');(rr/'training').mkdir()
        torch.save(dict(updates=10,contract=dict(supervision=supervision,visual_stride=1),
            configuration=dict(statistics_sha256=sha256(rr/'state_data/statistics.pt'))),rr/'training/best.pt')
    backend.prepare(root,'A',2,offline_anchor_batch=batch)
    prepared=json.loads((root/'A/backend/session.json').read_text())['residual_pair']
    if batch is None:assert 'cache_offline_planning_profile' not in prepared
    else:assert prepared['cache_offline_planning_profile']==profile(batch)
    assert prepared['cache_transfer_mode']=='episode' and prepared['cache_scheduler_index_mode']=='host'
    assert prepared['bases']['A']['checkpoint']==str(root/'A/training/best.pt')


def test_candidate_checks_stop_between_plan_batches_after_complete_real_history(tmp_path,monkeypatch):
    row,binding,entry,lease=cache_fixture(tmp_path,monkeypatch)
    checked=[]
    def check():
        checked.append(True)
        if len(checked)==2:raise InterruptedError('Owned STOP')
    lease.check=check;base=Base('q_reached')
    with pytest.raises(InterruptedError,match='STOP'):
        native_data.cache_residual_episode(base,row,entry,lease,cache_identity=binding,
            transfer_mode='episode',scheduler_index_mode='host',offline_planning_profile=profile(1))
    assert len([c for c in base.calls if c[0]=='observe'])==47
    assert len([c for c in base.calls if c[0]=='predict'])==1


@pytest.mark.parametrize('mismatch',['none','configuration','index','payload','invalid_profile'])
def test_backend_consumer_rejects_profile_mixing_even_after_rehash(tmp_path,monkeypatch,mismatch):
    from visuotactile import anchor_continue, state_resources
    from visuotactile.models.native_dp import TargetResidual
    from visuotactile.compact import sha256
    from visuotactile.fresh_contracts import FRESH_DATA_CONTRACT
    root=tmp_path/'B';nominal=root/'nominal';nominal.mkdir(parents=True)
    train=[f't{i}' for i in range(30)];val=[f'v{i}' for i in range(6)];ids=train+val
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    data=dict(splits=dict(train=train,validation=val,test=[]),visual_stride=1,supervision='u_expert',
        normalization_identity='u_expert_fresh_train_only_affine_v1',normalization_scope='selected train30 only')
    write(root/'state_data/data.json',data)
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    torch.save(stats,root/'state_data/statistics.pt');base=tmp_path/'base.pt';base.write_bytes(b'base')
    config=dict(version=TargetResidual.version,fresh_data_contract=FRESH_DATA_CONTRACT,
        bases={'B':dict(checkpoint=str(base),sha256=sha256(base))},
        residual_statistics_path=str(root/'state_data/statistics.pt'),initialization_seed=3,
        cache_offline_planning_profile=profile(32))
    index=dict(base_sha256=sha256(base),data_sha256=sha256(root/'state_data/data.json'),
        residual_statistics_sha256=sha256(root/'state_data/statistics.pt'),
        residual_normalization_identity=data['normalization_identity'],cache_offline_planning_profile=profile(32),episodes={})
    for identity in ids:
        payload=dict(time=torch.arange(3),cache_offline_planning_profile=profile(32))
        if mismatch=='payload' and identity==ids[-1]:payload['cache_offline_planning_profile']=profile(8)
        torch.save(payload,nominal/(identity+'.pt'));index['episodes'][identity]=dict(sha256=sha256(nominal/(identity+'.pt')))
    if mismatch=='configuration':config.pop('cache_offline_planning_profile')
    if mismatch=='index':index.pop('cache_offline_planning_profile')
    if mismatch=='invalid_profile':config['cache_offline_planning_profile']['seed']=7
    write(root/'configuration.json',dict(residual_pair=config));write(nominal/'index.json',index)
    write(nominal/'scales.json',dict(residual_statistics_sha256=index['residual_statistics_sha256'],
        residual_normalization_identity=index['residual_normalization_identity'],scales=[1.]*29))
    write(root/'session.json',dict(dino_sha256='dino',resource_root=str(tmp_path),training_deadline=123))
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext))
    monkeypatch.setattr(TargetResidual,'cuda',lambda self:self)
    if mismatch!='none':
        with pytest.raises(ValueError,match='profile'):anchor_continue.residual_setup(root)
    else:
        *_,configuration,lease=anchor_continue.residual_setup(root)
        assert configuration['cache_offline_planning_profile']==profile(32)
