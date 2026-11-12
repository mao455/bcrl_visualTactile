"""Exercise actual trainer scheduling/selection with tiny synthetic CPU tensors."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from visuotactile import anchor_continue
from visuotactile.models.native_dp import TargetResidual


class TinyResidual(torch.nn.Module):
    version = TargetResidual.version

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(29))
        self.register_buffer('action_scale', torch.ones(29))
        self.seen = [];self.had_state=[]

    def forward(self, inputs, state):
        self.had_state.append(state is not None)
        self.seen.extend(inputs['robot'][..., 0][inputs['valid']].tolist())
        raw = self.weight.expand(*inputs['valid'].shape, 29)
        return raw, {'last': raw[:, -1]}

    def compose(self, base, raw, contact, limits):
        return base+raw, raw

    objective = TargetResidual.objective


def record(identity, length):
    return dict(robot=torch.full((length, 1), float(identity)), visual=torch.zeros(length, 1),
        plans=torch.zeros(length, 1, 29), plan_valid=torch.ones(length, 1, dtype=torch.bool),
        tactile={'dummy': torch.zeros(length*2, 1)}, tactile_valid=torch.ones(length*2, dtype=torch.bool),
        base=torch.zeros(length, 29), target=torch.ones(length, 29),
        # Every synthetic tick is noncontact: counting only contact would fail.
        gate=torch.zeros(length, dtype=torch.bool), limits=torch.tensor([[-2., 2.]]).expand(29, -1))


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    root=tmp_path/'R01';root.mkdir()
    train=[f't{i}' for i in range(6)];validation=[f'v{i}' for i in range(4)]
    records={identity:record(i, i+1) for i,identity in enumerate(train)}
    records.update({identity:record(99, 1) for identity in validation})
    data={'splits':dict(train=train,validation=validation)}
    config=dict(maximum_control_tick_epochs=5,minimum_control_tick_epochs=1,
        microbatch=4,validation_epoch_interval=1,initialization_seed=3)
    session=dict(training_deadline=float('inf'),base_contracts={'R01':{}})
    lease=SimpleNamespace(block=nullcontext,wait_s=0.,work_s=0.)
    model=TinyResidual()
    monkeypatch.setattr(anchor_continue,'residual_setup',lambda *a,**kw:
        (session,config,data,records,model,{},lease))
    monkeypatch.setattr(anchor_continue.signal,'signal',lambda *a:None)
    monkeypatch.setattr(anchor_continue,'residual_curves',lambda *a:None)
    # Constant validation exercises the registered four-validation plateau.
    calls=[]
    def metrics(model,records,ids,lease):
        calls.append(list(ids))
        return {'all':{'normalized_equal_mse':1.}}, []
    monkeypatch.setattr(anchor_continue,'residual_metrics',metrics)
    return root,session,model,calls


def test_probe_is_four_complete_train_episodes_and_disposable(trainer):
    root,_,model,calls=trainer
    anchor_continue.residual_train(root,probe=True)
    finished=json.loads((root/'training_probe/finished.json').read_text())
    saved=torch.load(root/'training_probe/best.pt',map_location='cpu',weights_only=False)
    assert sorted(model.seen)==[0.]+[1.]*2+[2.]*3+[3.]*4
    assert finished['supervised_ticks']==finished['required_supervised_ticks']==10
    assert finished['target_reached'] and finished['control_tick_epochs']==1
    assert finished['effective_batch']==finished['train_episode_count']==4
    assert finished['updates']==saved['updates']==1
    assert finished['formal_updates']==saved['formal_updates']==0
    assert finished['disposable'] and saved['disposable']
    assert saved['train_ids']==['t0','t1','t2','t3']
    assert calls[0]==['v0','v1','v2','v3']
    assert finished['wall_s']>=finished['training_wall_s']>0
    assert finished['supervised_ticks_per_wall_s']>0
    assert not (root/'training').exists()
    with pytest.raises(ValueError,match='second residual'):anchor_continue.residual_train(root,probe=True)


def test_trainer_persists_explicit_preclip_loss_and_records_components(trainer, monkeypatch):
    from visuotactile.models.native_dp import PRECLIP_RESIDUAL_LOSS_VERSION
    root, _, model, _ = trainer
    setup = anchor_continue.residual_setup(root)
    configuration = setup[5]
    monkeypatch.setattr(anchor_continue, 'residual_setup', lambda *a, **kw:setup)
    model.loss_contract = dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=1.)
    configuration['loss_contract'] = dict(model.loss_contract)
    def compose_chain(base, raw, contact, limits):
        u_pre = base+raw
        return dict(u_exec=u_pre.clamp(-.01,.01), delta=raw, u_pre=u_pre)
    monkeypatch.setattr(model, 'compose_chain', compose_chain, raising=False)
    monkeypatch.setattr(model, 'objective_components',
        lambda *a, **kw:TargetResidual.objective_components(model, *a, **kw), raising=False)
    anchor_continue.residual_train(root, probe=True)
    saved = torch.load(root/'training_probe/last.pt', map_location='cpu', weights_only=False)
    rows = [json.loads(row) for row in (root/'training_probe/loss.jsonl').read_text().splitlines()]
    assert saved['residual_loss_version'] == PRECLIP_RESIDUAL_LOSS_VERSION
    assert saved['configuration']['loss_contract'] == model.loss_contract
    assert rows[0]['loss_components'] == dict(exec_fit=1., pre_fit=1., regularizer=0.)
    assert rows[0]['loss'] == 2.


def test_continuation_audits_loss_change_without_reinterpreting_old_weights():
    from visuotactile.models.native_dp import PRECLIP_RESIDUAL_LOSS_VERSION
    stats = dict(robot_mean=torch.zeros(178), robot_std=torch.ones(178),
        action_scale=torch.ones(29), action_offset=torch.zeros(29))
    old = TargetResidual(stats, [.1]*29)
    new = TargetResidual(stats, [.1]*29,
        loss_contract=dict(version=PRECLIP_RESIDUAL_LOSS_VERSION, lambda_pre=1.))
    saved = dict(residual_version=old.version, updates=3, model=old.state_dict(),
        configuration=dict(residual_statistics_sha256='statistics'))
    audit = anchor_continue.load_residual_continuation(new, saved, 'statistics')
    assert audit['loss_contract_before'] is None
    assert audit['loss_contract_after'] == new.loss_contract
    assert audit['loss_contract_changed'] and audit['loss_change_preserves_forward_execution']
    assert all(torch.equal(value, old.state_dict()[key]) for key, value in new.state_dict().items())
    saved['configuration']['loss_contract'] = new.loss_contract
    saved['residual_loss_version'] = PRECLIP_RESIDUAL_LOSS_VERSION
    with pytest.raises(ValueError, match='silently discard'):
        anchor_continue.load_residual_continuation(old, saved, 'statistics')


def test_trainer_current_weight_history_replays_prefix_after_each_update(trainer, monkeypatch):
    from visuotactile.models.native_dp import CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION
    root, _, model, _ = trainer
    setup = anchor_continue.residual_setup(root)
    _, config, data, records, _, configuration, _ = setup
    monkeypatch.setattr(anchor_continue, 'residual_setup', lambda *a, **kw:setup)
    data['splits']['train'] = ['t0']
    config.update(maximum_control_tick_epochs=1, minimum_control_tick_epochs=1, microbatch=1)
    records['t0'] = record(7, 80)
    records['t0']['time'] = torch.arange(80)/60
    records['t0']['supervision_mask'] = torch.arange(80)>=32
    model.history_contract = dict(version=CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION)
    configuration['history_contract'] = dict(model.history_contract)
    original = model.forward
    main_states=[];replay_calls=[]
    def forward(inputs, state=None, *, state_only=False):
        if state_only:
            replay_calls.append(len(inputs['valid'][0]))
        else:
            main_states.append((None if state is None else state['last'].detach().clone(), model.weight.detach().clone()))
        raw, next_state = original(inputs, state)
        # Real recurrent outputs are separate storage from mutable parameters.
        next_state = {key:value.clone() for key,value in next_state.items()}
        return (None if state_only else raw), next_state
    monkeypatch.setattr(model, 'forward', forward)
    anchor_continue.residual_train(root)
    saved = torch.load(root/'training/last.pt', map_location='cpu', weights_only=False)
    assert saved['updates'] == 2 and saved['supervised_ticks'] == 48
    assert saved['residual_history_version'] == CURRENT_WEIGHT_RESIDUAL_HISTORY_VERSION
    assert replay_calls == [32, 32, 32]
    assert len(main_states) == 3 and main_states[0][0] is None
    assert torch.equal(main_states[2][0][0], main_states[2][1])
    assert not torch.equal(main_states[1][1], main_states[2][1])


def test_flat_validation_stops_after_registered_minimum_and_four_checks(trainer):
    root,_,_,_=trainer
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    assert finished['control_tick_epochs']==4
    assert finished['supervised_ticks']==84 and finished['required_supervised_ticks']==105
    assert not finished['target_reached'] and not finished['disposable']
    assert finished['minimum_exposure_reached'] and finished['stop_reason']=='validation_plateau_after_minimum_exposure'
    assert finished['formal_updates']==finished['updates']==8
    assert finished['sampled_contact_anchors']==0


def test_plateau_cannot_bypass_minimum_and_common_exposure_is_saved(trainer):
    root,_,_,_=trainer
    _,config,*_=anchor_continue.residual_setup(root)
    config.update(minimum_control_tick_epochs=5,comparison_control_tick_epochs=[2])
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    assert finished['control_tick_epochs']==5 and finished['target_reached']
    common=torch.load(root/'training/comparison_epoch_2.pt',weights_only=False)
    assert common['epoch']==2 and common['supervised_ticks']==42 and common['updates']==4


def test_budget_stop_does_not_claim_full_exposure(trainer):
    root,session,_,_=trainer
    session['training_deadline']=0.
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    assert finished['stop_reason']=='fixed_wall_budget'
    assert not finished['target_reached']
    assert finished['supervised_ticks']<finished['required_supervised_ticks']


def test_selection_metric_counts_noncontact_and_equal_arm_hand(monkeypatch):
    model=TinyResidual()
    records={'a':dict(target=torch.zeros(2,29),gate=torch.tensor([True,False]))}
    prediction=torch.zeros(2,29)
    prediction[1,:7]=2.;prediction[1,7:]=4.
    monkeypatch.setattr(anchor_continue,'residual_prediction',lambda *a,**kw:prediction)
    metrics,_=anchor_continue.residual_metrics(model,records,['a'],None)
    # Tick0 = zero error; tick1 = .5*2^2 + .5*4^2 = 10.
    assert metrics['all']['normalized_equal_mse']==5.
    assert metrics['contact']['normalized_equal_mse']==0.
    assert metrics['noncontact']['normalized_equal_mse']==10.


def test_prefix_advances_history_without_optimizer_or_supervised_exposure(trainer):
    root,_,model,_=trainer
    _,config,data,records,*_=anchor_continue.residual_setup(root)
    data['splits']['train']=['t0'];config.update(maximum_control_tick_epochs=1,minimum_control_tick_epochs=1,microbatch=1,
        learning_rate=3e-4,weight_decay=.002)
    records['t0']=record(7,80);records['t0']['time']=torch.arange(80)/60
    records['t0']['supervision_mask']=torch.arange(80)>=64
    records['t0']['target'][:64]=1000.  # An invalid target must never affect loss.
    anchor_continue.residual_train(root)
    finished=json.loads((root/'training/finished.json').read_text())
    saved=torch.load(root/'training/last.pt',weights_only=False)
    assert model.seen==[7.]*80
    assert model.had_state==[False,True,True]
    assert finished['updates']==1 and finished['supervised_ticks']==16
    assert finished['control_ticks']==80 and finished['target_reached']
    assert saved['optimizer']['param_groups'][0]['lr']==3e-4
    assert saved['optimizer']['param_groups'][0]['weight_decay']==.002
    loss=json.loads((root/'training/loss.jsonl').read_text())
    assert loss['loss']==pytest.approx(1.) and loss['supervised_ticks']==16


@pytest.mark.parametrize('sampling',['uniform','stratified','recovery'])
def test_incremental_resume_after_final_group_preserves_next_updates(trainer,monkeypatch,sampling):
    root,session,model,_=trainer
    stratified=sampling!='uniform'
    _,config,*_=anchor_continue.residual_setup(root)
    config.update(continuation_from={'test':'CPU mocked selected source'},maximum_control_tick_epochs=3,
        minimum_control_tick_epochs=3,learning_rate=3e-4)
    if stratified:
        _,_,data,records,_,saved_config,_=anchor_continue.residual_setup(root)
        data['sampling_strata']=dict(core=['t0','t1','t2','t3'],historical_correction=['t4'],current_correction=['t5'])
        data['splits']['correction_dev']=['d'];records['d']=record(88,3)
        saved_config.update(sampling_strata=data['sampling_strata'])
        if sampling=='recovery':
            config['recovery_prefix']=dict(ticks=48,frontend_fraction=.5,backend_weight=3.)
            saved_config['recovery_prefix']=config['recovery_prefix']
        monkeypatch.setattr(anchor_continue,'residual_metrics',lambda *a,**kw:
            (dict(all={'normalized_equal_mse':1.},initial24={'normalized_equal_mse':1.}),[]))
    anchor_continue.residual_train(root)
    expected=torch.load(root/'training/last.pt',weights_only=False)
    other=root.parent/'resumed';other.mkdir();session['base_contracts']['resumed']={}
    with torch.no_grad():model.weight.zero_()
    original=anchor_continue.save;writes=[]
    def interrupted(path,payload):
        original(path,payload)
        if path.name=='last.pt':
            writes.append(path)
            if len(writes)==2:raise RuntimeError('simulated exit after completed epoch group')
    monkeypatch.setattr(anchor_continue,'save',interrupted)
    with pytest.raises(RuntimeError,match='simulated exit'):anchor_continue.residual_train(other)
    assert not (other/'training/finished.json').exists()
    monkeypatch.setattr(anchor_continue,'save',original)
    anchor_continue.residual_train(other)
    actual=torch.load(other/'training/last.pt',weights_only=False)
    assert actual['updates']==expected['updates']
    assert actual['supervised_ticks']==expected['supervised_ticks']
    if not stratified:assert actual['updates']==6 and actual['supervised_ticks']==63
    assert actual['episode_counts']==expected['episode_counts']
    if sampling=='recovery':
        loss_rows=[json.loads(line) for line in (other/'training/loss.jsonl').read_text().splitlines()]
        assert any(row['loss_weight_mass']>row['supervised_ticks'] for row in loss_rows)
        assert sum(row['supervised_ticks'] for row in loss_rows)==actual['supervised_ticks']
        for row in loss_rows:
            assert row['initial_recovery_supervised_ticks']+row['remaining_supervised_ticks']==row['supervised_ticks']
            assert row['loss_weight_mass']==row['supervised_ticks']+2*row['initial_recovery_supervised_ticks']
    assert torch.equal(actual['model']['weight'],expected['model']['weight'])
    for key in ('step','exp_avg','exp_avg_sq'):
        assert torch.equal(actual['optimizer']['state'][0][key],expected['optimizer']['state'][0][key])
    assert torch.equal(actual['schedule_rng'],expected['schedule_rng'])
    restored=json.loads((other/'training/restore.json').read_text())
    assert restored['passed'] and all(restored['checks'].values())
    # Reading an already completed round never performs an extra optimizer step.
    anchor_continue.residual_train(other)
    assert torch.equal(model.weight,actual['model']['weight'])


def test_selected_residual_reuses_parameters_but_rejects_changed_bounds():
    class Small(torch.nn.Module):
        version=TargetResidual.version
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(2,3)
            self.register_buffer('scales',torch.tensor([.1,.2,.3]))
    source=Small();target=Small()
    saved=dict(residual_version=TargetResidual.version,updates=20,configuration={'residual_statistics_sha256':'fixed'},model=source.state_dict())
    audit=anchor_continue.load_residual_continuation(target,saved,'fixed')
    assert audit['new_optimizer_updates']==0 and audit['statistics_and_delta_buffers_unchanged']
    assert all(torch.equal(value,target.state_dict()[key]) for key,value in source.state_dict().items())
    target.scales[0]=.11
    with pytest.raises(ValueError,match='buffer'):anchor_continue.load_residual_continuation(target,saved,'fixed')


def test_incremental_probe_keeps_formal_scheduler_horizon():
    schedule=dict(maximum_updates=2000,minimum_updates=1000,warmup_updates=100,
        lr_peak=1e-5,lr_floor=1e-6,validation_interval=500)
    assert anchor_continue.incremental_schedule(schedule,2000)['maximum_updates']==2000
    assert anchor_continue.incremental_schedule(schedule,1,smoke=True)['maximum_updates']==2000
    with pytest.raises(ValueError,match='shorten'):anchor_continue.incremental_schedule(schedule,8)


def test_matched_adaptation_preserves_learned_parameters_and_command_statistics():
    class Small(torch.nn.Module):
        version=TargetResidual.version
        def __init__(self,scales,adapter=None):
            super().__init__();self.layer=torch.nn.Linear(2,29);self.execution_adapter=adapter
            self.register_buffer('scales',torch.tensor(scales))
            self.register_buffer('action_scale',torch.arange(1,30).float())
    source=Small([.1]*29);target=Small([.2]*7+[.9]*22,{'version':'test'})
    saved=dict(residual_version=TargetResidual.version,updates=20,
        configuration={'residual_statistics_sha256':'fixed'},model=source.state_dict())
    with pytest.raises(ValueError,match='adapter change'):
        anchor_continue.load_residual_continuation(target,saved,'fixed')
    spec=dict(version='matched_ab_scales_v1',old=[.1]*29,new=[.2]*7+[.9]*22,
        requires_training_before_deployment=True)
    audit=anchor_continue.load_residual_continuation(target,saved,'fixed',scale_adaptation=spec)
    assert audit['requires_training_before_deployment'] and audit['new_optimizer_updates']==0
    assert torch.equal(source.layer.weight,target.layer.weight)
    assert torch.equal(source.layer.bias,target.layer.bias)
    assert torch.equal(source.action_scale,target.action_scale)
    assert torch.equal(saved['model']['scales'],torch.full((29,),.1))
    assert torch.equal(target.scales,torch.tensor(spec['new']))
    target.action_scale[0]=0
    with pytest.raises(ValueError,match='buffer'):
        anchor_continue.load_residual_continuation(target,saved,'fixed',scale_adaptation=spec)
    with pytest.raises(ValueError,match='scales differ'):
        anchor_continue.load_residual_continuation(target,saved,'fixed',scale_adaptation=dict(spec,old=[.2]*29))


def test_residual_offline_projection_uses_actual_pre_action_target():
    class Projected(TinyResidual):
        execution_adapter={'version':'test'}
        def forward(self,inputs,state,*,mask_force=False):
            return super().forward(inputs,state)
        def compose(self,base,raw,contact,limits,*,previous):
            assert torch.equal(previous,torch.full_like(previous,.37))
            return previous+raw,raw
    r=record(0,2);r['robot']=torch.zeros(2,178)
    r['robot'][:,58:87]=.37;r['robot'][:,116:145]=-.72
    model=Projected();lease=SimpleNamespace(block=nullcontext)
    predicted=anchor_continue.residual_prediction(model,r,lease)
    assert torch.equal(predicted,torch.full((2,29),.37))


def test_selection_metric_excludes_unlabeled_prefix(monkeypatch):
    model=TinyResidual();records={'a':dict(target=torch.zeros(2,29),gate=torch.tensor([True,True]),
        supervision_mask=torch.tensor([False,True]))}
    prediction=torch.ones(2,29);prediction[0]=1000
    monkeypatch.setattr(anchor_continue,'residual_prediction',lambda *a,**kw:prediction)
    metrics,_=anchor_continue.residual_metrics(model,records,['a'],None)
    assert metrics['all']['ticks']==1 and metrics['all']['normalized_equal_mse']==1.


def test_dagger_setup_keeps_original_stat_population_and_rejects_stale_cache(tmp_path,monkeypatch):
    from visuotactile import state_resources
    from visuotactile.native_data import supervision_mask_sha256
    root=tmp_path/'G11';root.mkdir();data_root=root/'state_data';data_root.mkdir()
    nominal=root/'nominal';nominal.mkdir();normal=tmp_path/'command_statistics';normal.mkdir()
    def write(path,value):path.write_text(json.dumps(value))
    data=dict(splits={'train':['old','new'],'validation':['val']},normalization_train_ids=['old'])
    write(data_root/'data.json',data)
    identity='u_expert_train166_affine_v1'
    write(normal/'data.json',dict(data,normalization_identity=identity))
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    torch.save(stats,normal/'statistics.pt')
    torch.save(dict(stats,action_scale=torch.full((29,),2.)),data_root/'statistics.pt')
    stat_sha=anchor_continue.sha256(normal/'statistics.pt')
    base=tmp_path/'updated_dp.pt';base.write_bytes(b'new DP identity')
    base_sha=anchor_continue.sha256(base)
    source_model=TargetResidual(stats,[.01]*29)
    source=tmp_path/'prior_selected.pt'
    torch.save(dict(residual_version=TargetResidual.version,updates=10,model=source_model.state_dict(),
        configuration={'residual_statistics_sha256':stat_sha}),source)
    continuation=dict(path=str(source),sha256=anchor_continue.sha256(source),teacher_supervision='u_expert',
        statistics_sha256=stat_sha,normalization_identity=identity,round_index=1,source_group='G11')
    pair=dict(version=TargetResidual.version,bases={'G11':{'checkpoint':str(base),'sha256':base_sha}},
        residual_statistics_path=str(normal/'statistics.pt'),initialization_seed=3,continuation_from=continuation)
    write(root/'configuration.json',{'residual_pair':pair})
    write(root/'session.json',dict(dino_sha256='fixed',resource_root=str(tmp_path),training_deadline=123))
    binding=dict(base_checkpoint_sha256=base_sha,statistics_sha256=anchor_continue.sha256(data_root/'statistics.pt'),
        data_sha256=anchor_continue.sha256(data_root/'data.json'),residual_statistics_sha256=stat_sha)
    rows={i:dict(time=torch.arange(3),supervision_mask=torch.ones(3,dtype=torch.bool),
        control_segment=torch.zeros(3,dtype=torch.long),cache_identity=dict(binding)) for i in ('old','new','val')}
    index=dict(base_sha256=base_sha,data_sha256=binding['data_sha256'],residual_statistics_sha256=stat_sha,
        residual_normalization_identity=identity,supervision_mask_sha256=supervision_mask_sha256(rows),episodes={})
    for key,row in rows.items():
        torch.save(row,nominal/(key+'.pt'));index['episodes'][key]={'sha256':anchor_continue.sha256(nominal/(key+'.pt'))}
    write(nominal/'index.json',index)
    write(nominal/'scales.json',dict(scales=[.01]*29,residual_statistics_sha256=stat_sha,residual_normalization_identity=identity))
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext))
    monkeypatch.setattr(TargetResidual,'cuda',lambda self:self)
    *_,model,config,_=anchor_continue.residual_setup(root)
    assert config['normalization_train_ids']==['old'] and torch.equal(model.action_scale,stats['action_scale'])
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in source_model.state_dict().items())
    _,_,smoke_data,verified_records,*_=anchor_continue.residual_setup(root,smoke=True)
    assert smoke_data['splits']['train']==['old'] and set(verified_records)=={'old','new','val'}
    # Even a fully rehashed cache file must identify the newly updated DP.
    rows['new']['cache_identity']['base_checkpoint_sha256']='0'*64
    torch.save(rows['new'],nominal/'new.pt');index['episodes']['new']['sha256']=anchor_continue.sha256(nominal/'new.pt')
    write(nominal/'index.json',index)
    with pytest.raises(ValueError,match='episode cache source'):anchor_continue.residual_setup(root)


def test_stratified_round_measures_update_zero_and_keeps_it_if_correction_worsens(trainer,monkeypatch):
    root,_,model,_=trainer
    session,config,data,records,_,saved_config,lease=anchor_continue.residual_setup(root)
    config.update(continuation_from={'test':'trained source'},maximum_control_tick_epochs=2,
        minimum_control_tick_epochs=2,learning_rate=3e-4)
    data['sampling_strata']=dict(core=['t0','t1','t2','t3'],historical_correction=['t4'],current_correction=['t5'])
    data['splits']['correction_dev']=['d'];records['d']=record(88,80)
    records['d']['time']=torch.arange(80)/60;records['d']['supervision_mask']=torch.arange(80)>=64
    saved_config.update(sampling_strata=data['sampling_strata'])
    measured=[]
    def metrics(model,records,ids,lease):
        trained=bool(model.weight.detach().abs().sum())
        measured.append((list(ids),trained))
        value=(2. if trained else 1.) if ids==['d'] else (.9 if trained else 1.)
        return dict(all={'normalized_equal_mse':value},initial24={'normalized_equal_mse':value}),[]
    monkeypatch.setattr(anchor_continue,'residual_metrics',metrics)
    anchor_continue.residual_train(root)
    chosen=torch.load(root/'training/best.pt',weights_only=False)
    done=json.loads((root/'training/finished.json').read_text())
    assert chosen['updates']==chosen['new_optimizer_updates']==0
    assert done['new_optimizer_updates']>0
    assert chosen['selection_baseline']==dict(old_validation=1.,correction_dev=1.,correction_initial24=1.)
    assert (['d'],False) in measured and (['d'],True) in measured
    assert done['coverage']['strata']['historical_correction']['supervised_ticks']>0
    assert not any(x==88 for x in model.seen)  # Held dev never reaches the actual optimizer.
    before=model.weight.detach().clone();count=len(model.seen)
    anchor_continue.residual_train(root)
    assert len(model.seen)==count and torch.equal(model.weight,before)


def test_multiround_probe_covers_correction_prefix_cost_at_formal_learning_rate(trainer):
    root,_,model,_=trainer
    _,config,data,*_=anchor_continue.residual_setup(root)
    config['learning_rate']=3e-4
    data['sampling_strata']=dict(core=['t0','t1','t2','t3'],historical_correction=['t4'],current_correction=['t5'])
    anchor_continue.residual_train(root,probe=True)
    saved=torch.load(root/'training_probe/last.pt',weights_only=False)
    assert saved['train_ids']==['t5','t4','t0','t1']
    assert set(model.seen)=={5.,4.,0.,1.}
    assert saved['optimizer']['param_groups'][0]['lr']==3e-4
    assert saved['formal_updates']==0 and saved['disposable']


def test_contact_phases_reuse_actions_and_mask_only_expert_error():
    pred=torch.arange(7,dtype=torch.float32)[:,None].expand(7,29)
    rec=dict(base=torch.zeros(7,29),target=torch.zeros(7,29),
        gate=torch.tensor([False,True,True,False,False,True,True]),
        supervision_mask=torch.tensor([False,False,True,True,True,True,True]))
    rows=anchor_continue.contact_phase_metrics(pred,rec,identity='one',split='correction_dev',masked=pred-.25)
    by_phase={r['contact_phase']:r for r in rows}
    assert {k:r['ticks'] for k,r in by_phase.items()}==dict(precontact=1,initialcontact=2,sustained=2,lostcontact=2)
    assert by_phase['precontact']['final_command_equal_mse_rad2'] is None
    assert by_phase['initialcontact']['supervised_ticks']==1
    assert by_phase['initialcontact']['final_command_equal_mse_rad2']==25.
    assert by_phase['lostcontact']['correction_rms_rad']==pytest.approx((12.5)**.5)
    assert by_phase['lostcontact']['correction_max_abs_rad']==4.
    assert all(r['force_zero_output_rms_rad']==.25 for r in rows)
    assert all(r['raw_saturation_fraction'] is None for r in rows)
    missing=anchor_continue.contact_phase_metrics(pred,rec,identity='one',split='train')
    assert all(not r['force_zero_available'] and r['force_zero_output_rms_rad'] is None for r in missing)


def test_actual_trainer_calls_projection_objective_and_persists_identity(trainer, monkeypatch):
    from visuotactile.execution_learning import PROJECTION_FY_LOSS
    root, _, model, _ = trainer
    setup=anchor_continue.residual_setup(root)
    for r in setup[3].values():r['target'].fill_(.005)
    model.loss_contract=dict(PROJECTION_FY_LOSS);setup[5]['loss_contract']=dict(PROJECTION_FY_LOSS)
    monkeypatch.setattr(anchor_continue,'residual_setup',lambda *a,**kw:setup)
    def compose(base,raw,contact,limits):
        pre=base+raw+1
        bounds=torch.stack((torch.full_like(pre,-.01),torch.full_like(pre,.01)),-1)
        return dict(u_exec=pre.clamp(-.01,.01),delta=raw,u_pre=pre,execution_bounds=bounds)
    monkeypatch.setattr(model,'compose_chain',compose,raising=False)
    monkeypatch.setattr(model,'objective_components',lambda *a,**kw:TargetResidual.objective_components(model,*a,**kw),raising=False)
    anchor_continue.residual_train(root,probe=True)
    saved=torch.load(root/'training_probe/last.pt',map_location='cpu',weights_only=False)
    rows=[json.loads(r) for r in (root/'training_probe/loss.jsonl').read_text().splitlines()]
    assert saved['residual_loss_version']==PROJECTION_FY_LOSS['version']
    assert saved['configuration']['loss_contract']==PROJECTION_FY_LOSS
    assert rows[0]['loss_components']['fy_fit']>rows[0]['loss_components']['exec_fit']
    assert (saved['model']['weight']<0).all()
