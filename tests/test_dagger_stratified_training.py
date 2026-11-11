"""Small correction pools must affect real sampling, selection and durable recovery."""
import copy
import json

import pytest
import torch

from visuotactile.anchor_sampling import (AnchorSampler,episode_exposure,
    relative_selection,sampling_strata,stratified_episode_order,recovery_prefix_rows,
    recovery_prefix_config,residual_recovery_weights)
from visuotactile.anchor_continue import reconcile_update_log,trained_source_updates
from visuotactile.state_image_offline import sampled_metrics


def record(length,prefix=0):
    return dict(time=torch.arange(length)/60,command=torch.zeros(length,29),
        supervision_mask=torch.arange(length)>=prefix,
        control_segment=(torch.arange(length)>=prefix).long())


def test_scarce_corrections_receive_mass_and_restored_decks_match():
    core=[f'core{i}' for i in range(166)]
    strata=dict(core=core,historical_correction=['historical'],current_correction=['current'])
    records={i:record(96) for i in core}
    records.update(historical=record(96,48),current=record(96,48))
    lengths={i:len(r['time']) for i,r in records.items()}
    rng=torch.Generator().manual_seed(51)
    sampler=AnchorSampler(lengths,rng,64,records=records,strata=strata)
    for _ in range(10):
        draws=sampler.draw()
        assert len(draws)==64
        assert all(t>=48 for i,t in draws if i in ('historical','current'))
    coverage=sampler.coverage()
    assert coverage['strata']['core']['sampled_anchors']==320
    assert coverage['strata']['historical_correction']['sampled_anchors']==128
    assert coverage['strata']['current_correction']['sampled_anchors']==192
    assert coverage['strata']['current_correction']['unique_anchors']==4
    assert coverage['strata']['current_correction']['repeated_samples']==188
    state=copy.deepcopy(sampler.state_dict());rng_state=rng.get_state()
    expected=[sampler.draw() for _ in range(7)]
    new_rng=torch.Generator();new_rng.set_state(rng_state)
    resumed=AnchorSampler(lengths,new_rng,64,records=records,strata=strata)
    resumed.load_state_dict(state)
    assert [resumed.draw() for _ in range(7)]==expected
    assert resumed.coverage()==sampler.coverage()


def test_empty_current_redistributes_without_loading_development():
    pools,weights=sampling_strata(['c','h'],dict(core=['c'],historical_correction=['h'],current_correction=[]))
    assert weights=={'core':.5,'historical_correction':.5}
    with pytest.raises(ValueError,match='partition'):
        sampling_strata(['c','h'],dict(core=['c'],historical_correction=['h','dev'],current_correction=[]))
    with pytest.raises(ValueError,match='half'):
        sampling_strata(['c','h'],dict(core=['c'],historical_correction=['h'],current_correction=[]),
            dict(core=.8,historical_correction=.1,current_correction=.1))


def test_backend_counts_supervised_ticks_and_repeated_episodes_not_prefix():
    ids=['c0','c1','c2','h','n']
    records={i:record(120) for i in ids[:3]}
    records.update(h=record(120,108),n=record(120,96))
    strata=dict(core=ids[:3],historical_correction=['h'],current_correction=['n'])
    rng=torch.Generator().manual_seed(71)
    order=stratified_episode_order(ids,records,strata,None,rng)
    counts={i:order.count(i) for i in ids}
    result=episode_exposure(records,counts,strata)
    core=result['strata']['core']['supervised_ticks']
    correction=sum(result['strata'][k]['supervised_ticks'] for k in ('historical_correction','current_correction'))
    assert correction>=core
    assert result['supervised_ticks']==sum(counts[i]*int(records[i]['supervision_mask'].sum()) for i in ids)
    assert result['repeated_samples']>0
    assert result['strata']['historical_correction']['unique_supervised_ticks']==12
    assert result['strata']['historical_correction']['supervised_ticks']%12==0


def test_selection_scales_each_component_to_real_update_zero():
    score,baseline,ratios=relative_selection(dict(old_validation=100.,correction_dev=2.,correction_initial24=.5))
    assert score==1 and set(ratios.values())=={1.}
    score,_,_=relative_selection(dict(old_validation=80.,correction_dev=4.,correction_initial24=1.),baseline)
    assert score==pytest.approx(1.4)  # Better Core does not hide doubled correction errors.
    with pytest.raises(ValueError):relative_selection(dict(old_validation=1.,correction_dev=None,correction_initial24=1.))


def test_invalid_student_prefix_is_not_offline_error():
    sample=dict(prediction=torch.ones(50,29),truth=torch.zeros(50,29),hold=torch.zeros(50,29),
        ee_mm=torch.zeros(50),rotation_deg=torch.zeros(50),valid=torch.arange(50)>=40)
    sample['prediction'][:40]=10000
    metrics=sampled_metrics(sample)
    assert metrics['ticks']==10 and metrics['arm_rmse']==metrics['hand_rmse']==1


def test_crash_tail_is_preserved_but_cannot_double_count_updates(tmp_path):
    path=tmp_path/'loss.jsonl'
    path.write_text(''.join(json.dumps(dict(update=i,loss=.1*i))+'\n' for i in range(1,6))+'{"update":6')
    assert reconcile_update_log(path,3)==3
    assert [json.loads(x)['update'] for x in path.read_text().splitlines()]==[1,2,3]
    audit=[json.loads(x) for x in path.with_name('loss.uncommitted.jsonl').read_text().splitlines()]
    assert [x['record']['update'] for x in audit[:2]]==[4,5]
    assert audit[-1]['record']['reason']=='partial_write'
    assert reconcile_update_log(path,3)==0


def test_update_zero_requires_explicit_trained_continuation_provenance(tmp_path):
    from visuotactile.compact import sha256
    assert trained_source_updates({'updates':0})==0
    source=tmp_path/'trained.pt';torch.save({'updates':44},source)
    assert trained_source_updates(dict(updates=0,new_optimizer_updates=0,
        configuration=dict(continuation_from={'path':str(source),'sha256':sha256(source)},
            continuation_audit={'source_updates':44,'source_checkpoint_updates':44})))==44


RECOVERY=dict(ticks=48,frontend_fraction=.5,backend_weight=3.)


def test_recovery_prefix_never_bridges_invalid_labels_or_control_switch():
    r=record(140,48)
    assert recovery_prefix_rows(r).nonzero().flatten().tolist()==list(range(48,96))
    r['supervision_mask'][61]=False
    assert recovery_prefix_rows(r).nonzero().flatten().tolist()==list(range(48,61))
    r['supervision_mask'][61]=True;r['control_segment'][60:]=2
    assert recovery_prefix_rows(r).nonzero().flatten().tolist()==list(range(48,60))
    r['supervision_mask'][:]=False
    assert not recovery_prefix_rows(r).any()
    with pytest.raises(ValueError):recovery_prefix_config(dict(RECOVERY,ticks=True))
    with pytest.raises(ValueError):recovery_prefix_config(dict(RECOVERY,backend_weight=float('nan')))


def test_recovery_anchor_decks_retain_all_labels_and_exact_resume():
    records=dict(c=record(240),h=record(240,36),n=record(240,48))
    lengths={i:len(r['time']) for i,r in records.items()}
    strata=dict(core=['c'],historical_correction=['h'],current_correction=['n'])
    rng=torch.Generator().manual_seed(619)
    sampler=AnchorSampler(lengths,rng,64,records=records,strata=strata,recovery_prefix=RECOVERY)
    for _ in range(20):sampler.draw()
    result=sampler.coverage()
    for name in ('historical_correction','current_correction'):
        recovery=result['strata'][name]['recovery_prefix']
        assert recovery['actual_initial_anchor_fraction']==.5
        assert recovery['initial_anchor_population']==recovery['unique_initial_anchors']==4
        assert recovery['remaining_anchor_population']==recovery['unique_remaining_anchors']>4
    assert result['unique_anchors']==result['anchor_population']
    state=copy.deepcopy(sampler.state_dict());rng_state=rng.get_state()
    expected=[sampler.draw() for _ in range(7)]
    new_rng=torch.Generator();new_rng.set_state(rng_state)
    resumed=AnchorSampler(lengths,new_rng,64,records=records,strata=strata,recovery_prefix=RECOVERY)
    resumed.load_state_dict(state)
    assert [resumed.draw() for _ in range(7)]==expected
    assert resumed.coverage()==sampler.coverage()
    changed=AnchorSampler(lengths,torch.Generator(),64,records=records,strata=strata,recovery_prefix=dict(RECOVERY,ticks=24))
    with pytest.raises(ValueError,match='recovery-prefix'):changed.load_state_dict(state)
    legacy=AnchorSampler(lengths,torch.Generator(),64,records=records,strata=strata)
    with pytest.raises(ValueError,match='recovery-prefix'):legacy.load_state_dict(state)


def test_short_correction_prefix_does_not_drop_valid_tail_or_invent_anchors():
    records=dict(c=record(48),h=record(62,49),n=record(110,1))
    # A valid segment ending before the next planner anchor has no early anchor;
    # later expert labels remain legal and must still be sampled as remaining.
    records['n']['supervision_mask'][4:24]=False
    records['n']['control_segment'][4:24]=2;records['n']['control_segment'][24:]=3
    strata=dict(core=['c'],historical_correction=['h'],current_correction=['n'])
    sampler=AnchorSampler({i:len(r['time']) for i,r in records.items()},torch.Generator().manual_seed(7),64,
        records=records,strata=strata,recovery_prefix=RECOVERY)
    samples=[s for _ in range(4) for s in sampler.draw()]
    assert all(t==60 for i,t in samples if i=='h')
    assert all(t>=24 for i,t in samples if i=='n')
    coverage=sampler.coverage()['strata']
    assert coverage['historical_correction']['recovery_prefix']['episodes_without_remaining_anchor']==1
    assert coverage['current_correction']['recovery_prefix']['episodes_without_initial_anchor']==1


def test_weighted_loss_supervises_composed_command_and_counts_physical_ticks():
    from visuotactile.models.native_dp import TargetResidual
    from types import SimpleNamespace
    mask=torch.tensor([[False,True,True,True,False],[True,True,False,False,False]])
    prefixes=[torch.tensor([False,True,False,False,True]),torch.zeros(5,dtype=torch.bool)]
    weights,early=residual_recovery_weights(prefixes,0,5,mask,3.)
    assert weights.tolist()==[[0.,3.,1.,1.,0.],[1.,1.,0.,0.,0.]]
    assert int(mask.sum())==5 and float(weights.sum())==7 and int(early.sum())==1
    base=torch.full((2,5,29),2.);raw=torch.zeros_like(base,requires_grad=True)
    target=torch.zeros_like(base);target[0,1]=4.;target[~mask]=10000
    model=SimpleNamespace(action_scale=torch.ones(29))
    loss=TargetResidual.objective(model,base+raw,target,raw,weights)
    # Every eligible final-command error is +/-2; residual-alone fitting would
    # wrongly have zero error on four rows and a much larger early-row error.
    assert float(loss)==pytest.approx(4.)
    loss.backward();assert not raw.grad[~mask].any()
    assert raw.grad[0,1,0]==pytest.approx(-3*float(raw.grad[0,2,0]))
    records=dict(c=record(80),h=record(80,16),n=record(80,32))
    counts=dict(c=2,h=3,n=4);strata=dict(core=['c'],historical_correction=['h'],current_correction=['n'])
    report=episode_exposure(records,counts,strata,recovery_prefix=RECOVERY)
    assert report['supervised_ticks']==80*2+64*3+48*4
    assert report['strata']['core']['recovery_prefix']['loss_weight_mass']==160
    historical=report['strata']['historical_correction']['recovery_prefix']
    assert historical['initial_supervised_tick_draws']==48*3
    assert historical['remaining_supervised_tick_draws']==16*3
    assert historical['loss_weight_mass']==48*3*3+16*3
    assert not historical['loss_weight_mass_is_exposure']


def test_missing_recovery_configuration_preserves_original_sampler_state():
    lengths=dict(c=120,h=144,n=156)
    records={i:record(n,24 if i!='c' else 0) for i,n in lengths.items()}
    strata=dict(core=['c'],historical_correction=['h'],current_correction=['n'])
    a=AnchorSampler(lengths,torch.Generator().manual_seed(51),64,records=records,strata=strata)
    b=AnchorSampler(lengths,torch.Generator().manual_seed(51),64,records=records,strata=strata,recovery_prefix=None)
    for _ in range(7):assert a.draw()==b.draw()
    assert a.state_dict()==b.state_dict()
    assert not any(key.startswith('recovery') for key in a.state_dict())
    assert torch.equal(a.generator.get_state(),b.generator.get_state())

