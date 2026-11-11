"""Small correction pools must affect real sampling, selection and durable recovery."""
import copy
import json

import pytest
import torch

from visuotactile.anchor_sampling import (AnchorSampler,episode_exposure,
    relative_selection,sampling_strata,stratified_episode_order)
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

