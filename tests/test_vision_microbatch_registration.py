"""Prepare-only registration; no Kit, model loads or CUDA work."""
import json

import pytest


@pytest.mark.parametrize('size',[2,16])
@pytest.mark.parametrize('exact_tail',[False,True])
def test_formal_prepare_registers_vision_outside_physics_backend(tmp_path,monkeypatch,size,exact_tail):
    import tools.tight_fresh_batch_run as formal
    import tools.tight_fresh_rollouts as serial
    calls=[]
    def prepare(root,output,count,seed):
        calls.append((count,seed));output.mkdir()
        (output/'plan.json').write_text(json.dumps(dict(kind='autonomous')))
        (output/'session.json').write_text(json.dumps(dict(expert_sha256='expert')))
    monkeypatch.setattr(serial,'prepare',prepare)
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    options=dict(vision_tail_profile=EXACT_TAIL_PROFILE) if exact_tail else {}
    plan=formal.prepare(tmp_path,tmp_path/'candidate',lanes=128,count=3,seed=17,vision_microbatch=size,**options)
    assert calls==[(3,17)] and plan['batch']['lanes']==128
    assert plan['batch']['vision_microbatch']==size
    assert 'vision_microbatch' not in plan['execution_backend']
    assert 'vision_tail_profile' not in plan['execution_backend']
    if exact_tail:assert plan['batch']['vision_tail_profile']==EXACT_TAIL_PROFILE
    else:assert 'vision_tail_profile' not in plan['batch']
    assert json.loads((tmp_path/'candidate/plan.json').read_text())==plan


def test_formal_prepare_rejects_invalid_vision_before_creating_plan(tmp_path):
    from tools.tight_fresh_batch_run import prepare
    for value in (0,-1,True,1.5,'8'):
        with pytest.raises(ValueError,match='vision_microbatch'):
            prepare(tmp_path,tmp_path/'bad',lanes=8,vision_microbatch=value)
    assert not (tmp_path/'bad').exists()


def test_unknown_tail_rejected_by_all_preparers_before_touching_data(tmp_path):
    from tools import tight_fresh_batch_run as formal,tight_fresh_batch_eval as probe,tight_fresh_expansion as expansion
    calls=[lambda:formal.prepare(tmp_path,tmp_path/'bad',lanes=8,vision_tail_profile={}),
        lambda:probe.prepare(tmp_path,tmp_path/'bad',8,2,vision_tail_profile={}),
        lambda:expansion.prepare(tmp_path,tmp_path/'bad',session_root=tmp_path,registry_id='new',seeds=[1],
            partition_manifest=tmp_path/'absent',frozen_families=tmp_path/'absent',schedules=['fixed40'],
            lanes=8,event_seed=2,round_index=9,vision_tail_profile={})]
    for call in calls:
        with pytest.raises(ValueError,match='vision_tail_profile'):call()
        assert not (tmp_path/'bad').exists()
