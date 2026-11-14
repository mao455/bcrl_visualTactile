"""Prepare-only registration; no Kit, model loads or CUDA work."""
import json

import pytest


@pytest.mark.parametrize('size',[2,16])
def test_formal_prepare_registers_vision_outside_physics_backend(tmp_path,monkeypatch,size):
    import tools.tight_fresh_batch_run as formal
    import tools.tight_fresh_rollouts as serial
    calls=[]
    def prepare(root,output,count,seed):
        calls.append((count,seed));output.mkdir()
        (output/'plan.json').write_text(json.dumps(dict(kind='autonomous')))
        (output/'session.json').write_text(json.dumps(dict(expert_sha256='expert')))
    monkeypatch.setattr(serial,'prepare',prepare)
    plan=formal.prepare(tmp_path,tmp_path/'candidate',lanes=128,count=3,seed=17,vision_microbatch=size)
    assert calls==[(3,17)] and plan['batch']['lanes']==128
    assert plan['batch']['vision_microbatch']==size
    assert 'vision_microbatch' not in plan['execution_backend']
    assert json.loads((tmp_path/'candidate/plan.json').read_text())==plan


def test_formal_prepare_rejects_invalid_vision_before_creating_plan(tmp_path):
    from tools.tight_fresh_batch_run import prepare
    for value in (0,-1,True,1.5,'8'):
        with pytest.raises(ValueError,match='vision_microbatch'):
            prepare(tmp_path,tmp_path/'bad',lanes=8,vision_microbatch=value)
    assert not (tmp_path/'bad').exists()
