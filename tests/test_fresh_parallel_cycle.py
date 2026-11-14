from pathlib import Path
import threading

import pytest

from tools.tight_fresh_cycle import paired_frontends, parallel_admission


@pytest.mark.parametrize('failed',[None,'A','B'])
def test_independent_frontends_join_and_preserve_both_outcomes(failed):
    barrier=threading.Barrier(2,timeout=2)
    seen={}
    def execute(name,args,completion,*,update_status):
        role=name[-1]
        assert args==['tools.tight_fresh_train','train','--root',Path('/round'),'--role',role,'--epochs',10]
        assert completion==Path('/round')/role/'training/finished.json'
        assert not update_status
        barrier.wait()
        seen[role]='failed' if role==failed else 'complete'
        if role==failed:raise RuntimeError('saved failed stage')
    result=paired_frontends(execute,Path('/round'),10)
    assert set(seen)=={'A','B'}
    assert result['complete']==(failed is None)
    assert set(result['errors'])==({failed} if failed else set())
    assert result['ended']>=result['started']


@pytest.mark.parametrize('free,ram,age,expected',[
    (22,24,1,True),(21.99,24,1,False),(22,23.99,1,False),
    (22,24,9,False),(22,24,-1,False)])
def test_parallel_joint_launch_admission_keeps_extra_headroom(free,ram,age,expected):
    row=dict(utc_epoch=100-age,gpu_free_mib=free*1024,ram_available_bytes=ram*1024**3)
    assert parallel_admission(row,100)['admitted'] is expected


def test_episode_staging_cannot_change_after_either_learner_starts(tmp_path):
    import json
    from tools.tight_fresh_cycle import configure_episode_io
    config=dict(version='frozen_episode_prefetch_v1',max_bytes=1024<<20,minimum_available_bytes=10<<30)
    for role in ('A','B'):(tmp_path/role).mkdir()
    (tmp_path/'B/training').mkdir()
    with pytest.raises(ValueError):configure_episode_io(tmp_path,config)
    assert not (tmp_path/'A/episode_io.json').exists()
    (tmp_path/'B/training').rmdir()
    assert configure_episode_io(tmp_path,config)==config
    assert json.loads((tmp_path/'A/episode_io.json').read_text())==json.loads((tmp_path/'B/episode_io.json').read_text())==config
    with pytest.raises(ValueError):configure_episode_io(tmp_path,dict(config,max_bytes=512<<20))
