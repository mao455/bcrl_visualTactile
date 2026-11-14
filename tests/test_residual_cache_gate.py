"""CPU admission, exact source binding, and hard finite owner behavior."""
import copy
import json
from pathlib import Path
import time

import pytest
import torch

from tools import tight_residual_cache_gate as gate
from test_selected_frontend_fit import artifacts, write


def cache_artifacts(artifacts):
    root,frozen,saved=artifacts;rr=root/'B'
    data=gate.read(rr/'state_data/data.json');cache=gate.read(rr/'cache_index.json')
    for identity in ('old1','new1'):
        path=root/(identity+'.h5');path.write_bytes(('train raw '+identity).encode())
        entry=data['entries'][identity];entry.update(path=str(path),episode=identity,source_sha256=gate.sha256(path))
        if identity=='old1':entry.pop('trajectory_kind');entry.pop('split')
        visual=rr/'state_data/perception'/(identity+'.pt');visual.write_bytes(('frozen patch '+identity).encode())
        cache['entries'][identity]=dict(path=str(visual),sha256=gate.sha256(visual),
            source_sha256=entry['source_sha256'],frames=entry['length'])
    write(rr/'state_data/data.json',data);cache['data_sha256']=gate.sha256(rr/'state_data/data.json')
    write(rr/'cache_index.json',cache)
    config=saved['configuration'];config.update(data_sha256=cache['data_sha256'],cache_index_sha256=gate.sha256(rr/'cache_index.json'))
    write(rr/'training/configuration.json',config);torch.save(saved,rr/'training/best.pt')
    write(rr/'training/finished.json',dict(complete=True,checkpoint_sha256=gate.sha256(rr/'training/best.pt')))
    torch.save(dict(data_sha256=cache['data_sha256'],records={i:dict(robot=torch.arange(10)) for i in ('old1','new1')}),rr/'state_data/numeric.pt')
    write(root/'session.json',dict(deadline=time.time()+3600))
    return root,frozen


def test_real_schema_prepare_is_cpu_only_best_bound_and_train_only(artifacts,tmp_path):
    root,frozen=cache_artifacts(artifacts);out=tmp_path/'gate'
    plan=gate.prepare(root,'B',frozen,root,out,anchor_batch=32,max_seconds=600)
    assert not torch.cuda.is_initialized() and plan['cuda_initialized'] is False
    assert plan['receipt']['epoch']==3 and plan['receipt']['paths']['checkpoint'].endswith('/best.pt')
    assert plan['receipt']['identities']==['old1','new1']
    assert set(plan['receipt']['original_sources'])=={'old1','new1'}
    projected=Path(plan['receipt']['projection']['root'])
    assert set(torch.load(projected/'numeric.pt',weights_only=False)['records'])=={'old1','new1'}
    assert gate.read(projected/'data.json')['splits']==dict(train=['old1','new1'],validation=[],test=[])
    assert plan['original_session_deadline']==gate.read(root/'session.json')['deadline']
    assert plan['max_seconds']==600 and plan['automatic_retry'] is False
    with pytest.raises(FileExistsError):gate.prepare(root,'B',frozen,root,out)
    # No fabricated fallback to last.pt.
    (root/'B/training/best.pt').unlink();torch.save({},root/'B/training/last.pt')
    with pytest.raises(FileNotFoundError):gate.prepare(root,'B',frozen,root,tmp_path/'missing_best')
    assert gate.read(tmp_path/'missing_best/prepare_error.json')['automatic_retry'] is False


@pytest.mark.parametrize('seconds',[None,True,0,59,901,10000])
def test_prepare_rejects_nonfinite_budget_before_output(tmp_path,seconds):
    with pytest.raises(ValueError,match='Finite'):
        gate.prepare(tmp_path,'B',tmp_path,tmp_path,tmp_path/'out',max_seconds=seconds)
    assert not (tmp_path/'out').exists()


def test_prepare_rejects_mutated_train_raw_before_gpu(artifacts,tmp_path):
    root,frozen=cache_artifacts(artifacts)
    (root/'old1.h5').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):
        gate.prepare(root,'B',frozen,root,tmp_path/'bad')
    assert not torch.cuda.is_initialized()
    assert (tmp_path/'bad/prepare_error.json').is_file()


def test_deadline_stop_and_source_binding_fail_closed(artifacts,tmp_path):
    root,frozen=cache_artifacts(artifacts);out=tmp_path/'gate'
    plan=gate.prepare(root,'B',frozen,root,out)
    gate.verify_sources(plan,Path(plan['source_tree']))
    with pytest.raises(TimeoutError):gate.check_bounds(plan,out,{'work_deadline':time.time()-1})
    (out/'STOP').touch()
    with pytest.raises(InterruptedError):gate.check_bounds(plan,out,{'work_deadline':time.time()+30})
    changed=copy.deepcopy(plan);changed['source_sha256']['visuotactile/native_data.py']='bad'
    with pytest.raises(ValueError,match='source bytes'):gate.verify_sources(changed,Path(plan['source_tree']))
    write(root/'session.json',dict(deadline=time.time()+3600))
    with pytest.raises(ValueError,match='SHA256'):gate.verify_sources(plan,Path(plan['source_tree']))


def owner_fixture(tmp_path,monkeypatch):
    from tools import tight_batch_owner as batch_owner
    from tools import tight_fresh_eval_recovery as recovery
    output=tmp_path/'gate';output.mkdir();resource=tmp_path/'resource';resource.mkdir()
    plan=dict(version=gate.VERSION,resource_root=str(resource),original_session_deadline=time.time()+3600,max_seconds=60)
    write(output/'plan.json',plan)
    monkeypatch.setattr(gate,'verify_sources',lambda *a:None)
    events=[]
    monkeypatch.setattr(batch_owner,'bind_worker_group',lambda pid:dict(pid=pid))
    monkeypatch.setattr(batch_owner,'cleanup_owned_group',lambda binding:events.append('cleanup') or dict(live_descendants_exited=True))
    monkeypatch.setattr(recovery,'stop_owned',lambda process:events.append('stop'))
    return output,resource,events


def test_owner_stops_owned_hung_worker_and_preserves_failure_without_retry(tmp_path,monkeypatch):
    output,resource,events=owner_fixture(tmp_path,monkeypatch)
    class Worker:
        pid=123456;returncode=None
        def __init__(self,*args,**kw):
            assert kw['start_new_session'] is True
            events.append('launch')
        def poll(self):return None
    monkeypatch.setattr(gate.subprocess,'Popen',Worker)
    checks=0
    def check(*args):
        nonlocal checks;checks+=1
        if checks>=3:raise TimeoutError('bounded')
    monkeypatch.setattr(gate,'check_bounds',check);monkeypatch.setattr(gate.time,'sleep',lambda n:None)
    with pytest.raises(ValueError,match='Bounded cache gate failed'):gate.owner(output,tmp_path)
    result=gate.read(output/'owner_finished.json');started=gate.read(output/'owner_started.json')
    assert result['complete'] is False and result['automatic_retry'] is False
    assert 'TimeoutError' in result['error']
    assert started['deadline']-started['utc']==60 and started['work_deadline']==started['deadline']-30
    assert events==['launch','stop','cleanup']
    with pytest.raises(FileExistsError):gate.owner(output,tmp_path)
    assert events==['launch','stop','cleanup']


def test_owner_requires_complete_passing_result_and_cleanup(tmp_path,monkeypatch):
    output,resource,events=owner_fixture(tmp_path,monkeypatch)
    class Worker:
        pid=123456;returncode=0
        def __init__(self,*a,**kw):write(output/'result.json',dict(complete=True,passed=False))
        def poll(self):return 0
    monkeypatch.setattr(gate.subprocess,'Popen',Worker)
    with pytest.raises(ValueError,match='Bounded'):gate.owner(output,tmp_path)
    assert gate.read(output/'owner_finished.json')['passed'] is False
    assert events==['stop','cleanup']
