"""Drive the formal dispatcher with a controlled clock, without Kit or CUDA."""
import contextlib
import json
from types import SimpleNamespace

from visuotactile.compact import sha256


def test_formal_driver_records_admission_separately_from_overlapping_wave_time(tmp_path,monkeypatch):
    from tools import tight_fresh_batch_run as batch
    import tools.tight_state as state
    import visuotactile.batched_rollout as rollout
    import visuotactile.state_resources as resources
    import visuotactile.models.native_vision as vision
    import visuotactile.native_evaluation as evaluation
    import torch
    session=tmp_path/'session.json';session.write_text(json.dumps(dict(deadline=10**12)))
    weight=tmp_path/'weight';weight.write_bytes(b'fixture')
    pair={key:str(weight) for key in ('frontend','backend')}
    pair.update({key+'_sha256':sha256(weight) for key in ('frontend','backend')})
    cases=[dict(name=f'case{i}',role='A') for i in range(2)]
    plan=dict(kind='takeover',test_used=False,batch=dict(version='whole_wave_complete_student_v1',lanes=8),
        execution_backend={},resource_root=str(tmp_path),source_root=str(tmp_path),source_sha256={},
        source_data_sha256={},source_session_sha256=sha256(session),pairs=dict(A=pair),cases=cases)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    clock=[100.];monkeypatch.setattr(batch.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(batch,'validate_takeover_plan',lambda plan:None)
    lease=SimpleNamespace(block=contextlib.nullcontext)
    monkeypatch.setattr(resources,'lease_for',lambda *args,**kwargs:lease)
    def environment(*args,**kwargs):
        clock[0]+=2
        return None,SimpleNamespace(num_envs=8),None,lease
    monkeypatch.setattr(state,'environment',environment)
    class Dino:
        def cuda(self):return self
        def eval(self):return self
    monkeypatch.setattr(vision,'NativeDinoSpatial',Dino)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:{})
    monkeypatch.setattr(evaluation,'exit_completed_kit_worker',lambda *args:None)
    def wave(*args):
        clock[0]+=10
        return [(case,dict(legal=True,software_complete=True,residual_ticks=10,steps=10)) for case in cases],dict(seconds=10,control_ticks=20,cases=2)
    monkeypatch.setattr(rollout,'run_wave',wave)
    calls=[]
    def admit(output,case,result,pair,plan,entries):
        calls.append(clock[0]);clock[0]+=3
        result['accepted_for_training']=True;entries[case['name']]={}
    monkeypatch.setattr(batch,'admit_capture',admit)
    batch.run(tmp_path)
    profile=json.loads((tmp_path/'wave_profiles.json').read_text())[0]
    result=json.loads((tmp_path/'result.json').read_text())
    assert calls==[112,115]
    assert profile['seconds']==10 and profile['admission_s']==6
    assert result['elapsed_seconds']==18 and result['accepted_training_trajectories']==2
    assert result['accepted_trajectories_per_hour']==400
