"""Dependency admission and paired fine-tuning configuration, without Kit."""
import copy
from pathlib import Path

import pytest

from tools import tight_dagger as owner


def dependency(tmp_path):
    root=tmp_path/'new';parent=tmp_path/'old';root.mkdir();parent.mkdir()
    owner.write(root/'registration.json',dict(predecessor=str(parent),predecessor_launch=dict(pid=321),
        predecessor_process=dict(pid=321,start_ticks=123)))
    owner.write(parent/'status.json',dict(state='COMPLETE',owned_workers_exited=True))
    owner.write(parent/'owned_cleanup.json',dict(verified=True))
    return root,parent


def test_predecessor_publishing_still_blocks_and_pid_reuse_does_not(tmp_path,monkeypatch):
    root,parent=dependency(tmp_path)
    monkeypatch.setattr(owner,'proc',lambda pid:dict(pid=pid,start_ticks=123))
    assert not owner.predecessor_finished(root)
    monkeypatch.setattr(owner,'proc',lambda pid:dict(pid=pid,start_ticks=456))
    assert owner.predecessor_finished(root)
    owner.write(parent/'status.json',dict(state='RUNNING',owned_workers_exited=True))
    assert not owner.predecessor_finished(root)


def test_missing_reap_receipt_blocks_even_after_owner_exit(tmp_path,monkeypatch):
    root,parent=dependency(tmp_path)
    monkeypatch.setattr(owner,'proc',lambda pid:None)
    (parent/'owned_cleanup.json').unlink()
    assert not owner.predecessor_finished(root)


def test_plan_uses_train_families_and_fixed_attempts_only():
    data=dict(splits=dict(train=['a','b','c'],validation=['v'],test=['t']),
        entries={k:dict(family=k) for k in ('a','b','c','v','t')})
    families=[dict(family=k,split='train' if k in ('a','b','c') else 'test',narrow_reset={}) for k in data['entries']]
    policy=dict(plan_seed=7,attempts_per_group=3,seed_start=100,event_seed_start=200,takeover_ticks=[60,120])
    first=owner.takeover_plan(data,families,policy)
    assert first==owner.takeover_plan(data,families,policy)
    assert {p['family'] for p in first}=={'a','b','c'}
    assert [p['takeover_tick'] for p in first]==[60,120,60]
    assert all(p['source_identity'] in data['splits']['train'] and p['cohort']=='dagger_train' for p in first)


@pytest.mark.parametrize('group,supervision,stage',[('G11','u_expert','command_residual'),('QG01','q_reached','configuration_residual')])
def test_backend_reuses_own_previous_transformer_and_command_affine(tmp_path,monkeypatch,group,supervision,stage):
    from tools import tight_finish
    root=tmp_path/'root';job=root/'round_01'/group;front=job/'dp';front.mkdir(parents=True)
    parent=tmp_path/'parent';(parent/stage).mkdir(parents=True)
    # Historical G10 frontend configuration has no residual_pair at all.
    owner.write(front/'configuration.json',dict(vision_pair=dict(groups=[group])))
    owner.write(parent/stage/'configuration.json',dict(residual_pair=dict(version='target_residual_transformer60_v1')))
    owner.write(root/'registration.json',dict(predecessor=str(parent)))
    owner.write(root/'configuration.json',dict(backend=dict(maximum_control_tick_epochs=30,minimum_control_tick_epochs=10,learning_rate=.0003)))
    stats=tmp_path/'commands.pt';stats.write_bytes(b'fixed expert-command affine')
    pair=dict(supervision=supervision,modality='G10' if group=='G11' else 'G00',backend='/selected/'+group+'.pt',
        backend_sha256='backendsha',residual_statistics_path=str(stats),residual_statistics_sha256=owner.sha(stats),scales=[.1]*29)
    owner.write(job/'initial_pair.json',pair)
    owner.write(front/'state_data/data.json',dict(supervision=supervision,splits=dict(train=['old','correction'],validation=['val'],test=['test']),normalization_train_ids=['old']))
    (front/'training').mkdir();(front/'training/best.pt').write_bytes(b'new paired frontend')
    captured={}
    def prepare(out,configuration):
        out.mkdir();captured.update(copy.deepcopy(configuration))
    monkeypatch.setattr(tight_finish,'residual_prepare',prepare)
    out=owner.prepare_backend(root,group)
    v=captured['residual_pair']
    assert v['bases'][group]['checkpoint']==str(front/'training/best.pt')
    assert v['continuation_from']==dict(path=pair['backend'],sha256='backendsha',teacher_supervision='u_expert',
        statistics_sha256=owner.sha(stats),normalization_identity='u_expert_train166_affine_v1',round_index=1,source_group=group)
    assert Path(v['residual_statistics_path']).read_bytes()==stats.read_bytes()
    metadata=owner.read(Path(v['residual_statistics_path']).parent/'data.json')
    assert metadata['supervision']=='u_expert' and metadata['normalization_train_ids']==['old']
    assert v['fixed_residual_scales']==pair['scales'] and v['dagger_round']==1


def test_follower_waits_without_integrating_or_launching_compute(tmp_path,monkeypatch):
    import time
    root,parent=dependency(tmp_path)
    owner.write(root/'configuration.json',{})
    owner.write(root/'collection_plan.json',[])
    registration=owner.read(root/'registration.json')
    registration.update(prepared_source='prepared',configuration_sha256=owner.sha(root/'configuration.json'),
        collection_plan_sha256=owner.sha(root/'collection_plan.json'),waiting_deadline=time.time()+60)
    owner.write(root/'registration.json',registration)
    monkeypatch.setattr(owner,'git',lambda *a:'prepared')
    monkeypatch.setattr(owner,'predecessor_finished',lambda r:False)
    monkeypatch.setattr(owner,'integrate_predecessor',lambda r:pytest.fail('Live predecessor must not be integrated'))
    monkeypatch.setattr(owner,'run',lambda r:pytest.fail('Live predecessor must not launch GPU phase'))
    def stop_after_first_observation(seconds):
        state=owner.read(root/'status.json')
        assert state['state']=='WAITING' and state['gpu_started'] is False
        (root/'STOP').touch()
    monkeypatch.setattr(owner.time,'sleep',stop_after_first_observation)
    owner.follow(root)
    assert owner.read(root/'status.json')['state']=='STOPPED'
