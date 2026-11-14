"""Small CPU checks for direct eager file reads and formal opt-in routing."""
import copy
import json
from pathlib import Path
import weakref

import pytest
import torch

from visuotactile.frozen_visual_io import profile,validate_profile,load_profile
from visuotactile.native_data import Episode
from visuotactile.training_resources import atomic_json


def test_direct_file_load_preserves_tensor_storage_layout_rng_and_default(tmp_path,monkeypatch):
    atomic_json(tmp_path/'data.json',dict(visual_pair=True,visual_stride=1,entries={'x':{}}))
    (tmp_path/'perception').mkdir()
    shard=dict(patches=torch.arange(72,dtype=torch.float16).reshape(3,4,6).transpose(1,2),
        depth=torch.arange(12,dtype=torch.float32).reshape(3,4),
        camera_time=torch.arange(9,dtype=torch.float64).reshape(3,3),camera_valid=torch.ones(3,3,dtype=torch.bool))
    torch.save(shard,tmp_path/'perception/x.pt');real_load=torch.load;calls=[];loaded=[]
    def read(path,**kwargs):
        assert isinstance(path,Path)  # No file-wide BytesIO duplicate.
        calls.append(kwargs);value=real_load(path,**kwargs);loaded.append(value);return value
    monkeypatch.setattr(torch,'load',read);before=torch.get_rng_state().clone()
    original=Episode(tmp_path,'x','I-HistoryDP');eager=Episode(tmp_path,'x','I-HistoryDP',frozen_visual_io=profile())
    assert [c['mmap'] for c in calls]==[True,False] and torch.equal(before,torch.get_rng_state())
    for key in shard:
        old=(original.record|original.perception)[key];new=(eager.record|eager.perception)[key]
        assert torch.equal(old,new) and old.dtype==new.dtype and old.stride()==new.stride()
        assert new is loaded[1][key], 'Eager tensors must not be cloned after file loading'
    assert eager.frozen_visual_io_receipt['extra_clone_bytes']==0
    assert not hasattr(original,'frozen_visual_io_receipt')
    assert validate_profile(None) is None and load_profile(tmp_path) is None
    for bad in (True,{},dict(profile(),torch_load_mmap=0),dict(profile(),extra=True)):
        with pytest.raises(ValueError):validate_profile(bad)
    atomic_json(tmp_path/'frozen_visual_io.json',profile());assert load_profile(tmp_path)==profile()


def test_metrics_loads_one_current_episode_without_changing_labels_or_rng(tmp_path,monkeypatch):
    from tools import tight_fresh_train as fresh
    from test_fresh_offline_metrics import TinyPolicy,record,conditions
    from visuotactile.offline_planning import profile as planning_profile
    rec=record();rec['source_root']=tmp_path;refs=[];seen=[]
    class CurrentEpisode:
        def __init__(self,root,identity,group,**kwargs):
            assert not any(r() is not None for r in refs)
            assert kwargs==dict(frozen_visual_io=profile())
            refs.append(weakref.ref(self));self.frozen_visual_io_receipt=dict(profile=profile(),identity=identity)
    def observe(model,record,**kwargs):
        if kwargs:assert isinstance(kwargs['episode'],CurrentEpisode)
        seen.append(bool(kwargs));return conditions(model,record)
    monkeypatch.setattr(fresh,'Episode',CurrentEpisode);monkeypatch.setattr(fresh,'supervised_conditions',observe)
    records={'x':rec,'y':rec};old=TinyPolicy();new=TinyPolicy();new.group='I-HistoryDP'
    expected=fresh.metrics(old,records,['x','y'],offline_planning=planning_profile(32))
    actual=fresh.metrics(new,records,['x','y'],offline_planning=planning_profile(32),frozen_visual_io=profile())
    for key in ('arm_rmse_rad','hand_rmse_rad','selection'):assert actual[key]==expected[key]
    assert torch.equal(torch.stack(old.draws),torch.stack(new.draws))
    assert seen==[False,False,True,True] and all(r() is None for r in refs)


def test_actual_b2_updates_release_prior_episodes_and_bind_io_identity(tmp_path,monkeypatch):
    from tools import tight_fresh_train as fresh
    from visuotactile import trajectory_training
    from test_full_trajectory_training import setup_trainer,make_records
    from test_paired_episode_io import equal_tree
    prepare=setup_trainer(tmp_path,monkeypatch);baseline=prepare('old');eager=prepare('eager')
    fresh.train(baseline,'B',1,trajectory_batch_profile=trajectory_training.profile())
    atomic_json(eager/'frozen_visual_io.json',profile());_,episodes=make_records();refs=[]
    class CurrentEpisode:
        def __init__(self,root,identity,group,**kwargs):
            assert sum(r() is not None for r in refs)<2, 'Previous B2 must be released before loading next visits'
            assert kwargs==dict(frozen_visual_io=profile())
            self.__dict__.update(copy.deepcopy(episodes[identity]).__dict__)
            self.frozen_visual_io_receipt=dict(profile=profile(),identity=identity)
            refs.append(weakref.ref(self))
    monkeypatch.setattr(fresh,'Episode',CurrentEpisode)
    fresh.train(eager,'B',1,trajectory_batch_profile=trajectory_training.profile())
    assert all(r() is None for r in refs) and len(refs)==3
    saved=[torch.load(root/'B/training/last.pt',map_location='cpu',weights_only=False) for root in (baseline,eager)]
    for key in ('model','ema','optimizer','schedule_rng','noise_rng','updates','anchors_seen','target_ticks','visits_seen'):
        equal_tree(saved[0][key],saved[1][key])
    assert saved[1]['configuration']['frozen_visual_io']==profile()
    rows=[json.loads(line) for line in (eager/'B/training/learning.jsonl').read_text().splitlines()]
    assert all(v['frozen_visual_io']['identity']==v['episode'] for row in rows for v in row['visits'])
    (eager/'frozen_visual_io.json').unlink()
    with pytest.raises(ValueError,match='source/config changed'):
        fresh.train(eager,'B',1,trajectory_batch_profile=trajectory_training.profile())


def test_cycle_routes_profile_to_b2_metrics_and_backend_only(tmp_path,monkeypatch):
    from tools import tight_fresh_cycle as cycle
    from test_fresh_parallel_backends import CycleFixture
    from visuotactile.trajectory_training import profile as batch_profile
    fixture=CycleFixture(tmp_path,monkeypatch);launch=cycle.subprocess.Popen
    def emit_manifest(command,**kwargs):
        proc=launch(command,**kwargs)
        if proc.stage=='aggregate':
            for role in 'AB':
                (fixture.output/role/'state_data').mkdir(parents=True)
                atomic_json(fixture.output/role/'state_data/data.json',dict(splits=dict(train=['x','y','z'])))
                atomic_json(fixture.output/role/'continuation.json',{})
        return proc
    monkeypatch.setattr(cycle.subprocess,'Popen',emit_manifest)
    fixture.run(trajectory_batch_profile=batch_profile(),frozen_visual_io=profile())
    assert json.loads((fixture.ledger/'registration.json').read_text())['frozen_visual_io']==profile()
    assert load_profile(fixture.output)==profile()  # Frontend reads the registered root profile.
    for name,command in fixture.calls:
        assert ('--frozen-visual-io-json' in command)==name.startswith('backend_prepare_')
    with pytest.raises(ValueError,match='explicit B2'):
        cycle.run(tmp_path,tmp_path,tmp_path/'bad',1,1,1,1,frozen_visual_io=profile())


def test_static_loop_eager_input_and_sequential_frontends_are_explicit():
    from tools import tight_fresh_research_loop as loop
    bundle=Path(__file__).resolve().parents[1]/'scripts/tight_fresh_20261115_01'
    config=json.loads((bundle/'loop_config.json').read_text());loop.validate_configuration(config)
    assert config['cycle']['parallel_frontends'] is False and config['cycle']['parallel_backends'] is True
    assert validate_profile(json.loads((bundle/config['inputs']['frozen_visual_io_json']).read_text()))==profile()
    context=dict(registration=dict(configuration=config),tree=Path('/frozen'),session=Path('/session'))
    paths=loop.round_paths(context['session'],Path('/ledger'),config['rounds'][0])
    commands=loop.commands(context,config['rounds'][0],paths,context['session'],None)
    assert '--parallel-frontends' not in commands['cycle'] and '--parallel-backends' in commands['cycle']
    assert '--frozen-visual-io-json' in commands['cycle']
    assert all('--frozen-visual-io-json' not in commands[k] for k in ('collection','collection_prepare'))
