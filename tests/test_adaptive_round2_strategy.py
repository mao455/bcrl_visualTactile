"""Round2 must use reviewed new students and original training-family starts."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'research_docs/evidence/tight_adaptive_dagger_20261119_01/round2_strategy.py'
    spec=importlib.util.spec_from_file_location('round2_strategy_under_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_rejects_unfinished_first_round_and_opened_test(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):m.derive()
    save(tmp_path/'round1_review_ready.json',dict(final_test_opened=True))
    with pytest.raises(AssertionError):m.derive()
    save(tmp_path/'round1_review_ready.json',dict(final_test_opened=False))
    (tmp_path/'final_test').mkdir()
    with pytest.raises(AssertionError):m.derive()


def test_role_specific_screens_cannot_recruit_development_or_replay_ticks(tmp_path,monkeypatch):
    m=module();root=tmp_path/'run';public=tmp_path/'evidence';config=tmp_path/'config'
    for key,value in [('ROOT',root),('PUBLIC',public),('CONFIG',config)]:monkeypatch.setattr(m,key,value)
    train=[28817000+4*i for i in range(8)]
    save(root/'round1_review_ready.json',dict(final_test_opened=False))
    save(config/'experiment.json',dict(training_seeds=train,development_seeds=[28826000],final_test_seeds=[28839000]))
    rows=[];waves=[]
    for seed in train+[28826000]:
        wave=root/f'wave{seed}';waves.append(str(wave));cases=[]
        for role in 'AB':
            name=role+'_new';case=dict(name=name,role=name,spec=dict(seed=seed));cases.append(case)
            save(wave/name/'result.json',dict(legal=True,software_complete=True,steps=20))
            rows.extend([dict(seed=seed,role=role+'_parent',success=role=='A',native_subgoals=0),
                         dict(seed=seed,role=name,success=False,native_subgoals=int(role=='B'))])
        save(wave/'plan.json',dict(cases=cases))
    save(root/'round01/evaluation/registration.json',dict(waves=waves,pairs={'frozen':'selected-round1-pairs'}))
    save(public/'round01/evaluation/per_case.json',rows)
    class File:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):return {}
    monkeypatch.setattr(m.h5py,'File',lambda *args:File())
    monkeypatch.setattr(m,'make_trigger',lambda cfg,fallback:SimpleNamespace(configuration=cfg,trigger_completed_tick=None))
    def observe(trigger,tick,transition,index):
        hits=dict(contact=6,transport=8,alignment=10,instability=12,post_subgoal=14)
        if tick==hits[trigger.configuration['stage']]:trigger.trigger_completed_tick=tick+1
    monkeypatch.setattr(m,'observe_transition',observe)
    strategy=m.derive();schedule=strategy['schedule']
    seeds=[seed for batch in schedule for seed in batch['families']]
    assert len(seeds)==len(set(seeds))==28
    assert set(seeds)<=set(range(28817000,28817030))
    assert all(row['seed'] in train for row in strategy['evidence'])
    assignments={seed:stage for batch in schedule for seed,stage in zip(batch['families'],batch['stages'])}
    assert assignments[train[0]]==dict(A='contact',B='post_subgoal')
    assert strategy['maximum_opportunities']==56
    assert strategy['source_pairs']=={'frozen':'selected-round1-pairs'}
    assert all(isinstance(stage,str) for pair in assignments.values() for stage in pair.values())
