import importlib.util
import json
from pathlib import Path

import torch


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/tight_fresh_night_20261116/finish26.py'
    spec=importlib.util.spec_from_file_location('night_finish26_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_incomplete_frontend_keeps_old_append_logs_independent(tmp_path, monkeypatch):
    m=module();source=tmp_path/'old';src=source/'A/training';src.mkdir(parents=True)
    dest=tmp_path/'new/A';dest.mkdir(parents=True)
    torch.save(dict(epoch=10,updates=2050,configuration=dict(source_sha256={})),src/'last.pt')
    torch.save(dict(model=dict(weight=torch.ones(2))),src/'best.pt')
    (src/'learning.jsonl').write_text('{"updates":2051}\n')
    (src/'interrupted_snapshot.pt').write_bytes(b'not formal')
    monkeypatch.setattr(m,'SOURCE',source)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'frozen-head\n')
    receipt=m.bind_frontend(dest,'A')
    assert receipt['durable_updates']==2050 and receipt['remaining_updates']==410
    with (dest/'training/learning.jsonl').open('a') as f:f.write('new record\n')
    assert (src/'learning.jsonl').read_text()=='{"updates":2051}\n'
    assert not (dest/'training/interrupted_snapshot.pt').exists()
    assert m.sha256(dest/'training/last.pt')==m.sha256(src/'last.pt')


def test_complete_frontend_is_reused_without_training_copy(tmp_path, monkeypatch):
    m=module();source=tmp_path/'old';src=source/'B/training';src.mkdir(parents=True)
    dest=tmp_path/'new/B';dest.mkdir(parents=True)
    (src/'best.pt').write_bytes(b'original selected checkpoint')
    (src/'finished.json').write_text(json.dumps(dict(complete=True,epochs=12,updates=2460,
        checkpoint_sha256=m.sha256(src/'best.pt'))))
    monkeypatch.setattr(m,'SOURCE',source)
    receipt=m.bind_frontend(dest,'B')
    assert receipt['complete'] and receipt['updates']==2460
    assert (dest/'training').is_symlink() and (dest/'training').resolve()==src


def test_real_training_recheck_removes_expert_takeover_and_keeps_cohorts_separate(tmp_path):
    from tools.tight_fresh_batch_run import append_training_rechecks
    source=dict(kind='takeover',cases=[dict(name='old_A',role='A',spec=dict(
        identity='old_A',family='hard1',split='train',seed=7,takeover_tick=60,
        training_expansion=dict(schedule='grasp_touch'),initial_configuration_sha256='old'))])
    (tmp_path/'plan.json').write_text(json.dumps(source))
    (tmp_path/'result.json').write_text(json.dumps(dict(software_complete=True)))
    plan=dict(kind='development',cases=[dict(name='fixed_A',role='A',spec=dict(family='dev1'))])
    data=dict(splits=dict(train=['t'],validation=['v']),entries=dict(t=dict(family='hard1'),v=dict(family='dev1')))
    append_training_rechecks(plan,tmp_path,data,300)
    assert plan['kind']=='development' and len(plan['cases'])==2
    fixed,recheck=plan['cases']
    assert fixed['diagnostic_cohort']=='fixed_development'
    assert recheck['diagnostic_cohort']=='training_family_recheck' and recheck['grasp_evaluation_ticks']==300
    assert not {'takeover_tick','training_expansion','initial_configuration_sha256'} & recheck['spec'].keys()
    assert recheck['spec']['seed']==7 and recheck['spec']['split']=='train'
    assert json.loads((tmp_path/'plan.json').read_text())==source
