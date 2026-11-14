"""Training-family diagnostics must use a real native reset without restoration."""
import json
from types import SimpleNamespace

import h5py
import pytest

from tools import tight_fresh_train_diagnostic as diagnostic
from visuotactile.compact import sha256


def population(tmp_path, monkeypatch):
    root=tmp_path/'source';(root/'A/state_data').mkdir(parents=True)
    (root/'session.json').write_text(json.dumps(dict(deadline=10**12)))
    originals=[f't{i}' for i in range(30)]
    entries={i:dict(family='family_'+i) for i in originals+['v']}
    for index,identity in enumerate(originals[:2]):
        path=tmp_path/(identity+'.h5')
        with h5py.File(path,'w') as stream:
            stream.attrs['result_json']=json.dumps(dict(success=True,legal=True,seed=10+index,
                event_seed=100+index,narrow_reset={'fixed':index}))
        entries[identity].update(path=str(path),source_sha256=sha256(path))
    data=dict(normalization_train_ids=originals,entries=entries,
        splits=dict(train=originals,validation=['v'],test=[]))
    (root/'A/state_data/data.json').write_text(json.dumps(data))
    monkeypatch.setattr(diagnostic,'completed_pair',lambda root,role:dict(role=role))
    return root,data


def test_registered_training_seed_runs_native_reset_without_source_restoration(tmp_path,monkeypatch):
    from visuotactile import state_benchmark,narrow_data,live_collision
    root,data=population(tmp_path,monkeypatch);output=tmp_path/'diagnostic'
    diagnostic.prepare(root,output,2)
    plan=json.loads((output/'plan.json').read_text())
    assert [x['role'] for x in plan['cases']]==['A','B','B','A']
    assert not plan['test_used'] and plan['kind']=='autonomous'
    calls=[]
    env=SimpleNamespace(clear_recorded_events=lambda:calls.append('clear'),
        cfg=SimpleNamespace(sim=SimpleNamespace(enable_scene_query_support=False)),_diagnostics=lambda:{})
    isolated=SimpleNamespace(reset=lambda seed:(calls.append(('reset',seed)),None),
        begin_events=lambda seed:calls.append(('events',seed)))
    monkeypatch.setattr(narrow_data,'configure_direct_reset',lambda env,reset:calls.append(('configuration',reset)))
    monkeypatch.setattr(live_collision,'clearance',lambda env:dict(legal=True))
    for case in plan['cases']:
        spec=case['spec'];assert spec['split']=='train'
        assert spec['diagnostic_cohort']=='original_training_diagnostic'
        assert 'dagger' not in case and 'source_data' not in case
        directory=tmp_path/case['name'];directory.mkdir()
        # Empty source deliberately rejects accidental observable-state restoration.
        state_benchmark.prepare(env,isolated,spec,{},directory)
        assert json.loads((directory/'restoration.json').read_text())['kind']=='fresh_native_reset'
    assert [x[1] for x in calls if isinstance(x,tuple) and x[0]=='reset']==[10,10,11,11]
    assert [x[1] for x in calls if isinstance(x,tuple) and x[0]=='events']==[100,100,101,101]


@pytest.mark.parametrize('fault',['heldout_family','takeover_source','changed_source'])
def test_bad_training_diagnostic_provenance_is_rejected(tmp_path,monkeypatch,fault):
    root,data=population(tmp_path,monkeypatch)
    if fault=='heldout_family':data['entries']['t0']['family']=data['entries']['v']['family']
    elif fault=='takeover_source':data['entries']['t0']['trajectory_kind']='expert_takeover_v1'
    else:data['entries']['t0']['source_sha256']='changed'
    (root/'A/state_data/data.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):diagnostic.prepare(root,tmp_path/'rejected',2)
