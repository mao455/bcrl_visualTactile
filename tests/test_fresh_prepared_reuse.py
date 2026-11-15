import json
from pathlib import Path
import pytest
from tools.tight_fresh_prepared_reuse import reuse
from visuotactile.compact import sha256

def setup(tmp_path):
    source=tmp_path/'old';root=tmp_path/'new';collection=tmp_path/'collection';output=tmp_path/'candidate'
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    write(collection/'plan.json',{'frozen':'plan'});write(collection/'result.json',{'software_complete':True})
    write(source/'prepared.json',{'collection_plan_sha256':sha256(collection/'plan.json'),'accepted_corrections':26})
    for p,reserve in ((source,32),(root,16)):
        write(p/'session.json',{'deadline':12345,'reserve_bytes':reserve})
        write(p/'metadata.json',{'same':'physical_task'})
    write(source/'gate.json',{'passed':True})
    write(source.with_name('old_stages')/'failure.json',{'error':'visual_cache disk reserve'})
    for role in ('A','B'):
        write(source/role/'state_data/data.json',{'entries':{'actual':'trajectory'}})
        write(source/role/'continuation.json',{'source':'existing checkpoint'})
        write(source/role/'cache_progress_index.json',{'entries':{'actual':{'path':'immutable_old_feature.pt'}}})
    return source,root,collection,output

def test_reuse_preserves_data_and_committed_cache_and_changes_only_new_resources(tmp_path):
    source,root,collection,output=setup(tmp_path);reuse(source,root,collection,output)
    for role in ('A','B'):
        assert (output/role/'state_data/data.json').read_bytes()==(source/role/'state_data/data.json').read_bytes()
        assert (output/role/'cache_progress_index.json').read_bytes()==(source/role/'cache_progress_index.json').read_bytes()
    assert json.loads((output/'session.json').read_text())['reserve_bytes']==16
    assert json.loads((source/'session.json').read_text())['reserve_bytes']==32
    assert not (output/'A/training').exists()
    with pytest.raises(FileExistsError):reuse(source,root,collection,output)

def test_started_training_cannot_be_repeated_by_cache_recovery(tmp_path):
    source,root,collection,output=setup(tmp_path);(source/'A/training').mkdir()
    with pytest.raises(ValueError,match='before any optimizer'):reuse(source,root,collection,output)
    assert not output.exists()
