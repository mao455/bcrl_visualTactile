import json
import pytest
from tools.tight_evaluate import run


def plan(tmp_path, *, queued=True, video=False):
    p={'kind':'autonomous','batch':{'lanes':2},'cases':[{'video':video} for _ in range(16)]}
    if queued:p['batch']['queue_profile']='independent_lane_queue_v1'
    (tmp_path/'plan.json').write_text(json.dumps(p));return p


def test_numerical_queue_uses_existing_real_consumer(tmp_path,monkeypatch):
    plan(tmp_path);called=[]
    monkeypatch.setattr('tools.tight_fresh_batch_run.run',lambda output,**kw:called.append((output,kw)))
    run(tmp_path);assert called==[(tmp_path,{'allow_heldout':False})]


def test_queue_video_requires_explicit_supported_path(tmp_path,monkeypatch):
    plan(tmp_path,video=True)
    monkeypatch.setattr('tools.tight_fresh_batch_run.run',lambda *a,**k:pytest.fail('No worker may start'))
    with pytest.raises(ValueError,match='explicit video=False'):run(tmp_path)


def test_synchronous_video_lane_contract_remains(tmp_path,monkeypatch):
    plan(tmp_path,queued=False)
    monkeypatch.setattr('tools.tight_fresh_batch_run.run',lambda *a,**k:pytest.fail('No worker may start'))
    with pytest.raises(AssertionError,match='One case per video lane'):run(tmp_path)
