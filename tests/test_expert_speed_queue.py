import json
import pytest
from tools.expert_speed_eval import ExpertSpeedEvaluation,report_body_hash


def setup_owner(tmp_path,state):
    parent=tmp_path/'parent';parent.mkdir();root=tmp_path/'new';root.mkdir()
    (parent/'status.json').write_text(json.dumps(state))
    (root/'session.json').write_text(json.dumps(dict(dependency_root=str(parent),deadline=10**12)))
    (root/'status.json').write_text('{}')
    return ExpertSpeedEvaluation(root),parent


def test_active_state_queue_remains_read_only_and_no_kit_is_started(tmp_path,monkeypatch):
    owner,parent=setup_owner(tmp_path,dict(state='evaluating',pid=1234))
    original=(parent/'status.json').read_bytes()
    owner.report=lambda:None
    class WaitReached(Exception):pass
    def wait(seconds):assert seconds==30;raise WaitReached()
    monkeypatch.setattr('tools.expert_speed_eval.time.sleep',wait)
    monkeypatch.setattr('tools.expert_speed_eval.subprocess.Popen',lambda *a,**kw:pytest.fail('GPU/worker launch during dependency wait'))
    with pytest.raises(WaitReached):owner.wait_dependency()
    assert (parent/'status.json').read_bytes()==original
    assert json.loads((owner.root/'status.json').read_text())['state']=='waiting_dependency'
    assert owner.attempts==[]


def test_incomplete_parent_exit_cannot_release_new_experiments(tmp_path):
    owner,parent=setup_owner(tmp_path,dict(state='blocked',pid=None))
    (parent/'owner_exit.json').write_text('{}')
    with pytest.raises(AssertionError,match='stopped incomplete'):owner.wait_dependency()
    assert owner.attempts==[]


def test_report_guard_ignores_only_the_generated_slow_result_section():
    a='Reviewed text\n<!-- SLOW_EXPERT_RESULTS_START -->\nwaiting\n<!-- SLOW_EXPERT_RESULTS_END -->\n'
    assert report_body_hash(a)==report_body_hash(a.replace('waiting','0/10'))
    assert report_body_hash(a)!=report_body_hash(a.replace('Reviewed text','Changed criteria'))
