import json
from unittest.mock import patch

from tools.anchor_interim_eval import counts, same_process, resume
from tools.anchor_continue_report import contact_count


def test_contact_dictionary_and_native_timeout_are_not_truthiness_counts():
    trials=[dict(success=False,contact={'object_contact':x},native_lift=False,
        native_subgoals=0,reasons={'timeout':True,'fall':False}) for x in (False,True)]
    result=counts(trials)
    assert result['contact']==1 and result['valid']==2 and result['timeout']==2
    assert result['success']==result['lift']==0
    assert contact_count(trials)==1


def test_pid_reuse_never_signals_new_process(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    record=dict(pid=42,start_ticks=100,parent_root=str(parent),resumed=False)
    (tmp_path/'parent_pause.json').write_text(json.dumps(record))
    with patch('tools.anchor_interim_eval.proc',return_value=dict(pid=42,start_ticks=101)), \
            patch('tools.anchor_interim_eval.os.kill') as kill:
        assert same_process(record) is None
        resume(tmp_path)
        kill.assert_not_called()
    assert json.loads((tmp_path/'parent_pause.json').read_text())['exact_parent_alive'] is False


def test_resume_is_idempotent(tmp_path):
    (tmp_path/'parent_pause.json').write_text(json.dumps(dict(resumed=True)))
    with patch('tools.anchor_interim_eval.os.kill') as kill:
        resume(tmp_path);kill.assert_not_called()


def test_evidence_review_hold_never_resumes_training_implicitly(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    record=dict(pid=42,start_ticks=100,parent_root=str(parent),resumed=False)
    (tmp_path/'parent_pause.json').write_text(json.dumps(record))
    (tmp_path/'session.json').write_text(json.dumps(dict(hold_parent_for_decision=True)))
    with patch('tools.anchor_interim_eval.os.kill') as kill:
        resume(tmp_path);kill.assert_not_called()
    assert not json.loads((tmp_path/'parent_pause.json').read_text())['resumed']
    assert json.loads((parent/'evaluation_interlude.json').read_text())['state']=='held_for_evidence_review'
