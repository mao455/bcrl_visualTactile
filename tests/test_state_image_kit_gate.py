import json

from tools.state_image_kit_gate import graceful_acceptance


def result(tmp_path):
    data=dict(software_complete=True,trials=[dict(legal=True,steps=181,action_exhausted=False,
        reasons={'max_successes':True},actuator_target_max_error=0.)])
    (tmp_path/'result.json').write_text(json.dumps(data))
    return data


def test_saved_result_and_zero_exit_do_not_replace_sdk_close_or_release(tmp_path):
    result(tmp_path)
    assert not graceful_acceptance(tmp_path,0,True)
    (tmp_path/'closed.json').write_text('{}')
    assert not graceful_acceptance(tmp_path,-6,True)
    assert not graceful_acceptance(tmp_path,0,False)
    assert graceful_acceptance(tmp_path,0,True)
    (tmp_path/'process_exit_requested.json').write_text('{}')
    assert not graceful_acceptance(tmp_path,0,True)


def test_restricted_helper_fds_remain_visible_without_fabricated_zero_counts(monkeypatch):
    from pathlib import Path
    import os
    from tools.state_image_kit_gate import proc
    original=Path.iterdir
    def denied(path):
        if path==Path('/proc',str(os.getpid()),'fd'):
            raise PermissionError(13,'Permission denied')
        return original(path)
    monkeypatch.setattr(Path,'iterdir',denied)
    row=proc(os.getpid())
    assert row['pid']==os.getpid() and row['start_ticks']>0
    assert row['fds'] is None and row['watches'] is None
    assert row['telemetry_missing']==['fd_directory_permission_denied']


def test_task_failure_can_pass_lifecycle_but_incomplete_episode_cannot(tmp_path):
    data=result(tmp_path);(tmp_path/'closed.json').write_text('{}')
    data['trials'][0]['reasons']={'timeout':True}
    (tmp_path/'result.json').write_text(json.dumps(data))
    assert graceful_acceptance(tmp_path,0,True)
    data['trials'][0]['action_exhausted']=True
    (tmp_path/'result.json').write_text(json.dumps(data))
    assert not graceful_acceptance(tmp_path,0,True)
