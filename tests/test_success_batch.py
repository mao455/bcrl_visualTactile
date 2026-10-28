import json
from visuotactile.collect_success_batch import classify_pre_action_rejection


def test_rejection_requires_explicit_diagnostic_and_no_started_trajectory(tmp_path):
    (tmp_path/'FAILED.txt').write_text('Some unknown startup error')
    (tmp_path/'metadata.json').write_text(json.dumps({'initial_geometry':{'valid_box_initialization':False}}))
    assert classify_pre_action_rejection(tmp_path) is None
    (tmp_path/'FAILED.txt').write_text('Initial object/table/hole collision overlap')
    assert classify_pre_action_rejection(tmp_path) is not None
    (tmp_path/'trajectories.h5').write_bytes(b'partial')
    assert classify_pre_action_rejection(tmp_path) is None
