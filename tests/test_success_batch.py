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


def _recovery_fixture(tmp_path, *, digest='new-digest', target=400):
    from pathlib import Path
    import visuotactile.finalize_collection as finalizer
    from visuotactile.collect_success_batch import recover_pending

    root=tmp_path/'data'/'close_v4'; run=tmp_path/'run';
    root.mkdir(parents=True); run.mkdir()
    (root/'reference_metadata.json').write_text('{}')
    (root/'manifest.json').write_text(json.dumps({
        'manifest_version':1, 'target_successes':target,
        'episodes':[{'id':'existing','content_digest':'existing-digest'}],
        'attempts':[{'id':'pending','run_id':'r1','outcome':'awaiting_audit'}],
    }))
    (run/'progress.json').write_text(json.dumps({'target':target}))
    source=root.parent.parent/'tmp'/'r1'/'pending'; source.mkdir(parents=True)
    (source/'data.h5').write_bytes(b'placeholder')
    info={'summary':{'outcome':'success'},'initial':{},'length':3,
          'effective_delay_counts':[3,0,0], 'content_digest':digest,
          'episode':'ep','exact_fields':143,'storage_version':2,
          'camera_ids':['fixed_high','fixed_side','wrist_rgbd'],
          'video_files':[], 'files':{}}
    return root,run,source,info,finalizer


def test_recovery_deduplicates_and_respects_target(tmp_path, monkeypatch):
    from visuotactile.collect_success_batch import recover_pending

    root,run,source,info,finalizer=_recovery_fixture(tmp_path, digest='existing-digest')
    monkeypatch.setattr(finalizer, 'finalize_stream', lambda *_: info)
    reports=recover_pending(root,run)
    manifest=json.loads((root/'manifest.json').read_text())
    assert reports == [{'id':'pending','outcome':'duplicate_success_not_counted'}]
    assert len(manifest['episodes']) == 1
    assert manifest['attempts'][0]['outcome'] == 'duplicate_success_not_counted'
    assert not source.exists()

    root,run,source,info,finalizer=_recovery_fixture(tmp_path/'target', digest='new-digest', target=1)
    monkeypatch.setattr(finalizer, 'finalize_stream', lambda *_: info)
    reports=recover_pending(root,run)
    manifest=json.loads((root/'manifest.json').read_text())
    assert reports == [{'id':'pending','outcome':'surplus_success_not_counted'}]
    assert len(manifest['episodes']) == 1
    assert manifest['attempts'][0]['outcome'] == 'surplus_success_not_counted'
    assert not source.exists()
