"""Independent final15 metadata fixtures; never touches actual runtime inputs."""
import json
import pytest
from outputs.diagnostics.tight_paired_dagger_20261122_01 import test_final_evidence_recovery15 as author

common=author.common


def pinset(f):
    return dict(small_file_sha256=f.binding['small_file_sha256'],
                local_retained_log_sha256=f.binding['local_retained_log_sha256'])


def test_failed13_wrapper_error_and_child_log_have_separate_provenance(tmp_path):
    f=author.closed.__wrapped__(tmp_path);root,audit=f.root,f.audit
    failure=audit/'driver_test_fragment13/failure.json'
    author.write(failure,dict(error="ValueError('evaluate_fragment13 failed; no retry or replacement is launched')"))
    rawlog=root/'cycle_logs_fragment13/evaluate_fragment13.log'
    rawlog.write_text('Traceback (most recent call last):\n  frozen child project guard\nValueError: Original suffix changed\n')
    boundary_path=audit/'ROOT_TEST_ADMISSION14_BOUNDARY.json'
    boundary=common.read_small(boundary_path)
    boundary['artifact_sha256']={path:common.sha(path) for path in boundary['artifact_sha256']}
    author.write(boundary_path,boundary)
    registry_path=audit/'TEST_ADMISSION14_REGISTRATION.json'
    registry=common.read_small(registry_path)
    registry.update(original13_driver_artifacts=boundary['artifact_sha256'],boundary_sha256=common.sha(boundary_path))
    author.write(registry_path,registry)
    author.write(audit/'ROOT_TEST_ADMISSION14_REVIEW.json',dict(
        status='source_and_actual_pre_native14_review_passed_inactive',registration_sha256=common.sha(registry_path)))
    f.binding['local_retained_log_sha256'][str(rawlog)]=common.sha(rawlog)
    author.repin(f)
    result,failed=common.admission14_gate(root,audit,f.binding,pinset(f),dict(driver_exit_utc=0))
    assert failed['additional_wave_attempts']==0 and failed['native_owner_started'] is False


@pytest.mark.parametrize('mutation',['duplicate_launch','unreaped','next_before_reap','false_software_success'])
def test_queue_lifecycle_semantics_cannot_be_rebound_as_success(tmp_path,mutation):
    f=author.closed.__wrapped__(tmp_path);owner=f.root/'owners/test_continuation15'
    if mutation=='duplicate_launch':
        author.write(owner/'launches/wave17_again.json',dict(index=17));f.pinned.add(owner/'launches/wave17_again.json')
    elif mutation=='next_before_reap':
        path=owner/'launches/wave21.json';row=common.read_small(path);row['utc']=0;author.write(path,row)
    else:
        path=owner/'stage_exits/wave20.json';row=common.read_small(path)
        row['child_reaped' if mutation=='unreaped' else 'software_completed']=False if mutation=='unreaped' else True
        author.write(path,row)
    author.repin(f)
    with pytest.raises(ValueError):common.closure_gate(f.root,f.audit)
