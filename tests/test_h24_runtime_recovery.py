import json

import pytest

from tools import h24_runtime_recovery as recovery
from tools import h24_comparison_run as owner


def fixture(root, name='trial'):
    output=root/name;output.mkdir()
    (output/'runtime_controls.json').write_text('{}')
    (output/'lifecycle.json').write_text('{"state":"launching"}')
    (root/(name+'_process_exit.json')).write_text('{"code":-6,"reaped":true}')
    (root/(name+'.log')).write_text('[Fatal] libc.so.6!abort libomniclient.so!omniClientFreeContent '
        'omniclient::provider_file::FileThread')
    return output


def test_sigabrt_without_exception_message_classified_only_before_scene(tmp_path):
    output=fixture(tmp_path)
    result=recovery.native_startup_evidence(tmp_path,'trial')
    assert result['confirmed_sigabrt'] and not result['task_started']
    assert not result['retry_authorized'] and not result['root_cause_proven']
    (output/'environment.json').write_text('{}')
    assert recovery.native_startup_evidence(tmp_path,'trial') is None


@pytest.mark.parametrize('replacement', ['FileThread unrelated', 'Traceback (most recent call last) '
    '[Fatal] libc.so.6!abort libomniclient.so!omniClientFreeContent omniclient::provider_file::FileThread'])
def test_unclassified_or_python_fault_stays_blocked(tmp_path,replacement):
    fixture(tmp_path);(tmp_path/'trial.log').write_text(replacement)
    assert recovery.native_startup_evidence(tmp_path,'trial') is None


def test_two_retained_faults_finish_missing_cohort_without_new_launch(tmp_path,monkeypatch):
    specs=[dict(identity=str(i),cohort='train' if i<4 else 'validation',seed=i) for i in range(12)]
    (tmp_path/'run_plan.json').write_text(json.dumps(dict(formal_cohort=specs)))
    (tmp_path/'kit_acceptance.json').write_text('{"persistent_reset_parity_passed":false}')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'unchanged')
    for i in (1,2):fixture(tmp_path,f'sample_3000_loop_worker_{i:02d}')
    calls=[]
    def existing(root,name,*a,**k):
        calls.append(name)
        if name.endswith('00'):return dict(trials=[dict(specs[0],legal=True,success=False)])
        raise RuntimeError('Existing attempt retained')
    monkeypatch.setattr(owner,'worker',existing)
    monkeypatch.setattr(owner,'startup_failure_evidence',recovery.native_startup_evidence)
    result=owner.cohort(tmp_path,'sample_3000',checkpoint)
    assert len(calls)==3 and len(result['trials'])==1 and len(result['missing'])==11
    assert [r['classification'] for r in result['missing'][:2]]==['pre_scene_native_fault']*2
    assert all(r['native_outcome'] is None for r in result['missing'])
    assert len({r['spec']['identity'] for r in result['missing']})==11
    assert owner.cohort(tmp_path,'sample_3000',checkpoint)==result and len(calls)==3
