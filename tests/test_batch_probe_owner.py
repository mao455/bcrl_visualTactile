import json
from pathlib import Path
import time
import pytest
from tools import tight_batch_probe_owner as owner


@pytest.mark.parametrize('valid',[True,False])
def test_finished_probe_never_accepts_an_autonomy_result(tmp_path,monkeypatch,valid):
    output=tmp_path/'probe';output.mkdir();tree=tmp_path/'tree';tree.mkdir()
    plan=dict(kind='engineering_batch_throughput_probe',test_used=False,source_sha256={},
        source_commit='frozen',resource_root=str(tmp_path),pairs={'A':{}},cases=[{}])
    (output/'plan.json').write_text(json.dumps(plan))
    (output/'session.json').write_text(json.dumps(dict(deadline=time.time()+600)))
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:'frozen')
    class Process:
        pid=12345;returncode=0
        def poll(self):return 0
    monkeypatch.setattr(owner.subprocess,'Popen',lambda *a,**k:Process())
    def wait(*a,**k):
        result=dict(software_complete=True,diagnostic_only=valid,autonomous_evaluation_eligible=not valid,
            results=[dict(role='A',results=[dict(success=None if valid else True,
                autonomous_evaluation_eligible=not valid)])])
        (output/'result.json').write_text(json.dumps(result));return None
    monkeypatch.setattr(owner,'wait_owned',wait)
    if valid:owner.run(output,tree)
    else:
        with pytest.raises(RuntimeError):owner.run(output,tree)
    result=json.loads((output/'owner_finished.json').read_text())
    assert result['software_complete'] is valid
    assert result['result_sha256']
    with pytest.raises(FileExistsError):owner.run(output,tree)
