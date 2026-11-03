import json

import pytest
import torch

from tools.h24_comparison_run import recover_interrupted_training, worker, cohort
from visuotactile.compact import sha256


def test_recovery_retains_failed_attempt_and_only_replays_uncommitted_steps(tmp_path):
    directory=tmp_path/'h24_train';directory.mkdir()
    config=directory/'configuration.json';config.write_text('{}')
    keys=('optimizer','ema','model','ema_optimization_step','schedule_rng','noise_rng',
        'torch_rng','cuda_rng','schedule_order','schedule_cursor','history')
    saved=dict.fromkeys(keys,0);saved.update(updates=2,config_sha256=sha256(config))
    checkpoint=directory/'last.pt';torch.save(saved,checkpoint);digest=sha256(checkpoint)
    original='{"update":1}\n{"update":2}\n{"update":3}\n{"up'
    (directory/'train.jsonl').write_text(original)
    (tmp_path/'stage').mkdir();(tmp_path/'stage'/'evidence').write_text('keep')
    (tmp_path/'stage.log').write_text('interrupted')
    recover_interrupted_training(tmp_path,'stage',directory)
    archive=tmp_path/'recovery/stage'
    assert (archive/'attempt/evidence').read_text()=='keep'
    assert (archive/'train_before_resume.jsonl').read_text()==original
    assert [json.loads(row)['update'] for row in (directory/'train.jsonl').read_text().splitlines()]==[1,2]
    assert sha256(checkpoint)==digest
    with pytest.raises(RuntimeError,match='bound'):
        recover_interrupted_training(tmp_path,'stage',directory)


def test_numerical_or_unclassified_failure_never_auto_resumes(tmp_path):
    directory=tmp_path/'h24_train';directory.mkdir()
    (directory/'last.pt').touch()
    (tmp_path/'stage_process_exit.json').write_text('{"code":1}')
    with pytest.raises(RuntimeError,match='Unclassified'):
        recover_interrupted_training(tmp_path,'stage',directory)
    assert not (tmp_path/'recovery').exists()


def test_completed_worker_is_adopted_without_launch(tmp_path):
    output=tmp_path/'stage';output.mkdir()
    (output/'result.json').write_text('{"software_complete":true,"updates":3000}')
    (tmp_path/'session.json').write_text('{}')
    result=worker(tmp_path,'stage','visuotactile.state_full_training',[])
    assert result['updates']==3000
    assert json.loads((tmp_path/'stage_committed.json').read_text())['recovered_completion']


def test_missing_cohort_is_durable_and_never_relaunched(tmp_path,monkeypatch):
    (tmp_path/'run_plan.json').write_text('{"formal_cohort":[]}')
    (tmp_path/'kit_acceptance.json').write_text('{"persistent_reset_parity_passed":false}')
    directory=tmp_path/'sample_3000_loop';directory.mkdir()
    result=directory/'result.json';result.write_text('{"software_complete":false,"trials":[],"missing":["startup"]}')
    (tmp_path/'sample_3000_loop_committed.json').write_text(json.dumps(dict(sha256=sha256(result))))
    monkeypatch.setattr('tools.h24_comparison_run.worker',lambda *a,**k:pytest.fail('repeat'))
    assert cohort(tmp_path,'sample_3000',tmp_path/'absent.pt')['missing']==['startup']


def test_completed_marker_rejects_changed_evidence(tmp_path):
    output=tmp_path/'stage';output.mkdir();(output/'result.json').write_text('{}')
    (tmp_path/'stage_committed.json').write_text('{"sha256":"wrong"}')
    with pytest.raises(ValueError,match='changed'):
        worker(tmp_path,'stage','unused',[])


def test_report_rmse_pools_valid_positions_and_separates_train_validation():
    from tools.h24_comparison_report import pooled_errors, loop_counts
    rows=[dict(valid12_positions=1,arm12_rmse_rad=1.,hand12_rmse_rad=2.),
        dict(valid12_positions=3,arm12_rmse_rad=3.,hand12_rmse_rad=4.)]
    assert pooled_errors(rows)['arm12_rmse_rad']==pytest.approx(7**.5)
    loop=dict(trials=[dict(cohort='train',legal=True,success=True,native_subgoals=2),
        dict(cohort='validation',legal=True,success=False,reasons=dict(timeout=True)),
        dict(cohort='validation',legal=False,success=False)],missing=[{}])
    counts=loop_counts(loop)
    assert counts['train']['successes']==1 and counts['train']['first_subgoal']==1
    assert counts['validation']['completed']==1 and counts['validation']['timeout']==1
    assert counts['validation']['registered']==8
