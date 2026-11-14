import copy
import json
from pathlib import Path

import pytest
import torch

from tools.tight_fresh_selected_fit import (latest_training_ids, run, sha256,
    validate_checkpoint, validate_inputs)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    root, source = tmp_path/'run', tmp_path/'frozen'
    role_root = root/'B'
    data = dict(aggregation=dict(new_training_ids=['new1']),
        splits=dict(train=['old1', 'new1'], validation=['val1'], test=['test1']),
        entries={identity:dict(split=split, family=family, trajectory_kind='expert_takeover_v1',
            source_sha256='source-'+identity, length=3) for identity,split,family in
            [('old1','train','train0'),('new1','train','train1'),
             ('val1','validation','val0'),('test1','test','test0')]})
    write(role_root/'state_data/data.json', data)
    write(role_root/'state_data/statistics.pt', {'fixture': True})
    write(role_root/'state_data/numeric.pt', {'fixture': True})
    visual = role_root/'state_data/perception/new1.pt'
    write(visual, {'fixture': True})
    write(role_root/'cache_index.json', dict(complete=True, stride=1,
        data_sha256=sha256(role_root/'state_data/data.json'), dino_sha256='dino',
        trainable_projection_cached=False, entries={'new1':dict(path=str(visual),
            sha256=sha256(visual), source_sha256='source-new1', frames=3)}))
    write(root/'gate.json', dict(passed=True))
    trainer = source/'tools/tight_fresh_train.py'
    trainer.parent.mkdir(parents=True)
    trainer.write_text('raise RuntimeError("check-only must never import trainer")\n')
    continuation = dict(target_window_contract='recovery_onset_commit12_v1')
    write(role_root/'continuation.json', continuation)
    config = dict(data_sha256=sha256(role_root/'state_data/data.json'),
        statistics_sha256=sha256(role_root/'state_data/statistics.pt'),
        cache_index_sha256=sha256(role_root/'cache_index.json'), dino_sha256='dino',
        continuation=continuation, source_sha256={'tools/tight_fresh_train.py':sha256(trainer)})
    write(role_root/'training/configuration.json', config)
    saved = dict(configuration=config, kind='dp', group='I-HistoryDP', epoch=3, updates=306,
        train_ids=['old1','new1'], validation_ids=['val1'], sampling_exposure={'old1':3,'new1':6},
        contract=dict(supervision='u_expert', visual_stride=1), ema_optimization_step=306)
    torch.save(saved, role_root/'training/best.pt')
    write(role_root/'training/finished.json', dict(checkpoint_sha256=sha256(role_root/'training/best.pt')))
    return root, source, saved


def test_latest_ids_reject_heldout_identity_and_family():
    data=dict(aggregation=dict(new_training_ids=['new']),
        splits=dict(train=['new'],validation=['v'],test=['t']),
        entries={i:dict(family=f,split=s,trajectory_kind='expert_takeover_v1') for i,f,s in
            [('new','train_family','train'),('v','val_family','validation'),('t','test_family','test')]})
    assert latest_training_ids(data)==['new']
    wrong=copy.deepcopy(data);wrong['aggregation']['new_training_ids']=['v']
    with pytest.raises(ValueError,match='exclusively training'):latest_training_ids(wrong)
    wrong=copy.deepcopy(data);wrong['entries']['new']['family']='test_family'
    with pytest.raises(ValueError,match='training-family'):latest_training_ids(wrong)


def test_preflight_reads_only_selected_visual_shards_and_binds_checkpoint(artifacts):
    root,source,saved=artifacts
    receipt=validate_inputs(root,'B',source)
    assert receipt['identities']==['new1']
    assert set(receipt['selected_visual_cache'])=={'new1'}
    assert not (root/'B/state_data/perception/val1.pt').exists()
    assert not (root/'B/state_data/perception/test1.pt').exists()
    validate_checkpoint(saved,receipt)
    changed=copy.deepcopy(saved);changed['sampling_exposure']['new1']=0
    with pytest.raises(ValueError,match='exposure'):validate_checkpoint(changed,receipt)
    changed=copy.deepcopy(saved);changed['contract']['supervision']='q_reached'
    with pytest.raises(ValueError,match='target/visual'):validate_checkpoint(changed,receipt)


@pytest.mark.parametrize('changed', ['trainer', 'visual', 'checkpoint', 'statistics', 'continuation'])
def test_preflight_rejects_artifact_drift(artifacts, changed):
    root,source,_=artifacts
    paths=dict(trainer=source/'tools/tight_fresh_train.py', visual=root/'B/state_data/perception/new1.pt',
        checkpoint=root/'B/training/best.pt', statistics=root/'B/state_data/statistics.pt',
        continuation=root/'B/continuation.json')
    if changed=='continuation':write(paths[changed], dict(target_window_contract=None))
    else:
        with paths[changed].open('ab') as stream:stream.write(b' changed')
    with pytest.raises(ValueError,match='differs'):validate_inputs(root,'B',source)


def test_check_only_reports_selected_exposure_without_importing_trainer_or_cuda(artifacts,tmp_path):
    root,source,_=artifacts
    assert not torch.cuda.is_initialized()
    out=tmp_path/'audit'
    result=run(root,'B',source,out,check_only=True)
    assert result['admission_only'] and result['metrics'] is None
    assert result['epoch']==3 and result['optimizer_updates']==0
    assert result['validation_evaluated'] is False and result['test_evaluated'] is False
    assert not torch.cuda.is_initialized()
    receipt=json.loads((out/'admission.json').read_text())
    assert receipt['sampling_exposure']=={'new1':6}
    with pytest.raises(FileExistsError):run(root,'B',source,out,check_only=True)
