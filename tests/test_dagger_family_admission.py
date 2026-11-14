"""Actual split membership governs correction admission, even with sparse maps."""
import copy
import json

import pytest
import torch

from visuotactile.narrow_data import aggregate_family_partitions, build_dagger_aggregate, validate_aggregate_families


def source_data():
    return dict(supervision='q_reached', splits=dict(train=['original','historical'],
        validation=['validation'], test=['test']),
        entries=dict(original={'family':'train_family'}, historical={'family':'train_family'},
            validation={'family':'validation_family','path':'/MUST_NOT_OPEN_VALIDATION'},
            test={'family':'test_family','path':'/MUST_NOT_OPEN_TEST'}),
        families={'train_family':'train'})


@pytest.mark.parametrize('family', ['validation_family','test_family','unseen_family',None])
def test_sparse_or_false_legacy_map_cannot_admit_heldout_or_unknown_payload(tmp_path,monkeypatch,family):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    data=source_data()
    if family is not None:
        data['families'][family]='train'  # Even a false legacy declaration is insufficient.
    (source/'data.json').write_text(json.dumps(data))
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:pytest.fail('Rejected family must not open payloads'))
    entry=dict(family=family,path='/MUST_NOT_OPEN_CORRECTION',trajectory_kind='expert_takeover_v1')
    with pytest.raises(ValueError,match='family'):
        build_dagger_aggregate(source,{'new':entry},out,'q_reached',include_geometry=False,
            include_frontend_tactile=False)
    assert not out.exists()
    assert json.loads((source/'data.json').read_text())==data


def test_known_train_family_is_allowed_without_any_legacy_family_registration():
    data=source_data();data['families']={}
    families=validate_aggregate_families(data,{'new':{'family':'train_family'}})
    assert families==dict(train_family='train',validation_family='validation',test_family='test')
    assert data['families']=={}


@pytest.mark.parametrize('fault', ['train_validation_family','validation_test_family','duplicate_episode','missing_family'])
def test_source_split_ambiguity_is_rejected(fault):
    data=source_data()
    if fault=='train_validation_family':data['entries']['validation']['family']='train_family'
    elif fault=='validation_test_family':data['entries']['test']['family']='validation_family'
    elif fault=='duplicate_episode':data['splits']['validation'].append('original')
    elif fault=='missing_family':del data['entries']['test']['family']
    with pytest.raises(ValueError,match='overlap|explicit family'):
        validate_aggregate_families(data,{'new':{'family':'train_family'}})


def test_duplicate_correction_identity_is_rejected_before_any_payload_load():
    with pytest.raises(ValueError,match='Duplicate'):
        validate_aggregate_families(source_data(),{'historical':{'family':'train_family'}})


def test_correction_development_partition_does_not_relabel_original_training_family():
    data=source_data()
    data['splits']['correction_dev']=['dev_correction']
    data['entries']['dev_correction']={'family':'train_family'}
    before=copy.deepcopy(data)
    assert aggregate_family_partitions(data)['train_family']=='train'
    assert validate_aggregate_families(data,{'new':{'family':'train_family'}})['test_family']=='test'
    assert data==before
