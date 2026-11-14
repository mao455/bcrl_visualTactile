import copy

import pytest

from tools.tight_fresh_study_onset import correction_cohorts


def population(count=54, new_count=1):
    original=[f'original{i}' for i in range(30)]
    corrections=[f'correction{i}' for i in range(count)]
    validation=[f'validation{i}' for i in range(6)]
    entries={i:dict(family=i) for i in original+validation}
    entries.update({name:dict(family=original[i%30],trajectory_kind='expert_takeover_v1')
        for i,name in enumerate(corrections)})
    return dict(normalization_train_ids=original,entries=entries,
        splits=dict(train=original+corrections,validation=validation,test=[]),
        aggregation=dict(new_training_ids=corrections[-new_count:]))


def test_anchor_uses_exact_new_population_instead_of_historical_counts():
    data=population();before=copy.deepcopy(data)
    cohorts=correction_cohorts(data,'matched_rear_output_anchor_study_v1')
    assert len(cohorts['all_corrections'])==54
    assert cohorts['current_corrections']==['correction53']
    assert data==before
    old=correction_cohorts(population(53,12),None)
    assert len(old['all53'])==53 and len(old['new12'])==12
    with pytest.raises(ValueError,match='Historical'):correction_cohorts(data,None)


@pytest.mark.parametrize('change',['overlap','unknown_family','duplicate','not_takeover','wrong_current','unknown_version'])
def test_population_rejects_changes_that_would_distort_fitting_comparison(change):
    data=population();version='matched_rear_output_anchor_study_v1'
    if change=='overlap':data['entries']['correction0']['family']='validation0'
    elif change=='unknown_family':data['entries']['correction0']['family']='unknown'
    elif change=='duplicate':data['splits']['train'].append('correction0')
    elif change=='not_takeover':data['entries']['correction0'].pop('trajectory_kind')
    elif change=='wrong_current':data['aggregation']['new_training_ids']=['original0']
    elif change=='unknown_version':version='unregistered'
    with pytest.raises(ValueError):correction_cohorts(data,version)
