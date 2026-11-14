import pytest

from tools.tight_fresh_visual_family_probe import family_folds, fold_population


def population():
    original=[f'original{i:02}' for i in range(30)]
    corrections=[f'correction{i:02}_{j}' for i in range(30) for j in range(2)]
    entries={identity:dict(family=f'family{i:02}') for i,identity in enumerate(original)}
    entries.update({f'correction{i:02}_{j}':dict(family=f'family{i:02}') for i in range(30) for j in range(2)})
    return dict(entries=entries),dict(original=original,corrections=corrections)


def test_every_training_family_is_evaluated_once_and_all_sibling_episodes_leave_fitting():
    data,pop=population()
    families=[data['entries'][i]['family'] for i in pop['original']]
    folds=family_folds(list(reversed(families)))
    assert folds==family_folds(families)
    evaluated=[]
    for fold in folds:
        heldout,train=fold_population(data,pop,fold)
        assert len(heldout)==12 and len(train['original_only'])==24
        assert len(train['with_corrections'])==72
        for ids in train.values():
            assert not set(ids)&set(heldout)
            assert not {data['entries'][i]['family'] for i in ids}&set(fold)
        evaluated.extend(heldout)
    assert sorted(evaluated)==sorted(pop['corrections'])


@pytest.mark.parametrize('families',[['f']*30,[str(i) for i in range(29)]])
def test_duplicate_or_missing_families_are_not_valid_folds(families):
    with pytest.raises(ValueError):family_folds(families)


def test_unknown_fold_family_is_rejected_before_feature_fitting():
    data,pop=population()
    with pytest.raises(ValueError):fold_population(data,pop,['unknown']+[f'family{i:02}' for i in range(5)])
