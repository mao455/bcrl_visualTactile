import h5py
import pytest

from test_training_expansion import setup_registry, write
from tools.tight_fresh_expansion_preregister import preregister


def check(setup,train,development):
    write(setup['source']/'session.json',dict(domain=setup['domain']))
    return preregister(setup['source'],setup['partition'],setup['families'],
        'future_session',train,development)


def test_metadata_only_preserves_six_validation_and_separates_requested_resets(setup_registry,monkeypatch):
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('No trajectory reads permitted'))
    proof=check(setup_registry,[6000,6001],[7000,7001])
    assert len(proof['existing_validation_families'])==6
    assert proof['existing_validation_identities']['A']==proof['existing_validation_identities']['B']
    assert proof['checks']['unique_new_train_configurations']==2
    assert proof['checks']['train_development_configuration_overlap']==[]
    assert not proof['created_session'] and proof['test_trajectories_opened']==0
    assert all(row['split']=='development_only' for row in proof['development'])


@pytest.mark.parametrize('seeds',[[6000],[1000],[7000,7000]])
def test_train_old_or_duplicate_development_seed_is_rejected(setup_registry,seeds):
    with pytest.raises(ValueError):check(setup_registry,[6000,6001],seeds)


def test_different_seed_cannot_conceal_matching_configuration(setup_registry,monkeypatch):
    from visuotactile.narrow_data import direct_configuration
    same=direct_configuration(setup_registry['domain'],6000)
    monkeypatch.setattr('tools.tight_fresh_expansion_preregister.direct_configuration',lambda domain,seed:same)
    with pytest.raises(ValueError,match='configuration'):check(setup_registry,[6000,6001],[7000])
