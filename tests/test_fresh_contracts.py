import pytest
from visuotactile.fresh_contracts import FRESH_DATA_CONTRACT, residual_normalization


def cohort(supervision='q_reached'):
    return dict(visual_stride=1, supervision=supervision,
        normalization_identity=supervision+'_fresh_train_only_affine_v1',
        normalization_scope='selected train30 only',
        splits=dict(train=[f't{i}' for i in range(30)], validation=[f'v{i}' for i in range(6)], test=[]))


def test_both_fresh_roles_share_only_command_normalization():
    for role in ('q_reached', 'u_expert'):
        assert residual_normalization(cohort(role), dict(fresh_data_contract=FRESH_DATA_CONTRACT)) == 'u_expert_fresh_train_only_affine_v1'
    assert residual_normalization({}, {}) == 'u_expert_train166_affine_v1'


@pytest.mark.parametrize('mutation', ['sparse', 'overlap', 'test', 'old_stats', 'wrong_train', 'scope'])
def test_fresh_contract_rejects_incompatible_data(mutation):
    data=cohort()
    if mutation == 'sparse': data['visual_stride']=12
    elif mutation == 'overlap': data['splits']['validation'][0]='t0'
    elif mutation == 'test': data['splits']['test']=['test0']
    elif mutation == 'old_stats': data['normalization_identity']='q_reached_train166_affine_v1'
    elif mutation == 'wrong_train': data['splits']['train'].pop()
    elif mutation == 'scope': data['normalization_scope']='all episodes'
    with pytest.raises(ValueError): residual_normalization(data, dict(fresh_data_contract=FRESH_DATA_CONTRACT))
