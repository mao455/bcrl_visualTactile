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


def aggregate():
    d=cohort();d['normalization_train_ids']=d['splits']['train'].copy()
    d['splits']['train'].append('takeover1')
    d['normalization_scope']='unchanged original training affine; no aggregate refit'
    d['aggregation']={'version':'expert_takeover_aggregate_v1','student_prefix_supervised':False}
    d['entries']={'takeover1':{'trajectory_kind':'expert_takeover_v1'}}
    return d


def test_aggregate_is_explicit_and_retains_original_affine():
    from visuotactile.fresh_contracts import FRESH_AGGREGATE_CONTRACT
    d=aggregate()
    assert residual_normalization(d,{'fresh_data_contract':FRESH_AGGREGATE_CONTRACT})=='u_expert_fresh_train_only_affine_v1'
    with pytest.raises(ValueError):residual_normalization(d,{'fresh_data_contract':FRESH_DATA_CONTRACT})


@pytest.mark.parametrize('bad',['refit','student_label','fake_episode','wrong_original','overlap'])
def test_aggregate_does_not_weaken_original_guards(bad):
    from visuotactile.fresh_contracts import FRESH_AGGREGATE_CONTRACT
    d=aggregate()
    if bad=='refit':d['normalization_scope']='all aggregated'
    elif bad=='student_label':d['aggregation']['student_prefix_supervised']=True
    elif bad=='fake_episode':d['entries']['takeover1']['trajectory_kind']='student'
    elif bad=='wrong_original':d['normalization_train_ids'].pop()
    elif bad=='overlap':d['splits']['validation'][0]='takeover1'
    with pytest.raises(ValueError):residual_normalization(d,{'fresh_data_contract':FRESH_AGGREGATE_CONTRACT})
