import copy

import pytest

from tools.tight_fresh_backend_study_report import check_exposure, check_onset_binding


def receipts():
    control = [dict(update=1, epoch=1, supervised_ticks=124, contact_ticks=70,
                    history_ticks=160, window_chunks=5, initial_recovery_supervised_ticks=12,
                    remaining_supervised_ticks=112, fixed_denominator=128, group_tail=False,
                    loss_weight_mass=124),
               dict(update=2, epoch=1, supervised_ticks=28, contact_ticks=10,
                    history_ticks=32, window_chunks=1, initial_recovery_supervised_ticks=4,
                    remaining_supervised_ticks=24, fixed_denominator=128, group_tail=True,
                    loss_weight_mass=28)]
    weighted = copy.deepcopy(control)
    for row in weighted:
        row['loss_weight_mass'] += 2 * row['initial_recovery_supervised_ticks']
    return control, weighted


def test_weighted_gradient_mass_is_not_additional_observation_exposure():
    control, weighted = receipts()
    report = check_exposure(control, weighted, 3)
    assert report['updates'] == 2
    assert report['supervised_ticks'] == report['control_loss_weight_mass'] == 152
    assert report['initial_recovery_ticks'] == 16
    assert report['weighted_loss_weight_mass'] == 184
    assert report['loss_weight_mass_is_exposure'] is False


@pytest.mark.parametrize('field', ['supervised_ticks', 'history_ticks', 'window_chunks',
                                  'initial_recovery_supervised_ticks', 'fixed_denominator'])
def test_changed_optimizer_window_is_rejected_even_when_totals_match(field):
    control, weighted = receipts()
    weighted[0][field] -= 1
    weighted[1][field] += 1
    with pytest.raises(ValueError, match='schedule or exposure'):
        check_exposure(control, weighted, 3)


def test_changed_weight_mass_is_rejected_without_changing_actual_exposure():
    control, weighted = receipts()
    weighted[0]['loss_weight_mass'] += 1
    with pytest.raises(ValueError, match='loss weight mass'):
        check_exposure(control, weighted, 3)


def onset_pair():
    source = dict(complete=True, test_used=False, device='cpu', arm='source',
        cuda_initialized=False,
        error_contract='physical_command_rmse_first_contiguous_legal12_next12_remaining_v1',
        source_sha256='analyzer', dependency_sha256={'model': 'dependency'},
        registration_sha256='registration', roles={'A': dict(frontend_sha256='dp',
            backend_sha256='source_rear', data_sha256='data', numeric_sha256='numeric',
            cache_index_sha256='cache_index', cases=[dict(identity='actual_takeover',
                cache_sha256='cache', legal_mask_sha256='legal_mask',
                bin_mask_sha256=dict(first12='first', next12='next', remaining='remaining'))])})
    weighted = copy.deepcopy(source)
    weighted['arm'] = 'weighted'
    weighted['roles']['A']['backend_sha256'] = 'weighted_rear'
    return source, weighted


def test_onset_comparison_accepts_same_history_and_masks_with_distinct_selected_rears():
    source, weighted = onset_pair()
    assert check_onset_binding(weighted, source, 'A', 'weighted', 'registration',
                               'weighted_rear') is weighted['roles']['A']


@pytest.mark.parametrize('field', ['legal_mask_sha256', 'bin_mask_sha256'])
def test_onset_comparison_rejects_changed_legal_rows_or_recovery_window(field):
    source, weighted = onset_pair()
    case = weighted['roles']['A']['cases'][0]
    if field == 'bin_mask_sha256':
        # Identical legal rows and totals do not permit shifting the first12 window.
        case[field]['first12'], case[field]['next12'] = case[field]['next12'], case[field]['first12']
    else:
        case[field] = 'different_legal_rows'
    with pytest.raises(ValueError, match='different records'):
        check_onset_binding(weighted, source, 'A', 'weighted', 'registration', 'weighted_rear')
