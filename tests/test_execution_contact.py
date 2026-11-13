from pathlib import Path

import pytest

from tools import tight_execution_contact as contact
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW


def test_only_unique_registered_root_is_allowed():
    assert contact.checked_root(contact.ROOT) == contact.ROOT
    with pytest.raises(ValueError):
        contact.checked_root(contact.COMPARATOR)
    with pytest.raises(ValueError):
        contact.checked_root(contact.ROOT / 'replacement')


def test_preserves_direction_correct_loss_and_current_weight_replay():
    configuration = contact.backend_contract()
    assert configuration['loss_contract']['regularizer_weight'] == 0
    assert configuration['loss_contract']['version'] == contact.PROJECTION_FY_LOSS['version']
    assert configuration['output_contract'] == contact.REACHABLE_CENTERED_OUTPUT
    assert configuration['history_contract'] == contact.HISTORY
    assert configuration['tick_window_contract']['target_supervised_ticks'] == 128
    assert configuration['learning_rate'] == 1e-4
    configuration['tick_window_contract']['target_supervised_ticks'] = 1
    assert contact.backend_contract()['tick_window_contract']['target_supervised_ticks'] == 128
    assert RECOVERY_ONSET_WINDOW == 'recovery_onset_commit12_v1'


@pytest.mark.parametrize('mode', ['prepare', 'run'])
def test_actual_entrypoint_dispatch(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(contact, mode, lambda root: calls.append(root))
    monkeypatch.setattr('sys.argv', ['tight_execution_contact', mode, '--root', str(contact.ROOT)])
    contact.main()
    assert calls == [Path(contact.ROOT)]
