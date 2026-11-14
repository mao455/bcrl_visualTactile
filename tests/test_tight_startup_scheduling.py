import pytest

from tools.tight_state import kit_startup_scheduling


def test_default_preserves_original_kit_arguments_bytewise():
    args, receipt = kit_startup_scheduling({})
    assert args == ('--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
                    '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')
    assert receipt['explicit_override'] is False


def test_opt_in_changes_only_task_scheduler_not_physics_threads():
    default, _ = kit_startup_scheduling({})
    args, receipt = kit_startup_scheduling({'TIGHT_KIT_TASKING_THREADS':'4'})
    assert args == default.replace('plugin/threadCount=2', 'plugin/threadCount=4')
    assert receipt['kit_tasking_threads'] == 4 and receipt['physics_threads'] == 2
    assert receipt['explicit_override'] is True


@pytest.mark.parametrize('value', ['', '0', '8', '4.0', '--/persistent/physics/numThreads=4'])
def test_unknown_or_injected_setting_fails_before_kit_start(value):
    with pytest.raises(ValueError, match='explicitly 2 or 4'):
        kit_startup_scheduling({'TIGHT_KIT_TASKING_THREADS':value})
