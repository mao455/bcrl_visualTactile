import json
from pathlib import Path

import pytest

from tools.tight_fresh_train import reconcile_learning_log
from visuotactile.compact import sha256


def row(update):
    return json.dumps(dict(epoch=(update-1)//2+1,updates=update,episode='example')).encode()+b'\n'


def test_restore_preserves_actual_crash_tail_and_excludes_from_durable_dose(tmp_path):
    path=tmp_path/'learning.jsonl';tail=row(3)+b'{"updates":4'
    path.write_bytes(row(1)+row(2)+tail)
    original=sha256(path)
    report=reconcile_learning_log(tmp_path,durable_update=2,completed_epochs=1,episodes_per_epoch=2,
        checkpoint_sha256='checkpoint')
    assert path.read_bytes()==row(1)+row(2)
    archive=Path(report['abandoned_path'])
    assert archive.read_bytes()==tail and report['original_log_sha256']==original
    assert report['known_discarded_update_ids']==[3] and report['abandoned_rows']==2
    assert report['discarded_updates_in_restored_model']==0
    assert report['abandoned_sha256']==sha256(archive)
    assert reconcile_learning_log(tmp_path,durable_update=2,completed_epochs=1,episodes_per_epoch=2) is None
    # A second real crash after retrying creates a different immutable archive.
    with path.open('ab') as stream:stream.write(row(3))
    again=reconcile_learning_log(tmp_path,durable_update=2,completed_epochs=1,episodes_per_epoch=2)
    assert again['abandoned_path']!=report['abandoned_path'] and archive.read_bytes()==tail


def test_restart_before_first_checkpoint_archives_all_actual_updates(tmp_path):
    path=tmp_path/'learning.jsonl';path.write_bytes(row(1))
    report=reconcile_learning_log(tmp_path,durable_update=0,completed_epochs=0,episodes_per_epoch=2)
    assert report['known_discarded_update_ids']==[1] and path.read_bytes()==b''


@pytest.mark.parametrize('payload',[row(1)+row(1),row(2),row(1)+b'broken\n'+row(2),row(1)+row(3)+row(2)])
def test_recovery_rejects_ambiguous_durable_logs_without_destroying_evidence(tmp_path,payload):
    path=tmp_path/'learning.jsonl';path.write_bytes(payload)
    with pytest.raises(ValueError):
        reconcile_learning_log(tmp_path,durable_update=2,completed_epochs=1,episodes_per_epoch=2)
    assert path.read_bytes()==payload and not list(tmp_path.glob('learning.abandoned.*'))


def test_recovery_requires_complete_epoch_and_present_durable_log(tmp_path):
    with pytest.raises(ValueError,match='completed-epoch'):
        reconcile_learning_log(tmp_path,durable_update=1,completed_epochs=1,episodes_per_epoch=2)
    with pytest.raises(ValueError,match='missing'):
        reconcile_learning_log(tmp_path,durable_update=2,completed_epochs=1,episodes_per_epoch=2)


def test_repeated_episode_deck_recovers_actual_length_and_preserves_crash_tail(tmp_path):
    from collections import Counter
    import torch
    from visuotactile.latest_takeover_sampling import episode_deck
    ids=['demo','historical','latest'];multiplicities=dict(demo=1,historical=1,latest=4)
    generator=torch.Generator().manual_seed(31)
    deck=episode_deck(ids,multiplicities,generator);width=len(deck)
    schedule_after=generator.get_state().clone()
    next_deck=episode_deck(ids,multiplicities,generator)
    def scheduled_row(update,identity):
        return json.dumps(dict(epoch=(update-1)//width+1,updates=update,episode=identity,
            episode_index=(update-1)%width+1,episodes_per_epoch=width)).encode()+b'\n'
    durable=b''.join(scheduled_row(index+1,identity) for index,identity in enumerate(deck))
    tail=b''.join(scheduled_row(width+index+1,identity) for index,identity in enumerate(next_deck[:3]))
    tail+=b'{"updates":10'
    path=tmp_path/'learning.jsonl';path.write_bytes(durable+tail)
    report=reconcile_learning_log(tmp_path,durable_update=width,completed_epochs=1,episodes_per_epoch=width)
    assert report['episodes_per_epoch']==6
    assert path.read_bytes()==durable and Path(report['abandoned_path']).read_bytes()==tail
    assert Counter(json.loads(line)['episode'] for line in path.read_text().splitlines())==multiplicities
    assert report['known_discarded_update_ids']==[7,8,9]
    # The same saved schedule state reconstructs all next-epoch visits, including
    # the discarded real updates, without counting those updates twice.
    generator.set_state(schedule_after)
    assert episode_deck(ids,multiplicities,generator)==next_deck


@pytest.mark.parametrize('changed', [dict(episode_index=2),dict(episodes_per_epoch=3)])
def test_recovery_rejects_wrong_expanded_deck_cursor_without_modifying_log(tmp_path,changed):
    saved=dict(epoch=1,updates=1,episode='latest',episode_index=1,episodes_per_epoch=6)
    saved.update(changed)
    path=tmp_path/'learning.jsonl';payload=json.dumps(saved).encode()+b'\n';path.write_bytes(payload)
    with pytest.raises(ValueError,match='deck length'):
        reconcile_learning_log(tmp_path,durable_update=0,completed_epochs=0,episodes_per_epoch=6)
    assert path.read_bytes()==payload and not list(tmp_path.glob('learning.abandoned.*'))
