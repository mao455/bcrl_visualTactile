import errno
from pathlib import Path

import pytest

from tools.kit_resource_preflight import allocate_probe, census, decision


LIMITS = dict(max_user_watches=524288, max_user_instances=4096)


def test_enospc_blocks_even_when_proc_count_looks_safe():
    result = decision(dict(watch_references=100, instance_fd_references=1), LIMITS,
        dict(requested=256, allocated=63, errno=errno.ENOSPC, cleanup_complete=True))
    assert result['state'] == 'blocked'
    assert 'bounded_inotify_allocation_failed' in result['reasons']


def test_near_exhaustion_blocks_even_when_small_probe_passes():
    result = decision(dict(watch_references=524190, instance_fd_references=86), LIMITS,
        dict(requested=16, allocated=16, errno=None, cleanup_complete=True))
    assert result['state'] == 'blocked'
    assert 'inotify_watch_headroom_below_20_percent' in result['reasons']


def test_resource_admission_never_certifies_full_episode_reliability():
    result = decision(dict(watch_references=100, instance_fd_references=1), LIMITS,
        dict(requested=16, allocated=16, errno=None, cleanup_complete=True))
    assert result['resource_admitted']
    assert result['kit_reliability_passed'] is False
    assert result['state'] == 'preflight'


def test_probe_is_bounded_and_releases_own_fds_and_paths(tmp_path):
    before = len(list(Path('/proc/self/fd').iterdir()))
    result = allocate_probe(2, tmp_path)
    assert result['cleanup_complete']
    assert not list(tmp_path.iterdir())
    assert len(list(Path('/proc/self/fd').iterdir())) == before
    # ENOSPC on an already-full shared host is a legitimate observed result.
    assert result['allocated'] == 2 or result['errno'] is not None
    with pytest.raises(ValueError):
        allocate_probe(1025, tmp_path)


def test_census_counts_zero_watch_instances_and_does_not_claim_unique(tmp_path):
    import os
    process = tmp_path/'123'; (process/'fd').mkdir(parents=True); (process/'fdinfo').mkdir()
    (process/'fd/7').symlink_to('anon_inode:inotify')
    (process/'fdinfo/7').write_text('pos:\t0\n')
    (process/'comm').write_text('fixture\n')
    (process/'status').write_text('Threads:\t2\nVmRSS:\t100 kB\n')
    result = census(tmp_path, os.getuid())
    assert result['instance_fd_references'] == 1
    assert result['watch_references'] == 0
    assert 'upper bound' in result['counting']
