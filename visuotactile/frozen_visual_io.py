"""Explicit eager/mapped immutable visual reads; no learned history cache."""
import json
from pathlib import Path

VERSION = 'frozen_visual_eager_v1'


def profile():
    return dict(version=VERSION,torch_load_mmap=False,
        lifetime='current_episode_or_current_B2_visits_only',learned_state_cache=False)


def prefetch_profile():
    return dict(version='frozen_visual_eager_prefetch_v2',torch_load_mmap=False,
        lifetime='current_B2_plus_at_most_one_next_B2',learned_state_cache=False,
        max_prefetch_bytes=3*1024**3,minimum_available_bytes=12*1024**3)


def mapped_profile():
    """Explicit zero-private-copy alternative; immutable pages share the OS cache.

    No asynchronous materialization: only the current one/two episodes are
    mapped. Learned history is still recomputed for every optimizer update.
    Old eager profiles and absent-profile behavior are unchanged.
    """
    return dict(version='frozen_visual_mapped_current_v3',torch_load_mmap=True,
        lifetime='current_episode_or_current_B2_visits_only',learned_state_cache=False)


def validate_profile(value):
    if value is None:return None
    if (not isinstance(value,dict) or value not in (profile(),prefetch_profile(),mapped_profile())
            or type(value.get('torch_load_mmap')) is not bool
            or value.get('learned_state_cache') is not False):
        raise ValueError('Exact frozen visual I/O profile required')
    return dict(value)


def load_profile(root):
    path=Path(root)/'frozen_visual_io.json'
    return validate_profile(json.loads(path.read_text())) if path.exists() else None
