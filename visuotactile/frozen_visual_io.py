"""Opt-in eager file reads of immutable visual tensors; no feature/history cache."""
import json
from pathlib import Path

VERSION = 'frozen_visual_eager_v1'


def profile():
    return dict(version=VERSION,torch_load_mmap=False,
        lifetime='current_episode_or_current_B2_visits_only',learned_state_cache=False)


def validate_profile(value):
    if value is None:return None
    if (not isinstance(value,dict) or value!=profile()
            or value.get('torch_load_mmap') is not False or value.get('learned_state_cache') is not False):
        raise ValueError('Exact frozen visual eager I/O profile required')
    return dict(value)


def load_profile(root):
    path=Path(root)/'frozen_visual_io.json'
    return validate_profile(json.loads(path.read_text())) if path.exists() else None
