"""Retained historical decision/compatibility helpers for existing controls.

Collection, training and publication owners are retired; use tools.tight_finish.
Full historical implementation remains in Git b85738e.
"""
from tools.runtime import CODE, read, write, sha
GATE="live_collision_mesh_bounds_v1"


def freeze(root):
    write(root/'source_freeze.json',{str(p):sha(p) for folder in ('visuotactile','tools','isaacsimenvs','evaluation','rl_games')
        for p in sorted((CODE/folder).rglob('*.py'))})


def check_freeze(root):
    for p,digest in read(root/'source_freeze.json').items():
        if sha(p)!=digest:raise RuntimeError('Frozen source drift: '+p)


def environment(root, evaluation=False, *, camera_content_validation=None):
    # Legacy callers share the current constructor; no current entry imports a
    # retired run/session loader from this historical pipeline.
    from tools.tight_state import environment as current_environment
    return current_environment(root, evaluation, camera_content_validation=camera_content_validation)
