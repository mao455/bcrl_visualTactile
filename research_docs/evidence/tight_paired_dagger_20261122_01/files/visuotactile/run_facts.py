"""Report sampler identities from the actual loaded checkpoint/configuration."""
import json
from pathlib import Path

from .compact import sha256


def sampler_facts(learning, role):
    import torch
    child=Path(learning)/role
    continuation=json.loads((child/'continuation.json').read_text())
    path=Path(continuation['path'])
    if sha256(path)!=continuation['sha256']:
        raise ValueError('Historical collection frontend identity changed')
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    # A missing field is reported as missing; do not silently substitute an
    # implementation fallback for a checkpoint fact.
    seed=saved['contract'].get('sampler_seed')
    configuration=json.loads((child/'training/configuration.json').read_text())
    planning=configuration.get('offline_planning')
    return dict(role=role,online_checkpoint=str(path),online_checkpoint_sha256=continuation['sha256'],
        checkpoint_sampler_seed=seed,offline_planning=planning,
        queue_rng='persistent_per_lane_across_episodes',
        episode_rng_position='not recoverable from a seed alone; use saved per-episode generator state',
        training_configuration_sha256=sha256(child/'training/configuration.json'))
