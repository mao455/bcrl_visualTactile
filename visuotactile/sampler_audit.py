"""Read actual per-episode generator state without drawing or reseeding."""
from pathlib import Path
import torch
from .checkpointing import save
from .compact import sha256
from .training_resources import atomic_json

def record(directory,base,stage,*,identity,vision_arithmetic):
    if stage not in ('start','end'):raise ValueError('Sampler audit stage required')
    directory=Path(directory);path=directory/f'sampler_rng_{stage}.pt'
    if path.exists():raise FileExistsError('Preserve original episode RNG state')
    state=base.generator.get_state().cpu().clone()
    save(path,dict(generator_state=state,initial_seed=base.generator.initial_seed(),device=str(base.generator.device)))
    atomic_json(directory/f'sampler_rng_{stage}.json',dict(identity=identity,stage=stage,
        initial_seed=base.generator.initial_seed(),checkpoint_sampler_seed=base.model.contract.get('sampler_seed'),
        state_path=path.name,state_sha256=sha256(path),state_bytes=state.numel(),
        next_step=getattr(base,'next_step',None),generator_lifetime='persistent_per_lane_and_model_across_queue_episodes',
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        vision_arithmetic=vision_arithmetic,extra_random_draws=0))
