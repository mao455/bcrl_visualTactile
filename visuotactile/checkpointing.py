"""Atomic complete model/EMA/optimizer/RNG checkpoints; no training entrypoint."""
import torch
from .state_history import STATE_CONTRACT

def save(path,payload):
    temporary = path.with_suffix('.partial.pt')
    torch.save(payload,temporary)
    temporary.replace(path)


def promote_snapshot(source, destination):
    """Atomically give a completed immutable snapshot its selected-best name.

    The next `save(source, ...)` replaces the inode, retaining an earlier best.
    No second tensor snapshot, optimizer serialization or RNG read is necessary.
    """
    import os
    import uuid
    from pathlib import Path
    source,destination=Path(source),Path(destination)
    if source.parent.resolve()!=destination.parent.resolve() or source==destination:
        raise ValueError('Selected snapshot must share its checkpoint directory')
    if source.is_symlink() or not source.is_file():raise ValueError('Completed regular checkpoint required')
    temporary=destination.with_name(destination.name+'.link_'+uuid.uuid4().hex+'.partial')
    try:
        os.link(source,temporary)
        os.replace(temporary,destination)
    finally:
        if temporary.exists():temporary.unlink()


def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):
        return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):
        return {name:cpu_snapshot(item) for name,item in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(cpu_snapshot(item) for item in value)
    return value
def training_state(model, averaged, optimizer, ema, generator, noise, **metadata):
    return cpu_snapshot(dict(contract=getattr(model,'contract',STATE_CONTRACT), model=model.state_dict(), ema=averaged.state_dict(),
        optimizer=optimizer.state_dict(), ema_optimization_step=ema.optimization_step,
        schedule_rng=generator.get_state(), noise_rng=noise.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [], **metadata))


def restore_training(saved, model, averaged, optimizer, ema, generator, noise):
    if saved['contract'] != getattr(model,'contract',STATE_CONTRACT):
        raise ValueError('Checkpoint contract mismatch')
    model.load_state_dict(saved['model']); averaged.load_state_dict(saved['ema'])
    optimizer.load_state_dict(saved['optimizer']); ema.optimization_step = saved['ema_optimization_step']
    generator.set_state(saved['schedule_rng']); noise.set_state(saved['noise_rng'])
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
