"""Atomic complete model/EMA/optimizer/RNG checkpoints; no training entrypoint."""
import torch
from .state_history import STATE_CONTRACT

def save(path,payload):
    temporary = path.with_suffix('.partial.pt')
    torch.save(payload,temporary)
    temporary.replace(path)


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
