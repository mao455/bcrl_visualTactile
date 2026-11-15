"""One immutable CPU snapshot per batch field, shared by numeric/compact rows.

The writer still takes bounded ownership before return. Tensor object memoization
avoids duplicate numeric/compact transfers; no mutable simulator view is queued.
"""
import numpy as np
import torch


def snapshot(value, memo=None):
    memo={} if memo is None else memo
    if isinstance(value,dict): return {k:snapshot(v,memo) for k,v in value.items()}
    if torch.is_tensor(value):
        key=id(value)
        if key not in memo:
            # .cpu() owns the GPU transfer, .copy() owns mutable CPU observations.
            cpu=value.detach().cpu().numpy()
            memo[key]=cpu if value.is_cuda else cpu.copy()
        return memo[key]
    return np.asarray(value)


def lane_arrays(value,lane,count):
    if isinstance(value,dict): return {k:lane_arrays(v,lane,count) for k,v in value.items()}
    return value[lane:lane+1] if value.ndim and value.shape[0]==count else value
