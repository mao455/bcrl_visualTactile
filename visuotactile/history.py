"""Shared causal indexing for dataset windows and independently resetting envs."""
from collections import deque
import numpy as np
import torch


def history_indices(t, length, stride=1):
    if t < 0 or length < 1 or stride < 1:
        raise ValueError("Nonnegative time, positive history length/stride required")
    raw = t - np.arange(length-1, -1, -1) * stride
    return raw.clip(min=0), raw >= 0


class OnlineHistory:
    def __init__(self, num_envs, length, stride=1, keys=("image", "proprio")):
        self.length, self.stride, self.keys = length, stride, tuple(keys)
        self.buffers = [deque(maxlen=(length-1)*stride+1) for _ in range(num_envs)]
        self.episodes = [None]*num_envs
        self.steps = [None]*num_envs

    def append(self, packet):
        for i, buf in enumerate(self.buffers):
            ep, step = int(packet["episode_id"][i]), int(packet["step_id"][i])
            if self.episodes[i] != ep:
                buf.clear()
                self.episodes[i], self.steps[i] = ep, None
            if self.steps[i] is not None and step != self.steps[i]+1:
                raise ValueError("History expects exactly one observation per control step")
            self.steps[i] = step
            buf.append({k: packet[k][i].detach().clone() for k in self.keys})

    def batch(self):
        result = {k: [] for k in self.keys}
        masks = []
        for buf in self.buffers:
            if not buf:
                raise ValueError("History is empty")
            idx, mask = history_indices(len(buf)-1, self.length, self.stride)
            for k in self.keys:
                result[k].append(torch.stack([buf[j][k] for j in idx]))
            masks.append(torch.as_tensor(mask, device=buf[0][self.keys[0]].device))
        return {**{k: torch.stack(v) for k,v in result.items()}, "valid": torch.stack(masks)}
