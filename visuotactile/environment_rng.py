"""Opt-in CPU-physics RNG stream isolation, without changing native draw order."""
from contextlib import contextmanager
import random
import numpy as np
import torch


class RandomStream:
    def __init__(self, seed):
        self.active = False
        with self._preserved():
            random.seed(seed); np.random.seed(seed)
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            self.state = self.snapshot()

    @staticmethod
    def snapshot():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

    @staticmethod
    def restore(state):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])

    @contextmanager
    def _preserved(self):
        state = self.snapshot()
        try:
            yield
        finally:
            self.restore(state)

    @contextmanager
    def scope(self):
        if self.active:
            raise RuntimeError('Environment RNG scopes must not nest')
        with self._preserved():
            self.active = True; self.restore(self.state)
            try:
                yield
            finally:
                self.state = self.snapshot(); self.active = False


class IsolatedEnvironment:
    """Every reset/step uses the same environment-owned stream for every policy.

    Only CPU simulation is admitted. CUDA DP noise has its own explicit generator.
    No promise of independent RNG *lanes* or cross-configuration PhysX parity.
    """
    def __init__(self, env, seed):
        if str(env.device) != 'cpu':
            raise ValueError('RNG contract v1 is CPU-physics only')
        self.env = env
        self.stream = RandomStream(seed)

    def reset(self, seed):
        self.stream = RandomStream(seed)
        with self.stream.scope():
            return self.env.reset(seed=seed)

    def step(self, command):
        with self.stream.scope():
            return self.env.step(command)
