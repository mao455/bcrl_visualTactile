"""Process-local simulation RNG scopes; default CPU contract remains unchanged."""
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


class DeviceRandomStream(RandomStream):
    """CPU/native draws plus one explicitly selected CUDA device, not all GPUs."""
    def __init__(self, seed, device):
        selected = torch.device(device)
        if selected.type != 'cuda' or selected.index is None:
            raise ValueError('DeviceRandomStream needs an explicit cuda:N device')
        self.device = selected
        self.active = False
        with self._preserved():
            random.seed(seed); np.random.seed(seed)
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            # A private Generator avoids manual_seed[_all] and other GPU streams.
            torch.cuda.set_rng_state(torch.Generator(device=selected).manual_seed(seed).get_state(), selected)
            self.state = self.snapshot()

    def snapshot(self):
        return (*RandomStream.snapshot(), dict(device=str(self.device), state=torch.cuda.get_rng_state(self.device).clone()))

    @staticmethod
    def restore(state):
        RandomStream.restore(state)
        torch.cuda.set_rng_state(state[3]['state'], state[3]['device'])


class LaneRandomStreams:
    """Independent streams for explicitly scoped lane reset callbacks.

    This utility does not intercept native vectorized step draws. Users must not
    describe a batch using IsolatedEnvironment as having independent event lanes.
    """
    def __init__(self, seeds, *, device='cpu'):
        self.device = str(device)
        self.streams = {}
        for lane, seed in enumerate(seeds):
            self.replace(lane, seed)

    def replace(self, lane, seed):
        if type(lane) is not int or lane < 0 or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Lane and seed must be nonnegative integer identities; seed < 2**32')
        self.streams[lane] = (RandomStream(seed) if self.device == 'cpu' else DeviceRandomStream(seed, self.device))

    def scope(self, lane):
        if lane not in self.streams:
            raise ValueError('Unregistered lane RNG identity')
        return self.streams[lane].scope()


class IsolatedEnvironment:
    """Every reset/step uses the same environment-owned stream for every policy.

    CPU v1 stays the default. GPU simulation requires a registered backend and
    scopes the selected CUDA device in addition to CPU. DP noise keeps its own
    explicit generator. Registered resets pass no seed to the simulator (which
    would globally reseed all CUDA devices); all draws use the owned scope.
    No promise of independent RNG *lanes* or cross-configuration PhysX parity.
    """
    def __init__(self, env, seed, *, execution_backend=None):
        from .execution_backend import execution_backend as resolve_backend
        requested = execution_backend if execution_backend is not None else getattr(env, 'execution_backend', None)
        self.backend = resolve_backend(requested) if requested is not None else None
        self.device = str(env.device)
        if self.backend is not None and self.backend['physics_device'] != self.device:
            raise ValueError('RNG device differs from the registered execution backend')
        if self.device != 'cpu' and self.backend is None:
            raise ValueError('RNG contract v1 is CPU-physics only')
        self.env = env
        self.stream = self._stream(seed)

    def _stream(self, seed):
        return RandomStream(seed) if self.device == 'cpu' else DeviceRandomStream(seed, self.device)

    def reset(self, seed):
        self.stream = self._stream(seed)
        with self.stream.scope():
            return self.env.reset(seed=seed if self.backend is None else None)

    def reset_lanes(self, lanes):
        """Native indexed reset only; caller owns scene forward and observations.

        No call to env.seed/reset and no global reseed. Other lane state is left
        to the native indexed reset contract. Event draws still share a batch.
        """
        ids = list(lanes)
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < self.env.num_envs for i in ids):
            raise ValueError('Reset requires unique registered lane indices')
        with self.stream.scope():
            return self.env._reset_idx(torch.tensor(ids, device=self.device, dtype=torch.long))

    def step(self, command):
        with self.stream.scope():
            return self.env.step(command)

    def begin_events(self, seed):
        """Opt-in post-reset event seed; native draws/distributions stay intact."""
        if not isinstance(seed, int) or seed < 0:
            raise ValueError('A nonnegative integer event seed is required')
        self.stream = self._stream(seed)
