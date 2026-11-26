"""Opt-in, once-per-episode fingerprints of actual first-plan tensors.

The recorder never forwards a model or draws a random number. Large tensors are
hashed only; small exact snapshots are saved once beside the JSON receipt. CPU
copies can synchronize CUDA and add wall time, but do not advance simulation.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compact import sha256

VERSION = 'first_native_plan_forensics_v1'
SMALL_TENSOR_BYTES = 65536


def tensor_snapshot(value):
    if not torch.is_tensor(value):
        raise TypeError('Forensics accepts actual tensors only')
    source_device = str(value.device)
    value = value.detach().cpu().contiguous().clone()
    raw = value.reshape(-1).view(torch.uint8).numpy()
    descriptor = dict(shape=list(value.shape), dtype=str(value.dtype), bytes=raw.nbytes)
    encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
    descriptor['sha256'] = hashlib.sha256(encoded+b'\0'+raw.tobytes()).hexdigest()
    descriptor['source_device'] = source_device
    return value, descriptor


class FirstPlanRecorder:
    def __init__(self, directory, *, binding=None):
        self.directory = Path(directory)
        self.binding = json.loads(json.dumps(binding or {}))
        self.pending = None
        self.receipts = []
        self.episodes = set()

    def before_sampling(self, episode, tick, tensors, generator):
        if tick != 0:
            raise ValueError('First-plan recording is restricted to the first episode tick')
        if self.pending is not None or episode in self.episodes:
            raise ValueError('Do not duplicate or overwrite an episode first-plan record')
        if 'generator_state_before' in tensors:
            raise ValueError('Generator state must come directly from its actual independent generator')
        began = time.perf_counter()
        # get_state is a read, not a draw. Record it after the already-computed
        # condition and immediately before the existing sampler call.
        values = dict(tensors, generator_state_before=generator.get_state())
        snapshots = {name: tensor_snapshot(value) for name, value in values.items()}
        self.pending = dict(episode_id=int(episode),step_id=int(tick),snapshots=snapshots,
            sampler_seed=generator.initial_seed(),sampler_device=str(generator.device),
            capture_wall_seconds=time.perf_counter()-began)
        self.episodes.add(episode)

    def after_sampling(self, episode, plan, generator):
        pending = self.pending
        if pending is None or pending['episode_id'] != episode:
            raise ValueError('First-plan output must match its pending episode input')
        began = time.perf_counter()
        for name, value in dict(plan=plan,generator_state_after=generator.get_state()).items():
            pending['snapshots'][name] = tensor_snapshot(value)
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.directory/f'episode_{episode:08d}_tick_000000'
        json_path, npz_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
        if json_path.exists() or npz_path.exists():
            raise FileExistsError('Preserve every first-plan forensic receipt')
        descriptors = {}; arrays = {}
        for name, (value, descriptor) in pending.pop('snapshots').items():
            descriptors[name] = descriptor
            if descriptor['bytes'] <= SMALL_TENSOR_BYTES:
                # Raw bytes preserve BF16 and every other dtype without lossy
                # NumPy conversion. Original shape/dtype are in the JSON.
                arrays[name] = value.reshape(-1).view(torch.uint8).numpy()
                descriptor.update(npz_key=name,npz_encoding='flat_uint8_exact_torch_storage')
        with npz_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        receipt = dict(version=VERSION,**pending,binding=self.binding,tensors=descriptors,
            tensor_runtime=dict(torch_version=str(torch.__version__),torch_cuda_build=torch.version.cuda,
                generator_state_format='Opaque torch.Generator state bytes for the recorded torch/runtime/device'),
            small_tensors=dict(path=npz_path.name,sha256=sha256(npz_path),maximum_tensor_bytes=SMALL_TENSOR_BYTES),
            output_capture_and_npz_write_wall_seconds=time.perf_counter()-began,
            scope='Actual first native observation and existing sampler inputs/output. No extra model forward or RNG draw; large RGB-D/features are hashes only. Hash equality is exact dtype/shape/bytes equality, not physical-state replay.',
            timing='Opt-in CPU snapshots/hash/write can synchronize CUDA and increase first-plan wall time. Simulation/control dt, tensor timestamps and random draws are unchanged. Exclude this overhead when comparing ordinary evaluation throughput.')
        with json_path.open('x') as stream:
            json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
        self.receipts.append(dict(episode_id=int(episode),json_path=str(json_path),json_sha256=sha256(json_path),
            npz_path=str(npz_path),npz_sha256=sha256(npz_path)))
        self.pending = None

    def summary(self):
        if self.pending is not None:
            raise ValueError('First-plan sampling has not completed')
        return dict(version=VERSION,episodes=len(self.receipts),receipts=self.receipts,
            ordinary_throughput_comparable=False,simulator_dt_changed=False,extra_rng_draws=0)
