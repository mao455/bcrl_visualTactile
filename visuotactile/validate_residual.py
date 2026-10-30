"""Read-only real checkpoint/cache/feedback equivalence on an observed prefix."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .manifest import ManifestReader
from .residual_controller import ResidualController
from .residual_data import MANIFEST
from .residual_session import SessionGate
from .train_residual import chunk_record, read_record
from .training_resources import atomic_json


def tensors(value):
    if isinstance(value, dict):
        return {name:tensors(item) for name,item in value.items()}
    return torch.as_tensor(np.asarray(value))[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    gate = SessionGate(args.root, gpu_stage=True)
    gate.before()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads((args.root / 'cache/index.json').read_text())
    episode = index['splits']['train'][0]
    record = read_record(args.root, episode, index)
    controller = ResidualController(args.root / 'training_pilot/b_best.pt')
    controller.base.generator.manual_seed(record['policy_noise_seed'])
    reader = ManifestReader(MANIFEST).reader(episode)
    errors, events = [], []
    prefix = min(13, record['steps'])
    with torch.no_grad():
        inputs = chunk_record(record, 0, prefix)
        def move(value):
            return {name:move(item) for name,item in value.items()} if isinstance(value, dict) else value.cuda()
        expected, _ = controller.model.forward_cached(**move(inputs))
        for step in range(prefix):
            gate.before()
            began = time.monotonic()
            actual = tensors(reader.read(reader.episodes[0], step))
            controller.act(actual['obs'])
            controller.acknowledge(actual['transition'], actual['transition']['submitted_action'], source='released_expert_rollin')
            base_error = float((controller.last['base'][0]-record['base'][step]).abs().max())
            plan_error = float((controller.last['plans'][0]-record['plans'][step]).abs().max())
            residual_error = float((controller.last['raw'][0]-expected[0, step].cpu()).abs().max())
            errors.append([base_error, plan_error, residual_error])
            events.append(dict(step=step, cursor=int(controller.last['cursor'][0]),
                generated_step=int(controller.last['generation_step'][0]),
                memory_updates=int(controller.last['memory_updates'][0]),
                contact=bool(controller.last['contact'][0]),
                previous_teacher_command_exact=bool(torch.equal(controller.last['previous_command'][0],
                                                                record['motion']['previous_command'][step]))))
            gate.pace(began)
    maximum = np.max(errors, axis=0).tolist()
    passed = max(maximum)<1e-4 and all(event['previous_teacher_command_exact'] for event in events)
    atomic_json(args.root / 'causal_validation.json', dict(passed=passed, episode=episode, steps=prefix,
        max_errors=dict(base=maximum[0], remaining_plan=maximum[1], residual_raw=maximum[2]), events=events,
        note='read-only expert observations and stored native acknowledgments; not a new rollout or performance test'))
    if not passed:
        raise AssertionError('Offline/online causal replay differs')


if __name__ == '__main__':
    main()
