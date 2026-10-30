"""Bounded training-only sampling checks on a completed long-horizon pilot."""
import argparse
import json
from pathlib import Path

import torch

from .compact import sha256
from .long_resources import Lease
from .models.long_horizon import LongHorizonConfig, LongHorizonPolicy
from .train_long import sampled_metrics
from .training_resources import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', choices=['G00-new', 'G10-new'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source = args.root/(args.group+'_pilot_fit3300')/'last.pt'
    output = source.parent/'sampling_review.json'
    if output.exists():
        raise ValueError('Preserve existing sampling review')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    lease = Lease(args.root)
    with lease.block():
        model = LongHorizonPolicy(LongHorizonConfig(**saved['configuration']['model'])).to('cuda:0')
    results = []
    for weights, steps in (('model', 16), ('ema', 16), ('ema', 100)):
        with lease.block():
            model.load_state_dict(saved[weights])
        model.cfg.inference_steps = steps
        metrics = sampled_metrics(model, args.root, saved['configuration']['train_episodes'], lease)
        results.append(dict(weights=weights, inference_steps=steps, metrics=metrics))
        atomic_json(output, dict(source_sha256=sha256(source), updates=saved['updates'],
            scope='same four complete training episodes; no test, no deployment change', results=results))
        print(args.group, weights, steps, json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
