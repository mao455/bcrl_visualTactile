"""Opt-in batching of independent offline plans on the native 5 Hz clock.

Conditions must already contain the current model's full causal 60 Hz history.
This changes denoiser batch shape, not history, supervision or optimizer steps.
The ordinary trainer/online controller do not import or enable this candidate.
"""
import torch

VERSION = 'offline_contiguous_anchor_batch_v1'
SEED = 7091
COMMIT = 12


def profile(max_anchors):
    if type(max_anchors) is not int or not 1 <= max_anchors <= 128:
        raise ValueError('Explicit integer offline anchor batch in [1,128] required')
    return dict(version=VERSION, max_anchors=max_anchors, seed=SEED,
        planning_interval_ticks=COMMIT, inference_steps=16, clip_sample=False,
        scheduler_index_mode='host', noise_draw_shape=[1, 24, 29],
        same_generator_per_row=True, split_at_missing_anchors=True,
        scope='offline inference only; unchanged full causal history and optimizer budget')


def validate_profile(value):
    if not isinstance(value, dict) or value != profile(value.get('max_anchors')):
        raise ValueError('Exact registered offline planning profile required')
    return value


def contiguous_blocks(ticks, max_anchors):
    """Return half-open row ranges, never joining across an unlabelled plan."""
    profile(max_anchors)
    if (not ticks or any(type(t) is not int or t < 0 or t % COMMIT for t in ticks)
            or ticks != sorted(set(ticks))):
        raise ValueError('Increasing unique integer anchors on the native planning clock required')
    blocks = []
    start = 0
    for index in range(1, len(ticks)):
        if index - start == max_anchors or ticks[index] != ticks[index-1] + COMMIT:
            blocks.append((start, index))
            start = index
    blocks.append((start, len(ticks)))
    return blocks


@torch.no_grad()
def predict_contiguous(model, conditions, anchors, *, configuration, check=None, receipt=None):
    """Preserve individual (1,H,29) random draws, including prefix/gap draws.

NativePolicy.predict's generator-list path draws rows sequentially. Repeating
the *same* generator here preserves the serial random stream on CUDA as well
as CPU. A scalar-generator batched randn is intentionally not equivalent.
CUDA denoiser kernels can still change floating-point rounding; live acceptance
is required. No random draw is made beyond the last supervised planning tick.
"""
    config = validate_profile(configuration)
    if (model.horizon != 24 or model.training or anchors.ndim != 1
            or anchors.dtype != torch.long or conditions.ndim != 2
            or len(conditions) != len(anchors)):
        raise ValueError('Eval-only H24 model and aligned condition/long anchor rows required')
    ticks = anchors.detach().cpu().tolist()
    blocks = contiguous_blocks(ticks, config['max_anchors'])
    generator = torch.Generator(device=conditions.device).manual_seed(SEED)
    cursor = 0
    skipped = 0
    plans = []
    for lo, hi in blocks:
        while cursor < ticks[lo]:
            if check is not None:
                check()
            torch.randn((1, model.horizon, 29), device=conditions.device, generator=generator)
            cursor += COMMIT
            skipped += 1
        if check is not None:
            check()
        plans.append(model.predict(conditions[lo:hi], [generator] * (hi-lo),
            inference_steps=16, clip_sample=False, scheduler_index_mode='host'))
        cursor = ticks[hi-1] + COMMIT
    if receipt is not None:
        receipt.update(profile=config, supervised_anchors=len(ticks), skipped_noise_draws=skipped,
            total_noise_draws=skipped+len(ticks), predict_calls=len(blocks),
            denoiser_calls=16*len(blocks), block_sizes=[hi-lo for lo, hi in blocks],
            first_anchor=ticks[0], last_anchor=ticks[-1])
    return torch.cat(plans)
