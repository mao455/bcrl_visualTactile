"""Explicit physical-tick gradient measure; recurrent chunks remain causal."""
import math

import torch


TICK_WINDOW_CONTRACT = dict(version='fixed_supervised_tick_window_v1',
    target_supervised_ticks=128, denominator='fixed_unweighted_target_ticks',
    flush='complete_episode_group', chunk_boundary='unsplit_32tick')


def validate_tick_window(value):
    if value is None:
        return None
    if value != TICK_WINDOW_CONTRACT:
        raise ValueError('Unknown residual tick-window contract')
    return dict(value)


class ResidualTickWindow:
    def __init__(self, contract):
        self.contract = validate_tick_window(contract)
        if self.contract is None:
            raise ValueError('Explicit tick-window contract required')
        self.target = self.contract['target_supervised_ticks']
        self.reset()

    def reset(self):
        self.report = dict(loss=0., supervised_ticks=0, contact_ticks=0,
            history_ticks=0, loss_weight_mass=0., window_chunks=0,
            initial_recovery_supervised_ticks=0, remaining_supervised_ticks=0,
            loss_components={})

    @property
    def pending(self):
        return self.report['supervised_ticks'] > 0

    @property
    def ready(self):
        return self.report['supervised_ticks'] >= self.target

    def backward(self, loss, optimizer, *, weight_mass, supervised_ticks,
                 contact_ticks, history_ticks, weight_report):
        if supervised_ticks <= 0 or not math.isfinite(weight_mass) or weight_mass < supervised_ticks:
            raise ValueError('Expected positive legal ticks with recovery weights >= 1')
        if not self.pending:
            optimizer.zero_grad(set_to_none=True)
        scale = weight_mass / self.target
        scaled = loss * scale
        scaled.backward()
        self.report['loss'] += float(scaled.detach())
        self.report['loss_weight_mass'] += weight_mass
        self.report['window_chunks'] += 1
        for key, value in dict(supervised_ticks=supervised_ticks,
                contact_ticks=contact_ticks, history_ticks=history_ticks).items():
            self.report[key] += value
        for key in ('initial_recovery_supervised_ticks', 'remaining_supervised_ticks'):
            self.report[key] += weight_report.get(key, 0)
        for key, value in weight_report.get('loss_components', {}).items():
            previous = self.report['loss_components'].get(key, 0.)
            self.report['loss_components'][key] = (max(previous, value)
                if key == 'label_roundoff_max_rad' else previous + scale * value)

    def step(self, model, optimizer):
        if not self.pending:
            raise ValueError('Cannot update an empty tick window')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite residual gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report = dict(self.report, gradient_norm=float(norm),
            fixed_denominator=self.target, group_tail=self.report['supervised_ticks'] < self.target,
            loss_weight_mass_is_exposure=False)
        self.reset()
        return report
