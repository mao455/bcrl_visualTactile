"""Opt-in complete-history replay of the latest real corrective suffixes."""
import torch
from .latest_takeover_sampling import sampling_plan, episode_deck


def plan(data, records, specification):
    contract, counts = sampling_plan(data, specification)
    unique = sum(int(records[i]['supervision_mask'].sum()) for i in counts)
    repeated = sum(n*int(records[i]['supervision_mask'].sum()) for i, n in counts.items())
    return contract, counts, unique, repeated


def exposure(records, counts, visits, *, selected_epoch=None):
    rows = {i: dict(multiplicity=n, visits=int(visits[i]),
                   valid_ticks_per_visit=int(records[i]['supervision_mask'].sum()),
                   supervised_ticks=int(visits[i])*int(records[i]['supervision_mask'].sum()))
            for i, n in counts.items()}
    return dict(episodes=rows, selected_epoch=selected_epoch,
                scope='Actual complete causal history visits; only clean executed expert actions supervised')


@torch.no_grad()
def latest_fit(model, records, data, lease):
    from .anchor_continue import residual_prediction
    bins = {name: [] for name in ('all_legal', 'first90_legal', 'control_ticks_before150')}
    rows = []
    for identity in data['aggregation']['new_training_ids']:
        r = records[identity]; pred = residual_prediction(model, r, lease).cpu()
        valid = r['supervision_mask'].cpu().bool(); indices = valid.nonzero().flatten()
        early = torch.zeros_like(valid); early[indices[:90]] = True
        masks = dict(all_legal=valid, first90_legal=early,
                     control_ticks_before150=valid & (torch.arange(len(valid)) < 150))
        e = pred-r['target'].cpu()
        row = dict(identity=identity, bins={})
        for name, mask in masks.items():
            if mask.any():
                values = e[mask].square()
                bins[name].append(values)
                row['bins'][name] = dict(ticks=int(mask.sum()),
                    arm_rmse_rad=float(values[:, :7].mean().sqrt()),
                    hand_rmse_rad=float(values[:, 7:].mean().sqrt()))
        rows.append(row)
    summary = {}
    for name, values in bins.items():
        if values:
            x = torch.cat(values)
            summary[name] = dict(ticks=len(x), arm_rmse_rad=float(x[:, :7].mean().sqrt()),
                                 hand_rmse_rad=float(x[:, 7:].mean().sqrt()))
        else: summary[name] = dict(ticks=0, arm_rmse_rad=None, hand_rmse_rad=None)
    return dict(summary=summary, episodes=rows, test_used=False,
                scope='Latest real expert suffixes; new selected DP inputs, current rear weights, real past teacher-forced history; not autonomy')
