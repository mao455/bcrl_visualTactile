"""Training-only recovery onset windows on the unchanged native clock."""
import torch

RECOVERY_ONSET_WINDOW = 'recovery_onset_commit12_v1'


def validate_target_window(value):
    if value not in (None, RECOVERY_ONSET_WINDOW):
        raise ValueError('Unknown DP target-window contract')
    return value


def with_target_window(records, value):
    """Attach loss metadata without changing history or target values."""
    validate_target_window(value)
    if value is None:
        return records
    return {key: dict(record, target_window_contract=value) for key, record in records.items()}


def window_mask(supervised, segment, anchors, horizon, contract=None):
    validate_target_window(contract)
    indices = anchors[:, None] + torch.arange(horizon, device=anchors.device)
    safe = indices.clamp_max(len(supervised)-1)
    starts = anchors
    if contract is not None:
        eligible = supervised.nonzero().flatten()
        if len(eligible):
            first = eligible[0]
            # Only the first legal suffix, within the plan's committed prefix.
            # No older plans or later mask gaps acquire counterfactual targets.
            onset = (anchors == first//12*12) & (anchors < first)
            starts = torch.where(onset, first, anchors)
    before = indices < starts[:, None]
    valid = ((indices < len(supervised)) & supervised[safe]
             & (segment[safe] == segment[starts, None]))
    return (valid | before).long().cumprod(-1).bool() & ~before
