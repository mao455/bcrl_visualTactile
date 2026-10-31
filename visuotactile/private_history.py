"""Versioned private base proposals with observed native delay timestamps."""
from collections import deque

import torch

from .action_history import ExecutionHistory


class PrivateProposalHistory(ExecutionHistory):
    def __init__(self):
        super().__init__(1)
        self.proposals = deque(maxlen=64)

    def observe(self, *args, **kwargs):
        raise ValueError('Private history requires observe_proposal, never actual/mixed command values')

    def observe_proposal(self, transition, proposal, packet):
        if proposal.shape != (1,29) or not torch.isfinite(proposal).all() or proposal.abs().max()>1:
            raise ValueError('Invalid private base proposal')
        trace = transition['control_trace']
        if not trace['valid'].all():
            raise ValueError('Native delay timing is required')
        for name in ('episode_id','step_id'):
            if not torch.equal(transition[name].cpu(),packet[name].cpu()):
                raise ValueError('Proposal timing belongs to another transition')
        if self.transition is not None and int(transition['step_id'][0]) != int(self.transition['step_id'][0])+1:
            raise ValueError('Private proposal transitions must be consecutive')
        submitted_time = transition['time_before_s'].cpu().clone()
        delayed_time = trace['selected_command_time_s'].cpu().clone()
        if abs(float(submitted_time[0]-packet['time_s'][0]))>1e-6 or delayed_time[0]>submitted_time[0]+1e-6:
            raise ValueError('Private proposal timestamps are not causal')
        self.proposals.append((float(submitted_time[0]),proposal.detach().cpu().clone()))
        candidates = [value for timestamp,value in self.proposals if abs(timestamp-float(delayed_time[0]))<=1e-6]
        if len(candidates)!=1:
            raise ValueError('Native selected timestamp has no unique private proposal')
        self.transition = dict(episode_id=packet['episode_id'].cpu().clone(),step_id=packet['step_id'].cpu().clone(),
            issued=proposal.detach().cpu().clone(),delayed=candidates[0].clone(),issued_time=submitted_time,
            delayed_time=delayed_time,completed_time=transition['time_after_s'].cpu().clone())


def private_action_rows(proposals, times, selected_times, start, stop):
    result = torch.zeros(stop-start,61)
    for step in range(start,stop):
        if step==0:
            continue
        previous = step-1
        distance = (times[:step]-selected_times[previous]).abs()
        matched = torch.nonzero(distance<=1e-6).flatten()
        if len(matched)!=1:
            raise ValueError('No unique past private proposal for native delayed timestamp')
        result[step-start] = torch.cat((proposals[previous],proposals[int(matched[0])],
            torch.tensor([float(times[step]-times[previous]),float(times[step]-selected_times[previous]),1.])))
    return result
