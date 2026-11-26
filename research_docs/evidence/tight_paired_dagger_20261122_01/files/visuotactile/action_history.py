"""Past native execution feedback for command baselines, never private plans."""
import torch


def history_contract(mode):
    if mode == "none":
        return dict(action_history_mode="none", action_history_version=0)
    if mode != "issued":
        raise ValueError("Unknown action history mode")
    return dict(action_history_mode="issued", action_history_version=1,
                training_source="past_executed_expert_teacher_forcing",
                deployment_source="past_executed_student_no_correction",
                command_history_alignment="transition[t-1] before obs[t]",
                command_history_units="canonical native commands; float64 simulation timestamps")


def execution_context(issued, delayed, issued_time, delayed_time, valid):
    if issued.shape != delayed.shape or issued.shape[-1] != 29 or issued.shape[:-1] != valid.shape:
        raise ValueError("Action history command/valid shapes disagree")
    if issued_time.shape != valid.shape or delayed_time.shape != valid.shape:
        raise ValueError("Action history timestamp shapes disagree")
    if not torch.isfinite(issued).all() or not torch.isfinite(delayed).all():
        raise ValueError("Non-finite native command history")
    if (issued.abs() > 1.000001).any() or (delayed.abs() > 1.000001).any():
        raise ValueError("History must contain clipped native commands")
    if not torch.isfinite(issued_time).all() or not torch.isfinite(delayed_time).all():
        raise ValueError("Non-finite command timestamp")
    valid = valid.bool()
    return dict(previous_issued_command=torch.where(valid[...,None], issued.detach(), 0.),
                previous_delayed_command=torch.where(valid[...,None], delayed.detach(), 0.),
                previous_command_time_s=torch.where(valid, issued_time.double(), 0.),
                previous_delayed_time_s=torch.where(valid, delayed_time.double(), 0.),
                action_history_valid=valid)


class ExecutionHistory:
    """Acknowledge actual completed transitions before the next policy call."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.transition = None

    def observe(self, transition, expected_command, episode_ids, step_ids):
        trace = transition["control_trace"]
        issued = transition["command_clipped_pre_delay"].detach().cpu()
        if not torch.equal(issued, expected_command.detach().cpu()):
            raise ValueError("Command-baseline history cannot include expert overrides or correction/mixed commands")
        if not torch.equal(transition["episode_id"].cpu(), episode_ids.cpu()) or not torch.equal(
                transition["step_id"].cpu(), step_ids.cpu()):
            raise ValueError("Execution acknowledgment belongs to a different action/episode")
        if not trace["valid"].all():
            raise ValueError("Native delayed-command telemetry is required")
        if self.transition is not None and torch.equal(self.transition["episode_id"], episode_ids.cpu()) and torch.equal(
                self.transition["step_id"], step_ids.cpu()):
            raise ValueError("Duplicate execution acknowledgment")
        self.transition = dict(episode_id=episode_ids.cpu().clone(), step_id=step_ids.cpu().clone(),
            issued=issued.clone(), delayed=trace["delayed_command"].detach().cpu().clone(),
            issued_time=transition["time_before_s"].cpu().clone(),
            delayed_time=trace["selected_command_time_s"].cpu().clone(),
            completed_time=transition["time_after_s"].cpu().clone())

    def context(self, packet):
        steps = packet["step_id"].cpu()
        valid = steps > 0
        if self.transition is None:
            if valid.any():
                raise ValueError("Missing prior execution acknowledgment")
            return execution_context(torch.zeros(self.num_envs,29), torch.zeros(self.num_envs,29),
                                     torch.zeros(self.num_envs,dtype=torch.float64),
                                     torch.zeros(self.num_envs,dtype=torch.float64), valid)
        previous = self.transition
        aligned = ((previous["episode_id"] == packet["episode_id"].cpu()) &
                   (previous["step_id"] == steps-1) &
                   ((previous["completed_time"]-packet["time_s"].cpu()).abs() <= 1e-6))
        if (valid & ~aligned).any():
            raise ValueError("Action history requires exactly the preceding completed native transition")
        return execution_context(previous["issued"], previous["delayed"], previous["issued_time"],
                                 previous["delayed_time"], valid)
