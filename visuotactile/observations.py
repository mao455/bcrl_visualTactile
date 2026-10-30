"""Versioned, explicit inference inputs and operator-provided task commands."""
import torch

PERCEPTION_VERSION = 2
CAMERA_IDS = ("global_oblique", "local_opposite")
# A declared task identity, NOT a success-derived phase or simulator goal pose.
TASK_IDS = ("tight_insertion",)
NOMINAL_KEYS_V2 = ("image", "depth_valid", "camera_valid", "camera_time_s", "time_s",
                   "proprio", "task_id", "goal_command", "goal_valid",
                   "previous_nominal_action", "previous_nominal_valid", "prior_plan", "prior_plan_valid")
ACTION_HISTORY_KEYS = ("previous_issued_command", "previous_delayed_command",
                       "previous_command_time_s", "previous_delayed_time_s", "action_history_valid")


def nominal_keys(version, action_history_mode="none"):
    if version == 1:
        return ("image", "proprio")
    if version == 2:
        if action_history_mode not in ("none", "issued"):
            raise ValueError("Unsupported action history mode")
        return NOMINAL_KEYS_V2 + (ACTION_HISTORY_KEYS if action_history_mode == "issued" else ())
    raise ValueError("Unsupported perception version")


def empty_plan_context(n, horizon=12, device="cpu"):
    return dict(previous_nominal_action=torch.zeros(n, 29, device=device),
                previous_nominal_valid=torch.zeros(n, dtype=torch.bool, device=device),
                prior_plan=torch.zeros(n, horizon, 29, device=device),
                prior_plan_valid=torch.zeros(n, horizon, dtype=torch.bool, device=device))


class TaskCommand:
    """External command: task ID plus optional base-frame xyz + 6D rotation.

    No env or simulation state accepted. An absent pose is masked, never filled
    from the teacher's goal. Pose estimators need their own validated adapter.
    """
    def __init__(self, task, goal=None):
        if task not in TASK_IDS:
            raise ValueError(f"Unregistered external task {task!r}; extend the versioned task vocabulary explicitly")
        self.task = task
        self.goal = torch.zeros(9)
        self.valid = goal is not None
        if goal is not None:
            if set(goal) != {"source", "frame", "position_m", "rotation_columns"}:
                raise ValueError("External goal requires source, frame, position_m, rotation_columns")
            if goal["source"] != "operator_command" or goal["frame"] != "robot_base":
                raise ValueError("Only explicit operator commands in robot_base are implemented")
            self.goal = torch.tensor(goal["position_m"] + goal["rotation_columns"], dtype=torch.float)
            if self.goal.shape != (9,) or not torch.isfinite(self.goal).all():
                raise ValueError("Expected finite xyz + first two rotation columns")
            a, b = self.goal[3:6], self.goal[6:9]
            if abs(float(a.norm())-1) > 1e-4 or abs(float(b.norm())-1) > 1e-4 or abs(float(a@b)) > 1e-4:
                raise ValueError("Command rotation columns must be orthonormal")

    def packet(self, n, device):
        return dict(task_id=torch.full((n,), TASK_IDS.index(self.task), dtype=torch.long, device=device),
                    goal_command=self.goal.to(device).expand(n, -1).clone(),
                    goal_valid=torch.full((n,), self.valid, dtype=torch.bool, device=device))

    def metadata(self):
        return dict(vocabulary=list(TASK_IDS), task=self.task, source="operator_task_selection",
                    pose_source="operator_command" if self.valid else "absent", frame="robot_base",
                    pose=self.goal.tolist() if self.valid else None,
                    limitation="Task identity alone does not specify hidden multi-goal pose or task stage")
