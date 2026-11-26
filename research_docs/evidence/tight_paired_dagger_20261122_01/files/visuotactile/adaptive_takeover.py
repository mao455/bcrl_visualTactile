"""Collection-only causal screens; no privileged truth enters the student."""
import numpy as np

VERSION = 'adaptive_stage_recovery_v1'
STAGES = ('contact', 'transport', 'alignment', 'instability', 'post_subgoal')


def profile(stage):
    if stage not in STAGES:
        raise ValueError('Unknown adaptive collection stage')
    return dict(version=VERSION, stage=stage, earliest_contact_tick=96,
                contact_stall_window=36, held_window=12, confirm_ticks=2,
                transport_travel_m=.035, near_root_m=.09,
                alignment_window=36, alignment_progress_m=.0015,
                keypoint_floor_m=.015, height_loss_m=.004,
                relative_slip_m=.008, minimum_elevation_m=.005)


def validate(value, fallback):
    if value != profile(value.get('stage')) or fallback != 10**9:
        raise ValueError('Exact adaptive screen and no fixed-time fallback required')
    return dict(value)


class AdaptiveTrigger:
    def __init__(self, configuration, fallback):
        self.configuration = validate(configuration, fallback)
        self.fallback_tick = self.scheduled_tick = fallback
        self.rows = []
        self.height = self.origin = self.first_held = None
        self.confirm = 0
        self.candidate = None
        self.trigger_completed_tick = self.at_trigger = self.last = None
        self.reason = 'requested_stage_not_reached'

    def observe(self, tick, pose, palm, force, *, keypoint_max_distance_m,
                goal_index_before, goal_index_after, goal_pose_before, goal_pose_after):
        if tick != len(self.rows) or tick >= self.scheduled_tick:
            raise ValueError('Observe every completed student transition once')
        pose, palm, force, goal = [np.asarray(v).copy() for v in
                                  (pose, palm, force, goal_pose_before)]
        if [v.shape for v in (pose, palm, force, goal)] != [(7,), (7,), (2, 5, 3), (7,)]:
            raise ValueError('Actual transition shapes required')
        if not all(np.isfinite(v).all() for v in (pose, palm, force, goal)) or not np.isfinite(keypoint_max_distance_m):
            raise ValueError('Finite recorded transition required')
        inv = palm[3:].copy(); inv[1:] *= -1
        v = pose[:3] - palm[:3]; cross = 2*np.cross(inv[1:], v)
        relative = v + inv[0]*cross + np.cross(inv[1:], cross)
        row = dict(z=float(pose[2]), xy=pose[:2], relative=relative,
                   fingers=int((np.linalg.norm(force, axis=-1) > .1).sum(-1).min()),
                   key=float(keypoint_max_distance_m), root=float(np.linalg.norm(pose[:3]-goal[:3])),
                   goal=int(goal_index_before),
                   changed=int(goal_index_before) != int(goal_index_after) or not np.array_equal(goal_pose_before, goal_pose_after))
        self.rows.append(row); n = len(self.rows); p = self.configuration
        if n == 48:
            self.height = float(np.median([r['z'] for r in self.rows[24:48]]))
            self.origin = np.median([r['xy'] for r in self.rows[24:48]], axis=0)
        if self.height is None:
            return None
        recent = self.rows[-p['held_window']:]
        held = len(recent) == p['held_window'] and all(r['fingers'] >= 2 and r['z'] >= self.height+.02 for r in recent)
        if held and self.first_held is None:
            self.first_held = n
        unchanged = all(not r['changed'] and r['goal'] == row['goal'] for r in recent)
        elevated = row['z'] >= self.height+p['minimum_elevation_m']
        loss = max(r['z'] for r in recent)-row['z']
        slip = float(np.linalg.norm(row['relative']-recent[0]['relative']))
        travel = float(np.linalg.norm(row['xy']-self.origin))
        contact_window = self.rows[-p['contact_stall_window']:]
        contact_stall = n >= p['earliest_contact_tick'] and len(contact_window) == p['contact_stall_window'] and self.first_held is None and sum(r['fingers'] >= 2 for r in contact_window) < 6
        window = self.rows[-p['alignment_window']:]
        near_stall = len(window) == p['alignment_window'] and held and row['root'] <= p['near_root_m'] and row['key'] > p['keypoint_floor_m'] and all(not r['changed'] and r['goal'] == row['goal'] for r in window) and min(r['key'] for r in window[:12])-min(r['key'] for r in window[-12:]) < p['alignment_progress_m']
        instability = self.first_held is not None and elevated and max(r['fingers'] for r in recent[:-1]) >= 2 and row['fingers'] < 2 and (loss >= p['height_loss_m'] or slip >= p['relative_slip_m'])
        conditions = dict(contact=contact_stall and row['goal'] == 0,
                          transport=held and travel >= p['transport_travel_m'] and row['goal'] == 0,
                          alignment=near_stall and row['goal'] == 0,
                          instability=instability,
                          post_subgoal=row['goal'] >= 1 and elevated and row['fingers'] >= 1)
        stage = p['stage']; qualifies = unchanged and conditions[stage]
        self.last = dict(completed_tick=n, stage=stage, held12=held, fingers=row['fingers'],
                         elevation_m=row['z']-self.height, root_m=row['root'], keypoint_m=row['key'],
                         height_loss12_m=loss, palm_slip12_m=slip, travel_xy_m=travel,
                         goal_index=row['goal'], qualifies=bool(qualifies))
        self.confirm = self.confirm+1 if qualifies else 0
        if self.confirm >= p['confirm_ticks']:
            self.trigger_completed_tick = self.scheduled_tick = n
            self.at_trigger = dict(self.last)
            self.reason = 'completed_student_'+stage
            return n
        return None

    def receipt(self):
        return dict(configuration=self.configuration, fallback_tick=self.fallback_tick,
                    actual_scheduled_takeover_tick=self.scheduled_tick,
                    trigger_completed_tick=self.trigger_completed_tick,
                    first_held_completed_tick=self.first_held,
                    completed_student_transitions_observed=len(self.rows),
                    settled_height_m=self.height, reason=self.reason,
                    at_trigger=self.at_trigger, last=self.last,
                    diagnostic_only=True, student_inputs_changed=False,
                    native_success_changed=False, no_expert_fallback=True,
                    phase_alignment='next action after causal confirmation; original planner phase retained',
                    limitation='A collection screen, not a proven harmful error or guaranteed expert recoverability')
