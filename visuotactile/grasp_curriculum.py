"""Causal collection screens and rotation-permitting holding diagnostics.

Truth is used only by the collector/diagnostics, never by the student. Neither
the trigger nor these metrics alter native success, termination or supervision.
"""
import numpy as np

EARLY_GRASP_VERSION = 'early_contact_recovery_v1'
EARLY_GRASP = dict(version=EARLY_GRASP_VERSION, minimum_tick=48,
                   weak_contact_ticks=12, contact_deadline=96, lift_deadline=150,
                   loss_ticks=2, rising_height_m=.005, lift_height_m=.02)


def validate_early(configuration, fallback_tick):
    if configuration != EARLY_GRASP or fallback_tick != 1_000_000_000:
        raise ValueError('Early grasp v1 requires its explicit fixed causal thresholds')
    return dict(configuration)


def longest_run(mask):
    best = count = 0
    for value in mask:
        count = count + 1 if value else 0
        best = max(best, count)
    return best


def holding_metrics(pose, force, *, window=60):
    """One second held: elevated throughout, >=90% two-finger ticks, gap<=3.

    Each contact tick requires both 120Hz samples, >=2 fingers above0.1N.
    The height floor is2cm above completed25-48 median. No palm orientation
    constraint: task-directed in-hand rotation is allowed. Report120 as well.
    """
    pose, force = np.asarray(pose), np.asarray(force)
    if len(pose) < 48:
        return dict(reference_available=False, sustained=False, longest_contact_height_ticks=0)
    height = float(np.median(pose[24:48, 2]))
    two = ((np.linalg.norm(force, axis=-1) > .1).sum(-1) >= 2).all(-1)
    elevated = pose[:, 2] >= height + .02
    held = two & elevated
    good = []
    for end in range(max(48, window), len(pose)+1):
        contact = two[end-window:end]
        if (elevated[end-window:end].all() and contact.sum() >= np.ceil(.9*window)
                and longest_run(~contact) <= 3):
            good.append(end)
    return dict(reference_available=True, height_reference_m=height, window_ticks=window,
                duration_seconds=window/60, sustained=bool(good),
                first_completed_tick=good[0] if good else None,
                qualified_windows=len(good), held_at_final_window=len(pose) in good,
                longest_contact_height_ticks=longest_run(held),
                native_success_changed=False, rotation_allowed=True)


class EarlyGraspTrigger:
    """Intervene on the NEXT actual action, including between DP plans.

    Original30 experts first touch at55..90 (median68), first elevated contact
    median119/q75=127.75. Twelve failed held0 prefixes first touch51..80. The
    deadlines are screens for expert help, not claims that slower starts fail.
    Contact losses use a short recent history and two completed weak ticks.
    """
    def __init__(self, configuration, fallback_tick):
        self.configuration = validate_early(configuration, fallback_tick)
        self.fallback_tick = self.scheduled_tick = fallback_tick
        self.poses = []; self.finger_counts = []; self.height = None
        self.first_contact = None; self.first_two = None; self.first_lift = None
        self.reason = 'no_early_recovery_condition'
        self.trigger_completed_tick = None; self.at_trigger = None

    def observe(self, tick, pose, palm, force):
        pose, palm, force = (np.asarray(x) for x in (pose, palm, force))
        if (tick != len(self.poses) or tick >= self.scheduled_tick
                or pose.shape != (7,) or palm.shape != (7,) or force.shape != (2, 5, 3)
                or not all(np.isfinite(x).all() for x in (pose, palm, force))):
            raise ValueError('Expected each finite completed real student transition once')
        self.poses.append(pose.copy())
        self.finger_counts.append(int((np.linalg.norm(force,axis=-1)>.1).sum(-1).min()))
        n = len(self.poses)
        if n == 48: self.height = float(np.median(np.asarray(self.poses)[24:48, 2]))
        if self.height is None: return None
        fingers = np.asarray(self.finger_counts)
        two = fingers >= 2; elevation = float(pose[2] - self.height)
        if fingers[-1] > 0 and self.first_contact is None: self.first_contact = n
        if two[-1] and self.first_two is None: self.first_two = n
        if two[-1] and elevation >= .02 and self.first_lift is None: self.first_lift = n
        reason = None
        # Do not interpret rotation, a missed old stable window or a native
        # height diagnostic as a dropped object. React to observed contact.
        if not two[-2:].any() and two[max(48, n-8):n-2].sum() >= 3:
            reason = 'recent_two_finger_contact_lost'
        elif elevation >= .005 and not two[-2:].any():
            reason = 'rising_without_reliable_contact'
        elif (self.first_contact is not None and n-self.first_contact >= 12
              and two[-12:].sum() < 3):
            reason = ('lifted_contact_degraded' if self.first_lift is not None else 'contact_not_established_after_touch')
        elif self.first_contact is None and n >= 96:
            reason = 'no_contact_by96'
        elif n >= 150 and self.first_lift is None:
            reason = 'no_contact_lift_by150'
        if reason is None: return None
        self.reason = reason; self.scheduled_tick = self.trigger_completed_tick = n
        self.at_trigger = dict(elevation_m=elevation, fingers_both_substeps=int(fingers[-1]),
            recent12_two_finger_ticks=int(two[-12:].sum()),
            object_palm_root_distance_m=float(np.linalg.norm(pose[:3]-palm[:3])))
        return n

    def receipt(self):
        return dict(configuration=self.configuration, fallback_tick=self.fallback_tick,
            actual_scheduled_takeover_tick=self.scheduled_tick, reason=self.reason,
            trigger_completed_tick=self.trigger_completed_tick,
            completed_student_transitions_observed=len(self.poses), settled_height_m=self.height,
            first_contact_completed_tick=self.first_contact, first_two_finger_completed_tick=self.first_two,
            first_contact_lift_completed_tick=self.first_lift, at_trigger=self.at_trigger,
            rule='Completed contact history only; next-action takeover, no12tick grid; expert recoverability must be measured',
            diagnostic_only=True, student_inputs_changed=False, native_success_changed=False,
            future_information_used=False, stable_grasp_claim=False)
