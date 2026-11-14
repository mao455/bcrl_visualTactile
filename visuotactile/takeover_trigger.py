"""Diagnostic-only causal takeover scheduling; never a student input or success rule."""
import numpy as np

VERSION = 'stable_grasp_then_delay_v1'
CONTACT_HEIGHT_VERSION = 'contact_height_then_delay_v1'
HELD_NEAR_GOAL_VERSION = 'held_near_goal_then_delay_v1'
NEVER_FALLBACK = 1_000_000_000


def validate_trigger(configuration, fallback_tick):
    if isinstance(configuration,dict) and configuration.get('version')==HELD_NEAR_GOAL_VERSION:
        if (set(configuration)!={'version','delay_ticks','max_keypoint_distance_m'}
                or type(configuration['delay_ticks']) is not int or configuration['delay_ticks']!=0
                or type(fallback_tick) is not int or fallback_tick!=NEVER_FALLBACK
                or isinstance(configuration['max_keypoint_distance_m'],bool)
                or not isinstance(configuration['max_keypoint_distance_m'],(int,float))
                or not np.isfinite(configuration['max_keypoint_distance_m'])
                or configuration['max_keypoint_distance_m']!=.06):
            raise ValueError('Held near-goal v1 requires explicit60mm, delay0 and no fallback')
        return dict(configuration)
    if not isinstance(configuration, dict) or set(configuration) != {'version', 'delay_ticks'}:
        raise ValueError('Explicit stable-grasp trigger version and delay required')
    delay = configuration['delay_ticks']
    if configuration['version'] not in (VERSION, CONTACT_HEIGHT_VERSION) or isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError('Invalid stable-grasp trigger contract')
    if isinstance(fallback_tick, bool) or not isinstance(fallback_tick, int) or fallback_tick < 48:
        raise ValueError('Trigger fallback must permit the completed 25-48 reference ticks')
    return dict(configuration)


def stable_windows(pose,palm,force,height):
    """Original 12tick contact/relative-pose thresholds, explicit height reference."""
    contact=np.linalg.norm(force,axis=-1)>.1
    two=(contact.sum(-1)>=2).all(-1)
    inverse=palm[:,3:].copy();inverse[:,1:]*=-1
    qv=inverse[:,1:];v=pose[:,:3]-palm[:,:3];cross=2*np.cross(qv,v)
    relative=v+inverse[:,:1]*cross+np.cross(qv,cross)
    q=pose[:,3:]
    relative_q=np.concatenate((inverse[:,:1]*q[:,:1]-(qv*q[:,1:]).sum(-1,keepdims=True),
        inverse[:,:1]*q[:,1:]+q[:,:1]*qv+np.cross(qv,q[:,1:])),axis=-1)
    stable=np.zeros(len(pose),bool)
    for stop in range(12,len(pose)+1):
        start=stop-12
        if not two[start:stop].all() or not (pose[start:stop,2]>=height+.02).all():continue
        drift=np.linalg.norm(relative[start:stop]-relative[start],axis=-1).max()
        cosine=np.clip(np.abs((relative_q[start:stop]*relative_q[start]).sum(-1)),0,1)
        stable[stop-1]=drift<=.01 and (2*np.arccos(cosine)).max()<=.2
    return stable,contact


class StableGraspTrigger:
    """Observe completed student transitions; switch on a future action index.

    Tick arguments and scheduled switches are zero-based action indices. A stable
    window ending at completed count N schedules action N+delay. The height
    reference is frozen after observing completed transitions 25 through 48.
    """
    def __init__(self, configuration, fallback_tick):
        self.configuration = validate_trigger(configuration, fallback_tick)
        if configuration['version'] != VERSION:
            raise ValueError('StableGraspTrigger requires the original stable-grasp contract')
        self.fallback_tick = fallback_tick
        self.poses, self.palms, self.forces = [], [], []
        self.height = None
        self.first_stable_completed_tick = None
        self.scheduled_tick = fallback_tick
        self.reason = 'fallback_no_completed_stable_window'

    def observe(self, tick, pose, palm, force):
        if tick != len(self.poses) or tick >= self.scheduled_tick:
            raise ValueError('Trigger observes each completed student transition exactly once')
        values = [np.asarray(x).copy() for x in (pose, palm, force)]
        if [x.shape for x in values] != [(7,), (7,), (2, 5, 3)] or not all(np.isfinite(x).all() for x in values):
            raise ValueError('Finite actual object/palm poses and both physics contact samples required')
        self.poses.append(values[0]); self.palms.append(values[1]); self.forces.append(values[2])
        completed = len(self.poses)
        if completed == 48:
            self.height = float(np.median(np.asarray(self.poses)[24:48, 2]))
        if self.height is None or self.first_stable_completed_tick is not None:
            return None
        stable, _ = stable_windows(np.asarray(self.poses[-12:]), np.asarray(self.palms[-12:]),
            np.asarray(self.forces[-12:]), self.height)
        if not stable[-1]:
            return None
        self.first_stable_completed_tick = completed
        candidate = completed + self.configuration['delay_ticks']
        if candidate <= self.fallback_tick:
            self.scheduled_tick = candidate
            self.reason = 'completed_student_stable_window_plus_delay'
            return candidate
        self.reason = 'fallback_before_requested_stable_delay'
        return None

    def receipt(self):
        return dict(configuration=self.configuration, fallback_tick=self.fallback_tick,
            actual_scheduled_takeover_tick=self.scheduled_tick, reason=self.reason,
            completed_student_transitions_observed=len(self.poses), settled_height_m=self.height,
            first_stable_window_end_completed_tick=self.first_stable_completed_tick,
            rule='12 ticks, both physics samples >=2 object-contact fingers >0.1N, >=2cm above median completed25-48, palm-relative drift<=1cm/0.2rad',
            indices='Schedule is zero-based action index; stable end is completed transition count',
            diagnostic_only=True, student_inputs_changed=False)


def contact_height_windows(pose, force, height):
    """Twelve actual lifted/contact transitions; makes no rigid-grasp claim."""
    pose, force = np.asarray(pose), np.asarray(force)
    if (pose.ndim != 2 or pose.shape[1:] != (7,)
            or force.shape != (len(pose), 2, 5, 3)
            or not np.isfinite(pose).all() or not np.isfinite(force).all()
            or not np.isfinite(height)):
        raise ValueError('Finite actual poses and both physics contact samples required')
    two = ((np.linalg.norm(force, axis=-1) > .1).sum(-1) >= 2).all(-1)
    instantaneous = two & (pose[:, 2] >= height + .02)
    result = np.zeros(len(pose), dtype=bool)
    for end in range(11, len(pose)):
        result[end] = instantaneous[end-11:end+1].all()
    return result


class ContactHeightTrigger:
    """Schedule from contact and height while retaining the old stability diagnosis.

    Relative object/palm motion is deliberately not an admission condition. This
    is a collection screen, not a revised definition of stable grasp or success.
    All inputs are already completed student transitions, never expert labels.
    """
    def __init__(self, configuration, fallback_tick):
        self.configuration = validate_trigger(configuration, fallback_tick)
        if configuration['version'] != CONTACT_HEIGHT_VERSION:
            raise ValueError('ContactHeightTrigger requires its explicit contract')
        self.fallback_tick = fallback_tick
        self.poses, self.palms, self.forces = [], [], []
        self.height = None
        self.first_contact_height_completed_tick = None
        self.frozen_stable_at_trigger = None
        self.scheduled_tick = fallback_tick
        self.reason = 'fallback_no_completed_contact_height_window'

    def observe(self, tick, pose, palm, force):
        if tick != len(self.poses) or tick >= self.scheduled_tick:
            raise ValueError('Trigger observes each completed student transition exactly once')
        values = [np.asarray(x).copy() for x in (pose, palm, force)]
        if [x.shape for x in values] != [(7,), (7,), (2, 5, 3)] or not all(np.isfinite(x).all() for x in values):
            raise ValueError('Finite actual object/palm poses and both physics contact samples required')
        self.poses.append(values[0]); self.palms.append(values[1]); self.forces.append(values[2])
        completed = len(self.poses)
        if completed == 48:
            self.height = float(np.median(np.asarray(self.poses)[24:48, 2]))
        if self.height is None or self.first_contact_height_completed_tick is not None:
            return None
        poses, palms, forces = (np.asarray(x[-12:]) for x in (self.poses, self.palms, self.forces))
        if not contact_height_windows(poses, forces, self.height)[-1]:
            return None
        self.first_contact_height_completed_tick = completed
        self.frozen_stable_at_trigger = bool(stable_windows(poses, palms, forces, self.height)[0][-1])
        candidate = completed + self.configuration['delay_ticks']
        if candidate <= self.fallback_tick:
            self.scheduled_tick = candidate
            self.reason = 'completed_student_contact_height_window_plus_delay'
            return candidate
        self.reason = 'fallback_before_requested_contact_height_delay'
        return None

    def receipt(self):
        return dict(configuration=self.configuration, fallback_tick=self.fallback_tick,
            actual_scheduled_takeover_tick=self.scheduled_tick, reason=self.reason,
            completed_student_transitions_observed=len(self.poses), settled_height_m=self.height,
            first_contact_height_window_end_completed_tick=self.first_contact_height_completed_tick,
            frozen_stable_window_at_trigger=self.frozen_stable_at_trigger,
            rule='12 ticks, both physics samples >=2 object-contact fingers >0.1N, >=2cm above median completed25-48; no relative-pose admission condition',
            indices='Schedule is zero-based action index; contact-height end is completed transition count',
            diagnostic_only=True, student_inputs_changed=False, stable_grasp_claim=False,
            native_success_changed=False)


class HeldNearGoalTrigger:
    """A completed contact-height window near the same current native target.

    Keypoint distance belongs to the transition's pre-action goal. Reject a
    goal/stage change in that transition; no future state or expert action is
    inspected. The12tick holding window permits rotation within the hand.
    """
    def __init__(self,configuration,fallback_tick):
        self.configuration=validate_trigger(configuration,fallback_tick)
        if self.configuration['version']!=HELD_NEAR_GOAL_VERSION:
            raise ValueError('HeldNearGoalTrigger requires its explicit contract')
        self.fallback_tick=fallback_tick;self.scheduled_tick=fallback_tick
        self.poses=[];self.palms=[];self.forces=[];self.height=None
        self.first_contact_height_completed_tick=None;self.first_held_near_goal_completed_tick=None
        self.reason='no_completed_held_near_goal_window';self.near_goal_at_trigger=None
        self.target_change_transitions=0

    def observe(self,tick,pose,palm,force,*,keypoint_max_distance_m,goal_index_before,goal_index_after,
                goal_pose_before,goal_pose_after):
        if tick!=len(self.poses) or tick>=self.scheduled_tick:
            raise ValueError('Trigger observes each completed student transition exactly once')
        values=[np.asarray(x).copy() for x in (pose,palm,force,goal_pose_before,goal_pose_after)]
        if ([x.shape for x in values]!=[(7,),(7,),(2,5,3),(7,),(7,)]
                or not all(np.isfinite(x).all() for x in values)):
            raise ValueError('Finite actual object/palm/contact and current native goal poses required')
        scalars=[np.asarray(x) for x in (keypoint_max_distance_m,goal_index_before,goal_index_after)]
        if any(x.shape!=() or not np.isfinite(x) or float(x)<0 for x in scalars):
            raise ValueError('Finite nonnegative native keypoint distance and goal indices required')
        distance,before,after=map(float,scalars)
        if not before.is_integer() or not after.is_integer():raise ValueError('Native goal stages must be integers')
        unchanged=before==after and np.array_equal(values[3],values[4])
        self.target_change_transitions+=not unchanged
        self.poses.append(values[0]);self.palms.append(values[1]);self.forces.append(values[2])
        completed=len(self.poses)
        if completed==48:self.height=float(np.median(np.asarray(self.poses)[24:48,2]))
        if self.height is None:return None
        held=contact_height_windows(np.asarray(self.poses[-12:]),np.asarray(self.forces[-12:]),self.height)[-1]
        if held and self.first_contact_height_completed_tick is None:self.first_contact_height_completed_tick=completed
        if not held or not unchanged or distance>self.configuration['max_keypoint_distance_m']:return None
        self.first_held_near_goal_completed_tick=completed;self.scheduled_tick=completed
        self.reason='completed_student_held_near_goal_window_plus_delay'
        self.near_goal_at_trigger=dict(native_keypoint_max_distance_m=distance,active_goal_index=int(before),
            goal_pose_w=values[3].tolist(),pre_post_goal_unchanged=True)
        return self.scheduled_tick

    def receipt(self):
        return dict(configuration=self.configuration,fallback_tick=self.fallback_tick,
            actual_scheduled_takeover_tick=self.scheduled_tick,reason=self.reason,
            completed_student_transitions_observed=len(self.poses),settled_height_m=self.height,
            first_contact_height_window_end_completed_tick=self.first_contact_height_completed_tick,
            first_held_near_goal_window_end_completed_tick=self.first_held_near_goal_completed_tick,
            near_goal_at_trigger=self.near_goal_at_trigger,target_change_transitions=self.target_change_transitions,
            rule='12 completed ticks, both physics samples >=2 object-contact fingers >0.1N, >=2cm above median completed25-48; current native keypoint max<=60mm; unchanged pre/post goal pose and stage; next action takeover',
            indices='Zero-based action index; a window ending at completed count N schedules action N',
            diagnostic_only=True,student_inputs_changed=False,stable_grasp_claim=False,
            native_success_changed=False,no_expert_fallback=True,future_information_used=False)


def observe_transition(trigger,tick,transition,index=0):
    """Shared live singleton-lane and HDF5 audit reader; extra truth is trigger-only."""
    def value(path):
        current=transition
        for key in path.split('/'):current=current[key]
        current=current[index]
        if hasattr(current,'detach'):current=current.detach().cpu().numpy()
        return current
    extra={}
    if getattr(trigger,'configuration',{}).get('version')==HELD_NEAR_GOAL_VERSION:
        extra=dict(keypoint_max_distance_m=value('native_keypoints_max_dist'),
            goal_index_before=value('diagnostics_before/successes'),goal_index_after=value('diagnostics_after/successes'),
            goal_pose_before=value('diagnostics_before/goal_pose_w'),goal_pose_after=value('diagnostics_after/goal_pose_w'))
    return trigger.observe(tick,value('diagnostics_after/object_pose_w'),value('diagnostics_after/palm_body_pose_w'),
        value('object_contact_force_w'),**extra)


def make_trigger(configuration, fallback_tick):
    validated = validate_trigger(configuration, fallback_tick)
    cls = {VERSION:StableGraspTrigger,CONTACT_HEIGHT_VERSION:ContactHeightTrigger,
        HELD_NEAR_GOAL_VERSION:HeldNearGoalTrigger}[validated['version']]
    return cls(validated, fallback_tick)
