"""Diagnostic-only causal takeover scheduling; never a student input or success rule."""
import numpy as np

VERSION = 'stable_grasp_then_delay_v1'


def validate_trigger(configuration, fallback_tick):
    if not isinstance(configuration, dict) or set(configuration) != {'version', 'delay_ticks'}:
        raise ValueError('Explicit stable-grasp trigger version and delay required')
    delay = configuration['delay_ticks']
    if configuration['version'] != VERSION or isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
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
