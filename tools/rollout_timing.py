"""Opt-in host timing only: no additional physics step, tensor op or RNG draw."""
import time
from tools.runtime import write


class Profile:
    def __init__(self,output):
        self.output=output;self.started=time.time();self.rows={};self.lease=None

    def call(self,name,fn,*args,subtract_wait=False,**kwargs):
        lease=self.lease
        wait=(lease.wait_s+lease.pacing_s) if lease and subtract_wait else 0
        start=time.perf_counter()
        try:return fn(*args,**kwargs)
        finally:
            elapsed=time.perf_counter()-start
            if lease and subtract_wait:elapsed-=lease.wait_s+lease.pacing_s-wait
            row=self.rows.setdefault(name,dict(calls=0,wall_s=0.,max_s=0.))
            row['calls']+=1;row['wall_s']+=elapsed;row['max_s']=max(row['max_s'],elapsed)

    def patch(self,base):
        old_lease=base.lease_for
        def lease(*a,**kw):self.lease=old_lease(*a,**kw);return self.lease
        base.lease_for=lease
        def wrap(obj,name,label,subtract=False):
            original=getattr(obj,name)
            def measured(*a,**kw):return self.call(label,original,*a,subtract_wait=subtract,**kw)
            setattr(obj,name,measured)
        wrap(base,'make_env','scene_construction',True)
        wrap(base,'prepare','reset_and_initial_state',True)
        wrap(base.IsolatedEnvironment,'step','physics_render_sensors')
        wrap(base,'append_numeric','numeric_recording')
        wrap(base.RolloutVideo,'append','video_encode')
        wrap(base.RolloutVideo,'close','video_flush')
        original=base.create_expert
        def expert(*a,**kw):
            player,config=self.call('expert_load',original,*a,**kw)
            wrap(player,'get_action','rl_inference_cpu')
            return player,config
        base.create_expert=expert
        original_exit=base.exit_completed_kit_worker
        def exit_worker(*a,**kw):self.dump();return original_exit(*a,**kw)
        base.exit_completed_kit_worker=exit_worker

    def dump(self):
        lease=self.lease
        write(self.output/'wall_profile.json',dict(started=self.started,finished=time.time(),categories=self.rows,
            resource_wait_s=lease.wait_s if lease else None,resource_pacing_s=lease.pacing_s if lease else None,
            scope='Host elapsed; physics category includes rendering/sensors. Scene/reset exclude lease waits. RL runs on CPU. No extra synchronization or simulation calls. Fsync/reaping follows this timestamp.',
            changes_physics_or_rng=False))
