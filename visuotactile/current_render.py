"""Reuse only a witnessed render at the current physical step during image read.

Native stepping and reset still render normally. This removes the extra render
inside read_student_camera_image, not the camera update or a sensor observation.
"""
from contextlib import contextmanager
import time

PROFILE='reuse_native_current_render_v1'


class CurrentRenderReuse:
    def __init__(self,env):
        if env.cfg.sim.render_interval!=env.cfg.decimation:
            raise ValueError('Current-render reuse requires native control-rate rendering')
        if getattr(env,'current_render_reuse',None) is not None:raise ValueError('Already installed')
        self.env=env;self.original=env.sim.render;self.reading=False;self.stamp=None
        self.stats=dict(render_calls=0,render_seconds=0.,redundant_reader_renders_avoided=0,
            reader_renders_required=0,invalidations=0)
        env.sim.render=self.render;env.current_render_reuse=self

    def invalidate(self):
        self.stamp=None;self.stats['invalidations']+=1

    def render(self,*args,**kwargs):
        current=int(self.env.sim.current_time_step_index)
        if self.reading and self.stamp==current:
            self.stats['redundant_reader_renders_avoided']+=1;return None
        if self.reading:self.stats['reader_renders_required']+=1
        start=time.perf_counter();result=self.original(*args,**kwargs)
        self.stats['render_seconds']+=time.perf_counter()-start;self.stats['render_calls']+=1
        self.stamp=int(self.env.sim.current_time_step_index)
        return result

    @contextmanager
    def observation(self):
        if self.reading:raise RuntimeError('Nested image read')
        self.reading=True
        try:yield
        finally:self.reading=False

    def close(self):
        self.env.sim.render=self.original;self.env.current_render_reuse=None
