from types import SimpleNamespace
from visuotactile.current_render import CurrentRenderReuse


def test_native_render_reuse_requires_current_tick_and_reset_invalidation():
    calls=[]
    sim=SimpleNamespace(current_time_step_index=0,render=lambda:calls.append(True))
    env=SimpleNamespace(sim=sim,cfg=SimpleNamespace(sim=SimpleNamespace(render_interval=2),decimation=2))
    original=sim.render;reuse=CurrentRenderReuse(env)
    sim.render()
    with reuse.observation():sim.render()
    assert len(calls)==1
    sim.current_time_step_index=2
    with reuse.observation():sim.render()
    assert len(calls)==2
    reuse.invalidate()
    with reuse.observation():sim.render()
    assert len(calls)==3
    sim.current_time_step_index=4;sim.render()
    with reuse.observation():sim.render()
    assert len(calls)==4
    assert reuse.stats['redundant_reader_renders_avoided']==2
    assert reuse.stats['reader_renders_required']==2
    reuse.close();assert sim.render is original and env.current_render_reuse is None
