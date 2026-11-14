"""Real formal recorder/bridge with fake physics: retirement order and cleanup."""
import json
from types import SimpleNamespace

import pytest

from visuotactile.inactive_render_products import PROFILE


class Guard:
    def __init__(self,env,output):
        self.num_envs=env.num_envs;self.gate_limit_ticks=None;self.env=env;self.output=output
        self.events=[];self.active=[True]*self.num_envs;self.closed=False;self.callback_error=None
    def snapshot(self):return dict(profile=PROFILE,active=self.active.copy(),closed=self.closed,
        callback_error=self.callback_error,products=[dict(enabled=self.closed or self.active[i]) for i in range(self.num_envs)])
    def begin_wave(self,wave):
        if wave:assert not any(self.active)
        self.events.append(('begin',wave));self.active=[True]*self.num_envs
    def observe(self,packet,*,control_tick):
        self.events.append(('observe',control_tick))
        return packet,dict(active_lanes=[i for i,a in enumerate(self.active) if a],control_tick=control_tick)
    def mask_inactive(self,packet):return packet
    def retire(self,lanes,*,reason,control_tick):
        if reason=='natural_termination':
            for lane in lanes:
                assert (self.output/f'case_{lane}'/'execution_complete.json').exists()
        for lane in lanes:assert self.active[lane];self.active[lane]=False
        self.events.append((reason,control_tick,list(lanes)))
    def close(self):self.closed=True


def test_formal_retirement_follows_copied_terminal_and_observes_first_frame_once(tmp_path,monkeypatch):
    import visuotactile.batched_rollout as rollout
    from test_batched_rollout import test_actual_wave_stops_each_lane_once_preserves_history_and_masks_delayed_labels as original_test
    run=rollout.run_wave;seen=[]
    def wrapped(env,player,lease,output,cases,pair,saved,dino,plan,wave):
        guard=Guard(env,output);seen.append(guard)
        reset=env.reset
        def reset_checked(*args,**kwargs):
            assert guard.events==[('begin',0)]
            return reset(*args,**kwargs)
        env.reset=reset_checked
        plan['batch']['inactive_render_profile']=PROFILE
        plan['execution_backend']=dict(version='tight_execution_backend_v1',physics_device='cpu',camera_backend='standard',
            kit_tasking_threads=2,physics_threads=2)
        return run(env,player,lease,output,cases,pair,saved,dino,plan,wave,render_guard=guard)
    monkeypatch.setattr(rollout,'run_wave',wrapped)
    original_test(tmp_path,monkeypatch,None,0,False)
    guard=seen[0]
    assert [event[1] for event in guard.events if event[0]=='observe']==list(range(8))
    assert ('unassigned',0,[2]) in guard.events
    assert ('natural_termination',2,[0]) in guard.events
    assert ('natural_termination',7,[1]) in guard.events
    assert not any(guard.active)
    assert (tmp_path/'wave_0000_inactive_render_initial.json').exists()


def test_all_illegal_wave_retires_all_products_without_actions(tmp_path,monkeypatch):
    import visuotactile.batched_rollout as rollout
    import visuotactile.live_collision as collision
    from test_batched_rollout import test_actual_wave_stops_each_lane_once_preserves_history_and_masks_delayed_labels as original_test
    run=rollout.run_wave;seen=[]
    class Complete(Exception):pass
    def wrapped(env,player,lease,output,cases,pair,saved,dino,plan,wave):
        guard=Guard(env,output);seen.append(guard)
        monkeypatch.setattr(collision,'clearance',lambda *args,**kwargs:dict(legal=False))
        plan['batch']['inactive_render_profile']=PROFILE
        plan['execution_backend']=dict(version='tight_execution_backend_v1',physics_device='cpu',camera_backend='standard',
            kit_tasking_threads=2,physics_threads=2)
        results,profile=run(env,player,lease,output,cases,pair,saved,dino,plan,wave,render_guard=guard)
        assert all(result['not_started'] for _,result in results)
        assert profile['control_ticks']==0 and env.research_step==0
        guard.begin_wave(1)
        raise Complete()
    monkeypatch.setattr(rollout,'run_wave',wrapped)
    with pytest.raises(Complete):original_test(tmp_path,monkeypatch,None,0,False)
    assert seen[0].events==[('begin',0),('observe',0),('initial_illegal',0,[0,1]),('unassigned',0,[2]),('begin',1)]


@pytest.mark.parametrize('fault',['none','body','close','callback'])
def test_managed_guard_restores_and_persists_failure_before_success(tmp_path,monkeypatch,fault):
    from tools.tight_fresh_batch_run import inactive_render_scope
    from visuotactile.inactive_render_products import InactiveRenderProducts
    guard=Guard(SimpleNamespace(num_envs=2),tmp_path)
    if fault=='callback':guard.callback_error='callback failed'
    close=guard.close
    def cleanup():
        close()
        if fault=='close':raise RuntimeError('restore failed')
    guard.close=cleanup
    monkeypatch.setattr(InactiveRenderProducts,'bind',lambda *args,**kwargs:guard)
    def invoke():
        with inactive_render_scope(None,PROFILE,tmp_path) as actual:
            assert actual is guard
            if fault=='body':raise RuntimeError('native failure')
    if fault=='none':invoke()
    else:
        with pytest.raises(RuntimeError):invoke()
    assert guard.closed
    receipt=json.loads((tmp_path/'inactive_render_cleanup.json').read_text())
    assert receipt['after']['closed'] is True
    assert (receipt['primary_error'] is not None)==(fault=='body')
    assert (receipt['cleanup_error'] is not None)==(fault in ('close','callback'))


def test_default_scope_binds_nothing_and_writes_nothing(tmp_path,monkeypatch):
    from tools.tight_fresh_batch_run import inactive_render_scope
    from visuotactile.inactive_render_products import InactiveRenderProducts
    monkeypatch.setattr(InactiveRenderProducts,'bind',lambda *args,**kwargs:pytest.fail('Default must not subscribe or bind'))
    with inactive_render_scope(None,None,tmp_path) as guard:assert guard is None
    assert not list(tmp_path.iterdir())
