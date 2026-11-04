"""Opt-in post-episode shutdown; all active rollout/control code stays unchanged."""
import argparse
import faulthandler
from pathlib import Path
import time

from tools.state_image_kit_gate import write


def disarm_stop_render_loop(sim):
    """Only call after the final rollout/writers finish, immediately before close.

    This installed Lab's STOP subscriber loops render() until the timeline plays
    again. Shutdown never resumes it. Unsubscribe the same handle that Lab's
    clear_instance normally clears, but before closing scene/sensors.
    """
    if not hasattr(sim, '_disable_app_control_on_stop_handle'):
        raise RuntimeError('Unrecognized Lab lifecycle; no guessed patch')
    previous=sim._disable_app_control_on_stop_handle
    sim._disable_app_control_on_stop_handle=True
    handle=sim._app_control_on_stop_handle
    if handle is not None:
        handle.unsubscribe()
        sim._app_control_on_stop_handle=None
    return dict(previous_disable=previous,subscriber_present=handle is not None,
        active_rollout_changed=False,scope='post-episode SDK lifecycle only')


def run(root, output, mode, job):
    from visuotactile import state_benchmark as base
    original=base.make_env
    # Open outside output: base.run deliberately requires a new attempt directory.
    with output.with_suffix('.shutdown_stacks.txt').open('x') as diagnostic:
        def make_env(*args,**kwargs):
            env=original(*args,**kwargs);close=env.close
            def closing():
                write(output/'shutdown_phase.json',dict(phase='enter_env_close',utc=time.time()))
                faulthandler.dump_traceback_later(10,repeat=True,file=diagnostic)
                evidence=disarm_stop_render_loop(env.sim)
                write(output/'shutdown_policy.json',dict(**evidence,utc=time.time()))
                close()
                write(output/'shutdown_phase.json',dict(phase='env_closed_app_close_next',utc=time.time()))
            env.close=closing
            return env
        base.make_env=make_env
        try:
            base.run(root,output,mode,job)
            write(output/'shutdown_phase.json',dict(phase='sdk_close_returned',utc=time.time()))
        finally:
            faulthandler.cancel_dump_traceback_later()
            base.make_env=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['expert','dp'],required=True)
    p.add_argument('--job',type=Path,required=True);a=p.parse_args();run(a.root,a.output,a.mode,a.job)
