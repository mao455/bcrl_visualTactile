"""Isolated lifecycle probes and evaluation; never changes shared SDK files."""
import argparse
import json
from pathlib import Path
import time

from visuotactile.training_resources import atomic_json


def run(root, output, job_path):
    job=json.loads(job_path.read_text());case=job.get('lifecycle',{})
    from isaaclab import app as launcher
    original_launcher=launcher.AppLauncher
    fast=case.get('fast_shutdown',True)
    def tracked_launcher(*args,**kwargs):
        kwargs['fast_shutdown']=fast
        launched=original_launcher(*args,**kwargs);close=launched.app.close
        def closing(*a,**kw):
            atomic_json(output/'shutdown_phase.json',dict(phase='app_close_entered',fast_shutdown=fast,utc=time.time()))
            close(*a,**kw)
            atomic_json(output/'shutdown_phase.json',dict(phase='app_close_returned',fast_shutdown=fast,utc=time.time()))
        launched.app.close=closing
        return launched
    launcher.AppLauncher=tracked_launcher
    from visuotactile import state_benchmark as base
    original_make=base.make_env
    def make_env(*args,**kwargs):
        env=original_make(*args,**kwargs);close=env.close
        def closing():
            atomic_json(output/'shutdown_phase.json',dict(phase='env_close_entered',utc=time.time()))
            if case.get('disarm_stop',True):
                sim=env.sim
                sim._disable_app_control_on_stop_handle=True
                handle=sim._app_control_on_stop_handle
                if handle is not None:handle.unsubscribe();sim._app_control_on_stop_handle=None
            # No timed faulthandler or other added in-process diagnostic thread.
            close()
            atomic_json(output/'shutdown_phase.json',dict(phase='env_close_returned',utc=time.time()))
        env.close=closing
        return env
    base.make_env=make_env
    # Match State's registered latent seed while retaining the original image
    # observer/action path and parameter tensors.
    from visuotactile import native_controller
    original_image=native_controller.NativeController
    class SeededImage(original_image):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.generator.manual_seed(job.get('inference',{}).get('sampler_seed',7091))
    native_controller.NativeController=SeededImage
    base.run(root,output,job.get('mode','dp'),job_path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(a.root,a.output,a.job)
