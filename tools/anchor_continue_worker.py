"""Read-only live input capture for strict ImageDP pairing; ordinary inference retained."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from tools.state_image_queue import read,write


def run(root,output,path):
    job=read(path);capture=job.get('capture_pair',False);rows={}
    from visuotactile import native_controller,state_benchmark as base
    from tools.final_dp_worker import run as evaluate
    cls=native_controller.NativeController;old_init,old_act=cls.__init__,cls.act
    def stash(key,value):
        if torch.is_tensor(value):rows[key]=value.detach().cpu().numpy().copy()
    def init(self,*args,**kwargs):
        # Same numerical settings for the Image duplicate gate and later Image evaluation.
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        old_init(self,*args,**kwargs)
        if not capture:return
        observe=self.model.observe;predict=self.model.predict
        def observed(robot,stamp,tick,frame,memory=None):
            if tick<12:
                stash(f'{tick:02d}_robot',robot);stash(f'{tick:02d}_normalized_robot',self.model.normalized_robot(robot))
                if frame is not None:
                    for k,v in frame.items():stash(f'{tick:02d}_frame_{k}',v)
            result=observe(robot,stamp,tick,frame,memory)
            if tick<12:
                stash(f'{tick:02d}_gru_native',result.native);stash(f'{tick:02d}_gru_visual',result.visual)
            return result
        def predicted(condition,generator,*a,**kw):
            first=self.next_step==0
            if first:
                stash('00_condition',condition);stash('00_rng_before',generator.get_state())
                count=[0]
                def hook(module,inputs):
                    stash(f'00_diffusion_{count[0]:02d}_input',inputs[0]);stash(f'00_diffusion_{count[0]:02d}_time',inputs[1]);count[0]+=1
                handle=self.model.denoiser.register_forward_pre_hook(hook)
            try:result=predict(condition,generator,*a,**kw)
            finally:
                if first:handle.remove()
            if first:stash('00_plan',result);stash('00_rng_after',generator.get_state())
            return result
        self.model.observe=observed;self.model.predict=predicted
    def act(self,packet,*a,**kw):
        tick=int(packet['step_id'][0])
        if capture and tick<12:
            for k in ('proprio','actuator_target_rad','tracking_error_rad','time_s','camera_time_s','camera_valid'):
                stash(f'{tick:02d}_packet_{k}',packet[k])
            if tick==0:
                stash('00_raw_image',packet['image']);stash('00_raw_depth_valid',packet['depth_valid'])
                stash('00_rgb_u8',(packet['image'][...,:3,:,:]*255).round().clamp(0,255).to(torch.uint8))
        result=old_act(self,packet,*a,**kw)
        if capture and tick<12:stash(f'{tick:02d}_command',result)
        if capture and tick==11:
            np.savez_compressed(output/'image_pair_inputs.npz',**rows)
            write(output/'image_pair_capture.json',dict(complete=True,fields=list(rows),ticks=12,read_only=True,expert_takeover=False))
        return result
    cls.__init__,cls.act=init,act
    evaluate(root,output,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;run(a.root,a.output,a.job)
