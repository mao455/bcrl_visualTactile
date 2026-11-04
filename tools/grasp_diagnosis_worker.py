"""One-factor K4 plan-overlap diagnostic on unchanged final EMA controllers."""
import argparse
from pathlib import Path

import numpy as np
import torch

from tools.state_image_queue import read
from visuotactile.position_control import overlap_transition


class OverlapPredictor:
    """Wrap the unchanged predictor; no extra random draws or changes to its inputs."""
    def __init__(self, predict, steps):
        if steps not in (0,4):raise ValueError('Only registered K0/K4')
        self.predict,self.steps=predict,steps
        self.previous=None;self.raw=None

    def __call__(self,*args,**kwargs):
        fresh=self.predict(*args,**kwargs)
        self.raw=fresh.detach().clone()
        result=overlap_transition(self.previous,fresh,12 if self.previous is not None else 0,self.steps)
        self.previous=result.detach().clone()
        return result


def run(root,output,job):
    from visuotactile import state_benchmark as base,native_controller
    from tools.final_dp_worker import run as evaluate
    config=read(job)
    assert config['ablation'] in ('precision_only','precision_K4')
    from tools.grasp_diagnosis_precision import install
    install()
    assert config['inference'].get('overlap_steps',0)==0,'No double blending'
    raw=[];blended=[];ticks=[]
    def instrument(cls,state):
        original_init,original_act=cls.__init__,cls.act
        def init(self,*a,**kw):
            original_init(self,*a,**kw)
            self.diagnostic_predictor=OverlapPredictor(self.model.predict,4 if config['ablation']=='precision_K4' else 0)
            self.model.predict=self.diagnostic_predictor
        def act(self,*a,**kw):
            command=original_act(self,*a,**kw)
            packet=a[1] if state else a[0];tick=int(packet['step_id'][0])
            if tick%12==0:
                raw.append(self.diagnostic_predictor.raw[0].cpu().numpy().copy())
                blended.append(self.plan[0].cpu().numpy().copy());ticks.append(tick)
            return command
        cls.__init__,cls.act=init,act
    instrument(base.HistoryController,True);instrument(native_controller.NativeController,False)
    original_trial=base.trial
    def trial(*a,**kw):
        result=original_trial(*a,**kw)
        if result.get('legal'):
            folder=output/result['identity']
            np.savez_compressed(folder/'ablation_plans.npz',raw_plans=np.stack(raw),blended_plans=np.stack(blended),ticks=np.array(ticks))
        return result
    base.trial=trial
    evaluate(root,output,job)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(a.root,a.output,a.job)
