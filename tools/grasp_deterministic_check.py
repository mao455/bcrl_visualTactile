"""One-anchor final-EMA reproducibility check in a fresh process, without Kit."""
import argparse
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import numpy as np

from tools.state_image_queue import read,write
from tools.grasp_diagnosis_analysis import BASE
from visuotactile.state_history import PersistentState,PersistentImage
from visuotactile.state_resources import lease_for


def run(root,output):
    session=read(root/'session.json');old=read(BASE/'state_image_20261104_02/session.json')
    identity=session['specs'][0]['identity'];cache=torch.load(old['numeric_cache'],map_location='cpu',weights_only=False)
    rec=cache['records'][identity];del cache
    lease=lease_for(root,deadline=session['deadline']);result={}
    for name in ('StateDP','ImageDP'):
        saved=torch.load(session['checkpoints'][name]['path'],map_location='cpu',weights_only=False)
        with lease.block():
            model=(PersistentState(saved['kind'],saved['contract']) if name=='StateDP' else PersistentImage(saved['group'],saved['contract'])).cuda().eval()
            model.load_state_dict(saved['ema']);del saved
            numeric={k:v.cuda() for k,v in rec.items() if torch.is_tensor(v)}
            if name=='ImageDP':
                from visuotactile.models.native_vision import NativeDinoSpatial
                from visuotactile.native_data import Episode
                dino=NativeDinoSpatial().cuda().eval();raw=Episode(session['source_root'],identity,'I-OfficialDP')
                patches=dino(raw.frame(0,'cuda:0')['rgb']);result['DINO']=patches.cpu().numpy();del dino
                numeric={k:numeric[k] for k in ('robot','command','time','joint_limits')}
                numeric.update(identity=identity,source_root=session['source_root'])
            condition,_,_=model.encode(numeric,0,1)
            plan=model.predict(condition,torch.Generator(device='cuda').manual_seed(7091))
            result[name+'_condition']=condition.cpu().numpy();result[name+'_plan']=plan.cpu().numpy()
        del model,numeric;torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.no_grad():run(a.root,a.output)
