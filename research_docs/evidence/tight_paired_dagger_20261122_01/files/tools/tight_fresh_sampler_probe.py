"""Same real condition/noise: measure only DDIM scheduler index transport."""
import argparse,json,time
from pathlib import Path
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.state_resources import lease_for
from visuotactile.precision import state_fp32
from visuotactile.models.native_dp import load_policy
from visuotactile.native_data import visual_records
from visuotactile.continuous_vision import encode_dense


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.output.exists():raise FileExistsError('Preserve measurement')
    session=json.loads((root/'session.json').read_text());state_fp32()
    lease=lease_for(root,deadline=session['deadline']);rows=[]
    with lease.block():
        for role in ['A','B']:
            path=root/role/'training/best.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
            model=load_policy(saved,'cuda:0').eval();dr=root/role/'state_data';data=json.loads((dr/'data.json').read_text())
            identity=data['splits']['train'][0];record=visual_records(dr,[identity],'G00',device='cuda:0')[identity]
            with torch.no_grad():condition,_=encode_dense(model,record,stop=25)
            for repeat in range(4):
                results={};seconds={}
                for mode in (['legacy','host'] if repeat%2==0 else ['host','legacy']):
                    generator=torch.Generator(device='cuda').manual_seed(7093+repeat)
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.no_grad():results[mode]=model.predict(condition,generator,scheduler_index_mode=mode)
                    torch.cuda.synchronize();seconds[mode]=time.perf_counter()-start
                error=float((results['legacy']-results['host']).abs().max())
                if error:raise ValueError('Changed native plan: '+str(error))
                rows.append(dict(role=role,repeat=repeat,seconds=seconds,max_abs_rad=error,checkpoint_sha256=sha256(path)))
    atomic_json(a.output,dict(passed=True,rows=rows,scope='Real three conditions, paired seeds; exact plans. Separate warmup and no autonomous throughput claim'))

if __name__=='__main__':main()
