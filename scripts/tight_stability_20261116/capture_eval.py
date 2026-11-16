"""Ordinary full-policy evaluation with read-only rear-input capture.

No shadow policy controls the simulator. Inputs come from the active DP's real
60 Hz RGB-D/history encoder, not stored hidden states from another checkpoint.
"""
import argparse,os,time
from pathlib import Path
import torch
from scripts.tight_stability_20261116.analyze_existing import read
from visuotactile.training_resources import atomic_json

def cpu_tree(x):
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    return x.detach().cpu().clone()

def slice_tree(x,i):
    if isinstance(x,dict):return {k:slice_tree(v,i) for k,v in x.items()}
    return x[i]

def cat_tree(rows):
    if isinstance(rows[0],dict):return {k:cat_tree([r[k] for r in rows]) for k in rows[0]}
    return torch.cat(rows,dim=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    plan=read(a.output/'plan.json')
    assert a.root.resolve()==Path(plan['resource_root']).resolve()
    assert len(plan['cases'])==plan['batch']['lanes']==16, 'This capture identity has one registered case per lane, no reuse'
    from visuotactile.gpu_batch_controller import GPUBatchedResidualController
    from tools.tight_fresh_batch_run import run
    original=GPUBatchedResidualController._act_neural_batch
    captured={};started=time.time();capture_seconds=0.
    def observed(self,packet,diagnostics,ids,commands):
        nonlocal capture_seconds
        rows=[]
        def hook(module,args):
            before=time.perf_counter();rows.append(cpu_tree(args[0]))
            nonlocal capture_seconds
            capture_seconds+=time.perf_counter()-before
        handle=self.lanes[0].residual.register_forward_pre_hook(hook)
        try: result=original(self,packet,diagnostics,ids,commands)
        finally:handle.remove()
        assert len(rows)==1
        for local,lane in enumerate(ids):
            c=self.lanes[lane];tick=int(packet['step_id'][lane]);values=captured.setdefault(lane,[])
            assert tick==len(values),'Missing, duplicated or reassigned capture tick'
            before=time.perf_counter()
            values.append(dict(inputs=slice_tree(rows[0],local),base=cpu_tree(c.nominal),
                executed=cpu_tree(c.command),limits=cpu_tree(c.base.limits),raw_expected_delta=cpu_tree(c.delta)))
            capture_seconds+=time.perf_counter()-before
        if sum(map(len,captured.values()))%128<16:
            atomic_json(a.output/'input_capture_heartbeat.json',dict(pid=os.getpid(),utc=time.time(),
                recorded_controls=sum(map(len,captured.values())),elapsed=time.time()-started))
        return result
    GPUBatchedResidualController._act_neural_batch=observed
    completed=False
    try:
        run(a.output);completed=True
    finally:
        GPUBatchedResidualController._act_neural_batch=original
        receipt=[]
        for lane,rows in captured.items():
            directory=a.output/plan['cases'][lane]['name'];output=directory/'rear_inputs.pt'
            assert not output.exists()
            data=dict(inputs=cat_tree([r['inputs'] for r in rows]),base=cat_tree([r['base'] for r in rows]),
                executed=cat_tree([r['executed'] for r in rows]),limits=rows[0]['limits'],
                delta=cat_tree([r['raw_expected_delta'] for r in rows]),lane=lane,
                checkpoint=plan['pairs'][plan['cases'][lane]['role']],complete=completed,
                version='read_only_full_rear_input_capture_v1')
            temp=output.with_suffix('.partial')
            with temp.open('xb') as stream:torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,output)
            receipt.append(dict(case=directory.name,controls=len(rows),bytes=output.stat().st_size))
        atomic_json(a.output/'input_capture_complete.json',dict(complete=completed,rows=receipt,
            capture_cpu_copy_seconds=capture_seconds,seconds=time.time()-started,
            scope='Read-only hook; all full-policy commands unchanged; independent current-weight memory must be recomputed during replay. Partial data never marked complete.'))

if __name__=='__main__':main()
