"""Isolated arithmetic/throughput probe, never alters the running experiment."""
import copy,json,time
from pathlib import Path
import torch

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.manual_seed(8721)

def serial(cell,x):
    h=None;values=[]
    for row in x:
        h=cell(row,h);values.append(h)
    return torch.stack(values)

def fused(cell,x):
    h=x.new_zeros((1,x.shape[1],cell.hidden_size))
    return torch._VF.gru(x,h,[cell.weight_ih,cell.weight_hh,cell.bias_ih,cell.bias_hh],True,1,0.,cell.training,False,False)[0]

rows=[]
for device,dtype,length in [('cpu',torch.float64,17),('cuda',torch.float32,300)]:
    for inputs in [256,512]:
        a=torch.nn.GRUCell(inputs,256).to(device=device,dtype=dtype)
        b=copy.deepcopy(a)
        x=torch.randn(length,2,inputs,device=device,dtype=dtype,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        times=[];outputs=[];gradients=[]
        for c,z,fun in [(a,x,serial),(b,y,fused)]:
            if device=='cuda':torch.cuda.synchronize()
            began=time.perf_counter();out=fun(c,z);out.square().mean().backward()
            if device=='cuda':torch.cuda.synchronize()
            times.append(time.perf_counter()-began);outputs.append(out.detach())
            gradients.append([p.grad.detach().clone() for p in c.parameters()]+[z.grad.detach().clone()])
        row=dict(device=device,dtype=str(dtype),length=length,input_size=inputs,
            maximum_output_error=float((outputs[0]-outputs[1]).abs().max()),
            maximum_gradient_error=max(float((u-v).abs().max()) for u,v in zip(*gradients)),
            serial_forward_backward_seconds=times[0],fused_forward_backward_seconds=times[1],
            parameters_unchanged=True,probe_only=True)
        rows.append(row);print(json.dumps(row),flush=True)
        del a,b,x,y,outputs,gradients
        if device=='cuda':torch.cuda.empty_cache()
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(rows=rows,scope='Synthetic arithmetic and speed only; no experimental optimizer update or deployed change'),indent=2))
