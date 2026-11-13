"""Batch only independent visual work; preserve every native history update.

All operations retain the established NativePolicy parameterization. The batch
axis is frame/view, never an attention axis mixing future observed frames.
"""
import torch
from torch.nn import functional as F
from .native_data import Episode


def read_batch(vision,tokens,ages,valid):
    """B independent reads over S views/frames, with their actual timestamps."""
    b=tokens.shape[0]
    values=tokens+vision.age(ages[...,None].float())[...,None,:]
    values=values.reshape(b,-1,128)
    padding=~valid[...,None].expand(*valid.shape,48).reshape(b,-1)
    if padding.all(-1).any():raise ValueError('No valid visual observations')
    result,_=vision.attention(vision.queries[None].expand(b,-1,-1),values,values,
        key_padding_mask=padding,need_weights=False)
    return result.flatten(1)


def frame_features(model,patches,depth,ages,valid):
    n=len(patches);v=model.vision
    rgb=v.rgb(patches.reshape(-1,384,18,32).float())
    dep=v.depth(depth.reshape(-1,2,18,32).float())
    tokens=F.adaptive_avg_pool2d(torch.cat((rgb,dep),1),(6,8)).flatten(-2).transpose(1,2)
    tokens=tokens.reshape(n,3,48,128)+v.position(v.coordinates)[None,None]+v.camera.weight[None,:,None]
    return tokens,read_batch(v,tokens,ages,valid)


def encode_dense(model,record,*,episode=None,start=0,stop=None):
    if model.contract.get('visual_stride')!=1 or model.modality!='G00':
        raise ValueError('Explicit every-tick G00 visual contract required')
    stop=len(record['time']) if stop is None else stop
    if not 0<=start<stop<=len(record['time']):raise ValueError('Invalid full-prefix window')
    episode=Episode(record['source_root'],record['identity'],model.group) if episode is None else episode
    if episode.visual_stride!=1:raise ValueError('Sparse cache cannot feed continuous model')
    device=next(model.parameters()).device
    robot=record['robot'][:stop].to(device);time=record['time'][:stop].to(device)
    ct=episode.record['camera_time'][:stop].to(device);cv=episode.record['camera_valid'][:stop].to(device)
    if ct.shape!=(stop,3) or not cv.all() or (ct>time[:,None]+1e-6).any():
        raise ValueError('Missing/future native visual frame')
    memory_native=memory_visual=None;tokens=[];native=[];visual=[]
    # Split at TBPTT boundary so the complete real prefix uses current weights,
    # but does not retain gradients. No learned state cache survives an update.
    for lo,hi in [(0,start),(start,stop)]:
        if lo==hi:continue
        with torch.set_grad_enabled(model.training and lo>=start):
            part=[];summaries=[]
            for a in range(lo,hi,32):
                b=min(a+32,hi)
                t,s=frame_features(model,episode.perception['patches'][a:b].to(device),
                    episode.perception['depth'][a:b].to(device),time[a:b,None]-ct[a:b],cv[a:b])
                part.append(t);summaries.append(s)
            part=torch.cat(part);summaries=torch.cat(summaries)
            inputs=model.native_input(model.normalized_robot(robot[lo:hi]))
            for j in range(hi-lo):
                memory_native=model.native_gru(inputs[j:j+1],memory_native)
                memory_visual=model.visual_gru(summaries[j:j+1],memory_visual)
                native.append(memory_native);visual.append(memory_visual)
            tokens.append(part)
    tokens=torch.cat(tokens);native=torch.cat(native);visual=torch.cat(visual)
    anchors=torch.arange(start,stop,device=device);anchors=anchors[anchors%12==0]
    if not len(anchors):raise ValueError('No planning anchors')
    rindex=anchors[:,None]+torch.arange(-12,1,device=device)
    rvalid=rindex>=0;ri=rindex.clamp_min(0)
    nr=model.normalized_robot(robot)
    rows=torch.cat((nr[ri],(time[anchors,None]-time[ri]).float()[...,None],rvalid.float()[...,None]),-1)
    rows=torch.where(rvalid[...,None],rows,0.).flatten(1)
    findex=anchors[:,None]+torch.arange(-2,1,device=device);fi=findex.clamp_min(0)
    valid=cv[fi]&(findex>=0)[...,None]
    readout=read_batch(model.vision,tokens[fi],time[anchors,None,None]-ct[fi],valid)
    condition=model.condition_projection(torch.cat((model.robot_encoder(rows),readout,native[anchors],visual[anchors]),-1))
    return condition,anchors
