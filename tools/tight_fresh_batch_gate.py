"""Real RGB-D lane parity with independent reset/history boundaries, no physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256


def gate(root,output,*,role='A',count=2):
    from tools.tight_fresh_rollouts import completed_pair
    from visuotactile.native_controller import ResidualController
    from visuotactile.batched_controller import BatchedResidualController,lane_tree
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32
    if output.exists():raise FileExistsError('Preserve gate evidence')
    if role not in ('A','B') or count not in (2,4):raise ValueError('Registered A/B role and two/four lanes required')
    state_fp32();s=json.loads((root/'session.json').read_text());lease=lease_for(root,deadline=s['deadline'])
    pair=completed_pair(root,role);saved=torch.load(pair['backend'],weights_only=False,map_location='cpu')
    data=json.loads((root/role/'state_data/data.json').read_text());ids=data['splits']['train'][:count]
    sources=[]
    for identity in ids:
        e=data['entries'][identity]
        with h5py.File(e['path']) as f:
            g=f['episodes'][e['episode']];metadata=json.loads(f.attrs['metadata_json'])
            read=lambda group:{k:torch.from_numpy(v[:25]) for k,v in group.items()}
            sources.append(dict(obs=read(g['obs']),diag=read(g['transition/diagnostics_before']),
                image=torch.cat((torch.from_numpy(g['frames/rgb_u8'][:25]).float()/255,torch.from_numpy(g['frames/depth_m'][:25])),2),
                depth_valid=torch.from_numpy(g['frames/depth_valid'][:25]),
                target=torch.from_numpy(g['transition/target_after_rad'][:25])))
    source_control_mode=metadata['action_control_mode']
    # Offline controller comparison only: source expert labels were postprocessed
    # native targets. No simulator is present or being reconfigured by this gate.
    metadata['action_control_mode']='absolute_joint_position_target_rad'
    metadata['joint_limits_rad']=metadata['joint_limits_rad']*count
    rows=[]
    with lease.block():
        dino=NativeDinoSpatial().cuda().eval()
        batch=BatchedResidualController(saved,metadata,dino,count)
        serial=[ResidualController(saved,metadata,dino) for _ in ids]
    for tick in range(25):
        # Restart only lane0 at the 13th observation; lane1's prefix continues.
        indices=[tick if tick<13 else tick-13]+[tick]*(count-1)
        packets=[];diagnostics=[];targets=[]
        for lane,j in enumerate(indices):
            source=sources[lane];p={k:v[j:j+1].clone() for k,v in source['obs'].items()}
            if lane==0 and tick>=13:p['episode_id']+=100000
            p.update(image=source['image'][j:j+1],depth_valid=source['depth_valid'][j:j+1])
            packets.append(p);diagnostics.append({k:v[j:j+1] for k,v in source['diag'].items()});targets.append(source['target'][j:j+1])
        packet={k:torch.cat([p[k] for p in packets]) for k in packets[0]}
        diag={k:torch.cat([d[k] for d in diagnostics]) for k in diagnostics[0]}
        with lease.block():
            prediction=batch.act(packet,diag,[True]*count)
            reference=torch.cat([c.act(p,d) for c,p,d in zip(serial,packets,diagnostics)])
            error=float((prediction-reference).abs().max())
            if error>2e-5:raise ValueError('Real complete-policy lane parity failed: '+str(error))
            for lane in range(count):
                serial[lane].base.command=targets[lane]
                batch.lanes[lane].base.command=targets[lane]
                serial[lane].observe_execution(dict(position_applied_rad=targets[lane]))
            batch.observe_execution(dict(position_applied_rad=torch.cat(targets)),[True]*count)
        rows.append(dict(tick=tick,max_abs_command_rad=error,steps=indices))
    atomic_json(output,dict(passed=True,pair=pair,role=role,lanes=count,source_ids=ids,rows=rows,visual_frames=batch.visual_frames,
        source_control_mode=source_control_mode,offline_controller_mode=metadata['action_control_mode'],
        max_abs_command_rad=max(row['max_abs_command_rad'] for row in rows),
        scope='Real recorded RGB-D and expert actual-history acknowledgements; lane0 reset while lane1 continues; no simulation outcome or speed claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=['A','B'],default='A');p.add_argument('--lanes',type=int,choices=[2,4],default=2)
    a=p.parse_args();torch.set_num_threads(2);gate(a.root,a.output,role=a.role,count=a.lanes)
