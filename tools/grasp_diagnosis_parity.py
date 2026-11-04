"""Final EMA checkpoints: raw recorded observation -> actual deployment controller.

Teacher feedback is explicitly used only in this offline numerical audit. No
simulation, action execution, optimizer, or expert recurrent-state import occurs.
"""
import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.state_image_queue import read, write, sha
from tools.grasp_diagnosis_analysis import BASE, PUBLIC
from visuotactile.state_history import PersistentState, PersistentImage, HistoryController
from visuotactile.native_controller import NativeController
from visuotactile.native_data import Episode
from visuotactile.models.native_vision import NativeDinoSpatial
from visuotactile.state_resources import lease_for
from visuotactile.state_history_training import labels


def error(a,b):
    a,b=torch.as_tensor(a).cpu(),torch.as_tensor(b).cpu()
    assert a.shape==b.shape,(a.shape,b.shape)
    return float((a.double()-b.double()).abs().max()) if a.numel() else 0.


def audit(root,precision=False):
    if precision:
        from tools.grasp_diagnosis_precision import install
        install()
    session=read(root/'session.json');old=read(BASE/'state_image_20261104_02/session.json')
    source=Path(session['source_root']);data=read(source/'data.json');lease=lease_for(root,deadline=session['deadline'])
    ids=[session['specs'][i]['identity'] for i in (0,4)]
    cache=torch.load(old['numeric_cache'],map_location='cpu',weights_only=False)
    records={i:cache['records'][i] for i in ids};del cache
    stats=torch.load(old['position_statistics'],map_location='cpu',weights_only=False)
    rows=[]
    for name in ('StateDP','ImageDP'):
        cp=session['checkpoints'][name]
        assert sha(cp['path'])==cp['sha256']
        saved=torch.load(cp['path'],map_location='cpu',weights_only=False)
        assert saved['updates']==10000
        # Remove optimizer/model duplicates; all evaluated parameters are immutable EMA.
        saved={k:v for k,v in saved.items() if k in ('ema','group','kind','contract','updates')}
        with lease.block():
            model=(PersistentState(saved['kind'],saved['contract']) if name=='StateDP' else
                PersistentImage(saved['group'],saved['contract'])).cuda().eval()
            model.load_state_dict(saved['ema'])
            dino=NativeDinoSpatial().cuda().eval() if name=='ImageDP' else None
        stat_errors={k:error(getattr(model,k),v) for k,v in stats.items()}
        assert max(stat_errors.values())==0.
        for identity in ids:
            entry=data['entries'][identity];rec=records[identity]
            with h5py.File(entry['path']) as f:
                g=f['episodes'][entry['episode']];metadata=json.loads(f.attrs['metadata_json'])
                obs={k:torch.from_numpy(v[:]) for k,v in g['obs'].items()}
                before={k:torch.from_numpy(v[:]) for k,v in g['transition/diagnostics_before'].items()}
                actor=torch.from_numpy(g['transition/expert_actor_input_diagnostic'][:])
                target=torch.from_numpy(g['transition/target_after_rad'][:])
                timestamps=torch.from_numpy(g['transition/time_before_s'][:])
            n=len(target);numeric={k:v.cuda() for k,v in rec.items() if torch.is_tensor(v)}
            offline_states=[]
            with torch.no_grad(),lease.block():
                if name=='StateDP':
                    conditions,anchors,_=model.encode(numeric)
                    features,_=model.native_features(numeric['actor'],numeric['robot'],numeric['geometry'])
                    all_hidden=model.history(features[None])[0][0].cpu()
                else:
                    numeric={k:numeric[k] for k in ('robot','command','time','joint_limits')}
                    numeric.update(identity=identity,source_root=str(source))
                    original=model.observe
                    def observe(*a,**kw):
                        memory=original(*a,**kw)
                        offline_states.append((memory.native.detach().cpu().clone(),memory.visual.detach().cpu().clone()))
                        return memory
                    model.observe=observe
                    conditions,anchors,_=model.encode(numeric)
                    model.observe=original
                conditions=conditions.cpu();anchors=anchors.cpu()
                y,valid=labels(rec,anchors,24)
                label_error=error(y[:,0],target[anchors])
                normalized=target*model.action_scale.cpu()+model.action_offset.cpu()
                roundtrip=error((normalized-model.action_offset.cpu())/model.action_scale.cpu(),target)
                controller=(HistoryController(saved,metadata,inference={'inference_steps':16,'clip_sample':False,'sampler_seed':7091})
                    if name=='StateDP' else NativeController(saved,metadata,dino=dino))
                controller.generator.manual_seed(7091)
            captured={};patch_rows=[]
            if name=='ImageDP':
                original=controller.model.observe
                def observe_live(robot,stamp,tick,frame,memory=None):
                    captured['robot']=robot.detach().cpu();captured['frame']=frame
                    return original(robot,stamp,tick,frame,memory)
                controller.model.observe=observe_live
                cached=Episode(source,identity,'I-HistoryDP');raw=Episode(source,identity,'I-OfficialDP')
            generator=torch.Generator(device='cuda').manual_seed(7091)
            max_errors=dict(robot=0.,geometry=0.,actor=0.,normalized_robot=0.,hidden=0.,condition=0.,
                raw_plan_rad=0.,bounded_plan_rad=0.,action_inverse_roundtrip_rad=roundtrip,label_index=label_error,
                timestamp=error(timestamps,rec['time']),camera_timestamp=0.,depth=0.,dino=0.,rgb_roundtrip=0.)
            per_tick=[];offline_plans=[];live_plans=[]
            for tick in range(n):
                packet={k:v[tick:tick+1] for k,v in obs.items()}
                with torch.no_grad(),lease.block():
                    if name=='ImageDP' and tick%12==0:
                        frame=raw.frame(tick,'cpu')
                        rgb=frame['rgb'].float()/255.
                        packet['image']=torch.cat((rgb,frame['depth'].float()),1)[None]
                        packet['depth_valid']=frame['depth_valid'][None]
                        max_errors['rgb_roundtrip']=max(max_errors['rgb_roundtrip'],error((rgb*255).round().to(torch.uint8),frame['rgb']))
                    if name=='StateDP':
                        action=controller.act(actor[tick:tick+1],packet,{k:v[tick:tick+1] for k,v in before.items()})
                        inputs=controller.last_inputs
                        max_errors['actor']=max(max_errors['actor'],error(inputs['actor'],rec['actor'][tick]))
                        max_errors['geometry']=max(max_errors['geometry'],error(inputs['geometry'],rec['geometry'][tick]))
                        observed_robot=inputs['robot'];hidden_error=error(controller.bank.lanes[0]['hidden'][0,0],all_hidden[tick])
                    else:
                        action=controller.act(packet);observed_robot=captured['robot']
                        hidden_error=max(error(controller.memory.native,offline_states[tick][0]),
                            error(controller.memory.visual,offline_states[tick][1]))
                        if tick%12==0:
                            expected=cached.frame(tick,'cuda:0');actual=captured['frame']
                            patch_error=error(actual['patches'],expected['patches']);depth_error=error(actual['depth'],expected['depth'])
                            time_error=error(actual['time'],expected['time'])
                            assert torch.equal(actual['valid'],expected['valid'])
                            patch_rows.append(dict(tick=tick,patch_max_abs=patch_error,depth_max_abs=depth_error,time_max_abs=time_error))
                            for key,v in [('dino',patch_error),('depth',depth_error),('camera_timestamp',time_error)]:max_errors[key]=max(max_errors[key],v)
                    robot_error=error(observed_robot,rec['robot'][tick])
                    max_errors['robot']=max(max_errors['robot'],robot_error)
                    max_errors['normalized_robot']=max(max_errors['normalized_robot'],error(
                        controller.model.normalized_robot(observed_robot.cuda()),model.normalized_robot(rec['robot'][tick].cuda())))
                    max_errors['hidden']=max(max_errors['hidden'],hidden_error)
                    line=dict(tick=tick,robot_max_abs=robot_error,hidden_max_abs=hidden_error)
                    if tick%12==0:
                        index=tick//12;condition_error=error(controller.last_condition,conditions[index:index+1])
                        pred=model.predict(conditions[index:index+1].cuda(),generator).cpu()
                        actual_plan=controller.raw_plan if name=='StateDP' else controller.plan
                        plan_error=error(actual_plan,pred);max_errors['raw_plan_rad']=max(max_errors['raw_plan_rad'],plan_error)
                        from visuotactile.position_control import bounded_targets
                        bounded=bounded_targets(pred[:,0],controller.limits)
                        max_errors['bounded_plan_rad']=max(max_errors['bounded_plan_rad'],error(action,bounded))
                        max_errors['condition']=max(max_errors['condition'],condition_error)
                        line.update(condition_max_abs=condition_error,plan_max_abs_rad=plan_error)
                        offline_plans.append(pred[0].numpy());live_plans.append(actual_plan[0].numpy())
                    # Teacher feedback for the numerical parity test only, never simulated.
                    controller.command=target[tick:tick+1].clone()
                    controller.observe_execution({'position_applied_rad':target[tick:tick+1].clone()})
                per_tick.append(line)
                if tick%24==0:write(root/'parity_progress.json',dict(model=name,identity=identity,tick=tick,ticks=n,completed=len(rows)))
            strict=('robot','geometry','actor','normalized_robot','hidden','condition','label_index','timestamp','camera_timestamp','depth','rgb_roundtrip')
            passed=(all(max_errors[k]<=2e-5 for k in strict) and max_errors['dino']<=.005 and
                max_errors['raw_plan_rad']<=1e-4 and max_errors['bounded_plan_rad']<=1e-4)
            folder=root/('parity_precision' if precision else 'parity')/name;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/(identity+'.npz'),anchors=anchors.numpy(),offline=np.stack(offline_plans),
                online=np.stack(live_plans),truth=y.numpy(),valid=valid.numpy())
            write(folder/(identity+'.json'),dict(per_tick=per_tick,vision=patch_rows))
            rows.append(dict(model=name,identity=identity,ticks=n,anchors=len(anchors),passed=passed,max_errors=max_errors,
                checkpoint_sha256=cp['sha256'],statistics_max_errors=stat_errors,
                evidence=str(folder/(identity+'.json')),vision=patch_rows))
            write(root/'parity_partial.json',dict(rows=rows))
            del controller,numeric;gc.collect();torch.cuda.empty_cache()
        del model,dino,saved;gc.collect();torch.cuda.empty_cache()
    result=dict(passed=all(r['passed'] for r in rows),rows=rows,formal_updates=0,precision_reconstruction=precision,
        scope='Full expert trajectories: raw stored obs and frames through actual deployment controllers vs frozen training encode; same EMA and sequential DDIM generator.',
        teacher_forced=True,task_success_evidence=False,
        tolerances=dict(input_hidden_condition=2e-5,raw_and_bounded_plan_rad=1e-4,dino=.005),
        no_kit_loaded=not any(k.split('.')[0] in ('omni','carb','isaacsim','isaaclab') for k in __import__('sys').modules))
    filename='parity_precision.json' if precision else 'parity.json'
    write(root/filename,result);write(PUBLIC/filename,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--precision',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.no_grad():audit(a.root,a.precision)
