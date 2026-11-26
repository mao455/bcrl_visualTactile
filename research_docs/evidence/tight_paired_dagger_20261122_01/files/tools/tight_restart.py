"""Fresh train30 selection and reproducible every-tick visual experiment.

No old student checkpoint is read. Source demonstrations remain immutable.
"""
import argparse
import json
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    import hashlib
    import h5py
    import numpy as np
    import torch
    from visuotactile.compact import sha256
    from visuotactile.native_data import limits_affine
    from visuotactile.narrow_data import actual_configuration_labels, record
    from visuotactile.position_control import continuous_image_contract, contract
    from visuotactile.training_resources import atomic_json
    from visuotactile.checkpointing import save
    if (root/'prepared.json').exists():
        raise FileExistsError('Fresh preparation cannot overwrite an existing experiment')
    source=Path('outputs/visuotactile/run/tight_vision_20261109_01/state_data').resolve()
    data=read(source/'data.json')
    # One full success per family, picked by median duration, not shortest success.
    candidates={};initial={};selected={};audit={};metadata=None
    for split,count in [('train',30),('validation',6)]:
        families={}
        for i in data['splits'][split]:families.setdefault(data['entries'][i]['family'],[]).append(i)
        candidates[split]=[]
        for family,ids in sorted(families.items()):
            ids=sorted(ids,key=lambda i:(data['entries'][i]['length'],i));i=ids[len(ids)//2]
            entry=data['entries'][i]
            with h5py.File(entry['path']) as f:
                g=f['episodes'][entry['episode']];b=g['transition/diagnostics_before']
                q=b['joint_position_rad'][0];obj=b['object_pose_w'][0];hole=b['hole_pose_w'][0]
                final=g['transition/diagnostics_after/successes'][-1]
                if not bool(g.attrs['complete']) or not bool(np.asarray(final).max()>0):
                    raise ValueError('Source is not an actual native complete success: '+i)
                initial[i]=dict(q=q.tolist(),object_pose=obj.tolist(),hole_pose=hole.tolist(),
                    initial_sha256=hashlib.sha256(np.concatenate((q,obj,hole)).tobytes()).hexdigest())
            candidates[split].append(i)
        ids=candidates[split]
        # Equal family opportunity, max-min scaled actual q/object/relative-hole
        # distance within the already approved narrow domain. No wider resets.
        x=np.array([initial[i]['q'][:7]+initial[i]['object_pose'][:3]+
            (np.array(initial[i]['hole_pose'][:3])-initial[i]['object_pose'][:3]).tolist() for i in ids])
        scale=np.maximum(x.std(0),1e-6);z=(x-x.mean(0))/scale
        chosen=[int(np.argmin((z*z).sum(1)))]
        while len(chosen)<min(count,len(ids)):
            dist=((z[:,None]-z[chosen][None])**2).sum(-1).min(1);dist[chosen]=-1
            chosen.append(int(np.argmax(dist)))
        selected[split]=[ids[k] for k in chosen]
        sx=x[chosen];delta=np.linalg.norm(sx[:,None]-sx[None],axis=-1);np.fill_diagonal(delta,np.inf)
        audit[split]=dict(available_families=len(ids),selected=len(chosen),one_per_family=True,
            identical_initial_states=len({initial[i]['initial_sha256'] for i in selected[split]})!=len(chosen),
            minimum_raw_mixed_coordinate_distance=float(delta.min()),
            metric_warning='Mixed rad/m norm is only a duplicate diagnostic; selection standardized by candidate train/validation population separately',
            initial_states={i:initial[i] for i in selected[split]})
    if len(selected['train'])!=30 or any(v['identical_initial_states'] for v in audit.values()):
        raise ValueError('Insufficient distinct initial families')
    if {data['entries'][i]['family'] for i in selected['train']}&{data['entries'][i]['family'] for i in selected['validation']}:
        raise ValueError('Initial family split leakage')
    ids=selected['train']+selected['validation'];records={};labels={}
    for i in ids:
        entry=data['entries'][i]
        if sha256(entry['path'])!=entry['source_sha256']:raise ValueError('Original demonstration hash changed: '+i)
        _,metadata,r=record(entry['path'],privileged=False)
        q,labels[i]=actual_configuration_labels(entry)
        r['q_reached']=q;records[i]=r
    root.mkdir(parents=True,exist_ok=True)
    dino=Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
    expert=Path('pretrained_assembly/tight_insertion/model.pth').resolve()
    old=read(Path('outputs/visuotactile/run/tight_vision_20261109_01/session.json'))
    session=dict(version='tight_fresh_every_tick_v1',resource_policy='full_gpu_v1',
        resources=dict(min_gpu_free_mib=4096,min_ram_gib=10,min_disk_gib=60),reserve_bytes=60*1024**3,
        expert_checkpoint=str(expert),expert_sha256=sha256(expert),dino_checkpoint=str(dino),dino_sha256=sha256(dino),
        precision_policy='unified_fp32',camera_content_validation=True,
        cache_deadline=time.time()+4*3600,deadline=time.time()+12*3600,
        domain=read(Path('configs/tight_state.json'))['domain'],
        authorization='2026-11-14 fresh30/every-frame vision/autonomous DAgger research; no inherited old deadline',
        visual_stride=1,control_hz=60,planning_hz=5,trainable_initialization='random; no student checkpoint read',
        source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(root/'session.json',session)
    atomic_json(root/'selection.json',dict(splits=selected,audit=audit,selection='one median-duration success per family; standardized max-min initial diversity',
        source_data_sha256=sha256(source/'data.json'),test_read=False,labels=labels))
    for role,supervision in [('A','q_reached'),('B','u_expert')]:
        dest=root/role;out=dest/'state_data';out.mkdir(parents=True)
        numeric={i:dict(r) for i,r in records.items()}
        # Keep command=actual expert u; clean_sample is chosen by supervision.
        definition=dict(visual_pair=True,visual_stride=1,supervision=supervision,
            contract=contract(24,True),normalization_identity=continuous_image_contract(supervision)['normalization_identity'],
            splits=dict(selected,test=[]),entries={i:data['entries'][i] for i in ids},
            source_data_sha256=sha256(source/'data.json'),normalization_scope='selected train30 only',
            policy_inputs=['robot178','DINO_RGB_every_control_tick','metric_depth_every_control_tick'],privileged_inputs=False)
        atomic_json(out/'data.json',definition)
        save(out/'numeric.pt',dict(contract=contract(24,True),data_sha256=sha256(out/'data.json'),records=numeric))
        x=torch.cat([records[i]['robot'] for i in selected['train']]);stats=dict(robot_mean=x.mean(0),robot_std=x.std(0).clamp_min(1e-4))
        y=torch.cat([records[i]['q_reached' if role=='A' else 'command'] for i in selected['train']])
        stats['action_scale'],stats['action_offset']=limits_affine(y)
        save(out/'statistics.pt',stats)
        atomic_json(dest/'session.json',dict(session,resource_root=str(root)))
    atomic_json(root/'metadata.json',metadata)
    atomic_json(root/'prepared.json',dict(train=30,validation=6,test_read=False,old_student_checkpoint_loaded=False,
        total_ticks=sum(len(r['time']) for r in records.values()),visual_updates_required=sum(len(r['time']) for r in records.values()),
        source_hashes_verified=True,all_q_next_alignment_passed=True,utc=time.time()))
    print(json.dumps(read(root/'prepared.json')),flush=True)


def cache(root):
    from visuotactile.native_data import cache_visual_pair
    from visuotactile.training_resources import atomic_json
    from visuotactile.compact import sha256
    cache_visual_pair(root/'A')
    # Share only the immutable backbone payload. Dataset-specific index stays bound.
    (root/'B/state_data/perception').symlink_to((root/'A/state_data/perception').resolve(),target_is_directory=True)
    index=read(root/'A/cache_index.json');index['data_sha256']=sha256(root/'B/state_data/data.json')
    atomic_json(root/'B/cache_index.json',index)
    atomic_json(root/'cache_complete.json',dict(frames=sum(v['frames'] for v in index['entries'].values()),stride=1,utc=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.root.resolve())

if __name__=='__main__':main()
