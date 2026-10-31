"""Read-only fixed400 diversity census and outcome-independent diagnostic selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .long_data import MANIFEST
from .training_resources import atomic_json


def summary(values):
    array = np.asarray(values, dtype=float)
    return dict(minimum=array.min(0).tolist(),maximum=array.max(0).tolist(),
                mean=array.mean(0).tolist(),std=array.std(0).tolist())


def census(source, output):
    manifest = json.loads(MANIFEST.read_text())
    splits = json.loads((source/'splits.json').read_text())
    membership = {identity:split for split,identities in splits.items() for identity in identities}
    rows, details, initial_vectors, descriptors = [], {}, {}, {}
    cameras, assets, randomization, reset_configs = {}, {}, {}, {}
    for entry in manifest['episodes']:
        identity = entry['id']
        path = MANIFEST.parent/entry['directory']/'data.h5'
        with h5py.File(path,'r') as stream:
            metadata = json.loads(stream.attrs['metadata_json'])
            episode = stream['episodes'][entry['episode']]
            length = entry['length']
            origin = np.asarray(metadata['env_origins_w'][0])
            def read(name):
                return episode[name][:length]
            initial = {key:dataset[0] for key,dataset in episode['transition/diagnostics_before'].items()}
            local = {}
            for key in ('object_pose_w','hole_pose_w','table_pose_w','goal_pose_w','base_pose_w'):
                value = initial[key].copy()
                value[:3] -= origin
                local[key] = value
            command = read('transition/command_clipped_pre_delay')
            force = np.linalg.norm(read('obs/contact_w_N'),axis=-1)
            gate = read('obs/contact_valid') & (force.max(-1)>.1)
            height = read('transition/diagnostics_after/object_pose_w')[:,2]-float(initial['object_initial_z_m'])
            lifted = read('transition/diagnostics_after/lifted_object')
            successes = read('transition/diagnostics_after/successes')
            applied_force = np.linalg.norm(read('transition/applied_wrench/force_w_N'),axis=-1)
            applied_torque = np.linalg.norm(read('transition/applied_wrench/torque_w_Nm'),axis=-1)
            row = dict(identity=identity,split=membership[identity],seed=entry['seed'],length=length,
                object_x_m=float(local['object_pose_w'][0]),object_y_m=float(local['object_pose_w'][1]),
                object_z_m=float(local['object_pose_w'][2]),table_z_m=float(local['table_pose_w'][2]),
                contact_steps=int(gate.sum()),contact_fraction=float(gate.mean()),
                native_lift_steps=int(lifted.sum()),height_above_5cm_steps=int((height>.05).sum()),
                height_peak_m=float(height.max()),first_subgoal_step=int(np.argmax(successes>0)) if (successes>0).any() else None,
                final_successes=int(successes.max()),retract_steps=int(read('transition/diagnostics_after/retract_phase').sum()),
                first_subgoal_or_later_steps=int((successes>0).sum()),
                applied_force_physics_samples=int((applied_force>1e-8).sum()),
                applied_torque_physics_samples=int((applied_torque>1e-8).sum()),
                applied_force_peak_N=float(applied_force.max()),applied_torque_peak_Nm=float(applied_torque.max()),
                command_mean=float(command.mean()),command_std=float(command.std()),saturation_fraction=float((abs(command)>=.99999).mean()),
                force_probability=float(read('transition/external_force_probability')[0]),
                torque_probability=float(read('transition/external_torque_probability')[0]),
                object_mass_kg=float(read('transition/external_object_mass_kg')[0]),
                hdf5_sha256=entry['files']['data.h5']['sha256'],command_sha256=hashlib.sha256(command.tobytes()).hexdigest())
            rows.append(row)
            vector = np.concatenate((initial['joint_position_rad'],initial['joint_velocity_rad_s'],
                                     local['object_pose_w'],local['hole_pose_w'],local['table_pose_w']))
            initial_vectors[identity] = vector
            sample_positions = np.linspace(0,length-1,32).astype(int)
            descriptors[identity] = command[sample_positions].flatten()
            details[identity] = dict(local_initial={key:value.tolist() for key,value in local.items()},
                initial_joint_position_rad=initial['joint_position_rad'].tolist(),
                initial_joint_velocity_rad_s=initial['joint_velocity_rad_s'].tolist(),
                command_mean_by_joint=command.mean(0).tolist(),command_std_by_joint=command.std(0).tolist(),
                command_saturation_by_joint=(abs(command)>=.99999).mean(0).tolist(),
                force_peak_by_sensor_N=force.max(0).tolist(),source_origin_w=origin.tolist(),
                recorded_initial_keys=list(initial),rng_before_step_present='torch_rng_before_step' in episode['transition'],
                full_delay_queue_present='action_queue' in episode['transition'],
                initial_sha256=hashlib.sha256(vector.tobytes()).hexdigest())
            for destination,key in ((cameras,'camera_rig'),(assets,'assets'),(randomization,'domain_randomization'),(reset_configs,'reset')):
                value = metadata.get(key,metadata['config'].get(key))
                if key=='camera_rig':
                    value = json.loads(json.dumps(value))
                    value['wrist'].pop('parent_prim',None)
                if key=='assets':
                    value = {name:value[name] for name in ('object_name','object_scale','object_friction',
                        'robot_friction','finger_tip_friction','table_friction','table_scale_range_x','table_scale_range_y')}
                encoded = json.dumps(value,sort_keys=True)
                digest = hashlib.sha256(encoded.encode()).hexdigest()
                destination.setdefault(digest,dict(value=value,count=0))['count'] += 1
    train = [row for row in rows if row['split']=='train']
    positions = np.stack([initial_vectors[row['identity']] for row in train])
    features = np.concatenate((positions,np.array([[row['length'],row['contact_fraction']] for row in train])),1)
    standardized = (features-features.mean(0))/np.maximum(features.std(0),1e-8)
    selected = [int(np.argmin(np.linalg.norm(standardized,axis=1)))]
    while len(selected)<4:
        nearest = np.min(np.stack([np.linalg.norm(standardized-standardized[index],axis=1) for index in selected]),axis=0)
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    chosen = [train[index]['identity'] for index in selected]
    scalar_keys = [key for key,value in rows[0].items() if isinstance(value,(float,int)) and key!='seed']
    distribution = {}
    for split in splits:
        subset = [row for row in rows if row['split']==split]
        distribution[split] = dict(episodes=len(subset),native_steps=sum(row['length'] for row in subset),
            statistics={key:summary([row[key] for row in subset if row[key] is not None]) for key in scalar_keys},
            initial_joint_position_rad=summary([details[row['identity']]['initial_joint_position_rad'] for row in subset]),
            initial_joint_velocity_rad_s=summary([details[row['identity']]['initial_joint_velocity_rad_s'] for row in subset]),
            object_pose_local=summary([details[row['identity']]['local_initial']['object_pose_w'] for row in subset]),
            hole_pose_local=summary([details[row['identity']]['local_initial']['hole_pose_w'] for row in subset]),
            table_pose_local=summary([details[row['identity']]['local_initial']['table_pose_w'] for row in subset]),
            command_mean_by_joint=summary([details[row['identity']]['command_mean_by_joint'] for row in subset]),
            command_std_by_joint=summary([details[row['identity']]['command_std_by_joint'] for row in subset]),
            command_saturation_by_joint=summary([details[row['identity']]['command_saturation_by_joint'] for row in subset]))
    train_ids = splits['train']
    train_initial = np.stack([initial_vectors[identity] for identity in train_ids])
    train_descriptors = np.stack([descriptors[identity] for identity in train_ids])
    scale = np.maximum(train_initial.std(0),1e-6)
    near = []
    for row in rows:
        identity = row['identity']
        distances = np.sqrt(np.mean(((train_initial-initial_vectors[identity])/scale)**2,axis=1))
        action_distance = np.sqrt(np.mean((train_descriptors-descriptors[identity])**2,axis=1))
        if identity in train_ids:
            distances[train_ids.index(identity)] = np.inf
            action_distance[train_ids.index(identity)] = np.inf
        near.append(dict(identity=identity,split=row['split'],nearest_initial_train=train_ids[int(distances.argmin())],
            standardized_initial_rms=float(distances.min()),nearest_action_train=train_ids[int(action_distance.argmin())],
            resampled_command_rmse=float(action_distance.min())))
    result = dict(source=str(MANIFEST),split_source=str(source/'splits.json'),splits=distribution,
        selected_train=chosen,selection='train-only standardized initial q/qd/object/hole/table + length/contact; central then farthest-point4; no student results',
        camera_configurations=list(cameras.values()),asset_configurations=list(assets.values()),
        randomization_configurations=list(randomization.values()),reset_configurations=list(reset_configs.values()),
        unique_initial_vectors=len({row['initial_sha256'] for row in details.values()}),
        unique_command_sequences=len({row['command_sha256'] for row in rows}),
        unique_hdf5=len({row['hdf5_sha256'] for row in rows}),nearest_train=near,
        selected_details={identity:details[identity] for identity in chosen},
        limitations=['successful-only collection; not unconditional expert success distribution',
            'gate is aggregate finger force, not verified object grasp',
            'phase duration measured by native flags, no inferred grasp labels',
            'nearest action uses32 normalized-progress samples, not full trajectory identity',
            'initial quaternion Euclidean coordinates used for selection, not an SO3 distance',
            'full simulator state/solver cache/sensor history not saved; no exact replay claim'])
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'dataset_distribution.json',result)
    with (output/'dataset_episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(episodes=len(rows),selected=chosen,unique_initial=result['unique_initial_vectors'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    census(arguments.source,arguments.output)
