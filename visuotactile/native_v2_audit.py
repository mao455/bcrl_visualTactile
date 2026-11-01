"""Read-only native data audit; no image decoding, simulator, or optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .compact import sha256
from .native_data import labels, limits_affine, robot_rows, geometry, object_dimensions
from .long_data import shift_actions
from .training_resources import atomic_json


def choose_coverage(rows, split, count=8):
    """Outcome-independent length quantiles, then geometry farthest points."""
    candidates = sorted((r for r in rows if r['split'] == split), key=lambda r: (r['length'], r['identity']))
    if len(candidates) < count:
        raise ValueError('Insufficient distinct initial states')
    chosen = [candidates[i] for i in np.linspace(0, len(candidates)-1, min(4, count), dtype=int)]
    values = np.asarray([r['initial_features'] for r in candidates])
    values = (values-values.mean(0))/np.maximum(values.std(0), 1e-4)
    while len(chosen) < count:
        selected = [candidates.index(r) for r in chosen]
        distances = ((values[:, None]-values[selected])**2).sum(-1).min(-1)
        distances[selected] = -1
        chosen.append(candidates[int(distances.argmax())])
    return [r['identity'] for r in chosen]


def audit(source, output, full_hash=False):
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    data = json.loads((source/'data.json').read_text())
    split_of = {identity: split for split, ids in data['splits'].items() for identity in ids}
    if len(split_of) != 400 or sum(map(len, data['splits'].values())) != 400:
        raise ValueError('Split identity overlap')
    stats = torch.load(source/'statistics.pt', map_location='cpu', weights_only=False)
    if sha256(source/'statistics.pt') != data['statistics_sha256']:
        raise ValueError('Statistics hash changed')
    rows, failures, records = [], [], []
    initial_hashes, command_hashes = {}, {}
    maxima = dict(command=0., target=0., robot=0., geometry=0., history=0.)
    for identity, entry in data['entries'].items():
        path = source/'numeric'/(identity+'.pt')
        if sha256(path) != entry['numeric_sha256']:
            raise ValueError('Numeric cache changed: '+identity)
        record = torch.load(path, map_location='cpu', weights_only=False)
        if full_hash and sha256(entry['path']) != entry['source_sha256']:
            raise ValueError('Source SHA mismatch: '+identity)
        with h5py.File(entry['path'], 'r') as f:
            g = f['episodes'][entry['episode']]
            m = json.loads(f.attrs['metadata_json'])
            def read(name):
                return torch.from_numpy(np.asarray(g[name][:]).copy())
            command = read('transition/command_clipped_pre_delay')
            n = len(command)
            reasons = {k: bool(v[-1]) for k, v in g['transition/termination_reasons'].items()}
            success = reasons['max_successes'] and bool(g.attrs['complete'])
            if not success or any(reasons[k] for k in reasons if k != 'max_successes'):
                failures.append(identity+': incomplete native success')
            times = read('obs/time_s'); target = read('obs/actuator_target_rad')
            delayed = read('transition/control_trace/delayed_command')
            past = shift_actions(command, delayed, read('transition/time_before_s'),
                                 read('transition/control_trace/selected_command_time_s'), times)
            diagnostics = {k: read('transition/diagnostics_before/'+k) for k in
                ('base_pose_w','object_pose_w','goal_pose_w','hole_pose_w','table_pose_w','palm_body_pose_w',
                 'object_velocity_w_m_s_rad_s','goal_velocity_w_m_s_rad_s','finger_position_w','finger_quat_wxyz')}
            robot = robot_rows(read('transition/diagnostics_before/joint_position_rad'),
                read('transition/diagnostics_before/joint_velocity_rad_s'), target, past, times-times[0])
            state = geometry(diagnostics, object_dimensions(m))[::12]
            for name, actual, expected in [('command', record['command'], command),
                    ('target', target, read('transition/target_before_rad')),
                    ('robot', record['robot'], robot), ('geometry', record['geometry'], state),
                    ('history', record['robot'][:,116:177], past)]:
                err = float((actual-expected).abs().max()); maxima[name] = max(maxima[name], err)
                if err > 1e-6: failures.append(identity+': '+name)
            if not torch.equal(command, read('transition/submitted_action').clamp(-1,1)):
                failures.append(identity+': submitted mismatch')
            if not torch.equal(read('obs/step_id'), torch.arange(n)):
                failures.append(identity+': tick mismatch')
            if not torch.allclose(times[1:]-times[:-1], torch.full_like(times[1:], 1/60), atol=2e-5, rtol=0):
                failures.append(identity+': clock mismatch')
            anchors = torch.arange(0,n,12); _, valid = labels(record, anchors)
            multiplicity = torch.zeros(n)
            for anchor in anchors: multiplicity[anchor:min(n,anchor+64)] += 1
            stage = read('transition/diagnostics_before/successes').flatten()
            contacts = read('transition/contact_substeps_after_w_N').norm(dim=-1).flatten(1).amax(1) > .1
            features = torch.cat((state[0,:7], state[0,7:14], robot[0,:29])).numpy()
            init_hash = hashlib.sha256(features.tobytes()).hexdigest()
            command_hash = hashlib.sha256(command.numpy().tobytes()).hexdigest()
            initial_hashes.setdefault(init_hash,[]).append(identity)
            command_hashes.setdefault(command_hash,[]).append(identity)
            row = dict(identity=identity, split=split_of[identity], length=n, success=success, reasons=reasons,
                subgoals=int(read('transition/diagnostics_after/successes')[-1]),
                retract=bool(read('transition/diagnostics_after/retract_succeeded')[-1]),
                source_seed=m['seed'], source_lane=m.get('source_env_index',0), source_envs=m.get('num_parallel_envs',1),
                initial_features=features.tolist(), initial_sha256=init_hash, command_sha256=command_hash,
                force_ticks=int(read('transition/external_force_nonzero').sum()),
                torque_ticks=int(read('transition/external_torque_nonzero').sum()),
                native_contact_ticks=int(contacts.sum()), contact_scope='aggregate fingertips, not object-specific',
                saturation_arm=float((command[:,:7].abs()>=.999).float().mean()),
                saturation_hand=float((command[:,7:].abs()>=.999).float().mean()),
                anchors=len(anchors), segments=(n+95)//96, valid_targets=int(valid.sum()),
                padded_targets=int((~valid).sum()), stage_ticks={str(int(v)):int((stage==v).sum()) for v in stage.unique()},
                stage_label_weight={str(int(v)):float(multiplicity[stage==v].sum()) for v in stage.unique()},
                source_hash_checked=full_hash)
            rows.append(row)
        if split_of[identity]=='train': records.append(record)
        atomic_json(output/'audit_progress.json',dict(completed=len(rows),required=400))
    reproduced = {}
    for name in ('robot','geometry'):
        values = torch.cat([r[name] for r in records])
        reproduced[name+'_mean']=values.mean(0); reproduced[name+'_std']=values.std(0).clamp_min(1e-4)
    commands=torch.cat([r['command'] for r in records])
    reproduced['action_scale'],reproduced['action_offset']=limits_affine(commands)
    stat_errors={k:float((stats[k]-v).abs().max()) for k,v in reproduced.items()}
    if max(stat_errors.values())>1e-6: failures.append('Train-only statistics mismatch')
    duplicates={kind:[ids for ids in mapping.values() if len(ids)>1] for kind,mapping in
                [('initial_observable',initial_hashes),('command',command_hashes)]}
    if any(len({split_of[i] for i in ids})>1 for groups in duplicates.values() for ids in groups):
        failures.append('Cross-split exact duplicate')
    summary={}
    for split in data['splits']:
        selected=[r for r in rows if r['split']==split]
        summary[split]=dict(episodes=len(selected),steps=sum(r['length'] for r in selected),
            lengths=np.percentile([r['length'] for r in selected],[0,25,50,75,100]).tolist(),
            features_mean=np.mean([r['initial_features'] for r in selected],0).tolist(),
            features_std=np.std([r['initial_features'] for r in selected],0).tolist(),
            successes=sum(r['success'] for r in selected), segments=sum(r['segments'] for r in selected),
            anchors=sum(r['anchors'] for r in selected), valid_targets=sum(r['valid_targets'] for r in selected),
            padded_targets=sum(r['padded_targets'] for r in selected))
    cohorts={s:choose_coverage(rows,s) for s in ('train','validation')}
    # Eight independent historical demonstrations. No test states used to choose a diagnostic.
    cohorts['replay']=cohorts['train'][:4]+cohorts['validation'][:4]
    result=dict(infrastructure_passed=not failures,offline_fit_passed=None,closed_loop_success=None,
        acceptance_passed=False,blocked_reason='Multi-trajectory physical replay pending' if not failures else '; '.join(failures),
        source=str(source.resolve()),source_data_sha256=sha256(source/'data.json'),statistics_sha256=sha256(source/'statistics.pt'),
        numeric_max_errors=maxima,statistics_max_errors=stat_errors,duplicates=duplicates,
        splits=summary,cohorts=cohorts,rows=rows,failures=failures,elapsed_s=time.monotonic()-began,
        sampling='Each anchor once per traversal; valid-target weighted four contiguous 96-tick segments/update; not episode-balanced',
        raw_integrity='all400 SHA256 re-read' if full_hash else 'prior full data Gate SHA evidence; numeric SHA and live HDF5 numeric fields re-read',
        caveat='Shared collection run seed is not independent RNG; exact observable duplicate test does not certify hidden-state independence')
    atomic_json(output/'data_audit.json',result)
    print(json.dumps({k:result[k] for k in ('infrastructure_passed','numeric_max_errors','statistics_max_errors','cohorts','elapsed_s')}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-hash',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    audit(a.source,a.output,a.full_hash)
