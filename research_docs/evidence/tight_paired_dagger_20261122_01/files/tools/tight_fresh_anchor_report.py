"""Read how nominal projection and the trained rear alter recorded recovery commands."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.tight_fresh_onset_report import BINS, PARTS, read_case
from visuotactile.compact import sha256
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
from visuotactile.execution_learning import ideal_execution_bounds
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
from visuotactile.training_resources import atomic_json


def summarize(values):
    result = {'ticks': len(values[0])}
    target, shadow, previous, anchor, nominal = values
    for part, section in PARTS.items():
        t, p, h, a, b = [v[:, section] for v in values]
        expert_increment, student_increment = t-h, p-h
        active = np.abs(expert_increment) > 1e-4
        expert_energy = np.square(expert_increment).sum()
        result[part] = dict(
            previous_target_rmse_rad=float(np.sqrt(np.square(expert_increment).mean())),
            clipped_nominal_rmse_rad=float(np.sqrt(np.square(a-t).mean())),
            full_student_shadow_rmse_rad=float(np.sqrt(np.square(p-t).mean())),
            nominal_projection_changed_fraction=float((np.abs(a-b)>1e-6).mean()),
            student_increment_rms_rad=float(np.sqrt(np.square(student_increment).mean())),
            student_on_expert_increment_projection=float((student_increment*expert_increment).sum()/expert_energy)
                if expert_energy else None,
            opposite_increment_fraction=float(((student_increment*expert_increment<0)&active).sum()/active.sum())
                if active.any() else None,
            active_expert_increment_coordinates=int(active.sum()))
    return result


def report(root, collection, output):
    plan=json.loads((collection/'plan.json').read_text())
    if plan['kind']!='takeover' or plan['test_used']:
        raise ValueError('Require actual training-only takeover data')
    data_path=root/'A/state_data/data.json'
    data=json.loads(data_path.read_text())
    records=torch.load(root/'A/state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
    roles={};cases=[]
    for role in ('A','B'):
        pair=plan['pairs'][role]
        for name in ('frontend','backend'):
            if sha256(pair[name])!=pair[name+'_sha256']:raise ValueError('Old model changed')
        saved=torch.load(pair['backend'],map_location='cpu',weights_only=False)
        configuration=saved['configuration']
        if (configuration.get('execution_adapter')!=REACHABLE_POSITION_EXECUTION
                or configuration.get('output_contract')!=REACHABLE_CENTERED_OUTPUT
                or configuration.get('base_sha256')!=pair['frontend_sha256']):
            raise ValueError('Recorded pair does not use the analyzed nominal-centered execution')
        del saved
        grouped={name:[] for name in BINS}
        for case in plan['cases']:
            if case['role']!=role:continue
            reference=read_case(collection,case,pair)
            if not reference['expert_takeover_success']:continue
            identity=case['name'];entry=data['entries'][identity]
            if identity not in data['splits']['train'] or entry['source_group']!=role:
                raise ValueError('Expected admitted training correction from the same full student')
            if sha256(entry['path'])!=entry['source_sha256']:raise ValueError('Admitted source changed')
            with h5py.File(collection/identity/'numeric.h5') as f:
                group=f['episode/transition/dagger']
                mask=group['expert_label_valid'][:].astype(bool)
                record=records[identity]
                if not np.array_equal(mask,record['supervision_mask'].numpy()):
                    raise ValueError('Actual expert mask differs from admitted data')
                previous=f['episode/obs/actuator_target_rad'][:]
                target=group['u_expert'][:].astype(np.float64)
                if not np.array_equal(target[mask],record['u_expert'].numpy()[mask]):
                    raise ValueError('Actual expert command differs from admitted data')
                nominal=group['nominal'][:].astype(np.float64)
                bounds=ideal_execution_bounds(torch.from_numpy(previous),record['joint_limits'],
                    REACHABLE_POSITION_EXECUTION).numpy().astype(np.float64)
                anchor=np.clip(nominal,bounds[...,0],bounds[...,1])
                shadow=group['student_proposed_target_rad'][:].astype(np.float64)
                if np.max(np.abs(np.clip(shadow,bounds[...,0],bounds[...,1])-shadow))>1e-6:
                    raise ValueError('Saved full student leaves the common reachable interval')
                indices=np.flatnonzero(mask)
                bins={}
                for name,rows in zip(BINS,(indices[:12],indices[12:24],indices[24:])):
                    if not len(rows):raise ValueError('This report requires a nonempty complete recovery bin')
                    values=tuple(v[rows] for v in (target,shadow,previous.astype(np.float64),anchor,nominal))
                    if not all(np.isfinite(v).all() for v in values):raise ValueError('Nonfinite recovery command')
                    grouped[name].append(values);bins[name]=summarize(values)
                cases.append(dict(case=identity,role=role,bins=bins,numeric_sha256=reference['numeric_sha256'],
                    admitted_source_sha256=entry['source_sha256']))
        roles[role]=dict(cases=len(grouped['first12']),frontend_sha256=pair['frontend_sha256'],
            backend_sha256=pair['backend_sha256'],bins={name:summarize(tuple(np.concatenate([v[i] for v in groups])
                for i in range(5))) for name,groups in grouped.items()})
    atomic_json(output,dict(roles=roles,cases=cases,plan_sha256=sha256(collection/'plan.json'),
        data_sha256=sha256(data_path),numeric_records_sha256=sha256(root/'A/state_data/numeric.pt'),
        source_sha256=sha256(Path(__file__)),test_used=False,model_forward_calls=0,simulator_starts=0,
        scope='Saved old full-student shadows on the same successful actual expert suffixes. Clipped nominal is the zero-raw-output anchor, not an independently executed policy. Projection fraction counts joint-ticks; opposite signs exclude expert increments <=1e-4 rad. Increment projection is a pooled dot product divided by expert squared norm. These errors and direction statistics do not identify the cause of autonomous failure.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.collection,a.output)
