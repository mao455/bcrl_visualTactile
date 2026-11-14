"""Bounded CPU-only DDIM seed sensitivity at fixed real contact histories."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from visuotactile.anchor_sampling import labels
from visuotactile.compact import sha256
from visuotactile.continuous_vision import encode_dense
from visuotactile.models.native_dp import load_policy
from visuotactile.native_data import Episode, visual_records
from visuotactile.position_control import bounded_targets
from visuotactile.training_resources import atomic_json


SEEDS = list(range(7091, 7096))
PARTS = dict(arm=slice(0, 7), hand=slice(7, 29))


def read(path):
    return json.loads(path.read_text())


def available_gib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) / 1024**2


def rms(value):
    return float(value.double().square().mean().sqrt())


def spread(plans, truth, joints):
    results = {}
    for part, section in PARTS.items():
        values = plans[:, :12, section].double()
        std = values.std(0, correction=0)
        span = values.max(0).values-values.min(0).values
        index = int(span.argmax()); joint = index % span.shape[1] + (0 if part=='arm' else 7)
        pairwise = [rms(values[i]-values[j]) for i in range(len(SEEDS)) for j in range(i)]
        results[part] = dict(seed_std_rms_rad=rms(std), seed_std_mean_rad=float(std.mean()),
            seed_std_max_rad=float(std.max()), max_seed_pair_abs_difference_rad=float(span.max()),
            max_difference_at=dict(plan_offset=index//span.shape[1], joint=joints[joint]),
            max_seed_pair_rmse_rad=max(pairwise),
            seed_errors=[dict(seed=seed, rmse_rad=rms(values[k]-truth[:12, section]),
                max_abs_rad=float((values[k]-truth[:12, section]).abs().max())) for k,seed in enumerate(SEEDS)])
    return results


def truth_motion(truth, previous):
    first = truth[:12].double()
    increments = torch.diff(torch.cat((previous[None].double(), first)), dim=0)
    return {part:dict(per_step_increment_rms_rad=rms(increments[:, section]),
        per_step_increment_max_abs_rad=float(increments[:, section].abs().max()),
        endpoint_displacement_rms_rad=rms(first[-1,section]-previous[section]),
        hold_previous_rmse_rad=rms(first[:,section]-previous[section])) for part,section in PARTS.items()}


@torch.inference_mode()
def run(round_root, output):
    if output.exists(): raise FileExistsError('Preserve earlier diagnostic results')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started=time.monotonic(); initial_available=available_gib()
    if initial_available<12: raise RuntimeError('Insufficient available RAM for CPU model load with10GiB reserve')
    plan=read(round_root/'development/plan.json')
    primary=read(round_root/'A/state_data/data.json')
    chosen=primary['normalization_train_ids'][:2]
    if len(chosen)!=2 or len({primary['entries'][i]['family'] for i in chosen})!=2:
        raise ValueError('Expected first two distinct original train30 families')
    registration=dict(requested_identities=chosen, seeds=SEEDS, cpu_threads=2,
        anchor_rule='floor(first fingertip-contact transition zero-based step_id/12)*12',
        condition='Recompute current complete frontend over every real visual and robot observation from0 through anchor; frozen all-frame DINO cache only.',
        sampling='DDIM16 eta0, clip_sampleFalse; independent CPU generator perseed advanced once by1x24x29 Gaussian draw for every previous5Hz anchor.',
        maximum_wall_seconds=300, minimum_available_ram_gib=10, test_used=False)
    rows=[]; models={}; limit=2; reduction=None
    for role in ('A','B'):
        directory=round_root/role; data_path=directory/'state_data'; data=read(data_path/'data.json')
        if chosen!=data['normalization_train_ids'][:2]: raise ValueError('A/B original cohort order differs')
        checkpoint=Path(plan['pairs'][role]['frontend'])
        identity=sha256(checkpoint)
        if identity!=plan['pairs'][role]['frontend_sha256']: raise ValueError('Selected model identity differs')
        saved=torch.load(checkpoint, weights_only=False, map_location='cpu')
        model=load_policy(saved, 'cpu').eval().requires_grad_(False)
        del saved
        models[role]=dict(checkpoint_sha256=identity, data_sha256=sha256(data_path/'data.json'),
            supervision=model.contract['supervision'], statistics_sha256=sha256(data_path/'statistics.pt'))
        contract=read(directory/'continuation.json').get('target_window_contract')
        records=visual_records(data_path, chosen[:limit], 'G00', device='cpu', target_window_contract=contract)
        for ordinal, name in enumerate(chosen[:limit]):
            if time.monotonic()-started>270 or available_gib()<11:
                raise RuntimeError('CPU audit bound reached before a registered case')
            began=time.monotonic(); record=records[name]; entry=data['entries'][name]
            with h5py.File(entry['path']) as f:
                group=f['episodes'][entry['episode']]
                joints=json.loads(f.attrs['metadata_json'])['joint_order']
                contact=np.linalg.norm(group['transition/object_contact_force_w'][:], axis=-1)>.1
                indices=np.flatnonzero(contact.any(axis=(1,2)))
                if not len(indices): raise ValueError('Expected a successful grasp with fingertip/object contact')
                first=int(indices[0]); anchor=first//12*12
                raw_label=torch.from_numpy(group['transition/diagnostics_after/joint_position_rad' if role=='A' else 'transition/target_after_rad'][anchor:anchor+12])
            episode=Episode(data_path,name,model.group)
            condition,anchors=encode_dense(model,record,episode=episode,stop=anchor+1)
            selected=condition[anchors==anchor]
            if selected.shape!=(1,512): raise ValueError('Expected one exact native-clock condition')
            target,valid=labels(record,torch.tensor([anchor]),24)
            if not valid[0,:12].all() or not torch.equal(target[0,:12],raw_label):
                raise ValueError('First12 labels must be actual, legal and semantically aligned')
            predictions=[]; inference_seconds=[]
            for seed in SEEDS:
                generator=torch.Generator(device='cpu').manual_seed(seed)
                for _ in range(anchor//12): torch.randn((1,model.horizon,29),generator=generator)
                tick=time.monotonic()
                predictions.append(model.predict(selected,generator,inference_steps=16,
                    clip_sample=False,scheduler_index_mode='host')[0])
                inference_seconds.append(time.monotonic()-tick)
            raw=torch.stack(predictions)
            bounded=bounded_targets(raw.flatten(0,1),record['joint_limits']).reshape_as(raw)
            truth=target[0]
            previous=record['robot'][anchor,:29] if role=='A' else record['robot'][anchor,58:87]
            row=dict(role=role,identity=name,family=entry['family'],source_recorded_sha256=entry['source_sha256'],
                first_detected_contact_transition_step_id=first,first_detected_contact_completed_tick=first+1,
                anchor_step_id=anchor,real_history_observations=anchor+1,noise_draws_before_anchor=anchor//12,
                condition_sha256=hashlib.sha256(selected.numpy().tobytes()).hexdigest(),
                full_prefix_recomputed=True,legal_first12_labels_match_raw_hdf5=True,
                label_semantics='actual configuration after each next control step; not settled configuration' if role=='A' else 'actual expert issued absolute joint target after each control step',
                raw_prediction=spread(raw,truth,joints),joint_limit_bounded_prediction=spread(bounded,truth,joints),
                bounded_changed_values=int((raw!=bounded).sum()),
                actual_label_first12_motion=truth_motion(truth,previous),
                ddim_seconds_per_seed=inference_seconds,wall_seconds=time.monotonic()-began,
                available_ram_gib=available_gib())
            rows.append(row)
            print(json.dumps(dict(role=role,identity=name,wall_seconds=row['wall_seconds'])),flush=True)
            del episode,condition,predictions
            if role=='A' and ordinal==0 and 4*(time.monotonic()-started)>285:
                limit=1; reduction='First role/family elapsed predicts over5minutes for all4cases; retained first original family for A/B.'
                break
        del model,records
    if torch.cuda.is_initialized(): raise ValueError('CPU audit unexpectedly initialized CUDA')
    result=dict(version='fixed_condition_cpu_ddim_seed_spread_v1',registration=registration,
        effective_identities=chosen[:limit],scope_reduction=reduction,models=models,cases=rows,
        source_sha256=sha256(Path(__file__)),continuous_encoder_sha256=sha256(Path('visuotactile/continuous_vision.py')),
        policy_sha256=sha256(Path('visuotactile/models/native_dp.py')),
        plan_sha256=sha256(round_root/'development/plan.json'),
        wall_seconds=time.monotonic()-started,initial_available_ram_gib=initial_available,
        process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_initialized=False,optimizer_updates=0,simulator_starts=0,test_used=False,
        interpretation='Fixed real successful expert histories and one precontact planning anchor per original training family. Full frontend only; backend and autonomous feedback are not evaluated. CPU RNG and CUDA RNG do not yield identical samples for the same seed. Seed spread can identify sensitivity magnitude; it cannot establish that randomness caused observed autonomous failures. Population standard deviation uses five predeclared seeds and is not an uncertainty interval.')
    atomic_json(output,result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--round-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.round_root,args.output)


if __name__=='__main__': main()
