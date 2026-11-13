"""Explicit fresh-cohort recovery stages; research decisions stay with the agent."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read, completed_pair


def prepare_collection(root, output, ticks, round_index, *, stable_delays=None, fallback_tick=360):
    import h5py
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.state_benchmark import dagger_configuration
    if output.exists():raise FileExistsError('Preserve started collection identity')
    if stable_delays is not None:
        from visuotactile.takeover_trigger import VERSION, validate_trigger
        if not stable_delays: raise ValueError('Nonempty stable-delay opportunities required')
        for delay in stable_delays: validate_trigger(dict(version=VERSION,delay_ticks=delay),fallback_tick)
        ticks=[fallback_tick]*len(stable_delays)
    if not ticks or any(t<1 for t in ticks):raise ValueError('Positive takeover ticks required')
    session=read(root/'session.json');pairs={r:completed_pair(root,r) for r in ['A','B']}
    data={r:read(root/r/'state_data/data.json') for r in pairs}
    if data['A']['splits']!=data['B']['splits']:raise ValueError('Different A/B sources')
    originals=data['A'].get('normalization_train_ids',data['A']['splits']['train'])
    if len(ticks)>len(originals):raise ValueError('One different training family per opportunity')
    cases=[]
    for index,tick in enumerate(ticks):
        identity=originals[index];entry=data['A']['entries'][identity]
        with h5py.File(entry['path']) as f:original=json.loads(f.attrs['result_json'])
        for role in (['A','B'] if index%2==0 else ['B','A']):
            suffix=f't{tick}' if stable_delays is None else f'stable_d{stable_delays[index]}'
            name=f'r{round_index:02d}_{role}_{identity}_{suffix}'
            spec=dict(identity=name,family=entry['family'],split='train',cohort='dagger_train',
                source_identity=identity,seed=original['seed'],event_seed=28150000+1000*round_index+index,
                narrow_reset=original['narrow_reset'],collision_gate='live_collision_mesh_bounds_v1',
                takeover_tick=tick)
            bridge=dict(version=ExpertPositionBridge.version,expert_speed=ExpertPositionBridge.speed_scheme)
            if stable_delays is not None:
                bridge['trigger']=dict(version=VERSION,delay_ticks=stable_delays[index])
            dagger_configuration(bridge,spec,data[role])
            cases.append(dict(name=name,role=role,spec=spec,dagger=bridge,video=index==0,
                source_data=str((root/role/'state_data/data.json').resolve())))
    output.mkdir(parents=True)
    atomic_json(output/'session.json',session)
    atomic_json(output/'plan.json',dict(kind='takeover',round_index=round_index,pairs=pairs,cases=cases,
        resource_root=str(root),test_used=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ['tools','visuotactile'] for p in sorted(Path(folder).rglob('*.py'))},
        reason=('Sample completed student stable-grasp states and subsequent deviations, fallback separately reported'
            if stable_delays is not None else 'Sample registered fixed-time approach/contact and later deviations'),
        selection='Native-successful real expert suffix only; all unsuccessful attempts retained',utc=time.time()))


def prepare_aggregate(root, collection, output):
    from visuotactile.narrow_data import build_dagger_aggregate
    from visuotactile.fresh_contracts import FRESH_AGGREGATE_CONTRACT,residual_normalization
    if output.exists():raise FileExistsError('Preserve existing aggregate')
    if not read(collection/'result.json').get('software_complete'):raise ValueError('Collection incomplete')
    entries=read(collection/'correction_entries.json')
    if not entries:raise ValueError('No successful actual expert recovery; analyze before training')
    plan=read(collection/'plan.json')
    output.mkdir(parents=True)
    for name in ['session.json','metadata.json']:shutil.copyfile(root/name,output/name)
    (output/'shared_resources.json').symlink_to(root/'shared_resources.json')
    for role in ['A','B']:
        destination=output/role;destination.mkdir()
        (destination/'shared_resources.json').symlink_to(root/'shared_resources.json')
        data=build_dagger_aggregate(root/role/'state_data',entries,destination/'state_data',
            'q_reached' if role=='A' else 'u_expert',include_geometry=role=='A',include_frontend_tactile=False)
        residual_normalization(data,dict(fresh_data_contract=FRESH_AGGREGATE_CONTRACT))
        atomic_json(destination/'session.json',dict(read(root/'session.json'),resource_root=str(output),
            cache_deadline=read(root/'session.json')['deadline']))
        pair=plan['pairs'][role]
        atomic_json(destination/'continuation.json',dict(path=pair['frontend'],sha256=pair['frontend_sha256'],
            statistics_sha256=sha256(destination/'state_data/statistics.pt'),
            source_root=str(root/role),reuse_index=str(root/role/'cache_index.json'),
            target_window_contract='recovery_onset_commit12_v1',learning_rate=3e-5,
            initialization='Current selected DP EMA; fresh paired AdamW/EMA; fixed original30 affine',
            previous_backend=pair['backend'],previous_backend_sha256=pair['backend_sha256']))
    atomic_json(output/'gate.json',dict(passed=True,inherited_input_gate=str(root/'gate.json'),
        inherited_input_gate_sha256=sha256(root/'gate.json'),new_capture_audits=str(collection),
        meaning='Exact same every-frame input path; actual new expert masks audited at collection'))
    atomic_json(output/'prepared.json',dict(kind=FRESH_AGGREGATE_CONTRACT,source_root=str(root),
        source_collection=str(collection),collection_plan_sha256=sha256(collection/'plan.json'),
        accepted_corrections=len(entries),shared_A_B_entries=sorted(entries),test_used=False,utc=time.time()))


def cache(output):
    from visuotactile.native_data import cache_visual_pair
    from visuotactile.precision import state_fp32
    state_fp32()
    cache_visual_pair(output/'A',reuse_index=read(output/'A/continuation.json')['reuse_index'])
    cache_visual_pair(output/'B',reuse_index=output/'A/cache_index.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','aggregate','cache'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collection',type=Path);p.add_argument('--ticks',type=int,nargs='+',default=[46,58,70,84,120,180])
    p.add_argument('--round',type=int,default=1)
    p.add_argument('--stable-delays',type=int,nargs='+');p.add_argument('--fallback-tick',type=int,default=360);a=p.parse_args()
    if a.stage=='prepare':prepare_collection(a.root.resolve(),a.output.resolve(),a.ticks,a.round,
        stable_delays=a.stable_delays,fallback_tick=a.fallback_tick)
    elif a.stage=='aggregate':prepare_aggregate(a.root.resolve(),a.collection.resolve(),a.output.resolve())
    else:cache(a.output.resolve())


if __name__=='__main__':main()
