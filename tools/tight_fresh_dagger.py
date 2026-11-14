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


def prepare_collection(root, output, ticks, round_index, *, stable_delays=None, fallback_tick=360, family_offset=0):
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
    if family_offset<0 or family_offset+len(ticks)>len(originals):raise ValueError('One different training family per opportunity within original train30')
    cases=[]
    for index,tick in enumerate(ticks):
        identity=originals[family_offset+index];entry=data['A']['entries'][identity]
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
    atomic_json(output/'plan.json',dict(kind='takeover',round_index=round_index,family_offset=family_offset,pairs=pairs,cases=cases,
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
    if plan.get('training_expansion_registry') is not None:
        from visuotactile.training_expansion import validate_collection_plan
        validate_collection_plan(plan)
    triggered_cases={c['name'] for c in plan['cases'] if c.get('dagger',{}).get('trigger') is not None}
    if triggered_cases:
        from tools.tight_fresh_report import audit_trigger_case
        for row in read(collection/'results.json'):
            if row['case'] in triggered_cases and row['result'].get('legal'):
                audit_trigger_case(collection/row['case'],row['result'])
    output.mkdir(parents=True)
    for name in ['session.json','metadata.json']:shutil.copyfile(root/name,output/name)
    (output/'shared_resources.json').symlink_to(root/'shared_resources.json')
    for role in ['A','B']:
        destination=output/role;destination.mkdir()
        (destination/'shared_resources.json').symlink_to(root/'shared_resources.json')
        data=build_dagger_aggregate(root/role/'state_data',entries,destination/'state_data',
            'q_reached' if role=='A' else 'u_expert',include_geometry=role=='A',include_frontend_tactile=False,
            **({'training_expansion_registry':plan['training_expansion_registry']}
                if plan.get('training_expansion_registry') is not None else {}))
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


def cache(output, *, reuse_index=None, payload_audit_memo=False):
    from visuotactile.native_data import cache_visual_pair
    from visuotactile.precision import state_fp32
    if type(payload_audit_memo) is not bool:raise ValueError('Explicit boolean payload-audit opt-in required')
    state_fp32()
    if not payload_audit_memo:
        cache_visual_pair(output/'A',reuse_index=reuse_index or read(output/'A/continuation.json')['reuse_index'])
        cache_visual_pair(output/'B',reuse_index=output/'A/cache_index.json')
        return
    from visuotactile.frozen_payload_audit import FrozenPayloadAudit
    audit=FrozenPayloadAudit();started=time.monotonic();phases=[]
    receipt=output/f'payload_audit_memo_{audit.pid}_{time.time_ns()}.json'
    def record(status,**extra):
        atomic_json(receipt,dict(status=status,started_utc=started_utc,utc=time.time(),
            seconds=time.monotonic()-started,statistics=audit.statistics(),phases=phases,**extra))
    started_utc=time.time();record('started')
    try:
        for role,index in [('A',reuse_index or read(output/'A/continuation.json')['reuse_index']),
                           ('B',output/'A/cache_index.json')]:
            cache_visual_pair(output/role,reuse_index=index,payload_audit=audit)
            phases.append(dict(role=role,statistics=audit.statistics(),seconds=time.monotonic()-started))
            record('role_complete')
        record('complete')
    except BaseException as error:
        record('failed',error=repr(error));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','aggregate','cache'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collection',type=Path);p.add_argument('--ticks',type=int,nargs='+',default=[46,58,70,84,120,180])
    p.add_argument('--reuse-index',type=Path)
    p.add_argument('--payload-audit-memo',action='store_true',help='Reuse only SHA digests fully computed in this cache process')
    p.add_argument('--round',type=int,default=1);p.add_argument('--family-offset',type=int,default=0)
    p.add_argument('--stable-delays',type=int,nargs='+');p.add_argument('--fallback-tick',type=int,default=360);a=p.parse_args()
    if a.payload_audit_memo and a.stage!='cache':p.error('--payload-audit-memo applies only to cache')
    if a.stage=='prepare':prepare_collection(a.root.resolve(),a.output.resolve(),a.ticks,a.round,
        stable_delays=a.stable_delays,fallback_tick=a.fallback_tick,family_offset=a.family_offset)
    elif a.stage=='aggregate':prepare_aggregate(a.root.resolve(),a.collection.resolve(),a.output.resolve())
    else:cache(a.output.resolve(),reuse_index=a.reuse_index,payload_audit_memo=a.payload_audit_memo)


if __name__=='__main__':main()
