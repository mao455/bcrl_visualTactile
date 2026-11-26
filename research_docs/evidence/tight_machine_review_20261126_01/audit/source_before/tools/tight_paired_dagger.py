"""One preregistered paired DAgger round; immutable historical inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
BASE = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01'
CONFIG = PROJECT/'configs/experiments/paired_dagger_20261122'
AUDIT = PROJECT/'outputs/diagnostics/tight_paired_dagger_20261122_01'


def read(path):
    return json.loads(Path(path).read_text())


def active_registry(root=ROOT):
    pointer=Path(root)/'active_training_expansion.json'
    return Path(read(pointer)['path']) if pointer.exists() else Path(root)/'training_expansion_registry.json'


def sources():
    return {str(p.relative_to(PROJECT)):sha256(p) for folder in ('tools','visuotactile')
            for p in sorted((PROJECT/folder).rglob('*.py'))}


def session_at(path):
    path.mkdir(parents=True, exist_ok=True)
    session = copy.deepcopy(read(ROOT/'session.json'))
    session.update(resource_root=str(path), owned_worker_roots=[str(ROOT)])
    atomic_json(path/'session.json', session)
    atomic_json(path/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
    return session


def register():
    import torch
    from visuotactile.state_benchmark import dagger_policy_identity
    from visuotactile.narrow_data import direct_configuration
    from visuotactile.training_expansion import make_registry, registry_binding
    if (ROOT/'registration.json').exists() or (ROOT/'owners').exists():
        raise FileExistsError('Keep every attempted experiment identity')
    ROOT.mkdir(parents=True,exist_ok=True); CONFIG.mkdir(parents=True,exist_ok=True); AUDIT.mkdir(parents=True,exist_ok=True)
    prior=read(ROOT/'session.json') if (ROOT/'session.json').exists() else None
    now=prior['started_utc'] if prior else time.time(); session=read(BASE/'session.json')
    session.update(version='paired_dagger_one_round_v1',started_utc=now,deadline=now+96*3600,
        resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],
        authorization='24 new shared full expert demonstrations; 48 own-student opportunities per A/B; one symmetric complete-student training round; 64 sealed paired final families',
        resource_policy='full_gpu_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    for key in ('collection_deadline','training_deadline','cache_deadline','offline_deadline'):
        session[key]=session['deadline']
    # This is an independent finite session, not an extension of any old run.
    atomic_json(ROOT/'session.json',session)
    pairs={}; ages={}; protected={}
    for role in 'AB':
        front=BASE/role/'training/best.pt';rear=BASE/role/'backend'/role/'training/last.pt'
        f=torch.load(front,map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(rear,map_location='cpu',weights_only=False,mmap=True)
        assert f['epoch']==50 and f['updates']==6000
        assert b['epoch']==60 and b['updates']==22763 and b['configuration']['base_sha256']==sha256(front)
        pairs[role]=dagger_policy_identity(b,str(rear))
        ages[role]=dict(frontend_epoch=f['epoch'],frontend_updates=f['updates'],
            backend_epoch=b['epoch'],backend_updates=b['updates'],
            frontend_continuation=f['configuration'].get('continuation'),
            backend_continuation=b.get('continuation_from'),
            historical_selected_backend=read(BASE/role/'backend'/role/'training/selected.json'))
        for p in (front,rear,BASE/role/'state_data/data.json',BASE/role/'state_data/statistics.pt',
                  BASE/role/'training/last.pt',BASE/role/'backend'/role/'training/best.pt'):
            protected[str(p.resolve())]=sha256(p)
    assert pairs['A']['supervision']=='q_reached' and pairs['B']['supervision']=='u_expert'
    atomic_json(ROOT/'parent_pairs.json',pairs);atomic_json(AUDIT/'parent_audit.json',ages)
    atomic_json(ROOT/'protected.json',protected)
    old_partition=PROJECT/'outputs/visuotactile/run/tight_state_20261107_01/composition_06/state_data/data.json'
    old_frozen=old_partition.parent.parent/'frozen_families.json'
    # The historical original partition predates the10cm family names. Preserve
    # every old holdout and explicitly add the current protected validation6.
    combined=read(old_partition); current=read(BASE/'A/state_data/data.json')
    frozen_rows=read(old_frozen); known={r['family'] for r in frozen_rows}
    for split in ('train','validation'):
        for identity in current['splits'][split]:
            entry=current['entries'][identity];family=entry['family']
            if identity not in combined['entries']:
                combined['entries'][identity]=entry
                combined['splits'][split].append(identity)
            if family not in known:
                seed=entry.get('seed',int(family.rsplit('_',1)[-1]))
                frozen_rows.append(dict(family=family,split=split,seed=seed,
                    narrow_reset=direct_configuration(session['domain'],seed)))
                known.add(family)
    partition=AUDIT/'all_historical_partitions.json';frozen=AUDIT/'all_frozen_families.json'
    atomic_json(partition,combined);atomic_json(frozen,frozen_rows)
    registry=make_registry('paired20261122',session['domain'],list(range(28910000,28910024)),
        {r:BASE/r/'state_data/data.json' for r in 'AB'},partition,frozen,created_utc=now)
    atomic_json(ROOT/'training_expansion_registry.json',registry)
    binding=registry_binding(ROOT/'training_expansion_registry.json')
    configs={}; cohorts={}
    for name,seeds in [('train',range(28910000,28910024)),('development',range(28911000,28911008)),('test',range(28912000,28912064))]:
        rows=[]
        for seed in seeds:
            reset=direct_configuration(session['domain'],seed)
            key=json.dumps(reset,sort_keys=True)
            assert key not in configs
            configs[key]=name
            rows.append(dict(seed=seed,family=f'paired_{name}_{seed}',narrow_reset=reset))
        cohorts[name]=rows
    # Check historical reset registrations and result filenames without opening any test trajectories.
    historical_seeds=set()
    for p in (PROJECT/'research_docs/evidence').rglob('registration.json'):
        text=p.read_text(errors='replace')
        for seed in range(28910000,28912064):
            if str(seed) in text:historical_seeds.add(seed)
    assert not historical_seeds, historical_seeds
    atomic_json(ROOT/'cohorts.json',cohorts)
    registration=dict(version='paired_dagger_one_round_v1',started_utc=now,session_deadline=session['deadline'],
        parents=pairs,parent_ages=ages,train_families=24,student_opportunities_per_role=48,
        opportunities='Each family once support/contact screen and once alignment screen, shared early instability/post-subgoal fallbacks; no fixed-time takeover',
        epochs=60,deploy_epoch=60,minimum_successful_tick_steps=50,
        training_budget='Preflight after collection, same DP updates per epoch and same rear updates per epoch; use larger actual pool and pad with explicit real replays, never masked synthetic updates',
        sampling_weights=dict(old_full_demo=3,old_correction=1,new_full_demo=3,own_new_correction=1),
        learning_rates=dict(dp=[1.5e-5]*48+[7.5e-6]*12,rear=[1e-4]*48+[5e-5]*12),
        inference_sampler_seed=7091,development_selection=False,final_test_families=64,
        milestone='Both A and B strict autonomous insertion success >= 32/64',
        final_software_rule='Every preregistered family in denominator; invalid initialization and software-incomplete are failures in primary table, separately listed; no outcome retries',
        expert_retry_rule='At most 3 independent same-reset expert attempts per train family until first valid full success; retain attempts/failures; never replace a family by outcome',
        inference='Full DP + Transformer, frozen encoder, 120/60/5Hz, DINO microbatch2 duplicate tail, host scheduler, native commit12 and strict native success, cumulative10 confirmation',
        data_policy='Keep original180; share only new full demonstrations, each role owns its new corrections; complete actual prefix as history, clean expert suffix only as targets',
        scope='Whole expanded-initialization plus own-correction plus extra-training package; no data-only causal claim',
        resource_policy='Coexist with external test; nice10 CPU8-15, bounded writers, one own Kit worker initially; reserve12GiB disk/10GiB RAM/4GiB VRAM; record external PIDs without signaling them',
        encoder_probe='Optional CPU readout after main work is covered; no encoder retraining',
        registry_binding=binding,cohorts_sha256=sha256(ROOT/'cohorts.json'),historical_seed_collision_count=0)
    atomic_json(ROOT/'registration.json',registration);atomic_json(CONFIG/'experiment.json',registration)
    # Expert collection has no dependency on training or test outcomes.
    owner=ROOT/'owners/experts';session_at(owner)
    output=ROOT/'experts';session_at(output)
    (output/'shared_resources.json').symlink_to(owner/'shared_resources.json')
    job=dict(version='tight_project_job_v1',operation='prepare',resource_root=str(owner),
        stages=[dict(entry='paired_experts',args=['--output',str(output)])])
    atomic_json(CONFIG/'experts.json',job)
    print(json.dumps(dict(root=str(ROOT),parents=pairs,ages=ages,expert_job=str(CONFIG/'experts.json'))),flush=True)


def experts(output):
    from tools.tight_state import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.narrow_data import record
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.native_evaluation import exit_completed_kit_worker
    output=Path(output);session=read(output/'session.json')
    with (output/'run_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),utc=time.time()),stream)
    app,env,player,lease=environment(output,camera_content_validation=True)
    env.speed_config=dict(ExpertPositionBridge.speed_scheme)
    rows=[]; entries={}
    families=read(active_registry())['families']
    subset=output/'family_subset.json'
    if subset.exists():
        selected=read(subset)
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(families):
            raise ValueError('Explicit nonempty registered expert family subset required')
        families={key:families[key] for key in selected}
    for family,item in families.items():
        for attempt in range(3):
            name=f'full_s{item["seed"]}_attempt{attempt:02d}'
            spec=dict(identity=name,family=family,seed=item['seed'],split='train',cohort='fresh',
                event_seed=item['seed']+100000+attempt,collision_gate='live_collision_mesh_bounds_v1',
                narrow_reset=item['narrow_reset'],layout=session['layout'])
            row=trial(env,lease,output/name,spec,{},player,session['expert_checkpoint'],
                expert_speed=ExpertPositionBridge.speed_scheme,capture_dataset=True,
                capture_compression='lzf',video=False)
            accepted=False
            if row.get('legal') and row.get('software_complete') and row.get('success'):
                path=Path(row['dataset_capture']['path']);visual=inspect_capture(path)
                record(path,episode=name,privileged=False)
                assert visual['passed']
                entries[name]=dict(path=str(path.resolve()),episode=name,length=row['steps'],
                    family=family,split='train',source_sha256=sha256(path),seed=item['seed'],
                    trajectory_kind='full_expert_demonstration_v1',expert_speed=ExpertPositionBridge.speed_scheme)
                accepted=True
            rows.append(dict(family=family,attempt=attempt,case=name,accepted=accepted,result=row))
            atomic_json(output/'results.json',rows);atomic_json(output/'expert_entries.json',entries)
            print(json.dumps(dict(case=name,success=row.get('success'),accepted=accepted,steps=row.get('steps'),completed_families=len(entries))),flush=True)
            if accepted:break
    atomic_json(output/'result.json',dict(software_complete=True,families=len(families),
        attempts=len(rows),accepted=len(entries),utc=time.time()))
    exit_completed_kit_worker(output)


def project_entry():
    from tools import tight_project
    tight_project.ENTRIES['paired_experts']=('prepare','tools.tight_paired_dagger',['experts'])
    sys.argv.pop(1)
    tight_project.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':
        return project_entry()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['register','experts'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='register':register()
    else:experts(args.output)


if __name__=='__main__':
    main()
