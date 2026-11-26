"""CPU registration and exact exposure preflight for one paired DAgger round."""
import argparse
import copy
from pathlib import Path
import shutil
import time

from tools.tight_paired_dagger import ROOT, BASE, CONFIG, AUDIT, read, session_at, active_registry
from tools.tight_paired_learning import PARENTS
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

SHARED = ROOT/'shared_demo_preparation'
LEARNING = ROOT/'learning'
SHARED_EXPERTS = ROOT/'experts_shared'


def merge_experts():
    assert not SHARED_EXPERTS.exists()
    folders=[ROOT/'experts', ROOT/'experts_admission01']
    entries={};rows=[];sources=[]
    for folder in folders:
        receipt=read(folder/'result.json')
        assert receipt['software_complete'] and read(ROOT/'owners'/folder.name/'project_exit.json')['children_reaped']
        new=read(folder/'expert_entries.json');assert not set(entries)&set(new)
        entries.update(new);rows.extend(read(folder/'results.json'))
        sources.append(dict(path=str(folder),result_sha256=sha256(folder/'result.json'),entries_sha256=sha256(folder/'expert_entries.json')))
    expected=set(read(active_registry())['families'])
    assert len(entries)==24 and {e['family'] for e in entries.values()}==expected
    SHARED_EXPERTS.mkdir()
    atomic_json(SHARED_EXPERTS/'expert_entries.json',entries)
    atomic_json(SHARED_EXPERTS/'results.json',rows)
    atomic_json(SHARED_EXPERTS/'result.json',dict(software_complete=True,families=24,attempts=len(rows),accepted=24,
        rejected_before_action=sum(bool(r['result'].get('not_started')) for r in rows),sources=sources,utc=time.time()))


def copy_contracts(destination):
    for name in ('metadata.json','frozen_visual_io.json','frozen_visual_hotset.json','cuda_allocator_policy.json',
                 'effective_exposure.json','bounded_extension.json','training_sampling.json','entry_exposure.json','visual_arithmetic.json'):
        shutil.copyfile(BASE/name,destination/name)
    diagnostic=PARENTS/'family_phase_diagnostics.json'
    if diagnostic.exists() and destination!=PARENTS:
        shutil.copyfile(diagnostic,destination/'family_phase_diagnostics.json')


def prepare_shared():
    from tools.tight_paired_pool import _materialize
    from visuotactile.training_expansion import registry_binding,read_registry
    assert not SHARED.exists()
    owner=ROOT/'owners/shared_demo_cache';session_at(owner)
    session=read(ROOT/'session.json');session.update(resource_root=str(SHARED),resource_policy='bounded_mean_v1')
    atomic_json(owner/'session.json',dict(session,resource_root=str(owner)))
    SHARED.mkdir();copy_contracts(SHARED);atomic_json(SHARED/'session.json',session)
    (SHARED/'shared_resources.json').symlink_to(owner/'shared_resources.json')
    full=read(SHARED_EXPERTS/'expert_entries.json');binding=registry_binding(active_registry());registered=read_registry(binding)
    for role in 'AB':
        child=SHARED/role;child.mkdir()
        _materialize(BASE/role/'state_data',child/'state_data',read(BASE/role/'state_data/data.json'),full,{},role,registered,binding)
        atomic_json(child/'session.json',session)
        (child/'shared_resources.json').symlink_to(owner/'shared_resources.json')
        (child/'resources').symlink_to(owner/'resources',target_is_directory=True)
        atomic_json(child/'gpu_schedule.json',dict(target_percent=40.,admission_percent=45.,min_gpu_free_mib=6144,min_ram_gib=14.))
        shutil.copyfile(BASE/'visual_arithmetic.json',child/'visual_arithmetic.json')
        atomic_json(child/'continuation.json',dict(reuse_index=str(BASE/role/'cache_index.json')))
    job=dict(version='tight_project_job_v1',operation='prepare',resource_root=str(owner),stages=[dict(entry='visual_cache',
        args=['--root',str(SHARED),'--output',str(SHARED),'--payload-audit-memo'])])
    atomic_json(CONFIG/'shared_demo_cache.json',job)
    atomic_json(SHARED/'prepared.json',dict(utc=time.time(),shared_experts=str(SHARED_EXPERTS),epochs=0,
        purpose='Reuse frozen original180 and extract shared24 features once while sustained training waits for external batch',
        resource_policy='bounded total GPU mean target40%; memory14GiB RAM/6GiB VRAM; no external process mutation'))


def complete_parents():
    import torch
    from visuotactile.state_benchmark import dagger_policy_identity
    pairs={};ages={}
    for role in 'AB':
        front=PARENTS/role/'training/best.pt';rear=PARENTS/role/'backend'/role/'training/best.pt'
        f=torch.load(front,map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(rear,map_location='cpu',weights_only=False,mmap=True)
        assert f['epoch']==b['epoch']==60 and f['updates']==7200
        assert b['configuration']['base_sha256']==sha256(front)
        assert read(PARENTS/role/'training/deployment.json')['checkpoint_sha256']==sha256(front)
        assert read(rear.parent/'deployment.json')['checkpoint_sha256']==sha256(rear)
        pairs[role]=dagger_policy_identity(b,str(rear))
        ages[role]=dict(initialization='Common deterministic seed; zero historical trainable ancestors',frontend_epochs=60,
            frontend_updates=f['updates'],backend_epochs=60,backend_updates=b['updates'],frontend_sha256=sha256(front),backend_sha256=sha256(rear))
    assert ages['A']['backend_updates']==ages['B']['backend_updates']
    old=ROOT/'parent_pairs.json';historical=ROOT/'historical_entry50_rear60_pairs.json'
    assert not historical.exists();shutil.copyfile(old,historical)
    atomic_json(old,pairs)
    atomic_json(ROOT/'actual_parent_ages.json',dict(roles=ages,historical_reference=str(historical),utc=time.time()))


def prepare_learning():
    from tools.tight_paired_pool import build_pool
    from visuotactile.latest_takeover_sampling import sampling_plan
    from visuotactile.trajectory_training import profile,budget
    from visuotactile.native_data import visual_records
    from visuotactile.paired_training_decks import build_pair,validate_pool_pair
    assert not LEARNING.exists()
    assert (ROOT/'actual_parent_ages.json').exists()
    owner=ROOT/'owners/main_train';session_at(owner)
    session=read(ROOT/'session.json');session.update(resource_root=str(LEARNING))
    LEARNING.mkdir();copy_contracts(LEARNING)
    atomic_json(LEARNING/'session.json',session);(LEARNING/'shared_resources.json').symlink_to(owner/'shared_resources.json')
    pairs=read(ROOT/'parent_pairs.json');data={};records={};sampling={};front_reg=dict(profile=profile(),roles={})
    for role in 'AB':
        child=LEARNING/role;child.mkdir();atomic_json(child/'session.json',session)
        (child/'shared_resources.json').symlink_to(LEARNING/'shared_resources.json')
        (child/'resources').symlink_to(LEARNING/'resources',target_is_directory=True)
        shutil.copyfile(BASE/'visual_arithmetic.json',child/'visual_arithmetic.json')
        data[role]=build_pool(role,child/'state_data',ROOT/'collection',experts=SHARED_EXPERTS)
        pair=pairs[role]
        atomic_json(child/'continuation.json',dict(path=pair['frontend'],sha256=pair['frontend_sha256'],
            statistics_sha256=sha256(child/'state_data/statistics.pt'),source_root=str(PARENTS/role),
            reuse_index=str(SHARED/role/'cache_index.json'),target_window_contract='recovery_onset_commit12_v1',learning_rate=1.5e-5,
            initialization='Selected matched parent EMA; fresh AdamW/EMA; original30 affine',
            previous_backend=pair['backend'],previous_backend_sha256=pair['backend_sha256']))
        records[role]=visual_records(child/'state_data',data[role]['splits']['train'],'G00',target_window_contract='recovery_onset_commit12_v1')
        sampling[role]=dict(version='paired_dagger_replay_v1',extra_replays=dict.fromkeys(data[role]['splits']['train'],0))
    validate_pool_pair(data['A'],data['B'],role='A',enabled=True)
    base_counts={r:sampling_plan(data[r],sampling[r])[1] for r in 'AB'}
    visits=max(sum(c.values()) for c in base_counts.values())
    if visits%2:visits+=1
    for role in 'AB':
        # Explicit extra real histories align B2 update budgets; no dummy lanes.
        ids=data[role]['paired_pool']['original_ids']
        missing=visits-sum(base_counts[role].values())
        for k in range(missing):sampling[role]['extra_replays'][ids[k%len(ids)]]+=1
        contract,counts=sampling_plan(data[role],sampling[role])
        cont=read(LEARNING/role/'continuation.json');cont['sampling_contract']=contract
        atomic_json(LEARNING/role/'continuation.json',cont)
        front_reg['roles'][role]=dict(**budget(sum(counts.values()),60),data_sha256=sha256(LEARNING/role/'state_data/data.json'),episode_multiplicities=counts)
    atomic_json(LEARNING/'full_trajectory_batch.json',front_reg)
    # The backend has its own exact 32chunk/128window actual update accounting.
    rear_sampling={r:dict(version='paired_dagger_replay_v1',extra_replays=dict.fromkeys(data[r]['splits']['train'],0)) for r in 'AB'}
    atomic_json(LEARNING/'backend_episode_sampling.json',rear_sampling)
    decks=build_pair(data,records,{r:sha256(LEARNING/r/'state_data/data.json') for r in 'AB'},sampling_by_role=rear_sampling)
    for r in 'AB':atomic_json(LEARNING/f'paired_group_deck_{r}.json',decks[r])
    atomic_json(LEARNING/'matched_learning_rates.json',{r:[1.5e-5]*48+[7.5e-6]*12 for r in 'AB'})
    atomic_json(LEARNING/'matched_backend_learning_rates.json',{r:[1e-4]*48+[5e-5]*12 for r in 'AB'})
    atomic_json(LEARNING/'gate.json',dict(passed=True,inherited_input_gate=str(BASE/'gate.json'),inherited_input_gate_sha256=sha256(BASE/'gate.json'),new_capture_audits=str(ROOT/'collection')))
    atomic_json(LEARNING/'budget_registration.json',dict(utc=time.time(),epochs=60,deploy_epoch=60,frontend=front_reg,
        backend_updates_per_epoch={r:decks[r]['updates_per_epoch'] for r in 'AB'},fixed_selection=True,
        supervision='Shared full demonstrations and own legal expert suffixes; every original trajectory retained',ancestors_counted=False))


def exposure_preflight(root):
    import collections
    import torch
    from visuotactile.native_data import visual_records
    from visuotactile.anchor_sampling import supervision_rows,sample_training_anchors
    from visuotactile.latest_takeover_sampling import sampling_plan,episode_deck
    from visuotactile.effective_exposure import Ledger,dp_ticks
    from visuotactile.entry_exposure import EntryLedger
    from visuotactile.paired_training_decks import group_windows,validate_deck
    root=Path(root);rows={}
    for role in 'AB':
        child=root/role;data=read(child/'state_data/data.json');cont=read(child/'continuation.json')
        records=visual_records(child/'state_data',data['splits']['train'],'G00',target_window_contract=cont['target_window_contract'])
        _,counts=sampling_plan(data,cont['sampling_contract'])
        masks={i:supervision_rows(r)[0] for i,r in records.items()}
        ledger=Ledger(masks);entries=EntryLedger(records);generator=torch.Generator().manual_seed(2026111402)
        anchors_profile=read(root/'training_sampling.json')['anchors'];update=0;per_epoch=[]
        for epoch in range(1,61):
            order=episode_deck(data['splits']['train'],counts,generator);before=update
            for offset in range(0,len(order),2):
                update+=1;contributions=[];selections=[]
                for pos in range(offset,min(offset+2,len(order))):
                    identity=order[pos];record=records[identity]
                    anchors,_=sample_training_anchors(record,anchors_profile,identity=identity,epoch=epoch,visit_index=pos+1)
                    contributions.append((identity,dp_ticks(record,anchors)));selections.append((identity,anchors))
                ledger.step(update,contributions);entries.step(update,entries.contributions(records,selections))
            per_epoch.append(update-before)
        front=ledger.summary({i:60*counts[i] for i in records});entry=entries.summary()
        assert front['eligible'] and entry['eligible']
        rear=Ledger(masks);visits=collections.Counter();rear_updates=0;rear_per_epoch=[]
        deck_path=root/f'paired_group_deck_{role}.json'
        if deck_path.exists():
            deck=validate_deck(read(deck_path),data,records,data_sha256=sha256(child/'state_data/data.json'),role=role)
            groups_by_epoch=deck['epoch_groups']
        else:
            _,rear_counts=sampling_plan(data,read(root/'backend_episode_sampling.json'))
            generator=torch.Generator().manual_seed(2026111404);groups_by_epoch=[]
            for _ in range(60):
                order=episode_deck(data['splits']['train'],rear_counts,generator)
                groups_by_epoch.append([order[j:j+4] for j in range(0,len(order),4)])
        for groups in groups_by_epoch:
            before=rear_updates
            for group in groups:
                visits.update(group)
                for window in group_windows(group,masks):
                    rear_updates+=1;rear.step(rear_updates,list(window['contributions'].items()))
            rear_per_epoch.append(rear_updates-before)
        rear_summary=rear.summary(dict(visits));assert rear_summary['eligible']
        rows[role]=dict(frontend=front,entry=entry,backend=rear_summary,
            frontend_updates_per_epoch=per_epoch,backend_updates_per_epoch=rear_per_epoch,
            data_sha256=sha256(child/'state_data/data.json'))
        print('exposure preflight',root.name,role,update,rear_updates,flush=True)
    for key in ('frontend_updates_per_epoch','backend_updates_per_epoch'):
        assert rows['A'][key]==rows['B'][key],key+' differs across roles'
    atomic_json(root/'complete_exposure_preflight.json',dict(passed=True,roles=rows,epochs=60,deploy_epoch=60,utc=time.time(),ancestors_counted=False))
    return rows


def make_learning_jobs():
    cache_owner=ROOT/'owners/main_cache';session_at(cache_owner)
    shared=LEARNING/'shared_resources.json';shared.unlink();shared.symlink_to(cache_owner/'shared_resources.json')
    (LEARNING/'resources').symlink_to(cache_owner/'resources',target_is_directory=True)
    atomic_json(CONFIG/'main_cache.json',dict(version='tight_project_job_v1',operation='prepare',resource_root=str(cache_owner),
        stages=[dict(entry='visual_cache',args=['--root',str(LEARNING),'--output',str(LEARNING),'--payload-audit-memo'])]))
    stages=[]
    for role in 'AB':stages.append(dict(entry='paired_frontend',args=['--root',str(LEARNING),'--role',role]))
    for role in 'AB':
        backend=LEARNING/role/'backend';child=backend/role
        stages.extend([dict(entry='paired_rear_prepare',args=['--root',str(LEARNING),'--role',role]),
            dict(entry='backend_cache',args=['--root',str(backend)]),dict(entry='paired_rear_train',args=['--root',str(child)]),
            dict(entry='backend_offline',args=['--root',str(child),'--job','image','--output',str(child/'offline')])])
    atomic_json(CONFIG/'main_train.json',dict(version='tight_project_job_v1',operation='train',resource_root=str(ROOT/'owners/main_train'),stages=stages))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['merge-experts','prepare-shared','complete-parents','prepare-learning','preflight-parents','preflight-learning','learning-jobs']);a=p.parse_args()
    {'merge-experts':merge_experts,'prepare-shared':prepare_shared,'complete-parents':complete_parents,'prepare-learning':prepare_learning,'preflight-parents':lambda:exposure_preflight(PARENTS),'preflight-learning':lambda:exposure_preflight(LEARNING),'learning-jobs':make_learning_jobs}[a.mode]()

if __name__=='__main__':main()
