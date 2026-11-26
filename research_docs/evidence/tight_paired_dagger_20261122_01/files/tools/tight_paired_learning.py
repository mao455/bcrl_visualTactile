"""Finite matched-parent reconstruction and final-epoch deployment wrappers."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.tight_paired_dagger import ROOT, BASE, CONFIG, AUDIT, PROJECT, read, session_at
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

PARENTS = ROOT / 'common180_parents'


def freeze_final(directory, kind):
    import torch
    from visuotactile.effective_exposure import Ledger
    directory = Path(directory)
    final = torch.load(directory/'last.pt', map_location='cpu', weights_only=False, mmap=True)
    assert final['epoch'] == 60 and final['updates'] > 0
    state = final['effective_exposure_state']
    # Ledger's state includes actual committed-step counts, never ancestors.
    visits = final['sampling_exposure'] if kind == 'frontend' else final['episode_counts']
    exposure = Ledger(state['legal'], state).summary(visits)
    assert exposure['eligible'] and exposure['minimum_tick_steps'] >= 50
    if kind == 'frontend':
        assert min(final['sampling_exposure'].values()) >= 50
        entry = final['entry_exposure_state']
        assert entry['updates'] == final['updates']
        slot_counts = [n for counts in entry['counts'].values() for n in counts.values()]
        assert slot_counts and min(slot_counts) >= 50, 'Selected DP native correction entry exposure insufficient'
    else:
        assert min(final['episode_counts'].values()) >= 50
    receipt = directory/'deployment.json'
    if receipt.exists():
        assert read(receipt)['checkpoint_sha256'] == sha256(directory/'best.pt')
        return
    for name in ('best.pt', 'selected.json'):
        source = directory/name
        if source.exists():
            destination = directory/('validation_'+name)
            assert not destination.exists()
            source.rename(destination)
    # Do not alias a training checkpoint that could later be atomically replaced.
    shutil.copyfile(directory/'last.pt', directory/'best.pt')
    row = dict(epoch=60, update=final['updates'], checkpoint_sha256=sha256(directory/'best.pt'),
        selection='preregistered_epoch60_only', validation_used_for_selection=False,
        final_checkpoint=str(directory/'last.pt'), final_sha256=sha256(directory/'last.pt'))
    if kind == 'backend':
        row.update(initialization_candidate=False, base_checkpoint_sha256=final['configuration']['base_sha256'])
    atomic_json(directory/'selected.json', row)
    atomic_json(receipt, row)


def prepare_baseline():
    import torch
    from visuotactile.models.native_dp import NativePolicy
    from visuotactile.position_control import continuous_image_contract
    from visuotactile.trajectory_training import profile, budget
    from visuotactile.latest_takeover_sampling import sampling_plan
    assert not PARENTS.exists(), 'Preserve every attempted baseline identity'
    PARENTS.mkdir()
    owner = ROOT/'owners/parent_train'
    session_at(owner)
    session = read(ROOT/'session.json')
    session.update(resource_root=str(PARENTS), trainable_initialization='Shared initialization seed2026111401; common original180; no historical trainable weights',
        authorization='Reconstruct A/B complete parents with common180 and matched training ages, then collect one own-student DAgger round')
    atomic_json(PARENTS/'session.json', session)
    (PARENTS/'shared_resources.json').symlink_to(owner/'shared_resources.json')
    for name in ('metadata.json','gate.json','frozen_visual_io.json','frozen_visual_hotset.json',
                 'backend_episode_sampling.json','cuda_allocator_policy.json','effective_exposure.json',
                 'bounded_extension.json','training_sampling.json','entry_exposure.json','visual_arithmetic.json'):
        shutil.copyfile(BASE/name, PARENTS/name)
    atomic_json(PARENTS/'matched_learning_rates.json', {r:[1.5e-5]*48+[7.5e-6]*12 for r in 'AB'})
    atomic_json(PARENTS/'matched_backend_learning_rates.json', {r:[1e-4]*48+[5e-5]*12 for r in 'AB'})
    registration = dict(profile=profile(), roles={})
    initials = {}
    for role in 'AB':
        child=PARENTS/role; child.mkdir()
        (child/'state_data').symlink_to(BASE/role/'state_data', target_is_directory=True)
        (child/'shared_resources.json').symlink_to(PARENTS/'shared_resources.json')
        shutil.copyfile(BASE/role/'cache_index.json', child/'cache_index.json')
        atomic_json(child/'session.json',session)
        statistics=torch.load(child/'state_data/statistics.pt',map_location='cpu',weights_only=False)
        torch.manual_seed(2026111401)
        model=NativePolicy('I-HistoryDP',24,modality='G00')
        model.contract=continuous_image_contract('q_reached' if role=='A' else 'u_expert')
        for key,value in statistics.items():getattr(model,key).copy_(value)
        initial=child/'common_seed_initialization.pt'
        torch.save(dict(contract=model.contract,ema=model.state_dict(),model=model.state_dict(),updates=0,epoch=0,
            initialization_seed=2026111401,ancestors=[]),initial)
        del model
        continuation=dict(path=str(initial),sha256=sha256(initial),statistics_sha256=sha256(child/'state_data/statistics.pt'),
            source_root=str(BASE/role),reuse_index=str(BASE/role/'cache_index.json'),
            target_window_contract='recovery_onset_commit12_v1',learning_rate=1.5e-5,
            initialization='Identical seed2026111401 random trainable parameters; fresh AdamW/EMA; common180 and frozen original30 affine',
            sampling_contract=read(PARENTS/'backend_episode_sampling.json'))
        atomic_json(child/'continuation.json',continuation)
        data=read(child/'state_data/data.json');_,counts=sampling_plan(data,continuation['sampling_contract'])
        assert len(data['splits']['train'])==180
        registration['roles'][role]=dict(**budget(sum(counts.values()),60),data_sha256=sha256(child/'state_data/data.json'),episode_multiplicities=counts)
        initials[role]=dict(path=str(initial),sha256=sha256(initial),updates=0,epoch=0)
    assert registration['roles']['A']['planned_optimizer_updates']==registration['roles']['B']['planned_optimizer_updates']
    atomic_json(PARENTS/'full_trajectory_batch.json',registration)
    stages=[]
    for role in 'AB':stages.append(dict(entry='paired_frontend',args=['--root',str(PARENTS),'--role',role]))
    for role in 'AB':
        backend=PARENTS/role/'backend'
        stages.extend([
            dict(entry='paired_rear_prepare',args=['--root',str(PARENTS),'--role',role]),
            dict(entry='backend_cache',args=['--root',str(backend)]),
            dict(entry='paired_rear_train',args=['--root',str(backend/role)]),
            dict(entry='backend_offline',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'offline')])])
    job=dict(version='tight_project_job_v1',operation='train',resource_root=str(owner),stages=stages)
    atomic_json(CONFIG/'parent_train.json',job)
    reason=dict(version='common180_parent_reconstruction_v1',registered_utc=time.time(),
        reason='Entry current-stage ages can align50/60 but R1 selected ancestor ages differ; old source path missing; reconstruct rather than claim aligned ancestral ages',
        original_parent_registration_preserved=str(ROOT/'parent_pairs.json'),initials=initials,
        train=180,epochs=60,deploy_epoch=60,encoder_frozen=True,
        frontend=registration,backend_initialization_seed=2026111404,
        sampling=dict(old_full_demo=3,old_correction=1),
        learning_rates=read(PARENTS/'matched_learning_rates.json'),backend_learning_rates=read(PARENTS/'matched_backend_learning_rates.json'),
        validation_selection=False,job=str(CONFIG/'parent_train.json'))
    atomic_json(PARENTS/'registration.json',reason)
    atomic_json(ROOT/'parent_reconstruction_amendment.json',reason)
    print(json.dumps(dict(prepared=str(PARENTS),job=str(CONFIG/'parent_train.json'))),flush=True)


def frontend(root,role):
    from tools.tight_fresh_train import train
    from visuotactile.trajectory_training import profile
    train(Path(root),role,60,offline_anchor_batch=64,trajectory_batch_profile=profile())
    freeze_final(Path(root)/role/'training','frontend')


def rear_prepare(root,role):
    from tools.tight_fresh_backend import prepare
    root=Path(root)
    prepare(root,role,60,continue_rear=root!=PARENTS,offline_anchor_batch=64,
        frozen_visual_io=read(root/'frozen_visual_io.json'),
        **(dict(paired_group_deck=read(root/f'paired_group_deck_{role}.json')) if (root/f'paired_group_deck_{role}.json').exists() else {}))


    # Reconstruction uses the same complete sixty epochs, irrespective of its
    # validation trend. No sampling or optimizer math changes here.
    if root==PARENTS:
        destination=root/role/'backend'
        for path in (destination/'session.json',destination/role/'session.json',destination/'configuration.json',destination/role/'configuration.json'):
            value=read(path);v=value['residual_pair'];v.pop('bounded_extension',None)
            v.update(minimum_control_tick_epochs=60,comparison_control_tick_epochs=[60])
            atomic_json(path,value)
        atomic_json(destination/'fixed_epoch_registration.json',dict(epochs=60,deploy_epoch=60,validation_selection=False))


def rear_train(root):
    from visuotactile.anchor_continue import residual_train
    residual_train(Path(root))
    freeze_final(Path(root)/'training','backend')


def project():
    from tools import tight_project
    tight_project.ENTRIES.update(paired_frontend=('train','tools.tight_paired_learning',['frontend']),
        paired_rear_prepare=('train','tools.tight_paired_learning',['rear-prepare']),
        paired_rear_train=('train','tools.tight_paired_learning',['rear-train']))
    sys.argv.pop(1);tight_project.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':return project()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare-baseline','frontend','rear-prepare','rear-train'])
    p.add_argument('--root',type=Path);p.add_argument('--role',choices=['A','B']);a=p.parse_args()
    if a.mode=='prepare-baseline':prepare_baseline()
    elif a.mode=='frontend':frontend(a.root,a.role)
    elif a.mode=='rear-prepare':rear_prepare(a.root,a.role)
    else:rear_train(a.root)

if __name__=='__main__':main()
