"""The paired opt-in admits full experts without weakening old aggregate gates."""
import copy

import pytest
import torch

from visuotactile.fresh_contracts import (FRESH_AGGREGATE_CONTRACT,
    FRESH_PAIRED_AGGREGATE_CONTRACT, residual_normalization)
from visuotactile.paired_training_decks import build_pair


@pytest.fixture(scope='module')
def paired():
    original = [f'old{i}' for i in range(180)]
    full = [f'new{i}' for i in range(24)]
    validation = [f'val{i}' for i in range(6)]
    speed = dict(kind='closed_loop', arm=.5, hand=1., phase='all', hand_target_cap_rad_s=2.)
    registry = dict(version='preregistered_training_families_v1', sha256='f'*64)
    data, records, sampling = {}, {}, {}
    for role in 'AB':
        own = [role+'corr']
        train = original + full + own
        entries = {i:dict(path='/actual/'+i, episode=i, source_sha256='e'*64,
            trajectory_kind='expert_takeover_v1' if i in original[30:] or i in own else 'expert_demonstration_v1',
            family='family_'+i, split='validation' if i in validation else 'train')
            for i in train + validation}
        for j, i in enumerate(full):
            entries[i].update(trajectory_kind='full_expert_demonstration_v1', seed=j, expert_speed=speed)
        entries[own[0]].update(source_group=role, family=entries[full[0]]['family'], seed=0)
        pool = dict(version='paired_own_training_pool_v1', role=role,
            original_ids=original, shared_full_ids=full, own_correction_ids=own)
        data[role] = dict(visual_stride=1, supervision='q_reached' if role=='A' else 'u_expert',
            normalization_identity=('q_reached' if role=='A' else 'u_expert')+'_fresh_train_only_affine_v1',
            normalization_train_ids=original[:30],
            normalization_scope='unchanged original30 training affine; no aggregate refit',
            splits=dict(train=train, validation=validation, test=[]), entries=entries, paired_pool=pool,
            aggregation=dict(version='expert_takeover_aggregate_v1', paired_pool_version='paired_own_student_pool_v1',
                student_prefix_supervised=False, original_training_ids=original,
                shared_full_training_ids=full, new_training_ids=own),
            pool_partitions=dict(old_full=original[:30], old_correction=original[30:], new_full=full, own_new_correction=own),
            training_expansion_registries=[registry],
            paired_pool_provenance=dict(version='paired_own_student_pool_v1', role=role,
                original180_ids=original, original30_affine_ids=original[:30], shared_full_ids=full,
                own_correction_ids=own, student_prefix_supervised=False, heldout_trajectories_opened=0,
                registry_binding=registry))
        records[role] = {i:dict(robot=torch.zeros(3,1), supervision_mask=torch.tensor([False, True, True])) for i in train}
        sampling[role] = dict(version='paired_dagger_replay_v1', extra_replays=dict.fromkeys(train,0))
    decks = build_pair(data, records, dict(A='dataA',B='dataB'), sampling_by_role=sampling)
    return data, {r:dict(fresh_data_contract=FRESH_PAIRED_AGGREGATE_CONTRACT,
        groups=[r], paired_group_deck=decks[r]) for r in 'AB'}


def test_explicit_paired_population_preserves_common_command_affine(paired):
    data, config = paired
    for role in 'AB':
        assert residual_normalization(data[role], config[role]) == 'u_expert_fresh_train_only_affine_v1'
        with pytest.raises(ValueError):
            residual_normalization(data[role], dict(fresh_data_contract=FRESH_AGGREGATE_CONTRACT))


@pytest.mark.parametrize('mutation', ['no_deck','wrong_role','scope','refit','partition','registry',
    'full_kind','speed','old_family','duplicate_family','own_family','foreign_owner','prefix','heldout','source'])
def test_paired_contract_rejects_unregistered_or_mixed_population(paired, mutation):
    ds, cs = paired
    data, config = copy.deepcopy(ds['A']), copy.deepcopy(cs['A'])
    if mutation=='no_deck': config.pop('paired_group_deck')
    elif mutation=='wrong_role': config['groups']=['B']
    elif mutation=='scope': data['normalization_scope']='refit aggregate'
    elif mutation=='refit': data['normalization_identity']='new affine'
    elif mutation=='partition': data['aggregation']['shared_full_training_ids']=[]
    elif mutation=='registry': data['training_expansion_registries']=[]
    elif mutation=='full_kind': data['entries']['new0']['trajectory_kind']='unverified_student'
    elif mutation=='speed': data['entries']['new0']['expert_speed']={}
    elif mutation=='old_family': data['entries']['new0']['family']='family_old0'
    elif mutation=='duplicate_family': data['entries']['new1']['family']='family_new0'
    elif mutation=='own_family': data['entries']['Acorr']['family']='family_old0'
    elif mutation=='foreign_owner': data['entries']['Acorr']['source_group']='B'
    elif mutation=='prefix': data['aggregation']['student_prefix_supervised']=True
    elif mutation=='heldout': data['splits']['validation'][0]='new0'
    elif mutation=='source': data['entries']['new0']['source_sha256']='changed'
    with pytest.raises(ValueError): residual_normalization(data, config)
