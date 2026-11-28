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


@pytest.fixture(scope='module')
def corrected_curve(paired):
    source, _ = paired
    data = {role:copy.deepcopy(d) for role,d in source.items()}
    records, sampling = {}, {}
    for role, d in data.items():
        full = d['paired_pool']['shared_full_ids']
        corrected = ['corrected'+str(j) for j in range(24)]
        aliases = {}
        for j, identity in enumerate(corrected):
            e = copy.deepcopy(d['entries'][full[j]])
            e.update(path='/corrected/'+identity, episode=identity, source_sha256=str(j)*64,
                layout_verified=True, actual_layout='verified100mm', capture_family='capture'+str(j))
            d['entries'][identity] = e
            aliases[e['capture_family']] = e['family']
        d['splits']['train'] += corrected
        # These historical fields intentionally share the exact full list.
        d['paired_pool']['shared_full_ids'] += corrected
        d['full_curve'] = dict(condition='plus_corrected24', corrected_full_ids=corrected,
            family_aliases=aliases, physical_new_families=0,
            old_corrections_preserved=True, terminal64_used=False)
        records[role] = {i:dict(robot=torch.zeros(3,1), supervision_mask=torch.tensor([False,True,True]))
            for i in d['splits']['train']}
        sampling[role] = dict(version='paired_dagger_replay_v1', extra_replays=dict.fromkeys(d['splits']['train'],0))
    decks = build_pair(data, records, dict(A='curveA',B='curveB'), sampling_by_role=sampling, epochs=30, curve=True)
    config = {r:dict(fresh_data_contract=FRESH_PAIRED_AGGREGATE_CONTRACT,
        groups=[r], paired_group_deck=decks[r]) for r in 'AB'}
    return data, config


def test_corrected_curve_accepts_same_family_verified_layout_without_refitting(corrected_curve):
    data, config = corrected_curve
    for role in 'AB':
        assert residual_normalization(data[role], config[role]) == 'u_expert_fresh_train_only_affine_v1'


@pytest.mark.parametrize('mutation', ['implicit','wrong_condition','layout','speed','family','seed','alias','independence','sealed','lost_corrections'])
def test_corrected_curve_rejects_unverified_or_mislabelled_extensions(corrected_curve, mutation):
    ds, cs = corrected_curve
    d, c = copy.deepcopy(ds['A']), copy.deepcopy(cs['A'])
    e = d['entries']['corrected0']
    if mutation=='implicit': d.pop('full_curve')
    elif mutation=='wrong_condition': d['full_curve']['condition']='old_only'
    elif mutation=='layout': e['layout_verified']=False
    elif mutation=='speed': e['expert_speed']={}
    elif mutation=='family': e['family']='family_old0'
    elif mutation=='seed': e['seed']=1000
    elif mutation=='alias': e['capture_family']='unknown'
    elif mutation=='independence': d['full_curve']['physical_new_families']=24
    elif mutation=='sealed': d['full_curve']['terminal64_used']=True
    elif mutation=='lost_corrections': d['full_curve']['old_corrections_preserved']=False
    with pytest.raises(ValueError): residual_normalization(d,c)
