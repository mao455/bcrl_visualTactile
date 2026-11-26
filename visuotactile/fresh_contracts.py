"""Explicit fresh-cohort identities; historical checkpoints retain their meaning."""

FRESH_DATA_CONTRACT = 'fresh30_every_native_visual_v1'
FRESH_AGGREGATE_CONTRACT = 'fresh30_actual_takeover_aggregate_v1'
FRESH_PAIRED_AGGREGATE_CONTRACT = 'fresh30_paired_full_and_own_takeover_aggregate_v1'


def _paired_population(data, configuration):
    """Admit the explicit paired pool; historical takeover-only paths stay strict.

    This only verifies the already materialized population and deck. Source
    execution, labels and masks are audited when the immutable pool is built;
    their source/data/mask bindings are checked again by the backend loader.
    No affine or supervision is computed here.
    """
    from .paired_training_decks import validate_pool, validate_deck
    groups = configuration.get('groups')
    if groups not in (['A'], ['B']):
        raise ValueError('Paired normalization requires one explicit own group')
    pool = validate_pool(data, role=groups[0])
    validate_deck(configuration.get('paired_group_deck'), data, role=groups[0])
    original, full, own = (pool[k] for k in ('original_ids', 'shared_full_ids', 'own_correction_ids'))
    affine = data['normalization_train_ids']
    aggregate = data['aggregation']
    provenance = data.get('paired_pool_provenance', {})
    expected_parts = dict(old_full=affine,
        old_correction=[i for i in original if i not in set(affine)],
        new_full=full, own_new_correction=own)
    if (len(original) != 180 or len(full) != 24
            or data.get('normalization_scope') != 'unchanged original30 training affine; no aggregate refit'
            or aggregate.get('paired_pool_version') != 'paired_own_student_pool_v1'
            or aggregate.get('original_training_ids') != original
            or aggregate.get('shared_full_training_ids') != full
            or aggregate.get('new_training_ids') != own
            or data.get('pool_partitions') != expected_parts
            or provenance.get('version') != 'paired_own_student_pool_v1'
            or provenance.get('role') != groups[0]
            or provenance.get('original180_ids') != original
            or provenance.get('original30_affine_ids') != affine
            or provenance.get('shared_full_ids') != full
            or provenance.get('own_correction_ids') != own
            or provenance.get('student_prefix_supervised') is not False
            or provenance.get('heldout_trajectories_opened') != 0
            or not provenance.get('registry_binding')
            or provenance['registry_binding'] not in data.get('training_expansion_registries', [])):
        raise ValueError('Paired aggregate requires its registered original180/shared24/own partitions and unchanged affine')
    entries = data['entries']
    speed = dict(kind='closed_loop', arm=.5, hand=1., phase='all', hand_target_cap_rad_s=2.)
    families = {}
    old_families = {entries[i].get('family') for i in original + data['splits']['validation']}
    for identity in full:
        entry = entries[identity]
        family = entry.get('family')
        if (entry.get('trajectory_kind') != 'full_expert_demonstration_v1'
                or entry.get('expert_speed') != speed or entry.get('split') != 'train'
                or not family or family in old_families or family in families
                or type(entry.get('seed')) is not int):
            raise ValueError('Paired shared full experts require24 new families and the original speed contract')
        families[family] = entry['seed']
    for identity in own:
        entry = entries[identity]
        if entry.get('split') != 'train' or families.get(entry.get('family')) != entry.get('seed'):
            raise ValueError('Paired own takeovers must belong to the registered new training families')


def residual_normalization(data, configuration):
    if configuration.get('fresh_data_contract') is None:
        return 'u_expert_train166_affine_v1'
    identity = configuration['fresh_data_contract']
    if identity not in (FRESH_DATA_CONTRACT, FRESH_AGGREGATE_CONTRACT, FRESH_PAIRED_AGGREGATE_CONTRACT):
        raise ValueError('Unknown fresh data identity')
    splits = data['splits']
    if (data.get('visual_stride') != 1 or len(splits['validation']) != 6 or splits.get('test')
            or len(set(splits['train'] + splits['validation'])) != len(splits['train'])+6):
        raise ValueError('Fresh initial cohort requires disjoint train30/validation6 and every-tick vision')
    if identity == FRESH_DATA_CONTRACT:
        if len(splits['train']) != 30 or data.get('normalization_scope') != 'selected train30 only':
            raise ValueError('Initial train30 statistics scope differs')
    elif identity == FRESH_PAIRED_AGGREGATE_CONTRACT:
        _paired_population(data, configuration)
    else:
        original=data.get('normalization_train_ids',[])
        extra=set(splits['train'])-set(original)
        if (len(original)!=30 or len(set(original))!=30 or not set(original)<=set(splits['train']) or not extra
                or data.get('normalization_scope')!='unchanged original training affine; no aggregate refit'
                or data.get('aggregation',{}).get('version')!='expert_takeover_aggregate_v1'
                or data.get('aggregation',{}).get('student_prefix_supervised') is not False
                or any(data.get('entries',{}).get(i,{}).get('trajectory_kind')!='expert_takeover_v1' for i in extra)):
            raise ValueError('Fresh aggregation requires fixed original30 affine and actual expert suffixes')
    if data.get('normalization_identity') != data.get('supervision', '') + '_fresh_train_only_affine_v1':
        raise ValueError('Fresh DP normalizer identity differs')
    return 'u_expert_fresh_train_only_affine_v1'
