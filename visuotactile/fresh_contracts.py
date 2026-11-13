"""Explicit fresh-cohort identities; historical checkpoints retain their meaning."""

FRESH_DATA_CONTRACT = 'fresh30_every_native_visual_v1'


def residual_normalization(data, configuration):
    if configuration.get('fresh_data_contract') is None:
        return 'u_expert_train166_affine_v1'
    if configuration['fresh_data_contract'] != FRESH_DATA_CONTRACT:
        raise ValueError('Unknown fresh data identity')
    splits = data['splits']
    if (data.get('visual_stride') != 1 or len(splits['train']) != 30
            or len(splits['validation']) != 6 or splits.get('test')
            or len(set(splits['train'] + splits['validation'])) != 36
            or data.get('normalization_scope') != 'selected train30 only'):
        raise ValueError('Fresh initial cohort requires disjoint train30/validation6 and every-tick vision')
    if data.get('normalization_identity') != data.get('supervision', '') + '_fresh_train_only_affine_v1':
        raise ValueError('Fresh DP normalizer identity differs')
    return 'u_expert_fresh_train_only_affine_v1'
