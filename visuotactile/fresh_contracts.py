"""Explicit fresh-cohort identities; historical checkpoints retain their meaning."""

FRESH_DATA_CONTRACT = 'fresh30_every_native_visual_v1'
FRESH_AGGREGATE_CONTRACT = 'fresh30_actual_takeover_aggregate_v1'


def residual_normalization(data, configuration):
    if configuration.get('fresh_data_contract') is None:
        return 'u_expert_train166_affine_v1'
    identity = configuration['fresh_data_contract']
    if identity not in (FRESH_DATA_CONTRACT, FRESH_AGGREGATE_CONTRACT):
        raise ValueError('Unknown fresh data identity')
    splits = data['splits']
    if (data.get('visual_stride') != 1 or len(splits['validation']) != 6 or splits.get('test')
            or len(set(splits['train'] + splits['validation'])) != len(splits['train'])+6):
        raise ValueError('Fresh initial cohort requires disjoint train30/validation6 and every-tick vision')
    if identity == FRESH_DATA_CONTRACT:
        if len(splits['train']) != 30 or data.get('normalization_scope') != 'selected train30 only':
            raise ValueError('Initial train30 statistics scope differs')
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
