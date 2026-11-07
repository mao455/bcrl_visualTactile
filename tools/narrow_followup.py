"""Retained historical decision/compatibility helpers for existing controls.

Collection, training and publication owners are retired; use tools.tight_finish.
Full historical implementation remains in Git b85738e.
"""
from tools.runtime import read, proc


def predecessor_released(parent):
    status=read(parent/'status.json')
    if status['state'] in ('blocked','paused'):
        raise RuntimeError('Predecessor did not complete normally: '+str(status.get('last_error')))
    if not (parent/'PIPELINE_COMPLETE.json').exists() or not status.get('owned_workers_exited'):
        return False
    if status['state']!='complete':raise RuntimeError('Contradictory predecessor completion')
    owner=read(parent/'owner.json')
    for name in ('owner','worker','monitor'):
        expected=owner.get(name)
        if expected:
            actual=proc(expected['pid'])
            if actual and actual['start_ticks']==expected['start_ticks']:return False
    return True


def dataset_union(original, additions):
    old={e['id']:e for e in original}
    if len(old)!=len(original):raise ValueError('Duplicate original episode')
    split={e['family']:e['split'] for e in original}
    for e in additions:
        if e['id'] in old:raise ValueError('Duplicate/overwritten episode')
        if e['family'] in split and split[e['family']]!=e['split']:raise ValueError('Family leakage')
        old[e['id']]=e;split[e['family']]=e['split']
    return list(old.values())
