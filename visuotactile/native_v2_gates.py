"""Fail-closed scientific admission, separate from software completion."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Gate:
    name: str
    infrastructure_passed: bool
    offline_fit_passed: bool | None
    closed_loop_success: dict | None
    acceptance_passed: bool
    blocked_reason: str | None

    def to_dict(self):
        return asdict(self)


def replay_gate(data, trials, required_ids):
    ids = [r['identity'] for r in trials]
    infrastructure = bool(data['infrastructure_passed']) and len(ids) == len(set(ids)) and set(ids) == set(required_ids)
    control = ('command', 'delayed', 'delay_index', 'target', 'force', 'torque')
    infrastructure &= all(all(r['comparison']['errors'][k]['first_exceeded'] is None for k in control) for r in trials)
    successes = sum(bool(r['success']) for r in trials)
    accepted = infrastructure and len(required_ids) >= 8 and successes == len(required_ids)
    reason = None if accepted else ('Data/control or cohort integrity failed' if not infrastructure else
        'Historical expert replay did not reliably complete native task; new optimizer updates blocked')
    return Gate('G0', infrastructure, None, dict(successes=successes, trials=len(trials), required=len(required_ids)), accepted, reason)


def data_gate(data, control):
    accepted = (data.get('infrastructure_passed') is True and control.get('passed') is True
                and control.get('episodes') == 400 and not data.get('failures'))
    return Gate('G0-A', accepted, None, None, accepted,
                None if accepted else 'Original data, train-only statistics or native command audit failed')


def online_expert_gate(trials, registered_ids, checkpoint_sha256, environment_sha256, minimum_successes=7):
    identities = [row.get('identity') for row in trials]
    integrity = (len(registered_ids) >= 8 and len(set(registered_ids)) == len(registered_ids)
                 and len(identities) == len(set(identities)) and set(identities) == set(registered_ids)
                 and 0 < minimum_successes <= len(registered_ids))
    integrity &= all(row.get('software_complete') is True and row.get('native_termination') is True
                     and row.get('policy') == 'released_online_expert'
                     and row.get('checkpoint_sha256') == checkpoint_sha256
                     and row.get('environment_sha256') == environment_sha256
                     and row.get('fresh_native_reset') is True for row in trials)
    successes = sum(row.get('success') is True for row in trials)
    accepted = integrity and successes >= minimum_successes
    return Gate('G0-B', bool(integrity), None,
                dict(successes=successes, trials=len(trials), required=minimum_successes,
                     checkpoint_sha256=checkpoint_sha256, environment_sha256=environment_sha256),
                bool(accepted), None if accepted else 'Fresh online expert reliability or cohort provenance failed')


def historical_replay_gate(data, trials, registered_ids):
    legacy = replay_gate(data, trials, registered_ids)
    metrics = dict(legacy.closed_loop_success, role='diagnostic_only', required_successes=None)
    return Gate('G0-C', legacy.infrastructure_passed, None, metrics, legacy.infrastructure_passed,
                None if legacy.infrastructure_passed else 'Historical cohort or numerical replay control mismatch')


def infrastructure_gate(numeric, online):
    accepted = (numeric.name == 'G0-A' and online.name == 'G0-B'
                and numeric.acceptance_passed and online.acceptance_passed)
    return Gate('G0', numeric.infrastructure_passed and online.infrastructure_passed, None,
                online.closed_loop_success, bool(accepted),
                None if accepted else 'Both G0-A numerical integrity and G0-B fresh expert reliability are required')


def require_training_foundation(root):
    numeric = Gate(**require_gate(root, 'G0-A'))
    online = Gate(**require_gate(root, 'G0-B'))
    combined = infrastructure_gate(numeric, online)
    if not combined.acceptance_passed:
        raise RuntimeError(combined.blocked_reason)
    from .native_v2_foundation import verify_foundation_provenance
    verify_foundation_provenance(root)
    return combined


def rollout_gate(name, trials, required_ids, minimum_successes=2):
    ids = [r['identity'] for r in trials]
    infrastructure = len(ids) == len(set(ids)) and set(ids) == set(required_ids)
    infrastructure &= all(r.get('software_complete', False) and r.get('native_termination', False) for r in trials)
    hashes={r.get('checkpoint_sha256') for r in trials}
    if len(hashes)>1:infrastructure=False
    count = sum(bool(r['success']) for r in trials)
    accepted = infrastructure and len(required_ids) >= 8 and count >= minimum_successes
    return Gate(name, infrastructure, None, dict(successes=count, trials=len(trials), required=len(required_ids),checkpoint_sha256=next(iter(hashes),None)),
        accepted, None if accepted else 'Insufficient autonomous full-task successes; investigate before expansion')


def visual_gate(g0, g1, g2, g3, repeat_results):
    repeat_ids = {r['identity'] for r in repeat_results if r['success']}
    checkpoint=(g3.closed_loop_success or {}).get('checkpoint_sha256')
    matched=checkpoint is not None and all(r.get('checkpoint_sha256')==checkpoint for r in repeat_results)
    matched &= (g2.closed_loop_success or {}).get('checkpoint_sha256')==checkpoint
    accepted = all(g.acceptance_passed for g in (g0,g1,g2,g3)) and len(repeat_ids) >= 2 and matched
    return Gate('G4', all(g.infrastructure_passed for g in (g0,g1,g2,g3)), g1.offline_fit_passed,
        dict(repeated_success_initials=len(repeat_ids)), accepted,
        None if accepted else 'State data/learning/validation/repeated success acceptance incomplete; visual training locked')


def require_gate(root, name):
    import json
    path = root/'gates'/(name+'.json')
    value = json.loads(path.read_text())
    if value.get('name') != name or value.get('acceptance_passed') is not True:
        raise RuntimeError(name+' blocks this stage: '+str(value.get('blocked_reason')))
    return value


def plateau(history, min_update=6000, patience=3):
    """Stop only after three milestones with no success OR stage progress."""
    if len(history)<patience or history[-1]['updates']<min_update:
        return False
    recent=history[-patience:]
    if any(r['successes'] for r in recent): return False
    return all(b['stage_score'] <= a['stage_score'] for a,b in zip(recent,recent[1:]))


def final_gate(g4, selected, trials, test_ids, *, configuration_frozen, prior_test_uses):
    """G5 validates identities separately from per-model native performance."""
    required=sorted(test_ids)[:20]
    integrity=g4.acceptance_passed and configuration_frozen and len(test_ids)==40 and len(set(test_ids))==40
    scores={}
    if set(selected)!={'S-NativeDP','I-OfficialDP','I-HistoryDP'}:integrity=False
    for group,checkpoint in selected.items():
        rows=[r for r in trials if r.get('group')==group]
        integrity &= sorted(r['identity'] for r in rows)==required
        integrity &= all(r.get('checkpoint_sha256')==checkpoint['sha256'] and r.get('software_complete') and r.get('native_termination') for r in rows)
        scores[group]=dict(successes=sum(bool(r['success']) for r in rows),trials=len(rows))
    success=bool(scores) and all(r['successes']>0 for r in scores.values())
    return Gate('G5',bool(integrity),None,dict(models=scores,prior_test_uses=prior_test_uses,virgin_test=not bool(prior_test_uses)),
        bool(integrity and success),None if integrity and success else
        'Formal protocol incomplete or at least one model lacks full native success; preserve all failures')
