"""Combine the closed round13 review and prior small receipts without rerunning work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE))


def run(root, output):
    bindings = {}

    def read(path):
        payload = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    cycle = read(root / 'completed_review/cycle.json')
    previous = cycle.pop('source')
    previous.pop('cases', None)
    cycle['previous_round'] = previous
    evidence = TREE / 'research_docs/evidence/tight_fresh_20261114'
    a = read(evidence / 'collection11_recovery/completed_cycle.json')['candidate']
    b = read(evidence / 'round09_behavior/report.json')['candidate']
    from tools.tight_fresh_round_compare import matched_legal_comparison
    assert a['execution'] == {key: b['execution'][key] for key in a['execution']} == cycle['candidate']['execution']
    source = dict(execution=a['execution'],
                  cases=[x for x in a['cases'] if x['role'] == 'A'] + [x for x in b['cases'] if x['role'] == 'B'],
                  pairs={'A': a['pairs']['A'], 'B': b['pairs']['B']})
    assert source['pairs'] == read(root / 'session.json')['followup']['selected_pairs']
    cycle['actual_sampling_sources'] = dict(pairs=source['pairs'],
                                            totals={'A': a['totals']['A'], 'B': b['totals']['B']})
    cycle['compared_to_actual_sampling_sources'] = matched_legal_comparison(source, cycle['candidate'])
    prior = read(evidence / 'prefetch_followup03/progress_1924.json')
    collection = prior['collection']
    collection.update(training_status_at_collection_close='not_started_then_completed',
                      student_autonomous_benefit='No overall promotion: A holding regressed; B one near-oriented case but zero native alignment or insertion.')
    owner = read(root / 'research_loop/owner_finished.json')
    assert owner['software_complete'] and owner['completed_rounds'] == 1
    profiles = []
    for path in sorted((root / 'new_family_13').glob('*/capture_complete.json')):
        profiles.append(read(path)['profile'])
    profile_keys = ('copy_s', 'submission_wait_s', 'worker_numeric_append_s', 'worker_compact_append_s',
                    'worker_flush_s', 'worker_finish_s', 'worker_verify_s', 'worker_hash_s', 'verification_queue_wait_s')
    result = dict(version='completed13_outcomes_and_efficiency_v1', cycle=cycle, collection=collection,
                  train_trajectories=prior['train_trajectories'], training_families=prior['training_families'],
                  validation_trajectories=prior['validation_trajectories'], frontends=prior['roles'],
                  owner={key: value for key, value in owner.items() if not isinstance(value, (dict, list))},
                  capture_worker_profiles=dict(captures=len(profiles),
                      sum_seconds={key: sum(row.get(key, 0.) for row in profiles) for key in profile_keys},
                      semantics='Overlapping writer/verifier task totals and queue waits, not additional rollout wall time.'),
                  decision=dict(overall_promotion=False, retained_baseline_roles={'A': 'round11', 'B': 'round09'},
                      next_collection_candidate_roles={'A': 'round11', 'B': 'round13'},
                      rationale='B13 reaches a held 8.4 degree / 21.7mm keypoint state in one development case before falling; use it for new recoverable late deviations, not as proven success. A13 holding regresses. Do not repeat the same late-heavy recipe as established improvement.'),
                  scope='Same repeated development cohort, legal A7/B8. Native lift, holding, alignment and insertion remain separate. No independent generalization claim.',
                  input_sha256=bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(totals=cycle['candidate']['totals'], seconds=cycle['seconds'],
                          full_owner_seconds=owner['seconds'], decision=result['decision']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
