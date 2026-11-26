"""Verify the lightweight deliverable using only the Python standard library."""
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.published_source import payload


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def yes(value):
    assert value in ('True','False')
    return value=='True'


def main():
    root = Path(__file__).resolve().parent
    if not (root/'primary_summary.json').exists():
        root = Path(__file__).resolve().parents[2]/'research_docs/evidence/tight_entry_final24_20261122_01'
    manifest = read(root/'published_sha256.json')
    for name,digest in manifest.items():
        assert hashlib.sha256(payload(root,name)).hexdigest()==digest, name
    summary = read(root/'primary_summary.json')
    registration = read(root/'registration.json')
    assert summary['complete'] and summary['episodes']==96
    assert summary['optimizer_updates']==summary['new_dagger_trajectories']==0
    assert summary['paired_initial_checks_passed']==24
    assert summary['inference_rng_start_end_verified']==96
    paired = rows(root/'per_initial_state.csv')
    rolls = rows(root/'rollouts.csv')
    assert len(paired)==48 and len(rolls)==96
    assert len({(r['group'],int(r['seed'])) for r in paired})==48
    assert {int(r['seed']) for r in paired}==set(registration['seeds'])
    for r in paired:
        old,new = yes(r['original_success']),yes(r['entry_success'])
        assert yes(r['both_success'])==(old and new)
        assert yes(r['only_original'])==(old and not new)
        assert yes(r['only_entry'])==(new and not old)
        assert yes(r['both_failure'])==(not old and not new)
        for suffix,role in [('original',r['group']+'_130_50'),('entry',r['group']+'_entry')]:
            rollout = next(x for x in rolls if x['role']==role and x['seed']==r['seed'])
            assert yes(r[suffix+'_success'])==yes(rollout['success'])
            assert yes(rollout['software_complete']) and yes(rollout['legal'])
    for c in summary['contrasts']:
        values = [r for r in paired if r['group']==c['group'] and
                  (c['cohort']=='all24' or r['cohort']==c['cohort'])]
        assert len(values)==c['n']
        for column in ('original_success','entry_success','both_success','only_original','only_entry','both_failure'):
            assert sum(yes(r[column]) for r in values)==c[column]
        assert abs((c['entry_success']-c['original_success'])/c['n']-c['paired_difference'])<1e-12
    for f in summary['funnels']:
        values = [r for r in rolls if r['role']==f['role'] and
                  (f['cohort']=='all24' or r['cohort']==f['cohort'])]
        assert len(values)==f['n']
        reached = [r for r in values if int(r['native_subgoals'])>=1]
        assert len(reached)==f['first_subgoal']==f['success_after_subgoal_denominator']
        assert sum(yes(r['success']) for r in reached)==f['success_after_subgoal']
    history = read(root/'entry_history/summary.json')
    assert history['complete'] and history['cases']==50 and history['legal_entry_slots']==334
    assert history['remaining_legal_slots']==284 and history['first_legal_slots']==50
    assert history['optimizer_updates']==history['new_dagger_trajectories']==history['physics_steps']==0
    assert len(rows(root/'entry_history/per_slot_errors.csv'))==1336
    assert len(read(root/'entry_history/noise_history_checks.json'))==50
    receipts = read(root/'entry_history/noise_history_receipts.json')
    assert len(receipts)==50
    for case in receipts:
        streams=case['streams']
        assert len(streams)==4 and len({(s['role'],s['policy']) for s in streams})==4
        for s in streams:
            assert s['planning_noise']==streams[0]['planning_noise']
            assert s['initial_rng_sha256']==streams[0]['initial_rng_sha256']
            assert s['source_visual_cache_sha256']==streams[0]['source_visual_cache_sha256']
            assert s['prefix_robot178_exact']
    print(f'PASS: {len(manifest)} file hashes; 96 rollouts / 24 states; paired counts; 50 histories / 334 slots; zero training and DAgger.')


if __name__=='__main__':
    main()
