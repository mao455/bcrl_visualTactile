"""Export matched new50 entry errors, with remaining slots excluding the first."""
from collections import defaultdict
from pathlib import Path
import shutil

import numpy as np

from scripts.tight_entry_final24_20261122.summarize import read, write, sha, csv_write

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
ROOT = PROJECT/'outputs/diagnostics/tight_entry_final24_20261122_01'
OUT = PROJECT/'research_docs/evidence/tight_entry_final24_20261122_01/entry_history'


def main():
    assert read(ROOT/'complete.json')['weights_unchanged']
    assert read(ROOT/'exit.json')['children_reaped']
    plan = read(ROOT/'entry_same_history_plan.json')
    selected = [c for c in plan['cases'] if c['new50']]
    rows = read(ROOT/'history_new50/rows.json')
    families = {c['identity']:c['family'] for c in read(ROOT/'input_verification.json')['cases']}
    assert len(rows)==334*4
    assert len({r['identity'] for r in rows})==50
    noise_checks = []
    noise_receipts = []
    for case in selected:
        documents = []
        for group in 'AB':
            for tag in ('original_R2','entry_repaired'):
                item = read(ROOT/f'history_new50/{group}/{tag}/{case["identity"]}.json')
                assert item['prefix_robot178_exact'] and item['frames']==case['stop']
                assert [r['absolute_tick'] for r in item['rows']]==case['ticks']
                documents.append(item)
        assert all(d['planning_noise']==documents[0]['planning_noise'] for d in documents)
        assert len({d['initial_rng_sha256'] for d in documents})==1
        assert len({d['source_visual_cache_sha256'] for d in documents})==1
        noise_checks.append(dict(identity=case['identity'],
            planning_calls=len(documents[0]['planning_noise']),
            all_four_streams_equal=True, prefix_robot178_exact=True))
        noise_receipts.append(dict(identity=case['identity'], streams=[
            {key:document[key] for key in ('role','policy','frames','initial_rng_sha256',
             'planning_noise','source_visual_cache_sha256','prefix_robot178_exact','pair')}
            for document in documents]))
    flattened = []
    for r in rows:
        value = {k:r[k] for k in ('role','policy','identity','native_anchor','slot','absolute_tick','first_legal')}
        value['family'] = families[r['identity']]
        for stage in ('nominal','residual','preprojection','final_command','projection_magnitude'):
            for part in ('arm','hand'):
                value[f'{stage}_{part}_rmse_rad'] = r[stage][part+'_rmse_rad']
        flattened.append(value)
    contrasts = []
    episode_rows = []
    for group in 'AB':
        original = {(r['identity'],r['absolute_tick']):r for r in rows
                    if r['role']==group and r['policy']=='original_R2'}
        candidate = [r for r in rows if r['role']==group and r['policy']=='entry_repaired']
        for scope in ('first_legal','remaining_legal','all_entry_commit'):
            values = [r for r in candidate if scope=='all_entry_commit' or
                (scope=='first_legal' and r['first_legal']) or
                (scope=='remaining_legal' and not r['first_legal'])]
            for stage in ('nominal','final_command'):
                for part in ('arm','hand'):
                    metric = part+'_rmse_rad'
                    before = np.array([original[(r['identity'],r['absolute_tick'])][stage][metric] for r in values])
                    after = np.array([r[stage][metric] for r in values])
                    per_episode = defaultdict(list)
                    for r,b,a in zip(values,before,after):
                        per_episode[r['identity']].append((float(b),float(a)))
                    ep_before,ep_after = [],[]
                    for identity,errors in per_episode.items():
                        b,a = np.array(errors).mean(0)
                        ep_before.append(b);ep_after.append(a)
                        episode_rows.append(dict(group=group,scope=scope,stage=stage,part=part,
                            identity=identity,family=families[identity],slots=len(errors),original_mean_slot_rmse_rad=b,
                            entry_mean_slot_rmse_rad=a,difference_rad=a-b))
                    delta = np.array(ep_after)-np.array(ep_before)
                    cluster = defaultdict(list)
                    for identity,difference in zip(per_episode,delta):
                        cluster[families[identity]].append(difference)
                    sums = np.array([np.sum(v) for v in cluster.values()])
                    counts = np.array([len(v) for v in cluster.values()])
                    rng = np.random.default_rng(20261122)
                    draws = rng.integers(0,len(cluster),(50000,len(cluster)))
                    boot = sums[draws].sum(1)/counts[draws].sum(1)
                    contrasts.append(dict(group=group,scope=scope,stage=stage,part=part,
                        target='q_reached' if group=='A' and stage=='nominal' else 'u_expert',
                        episodes=len(per_episode),slots=len(values),
                        original_mean_slot_rmse_rad=float(before.mean()),
                        entry_mean_slot_rmse_rad=float(after.mean()),
                        relative_change_percent=float((after.mean()/before.mean()-1)*100),
                        improved_slots=int((after<before).sum()),
                        improved_episodes=int((delta<0).sum()),
                        episode_equal_weight_difference_rad=float(delta.mean()),
                        family_clusters=len(cluster),
                        family_paired_bootstrap95_rad=np.quantile(boot,[.025,.975]).tolist()))
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/'per_slot_errors.csv',flattened)
    csv_write(OUT/'paired_errors.csv',contrasts)
    csv_write(OUT/'per_episode_errors.csv',episode_rows)
    write(OUT/'noise_history_checks.json',noise_checks)
    write(OUT/'noise_history_receipts.json',noise_receipts)
    write(OUT/'summary.json',dict(complete=True,comparisons=contrasts,
        cases=50,legal_entry_slots=334,first_legal_slots=50,remaining_legal_slots=284,
        noise_and_recorded_histories_verified=True,optimizer_updates=0,new_dagger_trajectories=0,
        physics_steps=0,aggregation='Mean of per-slot arm7/hand22 RMSE, radians. First legal action excluded from remaining. Supplemental bootstrap resamples complete family blocks after equal-weight aggregation within each episode; 50000 draws, seed20261122. Describes variation within these training examples, not autonomous recovery or independent generalization.',
        raw_root=str(ROOT/'history_new50')))
    for name in ('entry_same_history_plan.json','input_verification.json','job.json','job_started.json',
                 'session.json','complete.json','exit.json'):
        shutil.copy2(ROOT/name,OUT/name)
    shutil.copy2(ROOT/'history_new50/summary.json',OUT/'production_summary.json')
    write(OUT/'raw_case_hashes.json',{str(p):sha(p) for p in sorted((ROOT/'history_new50').glob('*/*/*.json'))})
    print(OUT/'summary.json')


if __name__=='__main__':
    main()
