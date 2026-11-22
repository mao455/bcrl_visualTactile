"""Describe heldout trajectories with the previously frozen retrospective rules.

This adds no trigger, pilot seed, training update, policy inference, or physics.
It does not change the original train/development census or test outcomes.
"""
import csv
import gzip
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

from scripts.tight_premature_scope_20261122 import trajectory_audit as audit


PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'outputs/diagnostics/tight_premature_scope_20261122_01'
OUT = BASE / 'terminal'
PUBLIC = PROJECT / 'research_docs/evidence/tight_premature_scope_20261122_01/terminal'
SOURCE = PROJECT / 'research_docs/evidence/tight_entry_final24_20261122_01'
PINNED = {
    PROJECT / 'scripts/tight_premature_scope_20261122/trajectory_audit.py':
        '72787cec47b3c2a233da6b7e13bdddea987b50ac69bfdbd05b6eee910c72f237',
    BASE / 'rules_registration.json':
        'd2416318ce254f4b6c440869d8c88a8b1ba8c90f1b165bd37788e357fe74fa19',
    BASE / 'rules_addendum.json':
        '1cc8a8de06f1ca644ad8e162e5b627fe459a4480f1032b275e03f7b6097da1ab',
}
ROLES = ('A_130_50', 'A_entry', 'B_130_50', 'B_entry')


def verify_pins():
    for path, digest in PINNED.items():
        assert audit.sha(path) == digest, f'Frozen method changed: {path}'


def summarize(rows):
    pools = {'all_terminal': rows}
    pools.update({role:[row for row in rows if row['policy'] == role] for role in ROLES})
    pools.update({cohort:[row for row in rows if row['terminal_cohort'] == cohort]
                  for cohort in ('reserved8', 'fresh16')})
    pools.update({role+'/'+cohort:[row for row in rows
        if row['policy'] == role and row['terminal_cohort'] == cohort]
        for role in ROLES for cohort in ('reserved8', 'fresh16')})
    # The train/development summarizer's "all" pool is the same descriptive
    # aggregation for any supplied rows; its named source pools are not reused.
    return {name:audit.summarize(pool)['all'] for name, pool in pools.items()}


def main():
    verify_pins()
    primary = audit.read(SOURCE / 'primary_summary.json')
    frozen = audit.read(SOURCE / 'registration.json')
    assert primary['complete'] and primary['episodes'] == 96
    assert primary['software_incomplete'] == 0
    assert audit.read(SOURCE/'project_exit.json')['children_reaped']
    rules = audit.read(BASE / 'rules_registration.json')
    assert rules['tunable_using_terminal'] is False
    table = list(csv.DictReader((SOURCE / 'rollouts.csv').open()))
    assert len(table) == len({(row['role'], row['seed']) for row in table}) == 96
    assert set(row['role'] for row in table) == set(ROLES)
    jobs = []
    for row in table:
        path = Path(row['result_path'])
        result = audit.read(path)
        assert result['complete_student_policy'] == frozen['pairs'][row['role']]
        assert result['seed'] == int(row['seed'])
        assert result['legal'] and result['software_complete']
        jobs.append(dict(cohort='terminal',terminal_cohort=row['cohort'],
            policy=row['role'],identity=f'{row["role"]}_{row["seed"]}',
            seed=int(row['seed']),family=result['family'],wave=path.parent.parent.name,
            path=str(path.parent/'numeric.h5'),group='episode',result_path=str(path)))
    OUT.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir(parents=True, exist_ok=False)
    audit.write(OUT/'source_registration.json',dict(
        scope='All 96 completed heldout trajectories; frozen descriptive rules only',
        started_utc=time.time(),source_primary_sha256=audit.sha(SOURCE/'primary_summary.json'),
        rules_sha256=audit.sha(BASE/'rules_registration.json'),
        addendum_sha256=audit.sha(BASE/'rules_addendum.json'),
        analysis_script_sha256=audit.sha(PROJECT/'scripts/tight_premature_scope_20261122/trajectory_audit.py'),
        wrapper_sha256=audit.sha(__file__),jobs=jobs,
        use='Descriptive only: cannot choose thresholds, intervention triggers, pilot seeds, models, or promote candidates',
        opening_event='native_open_command remains unknown; no zero-pose proxy inferred by this frozen code'))
    rows, hashes, total_ticks = [], [], 0
    began = time.time()
    try:
        with gzip.open(OUT/'derived_ticks.csv.gz','wt',newline='') as stream:
            writer = None
            for index, job in enumerate(jobs):
                path = Path(job['path'])
                with h5py.File(path,'r') as source:
                    metadata = json.loads(source.attrs['metadata_json'])
                    result = audit.read(job['result_path'])
                    row, columns = audit.analyze(source[job['group']],metadata,result,rules,
                                                 include_commands=True)
                assert row['ticks'] == result['steps']
                assert row['native_max_subgoals'] == result['native_subgoals']
                row.update(job)
                rows.append(row)
                total_ticks += row['ticks']
                if writer is None:
                    fields = ['cohort','terminal_cohort','policy','identity'] + list(columns)
                    fields += ['object_palm_reference_translation_mm','object_palm_reference_angle_deg',
                        'fk_final_target_palm_z_m','fk_final_target_palm_down_over12_mm',
                        'final_command_hand_distance_from_reference_rad',
                        'dagger_control_source_recorded','dagger_legal_expert_label_recorded',
                        'dagger_segment_id_recorded']
                    writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(fields)))
                    writer.writeheader()
                for tick in range(row['ticks']):
                    values={key:(value[tick].item() if isinstance(value[tick],np.generic)
                        else value[tick]) for key,value in columns.items()}
                    values.update({key:job[key] for key in ('cohort','terminal_cohort','policy','identity')})
                    writer.writerow(values)
                for input_path in (path, Path(job['result_path'])):
                    hashes.append(dict(path=str(input_path),sha256=audit.sha(input_path),
                                       bytes=input_path.stat().st_size))
                if index % 12 == 11:
                    print(json.dumps(dict(trajectories=index+1,total=96,ticks=total_ticks)),flush=True)
        assert len(rows)==96 and total_ticks==54751
        lookup={(row['policy'],row['seed']):row for row in rows}
        transitions=list(csv.DictReader((SOURCE/'per_initial_state.csv').open()))
        A_original_successes=[int(row['seed']) for row in transitions
            if row['group']=='A' and row['original_success']=='True']
        A_entry_postfail=[int(row['seed']) for row in transitions
            if row['group']=='A' and int(row['entry_subgoals'])>=1 and row['entry_success']=='False']
        details = dict(
            A_original_success_entry_destinations=[lookup['A_entry',seed] for seed in A_original_successes],
            A_entry_post_subgoal_failures=[lookup['A_entry',seed] for seed in A_entry_postfail],
            A_original_success_references=[lookup['A_130_50',seed] for seed in A_original_successes])
        summary=summarize(rows)
        verify_pins()
        audit.write(OUT/'trajectories.json',rows)
        audit.export_rows(rows,OUT/'trajectories.csv')
        audit.write(OUT/'summary.json',summary)
        audit.write(OUT/'A_transition_details.json',details)
        audit.write(OUT/'source_sha256.json',hashes)
        audit.write(OUT/'receipt.json',dict(trajectories=96,initial_states=24,
            ticks=total_ticks,wall_s=time.time()-began,
            description_only=True,thresholds_changed=False,triggers_selected=False,pilot_seeds_selected=False,
            optimizer_updates=0,new_dagger_trajectories=0,new_physics_steps=0,new_policy_inferences=0,
            scripts_and_rules_unchanged=True,pinned_sha256={str(k):v for k,v in PINNED.items()}))
        for name in ('source_registration.json','trajectories.json','trajectories.csv',
                     'summary.json','A_transition_details.json','source_sha256.json',
                     'receipt.json','derived_ticks.csv.gz'):
            shutil.copy2(OUT/name,PUBLIC/name)
        print(json.dumps({name:dict(n=summary[name]['n'],
            exclusive=summary[name]['exclusive'],outcomes=summary[name]['outcomes'])
            for name in ('all_terminal',)+ROLES},ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        audit.write(OUT/'failure.json',dict(error=repr(exc),utc=time.time()))
        raise


if __name__ == '__main__':
    main()
