#!/usr/bin/env python3
"""Summarize the closed collection from small receipts; never reopen raw data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    collection=root/'new_family_11';plan=read(collection/'plan.json')
    cases={case['name']:case for case in plan['cases']}
    rows=read(collection/'results.json');entries=read(collection/'correction_entries.json')
    final=read(collection/'result.json');owner=read(collection/'owner_finished.json')
    assert final['software_complete'] and owner['software_complete']
    total=Counter();schedules={};roles={};phases=Counter();fields=Counter()
    for row in rows:
        result=row['result'];dagger=result.get('dagger',{});case=cases[row['case']]
        accepted=result.get('accepted_for_training') is True
        counts=dict(attempts=1,legal=int(result.get('legal') is True),
            triggered=int(dagger.get('takeover_triggered') is True),accepted=int(accepted),
            accepted_expert_ticks=dagger.get('expert_valid_ticks',0) if accepted else 0)
        for target in (total,schedules.setdefault(case['expansion_schedule'],Counter()),roles.setdefault(row['role'],Counter())):
            target.update(counts)
        if accepted:
            receipt=read(collection/row['case']/'capture_complete.json')
            assert receipt['status']=='source_verified' and receipt['dataset_capture']==result['dataset_capture']
            fields[str(result['dataset_capture']['verified_source_fields'])]+=1
            phases[dagger.get('recovery_coverage',{}).get('phase','unknown')]+=1
    assert total['accepted']==len(entries)==45
    waves=read(collection/'wave_profiles.json')
    wall_components={key:sum(wave[key] for wave in waves) for key in
        ('controller_s','environment_s','expert_s','record_submit_s','finish_audit_s','admission_s')}
    return dict(version='collection11_closed_receipts_v1',collection_source_commit=plan['source_commit'],
        total=dict(total),schedules=schedules,roles=roles,accepted_switch_diagnostic_phases=phases,
        accepted_families=len({entry['family'] for entry in entries.values()}),
        accepted_verified_field_counts=fields,collection_seconds=final['elapsed_seconds'],
        accepted_per_hour=total['accepted']*3600/final['elapsed_seconds'],
        wave_wall_components_seconds=wall_components,
        wall_component_fraction={key:value/final['elapsed_seconds'] for key,value in wall_components.items()},
        timing_semantics='Sums of main-path timers across all four waves; environment includes sensors/rendering. Async verifier/writer internals overlap and are not added.',
        original_cycle_failure=read(root/'research_loop_rgb01/round_11/collect_cycle/owner_finished.json')['error'],
        training_status_at_collection_close='not_started',student_autonomous_benefit='not_yet_evaluated',
        phase_semantics='Schedules and recorded switch diagnostics; held conditions are not native lift/alignment success.',
        raw_payloads_opened=0,input_sha256={str(path.relative_to(root)):sha(path) for path in
            [collection/name for name in ('plan.json','results.json','result.json','owner_finished.json','correction_entries.json','wave_profiles.json')]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(summarize(args.root),indent=2))
