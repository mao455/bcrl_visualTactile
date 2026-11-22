"""Capture final receipts and export a portable lightweight evidence manifest."""
import csv
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from scripts.tight_entry_final24_20261122.summarize import read,write,sha

PROJECT=Path('/home/CNS2025331827/docs/bc+rl_RSS')
ROOT=PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01'
DIAG=PROJECT/'outputs/diagnostics/tight_entry_final24_20261122_01'
OUT=PROJECT/'research_docs/evidence/tight_entry_final24_20261122_01'


def resources(root):
    samples=[__import__('json').loads(s) for s in (root/'resources/shared.jsonl').read_text().splitlines()]
    return dict(samples=len(samples),
        maximum_sample_gap_s=max(b['utc_epoch']-a['utc_epoch'] for a,b in zip(samples,samples[1:])),
        minimum_sampled_ram_available_gib=min(s['ram_available_bytes'] for s in samples)/2**30,
        minimum_sampled_gpu_free_mib=min(s['gpu_free_mib'] for s in samples),
        scope='Sampled telemetry, not continuous guarantees. Process categorization can classify the new wrapper as external; no claim of exact ownership resource attribution.')


def main():
    freeze=read(ROOT/'registration.json')
    assert read(ROOT/'project_exit.json')['children_reaped'] and read(DIAG/'exit.json')['children_reaped']
    assert read(DIAG/'complete.json')['weights_unchanged']
    for path,digest in read(DIAG/'input_verification.json')['pairs'].items():
        assert sha(path)==digest
    for pair in freeze['pairs'].values():
        for name in ('frontend','backend'):
            assert sha(pair[name])==pair[name+'_sha256']
    for path,digest in freeze['source_sha256'].items():
        assert sha(PROJECT/path)==digest
    for name in ('monitor.json','monitor_exit.json','project_status.json'):
        shutil.copy2(ROOT/name,OUT/name)
    shutil.copy2(PROJECT/'configs/experiments/entry_final24/evaluate.json',OUT/'job.json')
    history=OUT/'entry_history'
    for name in ('monitor.json','monitor_exit.json'):
        shutil.copy2(DIAG/name,history/name)
    source=OUT/'source'
    for path in [PROJECT/'tools/tight_heldout_evaluate.py', PROJECT/'tools/tight_project.py',
                 *sorted((PROJECT/'scripts/tight_entry_final24_20261122').glob('*.py'))]:
        destination=source/path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    shutil.copy2(PROJECT/'scripts/tight_entry_final24_20261122/verify_published.py',OUT/'verify_published.py')
    report=PROJECT/'research_docs/TIGHT_ENTRY_FINAL24_20261122.md'
    shutil.copy2(report,OUT/'REPORT.md')
    closure=dict(complete=True,closed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        main_rollouts=96,main_states=24,main_software_incomplete=0,main_failed_attempts=0,
        diagnostic_cases=50,diagnostic_policies=4,diagnostic_physics_steps=0,
        optimizer_updates=0,new_dagger_trajectories=0,all_owners_and_monitors_reaped=True,
        frozen_weights_and_sources_unchanged=True,evaluation_resources=resources(ROOT),
        diagnostic_resources=resources(DIAG),
        final_report_sha256=sha(report),registration_sha256=sha(ROOT/'registration.json'))
    write(OUT/'closure.json',closure)
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name!='published_sha256.json' and '__pycache__' not in p.parts}
    write(OUT/'published_sha256.json',files)
    print(closure)


if __name__=='__main__':
    main()
