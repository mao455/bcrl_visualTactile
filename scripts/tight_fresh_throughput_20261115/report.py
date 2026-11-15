"""Small read-only throughput report from published receipts, never raw rescans."""
import argparse,json,time
from pathlib import Path

BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
def read(path):return json.loads(path.read_text()) if path.exists() else None
def collect(path):
    owner=read(path/'owner_finished.json');result=read(path/'result.json')
    rows=read(path/'results.json') or [];waves=read(path/'wave_profiles.json') or []
    entries=read(path/'correction_entries.json') or {};plan=read(path/'plan.json') or {}
    complete=bool(owner and owner.get('software_complete') and result and result.get('software_complete'))
    progress=read(path/'batch_progress.json')
    elapsed=result.get('elapsed_seconds') if complete else None
    return dict(path=str(path),complete=complete,owner=owner,reported_result=result,
        frozen_commit=plan.get('source_commit'),lanes=plan.get('batch',{}).get('lanes'),
        registered_opportunities=len(plan.get('cases',[])),accepted=len(entries),
        actual_control_steps=sum(r['result'].get('steps',0) for r in rows),
        valid_expert_steps=sum(r['result'].get('dagger',{}).get('expert_valid_ticks',0)
            for r in rows if r['result'].get('accepted_for_training')),
        active_steps_per_second=(sum(r['result'].get('steps',0) for r in rows)/elapsed if elapsed else None),
        accepted_per_hour=len(entries)*3600/elapsed if elapsed else None,
        observed_progress=progress,profiles=waves,
        recovery=read(path/'closed_capture_recovery.json') and dict(
            accepted=read(path/'closed_capture_recovery.json')['accepted'],
            valid_expert_steps=read(path/'closed_capture_recovery.json')['successful_expert_ticks']),
        interpretation='Only a completed owner plus admission receipt establishes throughput. Partial successes remain recoverable evidence; no task-failure or whole-batch completion inference.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases={'old_matched4':'tight_fresh_20261115_04/throughput_baseline15_01',
        'queue_matched4':'tight_fresh_20261115_04/throughput_queue15_03',
        'current_render_matched4':'tight_fresh_20261115_07/throughput_preflight/render_capture',
        'formal24_interrupted':'tight_fresh_20261115_07/new_family_15',
        'formal18_interrupted':'tight_fresh_20261115_08/new_family_15',
        'formal18_camera_fix':'tight_fresh_20261115_09/new_family_15'}
    report=dict(generated_utc=time.time(),version='receipt_only_collection_throughput_v1',
        runs={k:collect(BASE/v) for k,v in cases.items()},
        caveats=['Matched4 keeps policies, initial families and early-grasp rule; actual physical lengths differ.',
            'Old collector overlapped training; render acceptance ran serially. This is combined workflow improvement, not an isolated causal estimate.',
            'Formal09 includes five unresolved hard families and four narrow families; its success rate is not a paired baseline.',
            'Expert recovery success is not autonomous student success. No mass/friction generalization.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{q:v[q] for q in ('complete','accepted','actual_control_steps','accepted_per_hour')} for k,v in report['runs'].items()}))
