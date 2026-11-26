"""Aggregate measured component timers without summing nested wall/GPU costs."""
import argparse
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read


def report(root, output):
    if not read(root/'result.json').get('software_complete'):
        raise ValueError('Only complete registered queues can become final profiles')
    plan=read(root/'plan.json');rows=read(root/'results.json')
    if {r['case'] for r in rows}!={c['name'] for c in plan['cases']}:
        raise ValueError('Profile denominator differs from registered queue')
    cases=[]
    for row in rows:
        value=row['result'];steps=value.get('steps',0)
        if not value.get('legal') or not value.get('software_complete'):
            raise ValueError('Separate incomplete/illegal cases before throughput claims')
        measured=value['native_inference']['summary']
        for name in ['dino_rgb','learned_visual_and_history','residual_forward_compose']:
            if measured[name]['wall_s']['count']!=steps:
                raise ValueError('Every-control-frame/controller timing count differs')
        components={name:dict(calls=v['wall_s']['count'],total_wall_s=v['wall_s']['count']*v['wall_s']['mean'],
            mean_wall_s=v['wall_s']['mean']) for name,v in measured.items()}
        cases.append(dict(case=row['case'],role=row['role'],steps=steps,success=value['success'],
            expert_shadow=value['expert_shadow_computed'],wall_profile=value['wall_profile'],
            components=components,timing=row['timing'],result_sha256=sha256(root/row['case']/'result.json')))
    totals={}
    for role in ['A','B']:
        chosen=[c for c in cases if c['role']==role]
        seconds=sum(c['timing']['total_seconds'] for c in chosen);steps=sum(c['steps'] for c in chosen)
        timers={name:sum(c['wall_profile'].get(name,0) for c in chosen)
            for name in ['expert_inference_s','environment_step_render_s','numeric_write_s','compact_append_s','finalize_verify_s','setup_s']}
        nested={name:sum(c['timing']['nested_environment_components'].get(name,0) for c in chosen)
            for name in ['render_s','physics_step_s','scene_update_s']}
        totals[role]=dict(cases=len(chosen),steps=steps,total_seconds=seconds,control_ticks_per_wall_second=steps/seconds,
            resource_wait_s=sum(c['timing']['resource_wait_seconds'] for c in chosen),
            wall_components=timers,nested_environment_components=nested,
            controller_components={name:sum(c['components'][name]['total_wall_s'] for c in chosen) for name in chosen[0]['components']})
    atomic_json(output,dict(complete=True,kind=plan['kind'],source_commit=plan['source_commit'],
        plan_sha256=sha256(root/'plan.json'),results_sha256=sha256(root/'results.json'),totals=totals,cases=cases,
        interpretation='Actual wall intervals and call counts. Controller subregions overlap combined controller. Render/physics/scene nest inside environment step. CUDA launches are asynchronous: depth-transfer wall can include preceding DINO completion; neither timer alone is isolated device compute. Setup includes waits. Queue startup is outside per-case totals. Collection includes lossless RGB-D and expert queries; ordinary autonomous evaluation does not.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.output)
