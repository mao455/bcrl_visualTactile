"""Report natural four-lane outcomes and measured throughput as engineering evidence."""
import argparse
from pathlib import Path
from tools.tight_fresh_rollouts import read
from tools.tight_fresh_phases import case
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def report(root,output):
    if not read(root/'result.json').get('software_complete'):raise ValueError('Incomplete batch')
    plan=read(root/'plan.json');roles={}
    for role in ['A','B']:
        directory=root/role;result=read(directory/'result.json');rows=[]
        if not result.get('software_complete') or len(result['results'])!=len(plan['cases']):
            raise ValueError('Batch role opportunity count differs')
        if sorted(r['lane'] for r in result['results'])!=list(range(len(plan['cases']))):
            raise ValueError('Missing or duplicate batch lane')
        for row in result['results']:
            if row['pair']!=plan['pairs'][role]:raise ValueError('Unexpected complete pair')
            if row['identity']!=plan['cases'][row['lane']]['identity']:
                raise ValueError('Batch reset identity differs')
            if row.get('legal'):
                metrics=read(directory/f"lane_{row['lane']}_inference.json")['summary']
                for name in ['dino_rgb','learned_visual_and_history','residual_forward_compose']:
                    if metrics[name]['wall_s']['count']!=row['steps']:
                        raise ValueError('Every-control-frame inference count differs')
            if row.get('legal') and row.get('visual_frames')!=row['steps']:raise ValueError('Dropped native visual frames')
            rows.append(case(directory,numeric_path=directory/f"lane_{row['lane']}_numeric.h5",result=row,
                case_name=role+'_'+row['identity']))
            if row.get('legal') and rows[-1]['steps']!=row['steps']:
                raise ValueError('Numeric record length differs from real control ticks')
        if result['profile']['steps']!=sum(r['steps'] for r in result['results']):
            raise ValueError('Profile step denominator differs')
        roles[role]=dict(profile=result['profile'],cases=rows,pair=plan['pairs'][role],
            legal=sum(r['legal'] for r in rows),successes=sum(r.get('success',False) for r in rows),
            stable_grasps=sum(r.get('settled_height_stable_grasp',False) for r in rows))
    atomic_json(output,dict(roles=roles,plan_sha256=sha256(root/'plan.json'),
        source_commit=plan['source_commit'],neural_batch=plan.get('neural_batch',False),
        startup=read(root/'startup_timing.json'),test_used=False,
        interpretation='Engineering cohort; native batched RNG/physics is not exact serial trajectory replay. Each active lane sees every real 60Hz RGB-D frame; DP plans at 5Hz. Aggregate wall throughput, not summed overlapping lane timers.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.output)
