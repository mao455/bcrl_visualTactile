"""Finite diagnostic: reuse one Kit and time real complete-policy trials."""
import argparse, json, os, time, traceback
from pathlib import Path
from collections import defaultdict


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();root=args.root;plan=json.loads(args.plan.read_text())
    from tools.tight_state import environment
    from visuotactile.training_resources import atomic_json
    from visuotactile.state_resources import lease_for
    from visuotactile import native_evaluation, expert
    totals=defaultdict(float);counts=defaultdict(int)
    def wrap(obj,name,key):
        old=getattr(obj,name)
        def call(*a,**kw):
            t=time.perf_counter()
            try:return old(*a,**kw)
            finally:totals[key]+=time.perf_counter()-t;counts[key]+=1
        setattr(obj,name,call)
    wrap(native_evaluation,'make_env','environment_create_s');wrap(expert,'create_expert','expert_load_s')
    lease=lease_for(Path(plan['resource_root']),deadline=plan['compute_deadline'])
    t=time.perf_counter();app,env,player,lease=environment(root,evaluation=True,lease=lease)
    initialization=dict(total_s=time.perf_counter()-t,components=dict(totals))
    atomic_json(root/'initialization.json',initialization)
    import numpy as np,torch
    import visuotactile.state_benchmark as benchmark
    from visuotactile import native_controller
    from visuotactile.models.native_vision import NativeDinoSpatial
    wrap(benchmark,'prepare','prepare_s');wrap(torch,'load','torch_load_nested_s')
    wrap(NativeDinoSpatial,'__init__','dino_construct_nested_s')
    wrap(native_controller.ResidualController,'__init__','controller_construct_s')
    wrap(env.sim,'step','physics_step_s');wrap(env.sim,'render','render_s')
    wrap(env.scene,'update','scene_update_s')
    wrap(benchmark.RolloutVideo,'append','video_append_s');wrap(benchmark.RolloutVideo,'close','video_close_s')
    all_rows=[]
    for case in plan['cases']:
        t=time.perf_counter();before=dict(totals);before_counts=dict(counts);before_wait=lease.wait_s
        out=root/case['name'];source=Path(case['source']);spec=json.loads((source/'started.json').read_text())['spec']
        checkpoint=case.get('checkpoint',str(root/'session.json'))
        load_t=time.perf_counter();saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if case['mode']=='student' else None
        load_s=time.perf_counter()-load_t
        replay=np.load(source/'trace.npz')['command'] if case['mode']=='replay' else None
        result=benchmark.trial(env,lease,out,spec,{},player,checkpoint,saved=saved,replay=replay,
            position=True,video=case.get('video',False),expert_checkpoint=plan['expert_checkpoint'],
            numeric_buffer_steps=32 if plan.get('buffered_numeric') else 0)
        row=dict(name=case['name'],source=str(source),mode=case['mode'],total_s=time.perf_counter()-t,
            checkpoint_load_s=load_s,resource_wait_s=lease.wait_s-before_wait,
            components={k:totals[k]-before.get(k,0) for k in totals},
            calls={k:counts[k]-before_counts.get(k,0) for k in counts},result=result)
        atomic_json(out/'runtime_profile.json',row);all_rows.append(row)
        atomic_json(root/'runtime_results.json',dict(initialization=initialization,cases=all_rows))
        print('PROBE_COMPLETE',case['name'],result.get('steps'),result.get('success'),row['total_s'],flush=True)
    atomic_json(root/'result.json',dict(software_complete=True,worker_error=None,utc=time.time(),cases=len(all_rows)))
    atomic_json(root/'complete.json',dict(utc=time.time(),cases=len(all_rows)))
    from visuotactile.native_evaluation import exit_completed_kit_worker
    exit_completed_kit_worker(root)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        # trial closes its writers in finally; avoid Kit destructor hangs on errors.
        import sys
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
