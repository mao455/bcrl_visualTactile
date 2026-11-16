"""Read-only 60 Hz actual RGB video; no extra camera exposure or policy calls."""
import argparse,time
from pathlib import Path
from tools.tight_fresh_rollouts import read
from visuotactile.training_resources import atomic_json

def main(*,allow_heldout=False):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.name=='updated_evaluation' and (a.output.parent/'paired_request.json').exists():
        from scripts.tight_fulltask_20261116.paired_eval import upgrade_plan,install
        upgrade_plan(a.output);install(a.output)
    from visuotactile.gpu_batch_controller import MultiPolicyController
    from visuotactile.video import RolloutVideo,verify_video
    from visuotactile import native_evaluation
    from tools.tight_fresh_batch_run import run
    plan=read(a.output/'plan.json');assert plan['kind']=='autonomous'
    assert len(plan['cases'])==plan['batch']['lanes'],'One case per video lane; no reuse'
    original=MultiPolicyController.act;original_exit=native_evaluation.exit_completed_kit_worker
    videos={};spent=0.;flushed=False
    def act(self,packet,diagnostics,active):
        nonlocal spent
        before=time.perf_counter();ids=[i for i,x in enumerate(active) if x and plan['cases'][i].get('video',True)]
        rgb=packet['image'][ids,:,:3].detach().cpu()
        for j,lane in enumerate(ids):
            case=plan['cases'][lane];tick=int(packet['step_id'][lane])
            if lane not in videos:
                videos[lane]=RolloutVideo(a.output/case['name']/'autonomous.mp4',case['role'],case['spec']['seed'],
                    'selected',['fixed0','fixed1','wrist'])
            assert videos[lane].frames==tick
            videos[lane].append({'image':rgb[j:j+1]},tick,status=f"{case['role']} | AUTONOMOUS full DP + Transformer | seed {case['spec']['seed']}")
        spent+=time.perf_counter()-before
        return original(self,packet,diagnostics,active)
    def flush(complete):
        nonlocal flushed
        if flushed:return
        rows=[]
        for lane,video in videos.items():
            case=plan['cases'][lane];row=video.close(complete)
            if complete:
                result=read(a.output/case['name']/'result.json');assert row['frames']==result['steps']
                row.update(case=case['name'],success=result.get('success'),result=str(a.output/case['name']/'result.json'))
                row['decode']=verify_video(video.path,row['frames']);rows.append(row)
        atomic_json(a.output/'videos.json',dict(complete=complete,rows=rows,recording_seconds=spent,
            policy_unchanged=True,frame_source='Actual pre-action RGB observation, all control steps, no reset/terminal substitution'))
        flushed=True
    def completed(output):
        flush(True);original_exit(output)
    MultiPolicyController.act=act;native_evaluation.exit_completed_kit_worker=completed
    try:run(a.output,allow_heldout=allow_heldout)
    finally:
        MultiPolicyController.act=original;native_evaluation.exit_completed_kit_worker=original_exit;flush(False)
if __name__=='__main__':main()
