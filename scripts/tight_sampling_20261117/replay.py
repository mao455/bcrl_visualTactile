"""Reuse two saved A baseline histories; new ordinary policy inference only."""
import os, sys, subprocess, time, traceback
import torch
from .control import ROOT, OUT, read
from scripts.tight_fork_20261117.common import ROOT as CAPTURE
from scripts.tight_fork_20261117.replay import acknowledge
from scripts.tight_layout_20261116.publication import write, table, sha
REPLAY=ROOT/'ordinary_replay'

def worker():
    from visuotactile.gpu_batch_controller import GPUBatchedResidualController
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32
    state_fp32();torch.set_num_threads(2)
    lease=lease_for(REPLAY,deadline=read(ROOT/'session.json')['deadline'])
    previous=CAPTURE/'replay_fp32/outputs.pt'
    old=torch.load(previous,weights_only=False,map_location='cpu')
    old={(model,r['lane'],r['tick']):r for model in ['baseline','candidate'] for r in old['A_baseline_'+model]}
    generators=torch.load(CAPTURE/'initial_generators.pt',weights_only=False)
    ids=[1,5];active=[i in ids for i in range(8)];checkpoint=ROOT/'A/backend/A/training/best.pt'
    pair_paths=read(CAPTURE/'capture/plan.json')['pairs']
    scales={label:torch.load(path,map_location='cpu',weights_only=False,mmap=True)['model']['action_scale'].double().reshape(-1)
        for label,path in [('baseline',pair_paths['A_baseline']['backend']),('candidate',pair_paths['A']['backend']),('ordinary',checkpoint)]}
    with lease.block():
        c=GPUBatchedResidualController(torch.load(checkpoint,map_location='cpu',weights_only=False),read(CAPTURE/'replay_metadata.json'),None,8,
            neural_batch=True,scheduler_index_mode='host',vision_microbatch=4)
        for i in ids:c.lanes[i].base.generator.set_state(generators['A_baseline'][i])
    rows=[];start=time.time();input_hashes={}
    for tick in range(121):
        path=CAPTURE/'inputs'/f'{tick:03d}.pt';input_hashes[path.name]=sha(path)
        snap=torch.load(path,weights_only=False);p=snap['packet'];d=snap['diagnostics']
        with lease.block(),torch.no_grad():
            output=c.act(p,d,active,current_features=snap['features'][ids].cuda())
            for lane in ids:
                ctl=c.lanes[lane]
                assert ctl.base.memory.next_tick==tick+1
                assert len(ctl.base.memory.robot)==min(tick+1,13)
                assert len(ctl.base.memory.frames)==min(tick+1,3)
                base=old['baseline',lane,tick];onset=old['candidate',lane,tick]
                assert torch.equal(ctl.base.generator.get_state().cpu(),base['generator'].cpu())
                # External robot history remains recorded, despite shadow predictions.
                assert torch.equal(ctl.base.last_robot.cpu(),base['robot'].cpu())
                row=dict(seed=28824002 if lane==1 else 28824006,tick=tick,plan_cursor=tick%12)
                for key,v in [('nominal',ctl.nominal),('delta',ctl.delta),('command',output[lane])]:
                    a=base[key].double().cpu().reshape(-1);b=onset[key].double().cpu().reshape(-1);v=v.double().cpu().reshape(-1)
                    if key=='delta':
                        a=a/scales['baseline'];b=b/scales['candidate'];v=v/scales['ordinary'];key='proposal_rad'
                    for label,x,y in [('ordinary_vs_baseline',v,a),('onset_vs_baseline',b,a),('ordinary_vs_onset',v,b)]:
                        row[key+'_'+label+'_rms']=float((x-y).square().mean().sqrt())
                rows.append(row);acknowledge(ctl,snap['command'][lane],snap['command'][lane],p['time_s'][lane])
    bins=[]
    for seed in [28824002,28824006]:
        for low,high in [(0,59),(60,83),(84,120)]:
            rr=[r for r in rows if r['seed']==seed and low<=r['tick']<=high]
            bins.append(dict(seed=seed,start=low,end=high,**{k:sum(r[k] for r in rr)/len(rr) for k in rr[0] if k.endswith('_rms')}))
    table(OUT/'same_history_ticks.csv',rows);table(OUT/'same_history_bins.csv',bins)
    write(OUT/'same_history.json',dict(checkpoint_sha256=sha(checkpoint),reused_outputs_sha256=sha(previous),input_sha256=input_hashes,
        rows=242,units='All output RMS values radians; normalized residual divided by each model action_scale.',seconds=time.time()-start,generator_state_exact=True,external_robot_history_exact=True,full_prefix=True,bins=bins,
        prior_self_replay='fork_diagnosis/same_input.json (self replay command error <1e-6rad, same FP32 code)',
        interpretation='Two opened baseline sensor/actual-command histories only. Model-own recurrent/visual state, nominal and plan recomputed from tick0, same planning phase and diffusion generator states. No optimizer, no teacher labels from old outputs, no closed-loop success claim.'))

def run():
    from scripts.tight_layout_20261116.pilot import stop
    assert read(ROOT/'owner_exit.json')['children_reaped']
    assert read(ROOT/'evaluation_recovery01/owner_exit.json')['children_reaped']
    REPLAY.mkdir(exist_ok=False)
    session=read(ROOT/'session.json');session.update(resource_root=str(REPLAY),owned_worker_roots=[str(REPLAY)])
    write(REPLAY/'session.json',session);monitor=child=None
    try:
        with (REPLAY/'monitor.log').open('x') as f:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(REPLAY)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.time()+30
        while not (REPLAY/'shared_resources.json').exists():
            assert monitor.poll() is None and time.time()<until;time.sleep(.25)
        with (REPLAY/'worker.log').open('x') as f:child=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker','--root',str(REPLAY)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            assert monitor.poll() is None and time.time()<session['deadline'];time.sleep(2)
        assert child.returncode==0
        write(REPLAY/'complete.json',dict(complete=True,utc=time.time()))
    except BaseException:
        write(REPLAY/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
    finally:
        stop(child);(REPLAY/'STOP_MONITOR').touch()
        if monitor:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:stop(monitor)
        write(REPLAY/'owner_exit.json',dict(children_reaped=True,utc=time.time()))

if __name__=='__main__':globals()[sys.argv[1]]()
