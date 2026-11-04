"""CPU-only audit and preparation; never launches or initializes a trainer."""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import time
import torch

from tools.state_image_queue import CODE,read,write,sha


def prepare(root):
    root=root.resolve();root.mkdir(exist_ok=False,parents=True)
    old=CODE/'outputs/visuotactile/native_dp_v2/state_image_20261104_02'
    s=read(old/'session.json');data=read(Path(s['source_root'])/'data.json')
    cache=torch.load(s['numeric_cache'],map_location='cpu',weights_only=False)
    ids=data['splits']['train'];lengths={i:len(cache['records'][i]['time']) for i in ids};del cache
    schedule=[(i,t) for i in ids for t in range(0,lengths[i],96)]
    anchors=sum(math.ceil(n/12) for n in lengths.values());assert len(schedule)==832
    old_draws={};checkpoints={}
    for job in ('state','image'):
        path=old/(job+'_train')/'milestone_10000.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['updates']==10000 and saved['contract']['horizon']==24
        assert saved['contract']['action']=='absolute_joint_position_target_rad'
        assert saved['ema_optimization_step']==10000 and saved['optimizer']['state']
        assert all(torch.isfinite(v).all() for kind in ('model','ema') for v in saved[kind].values() if torch.is_tensor(v))
        generator=torch.Generator().manual_seed(20261102);order=[];cursor=0;draws=0
        counts={i:[0]*math.ceil(n/12) for i,n in lengths.items()}
        for _ in range(20000):
            if cursor==len(order):order=torch.randperm(len(schedule),generator=generator).tolist();cursor=0
            identity,t=schedule[order[cursor]];cursor+=1
            for anchor in range(t,min(t+96,lengths[identity]),12):counts[identity][anchor//12]+=1;draws+=1
        assert order==saved['schedule_order'] and cursor==saved['schedule_cursor']
        assert torch.equal(generator.get_state(),saved['schedule_rng'])
        rows=[read(old/(job+'_train')/'metrics.json')[i] for i in range(len(read(old/(job+'_train')/'metrics.json')))]
        old_draws[job]=dict(anchors=draws,anchor_equivalent_epochs=draws/anchors,segments=20000,
            segment_equivalent_epochs=20000/832,average_anchors_per_update=draws/10000,
            per_episode_anchor_counts={i:sum(v) for i,v in counts.items()},per_anchor_counts=counts,
            exact_old_sampler_reconstruction=True,last_lr=[g['lr'] for g in saved['optimizer']['param_groups']],
            milestones={str(r['update']):r['validation']['arm_rmse'] for r in rows})
        checkpoints[job]=dict(path=str(path.resolve()),sha256=sha(path),updates=10000,
            config=saved['configuration'],model_ema_optimizer_rng_present=True)
        del saved
        dest=root/(job+'_audit');dest.mkdir();shutil.copyfile(old/(job+'_audit')/'calibration.json',dest/'calibration.json')
        write(dest/'result.json',dict(passed=True,reused_frozen_calibration=True,calibration_sha256=sha(dest/'calibration.json')))
    now=time.time();session={k:s[k] for k in ('source_root','data_sha256','numeric_cache','position_statistics','expert_checkpoint','expert_sha256','resource_policy','resource_target_percent','specs')}
    session.update(run_id=root.name,old_root=str(old),baseline_root=str(CODE/'outputs/visuotactile/native_dp_v2/final_dp_20261104_01'),
        started_epoch=now,deadline=now+72*3600,training_deadline=now+72*3600,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        owned_worker_roots=[str(root)],checkpoints=checkpoints,lengths=lengths,anchor_population=anchors,segment_population=832,
        effective_batch=64,microbatch=4,lr_peak=3e-5,lr_floor=3e-6,warmup=200,milestones=[50,100,150,200,300],
        old_exposure=old_draws,maximum_anchor_epochs=300,minimum_anchor_epochs=150,
        targets={job:{str(e):math.ceil((e*anchors-old_draws[job]['anchors'])/64) for e in (50,100,150,200,300)} for job in old_draws},
        training_resume_limit=2,kit_maximum_starts=28,kit_recovery_limit=2,closed_loop_ordinals=[4,5,6,7],
        state_improvement_rule='At least one complete success or first subgoal, or higher native lift count on fixed validation4; otherwise Image remains pending_state_evidence and checkpoints retained.',
        stage_expansion_rule='At150/200 stop extension if both latest two validation selection-MSE gains <0.5%; always preserve best EMA and all milestones.',
        rollback_rule='Nonfinite: retain last checkpoint and stop. Two validations >20% worse than original: select saved best and stop extension; never silently reset exposure.',
        image_pair_gate='Two independent same-old-checkpoint live starts: raw RGB-D, DINO, history/condition, RNG, first12control must agree before Image continuation.')
    write(root/'session.json',session)
    # Reuse old immutable data/core hashes, plus new files frozen after implementation.
    write(root/'offline_freeze.json',read(old/'offline_freeze.json'))
    freeze=root/'freeze';freeze.mkdir();shutil.copyfile(old/'freeze/benchmark.json',freeze/'benchmark.json')
    previous=CODE/'outputs/visuotactile/native_dp_v2/final_dp_20261104_01'
    shutil.copyfile(previous/'freeze/user.config.json',freeze/'user.config.json')
    write(root/'status.json',dict(state='preparing',stage='sampling_contract',training_updates=0,pid=None))
    write(root/'audit.json',dict(train_episodes=320,validation_episodes=40,test_episodes=40,
        train_ticks=sum(lengths.values()),anchors=anchors,segments=832,old_exposure=old_draws,
        unchanged=['H24','commit12','absolute29rad','model architecture','Loss B and calibrated auxiliary weights','train-only statistics','original labels/split'],
        sampler_change='New episode-balanced shuffled decks; one native anchor per distinct episode in each effective batch. Full current-weight causal prefix, original96tick TBPTT boundaries.',
        scheduler_change='Old hand-coded cosine ended at0; preserve model/EMA/AdamW moments/step/RNG, replace scheduler by200step3e-6→3e-5 warmup and cosine to3e-6 at cumulative300anchor epochs. New sampler algorithm starts at old schedule RNG state.'))
    print({k:session[k] for k in ('run_id','anchor_population','targets')})
    print({j:{k:v for k,v in d.items() if k not in ('per_episode_anchor_counts','per_anchor_counts')} for j,d in old_draws.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);prepare(a.root)
