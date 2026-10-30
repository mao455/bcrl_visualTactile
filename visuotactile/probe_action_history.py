"""Bounded real-prefix fit and executed-history replay; never formal training."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from .action_history import ExecutionHistory
from .audit_baselines import DiagnosticBudget, command_errors, sample_frozen
from .compact import EpisodeFile, sha256
from .manifest import ManifestSequenceDataset
from .models.nominal import PolicyConfig
from .rollout import load_command_controller
from .train_dp import _make_model, _condition, _move, _save
from .training_resources import atomic_json


def initialize_history_model(payload, device):
    config=PolicyConfig.from_dict({**payload['config'],'action_history_mode':'issued','vision_checkpoint_images':3})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(730)
        model=_make_model(payload['variant'],config)
    parameters=model.state_dict()
    for name,value in payload['model'].items():
        if parameters[name].shape==value.shape:
            parameters[name]=value
        elif name.endswith('context.0.weight'):
            parameters[name].zero_()
            parameters[name][:,:value.shape[1]]=value
        else:
            raise ValueError('Unexpected warm-start shape change: '+name)
    model.load_state_dict(parameters)
    return model.to(device)


@torch.no_grad()
def assess(model, variant, data, budget, seeds=(190,191,192)):
    model.eval()
    with budget.block():
        condition,_=_condition(model,variant,data,None)
    indices=torch.arange(0,condition.shape[1],6,device=condition.device)
    target=data['action_target'][indices]
    valid=data['action_valid'][indices]
    rows=[]
    for seed in seeds:
        generator=torch.Generator(device=condition.device).manual_seed(seed)
        noise=torch.randn(target.shape,device=condition.device,generator=generator)
        with budget.block():
            prediction=sample_frozen(model,condition[0,indices],noise,8)
        rows.append(command_errors(prediction,target,valid))
    return dict(noise_seeds=list(seeds),samples=rows,
                mean_first6_arm_rmse=sum(row['first6_arm_rmse'] for row in rows)/len(rows))


def verify_real_execution_context(dataset, identity, model, variant, device, budget, limit):
    row=dataset.reader.entries[identity]
    history=ExecutionHistory(1)
    state=None
    max_difference=0.
    observations=[]
    with EpisodeFile(dataset.datasets[identity].path) as stream:
        group=stream['episodes'][row['episode']]
        for step,chunk in enumerate(dataset.chunks(identity,1)):
            if step>=limit:
                break
            packet={name:value[:,0] for name,value in chunk['obs'].items() if name!='valid'}
            packet.update(episode_id=torch.zeros(1,dtype=torch.long),step_id=torch.tensor([step]))
            online=history.context(packet)
            for name,value in online.items():
                torch.testing.assert_close(value,packet[name],atol=0,rtol=0)
            observations.append(packet)
            transition={name:torch.from_numpy(group['transition/'+name][step:step+1]) for name in
                        ('command_clipped_pre_delay','time_before_s','time_after_s')}
            transition.update(episode_id=packet['episode_id'],step_id=packet['step_id'],control_trace={
                name:torch.from_numpy(group['transition/control_trace/'+name][step:step+1]) for name in
                ('valid','delayed_command','selected_command_time_s')})
            history.observe(transition,transition['command_clipped_pre_delay'],packet['episode_id'],packet['step_id'])
            with torch.no_grad(),budget.block():
                condition,state=_condition(model,variant,_move(chunk,torch.device(device)),state)
            observations[-1]['condition']=condition.cpu()
        chunk=next(dataset.chunks(identity,limit))
        with torch.no_grad(),budget.block():
            packed,_=_condition(model,variant,_move(chunk,torch.device(device)),None)
        sequential=torch.cat([packet['condition'] for packet in observations],1)
        max_difference=float((sequential-packed.cpu()).abs().max())
    return dict(execution_context_exact=True,observations=len(observations),
                full_vs_step_condition_max_error=max_difference,
                source='actual expert transitions replayed as recorded execution; no student execution claim')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=240)
    parser.add_argument('--prefix',type=int,default=24)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if not 1<=args.updates<=300 or not 8<=args.prefix<=32:
        parser.error('Bounded probe requires <=300 updates and 8..32 prefix observations')
    args.output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parent
    sources=[source_root/name for name in ('probe_action_history.py','audit_baselines.py',
        'action_history.py','dataset.py','manifest.py','train_dp.py','rollout.py','observations.py')]
    sources+=sorted((source_root/'models').glob('*.py'))
    source_hashes={str(path.relative_to(source_root.parent)):sha256(path) for path in sources}
    atomic_json(args.output/'plan.json',dict(updates=args.updates,prefix=args.prefix,
        manifest_sha256=sha256(args.manifest),device=args.device,training_seed=730,
        source_sha256=source_hashes))
    torch.set_num_threads(2)
    torch.manual_seed(730)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    budget=DiagnosticBudget(args.output/'resources',args.device)
    report=dict(protocol='diagnostic v3 teacher-forced executed history; single recorded expert prefix',
                updates_per_variant=args.updates,prefix_observations=args.prefix,variants={},
                initialization='v2 weights copied explicitly; new history context columns zero; new history MLP random',
                limitation='fitting a recorded prefix is not closed-loop learning or evidence of generalization')
    try:
        for variant in ('b0','b1'):
            payload=torch.load(args.root/variant/'best.pt',map_location='cpu',weights_only=False)
            identity=payload['train_episodes'][0]
            dataset=ManifestSequenceDataset(args.manifest,ids=[identity],stage='production',
                include_tactile=variant=='b1',action_history_mode='issued')
            with budget.block():
                model=initialize_history_model(payload,args.device)
            replay=verify_real_execution_context(dataset,identity,model,variant,args.device,budget,args.prefix)
            data=_move(next(dataset.chunks(identity,args.prefix)),torch.device(args.device))
            before=assess(model,variant,data,budget)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-6)
            losses=[]
            gradient_max=0.
            encoder=model.encoder if variant=='b0' else model.visual
            started=time.perf_counter()
            for update in range(args.updates):
                model.train()
                with budget.block():
                    optimizer.zero_grad(set_to_none=True)
                    condition,_=_condition(model,variant,data,None)
                    loss=model.loss_from_condition(condition[0],data['action_target'],data['action_valid'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite diagnostic fit')
                    loss.backward()
                    gradient=encoder.action_history[0].weight.grad
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError('Non-finite history gradient')
                    gradient_max=max(gradient_max,float(gradient.norm()))
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Non-finite diagnostic gradient')
                    optimizer.step()
                    losses.append(float(loss))
                if (update+1)%40==0:
                    print(variant,'updates',update+1,'loss',sum(losses[-20:])/len(losses[-20:]),flush=True)
            after=assess(model,variant,data,budget)
            ablated=copy.copy(data)
            ablated['obs']={**data['obs'],'action_history_valid':torch.zeros_like(data['obs']['action_history_valid'])}
            without_history=assess(model,variant,ablated,budget)
            target=args.output/variant
            target.mkdir()
            options=SimpleNamespace(variant=variant,chunk_size=args.prefix,batch_size=1,seed=41,amp='off',
                                    training_source_sha256=source_hashes)
            _save(target/'diagnostic.pt',model,optimizer,model.cfg,options,payload['manifest_sha256'],
                  (payload['train_episodes'],payload['validation_episodes'],payload['test_episodes']),
                  0,[],encoder.proprio_mean,encoder.proprio_std,metadata=payload['metadata'],optimizer_updates=args.updates)
            saved=torch.load(target/'diagnostic.pt',map_location='cpu',weights_only=False)
            saved.update(diagnostic_only=True,warm_start_sha256=sha256(args.root/variant/'best.pt'),
                         diagnostic_training=dict(episode=identity,prefix=args.prefix,updates=args.updates,
                                                  lr=3e-4,training_seed=730))
            torch.save(saved,target/'diagnostic.pt')
            with budget.block():
                restored=load_command_controller(target/'diagnostic.pt',1,device=args.device)
            restored_result=assess(restored.model,variant,data,budget)
            if after!=restored_result:
                raise AssertionError('Saved history policy differs on fixed-noise sampling')
            report['variants'][variant]=dict(episode=identity,source_sha256=saved['warm_start_sha256'],
                replay=replay,before=before,after=after,history_masked=without_history,
                history_gradient_max=gradient_max,loss_first20=sum(losses[:20])/len(losses[:20]),
                loss_last20=sum(losses[-20:])/len(losses[-20:]),save_reload_exact=True,
                wall_seconds=time.perf_counter()-started,config=asdict(model.cfg))
            atomic_json(args.output/'report.json',report)
            print(variant,'first6 arm',before['mean_first6_arm_rmse'],'->',after['mean_first6_arm_rmse'],flush=True)
            del model,restored,optimizer,data,payload,dataset,saved
            if torch.device(args.device).type=='cuda':
                torch.cuda.empty_cache()
    finally:
        report['budget']=budget.summary()
        atomic_json(args.output/'report.json',report)


if __name__=='__main__':
    main()
