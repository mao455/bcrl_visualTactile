"""Evidence-driven finite repair continuation, with unchanged absolute deadline."""
import argparse,copy,json,os,subprocess,time,traceback
from pathlib import Path
import torch
from tools.runtime import RUN,read,write,sha,kill_owned,supervised_owner
from tools.tight_execution_learning import GROUPS,HISTORY,PARENT,model_and_records,probe_sequence,evaluate,write_csv
from visuotactile.training_resources import atomic_json
from visuotactile.execution_learning import PROJECTION_FY_LOSS
from visuotactile.residual_output import REACHABLE_TAIL_OUTPUT
from visuotactile.models.native_dp import TargetResidual
from visuotactile.native_data import residual_batch
from visuotactile.anchor_sampling import supervision_rows,recovery_prefix_rows
from visuotactile.anchor_continue import residual_prefix_state


def summary(rows):
    def mass(pred):return sum(r['normalized_weighted_exec_error_mass']*.5/(7 if r['part']=='Arm' else 22) for r in rows if pred(r))
    rr=[r for r in rows if r['partition']=='train' and r['stratum']=='recovery_initial48' and r['part']=='Hand']
    return dict(dev=mass(lambda r:r['partition']=='dev'),core=mass(lambda r:r['stratum']=='core'),
        train=mass(lambda r:r['partition']=='train'),direction=sum(r['direction_correct_coordinates'] for r in rr)/max(1,sum(r['moving_coordinates'] for r in rr)),
        saturation=sum(r['saturated_coordinates'] for r in rows)/sum(r['joint_coordinates'] for r in rows))


def comparison(candidate,old):
    dev=candidate['dev']/max(old['dev'],1e-30);core=candidate['core']/max(old['core'],1e-30)
    direction=candidate['direction']-old['direction'];sat=candidate['saturation']-old['saturation']
    return dict(dev_exec_ratio=dev,core_exec_ratio=core,recovery_hand_direction_difference=direction,
        saturation_fraction_difference=sat,pass_all=dev<=1.02 and core<=1.05 and direction>=-.01 and sat<=.01,
        tail_note='For linear tails, |tanh(z)|>=.95 marks tail occupancy, not vanishing tail gradient')


def train_candidate(root,spec):
    from visuotactile.state_resources import lease_for
    reg=read(root/'adaptive_runtime.json');manifest=read(root/'manifest.json');dest=root/spec['name'];dest.mkdir(exist_ok=False)
    write(dest/'registration.json',dict(spec,started=time.time(),deadline=reg['compute_deadline'],
        disposable=True,optimizer='fresh identical AdamW for each group/candidate',history=HISTORY,
        same_sequence=True,test22_reads=0,mathematical_loss_effect_isolated=spec['output_contract'] is None))
    lease=lease_for(root,reg['compute_deadline']-10800);result={}
    for group in GROUPS:
        _,records,payload=model_and_records(manifest,group)
        torch.manual_seed(2026111301);torch.cuda.manual_seed_all(2026111301)
        model=TargetResidual(payload['model'],payload['model']['scales'],execution_adapter=payload['configuration']['execution_adapter'],
            loss_contract=spec['loss_contract'],history_contract=HISTORY,output_contract=spec['output_contract'])
        model.load_state_dict(payload['model'])
        with lease.block():model.cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=1e-4)
        sequence=probe_sequence(manifest,records,spec['updates']);rows=[];checkpoints=[];began=time.time()
        for update,item in enumerate(sequence,1):
            if time.time()>reg['adaptive_probe_deadline']:raise TimeoutError('Registered adaptive optimization window exhausted')
            with lease.block():
                record=records[item['identity']];a,b=item['start'],item['stop'];model.train()
                prefix=[residual_batch([record],t,min(t+32,a),'cuda') for t in range(0,a,32)]
                state=residual_prefix_state(model,prefix);inputs=residual_batch([record],a,b,'cuda');raw,_=model(inputs,state)
                chain=model.compose_chain(record['base'][None,a:b].cuda(),raw,None,record['limits'].cuda(),previous=inputs['robot'][...,58:87])
                mask,_=supervision_rows(record);early=recovery_prefix_rows(record,48) if item['bank']!='core' else torch.zeros_like(mask)
                weight=mask[a:b].float()*(1+2*early[a:b].float())
                parts=model.objective_components(chain['u_exec'],record['target'][None,a:b].cuda(),chain['delta'],weight[None].cuda(),
                    u_pre=chain['u_pre'],execution_bounds=chain.get('execution_bounds'))
                gz=torch.autograd.grad(parts['total'],raw,retain_graph=True)[0]
                opt.zero_grad(set_to_none=True);parts['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['clip'])
                if not torch.isfinite(norm) or not torch.isfinite(parts['total']):raise ValueError('Nonfinite adaptive candidate')
                opt.step()
            rows.append(dict(update=update,supervised_ticks=int(mask[a:b].sum()),exec_fit=float(parts['exec_fit'].detach()),
                main_fit=float(parts.get('fy_fit',parts['exec_fit']).detach()),regularizer=float(parts['regularizer'].detach()),
                parameter_norm=float(norm),logit_gradient_l2=float(gz.norm()),elapsed_s=time.time()-began))
            if update%20==0:write(root/'status.json',dict(state='RUNNING',stage='adaptive_probe',candidate=spec['name'],group=group,updates=update,utc=time.time()))
            if update in sorted(set([200,400,spec['updates']])):
                tag=spec['name']+'_u'+str(update)
                with lease.block():metrics=evaluate(model,records,manifest,group,root,tag)
                checkpoints.append(dict(update=update,summary=summary(metrics)))
                write(dest/(group+'_checkpoints.json'),checkpoints)
        target=dest/group;target.mkdir()
        cfg=dict(payload['configuration'],loss_contract=spec['loss_contract'],history_contract=HISTORY,output_contract=spec['output_contract'],disposable=True)
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},updates=spec['updates'],configuration=cfg,
            residual_version=model.version,residual_loss_version=(spec['loss_contract'] or {}).get('version'),
            residual_history_version=HISTORY['version'],residual_output_version=(spec['output_contract'] or {}).get('version'),
            contract=payload['contract'],disposable=True),target/'probe.pt')
        write_csv(target/'training.csv',rows)
        result[group]=dict(summary=checkpoints[-1]['summary'],checkpoints=checkpoints,updates=spec['updates'],
            supervised_ticks=sum(r['supervised_ticks'] for r in rows),optimization_wall_s=rows[-1]['elapsed_s'],
            gradient_clip_fraction=sum(r['parameter_norm']>spec['clip'] for r in rows)/len(rows),
            checkpoint=str(target/'probe.pt'),sha256=sha(target/'probe.pt'))
        write(dest/'results.json',result);del model,opt;torch.cuda.empty_cache()
    return result


@supervised_owner
def probe_run(root):
    reg=read(root/'registration.json');monitor=None
    if (root/'adaptive_probe_launch.json').exists():raise ValueError('No implicit repeat of adaptive probe')
    write(root/'adaptive_probe_launch.json',dict(pid=os.getpid(),utc=time.time(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    specs=[dict(name='old800',loss_contract=None,output_contract=None,lr=3e-4,clip=1.,updates=800),
           dict(name='tail800',loss_contract=PROJECTION_FY_LOSS,output_contract=REACHABLE_TAIL_OUTPUT,lr=3e-4,clip=1.,updates=800)]
    write(root/'adaptive_probe_plan.json',dict(specs=specs,conditional_adjustment='If executed dev/Core/direction fail, compare tail with zero residual penalty and LR1e-4, 1600 updates; preserve every comparison',
        maximum_total_wall_s=7200,original_gates_preserved=True,capacity_floor_fraction_A=read(root/'capacity.json')['groups'][0]['floor_fraction_of_actual']))
    reg['adaptive_probe_deadline']=min(time.time()+7200,reg['compute_deadline']-14400)
    write(root/'adaptive_probe_budget.json',dict(deadline=reg['adaptive_probe_deadline'],registered=time.time(),absolute_deadline=reg['absolute_deadline']))
    # New amendment is separate; never overwrite the initial 24h registration.
    write(root/'adaptive_runtime.json',reg)
    try:
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        with (root/'adaptive_monitor.log').open('a') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        began=time.time()
        while not (root/'shared_resources.json').exists() or time.time()-read(root/'shared_resources.json')['utc_epoch']>3:
            if monitor.poll() is not None or time.time()-began>30:raise RuntimeError('Resource monitor startup')
            time.sleep(.2)
        results={}
        for spec in specs:results[spec['name']]=train_candidate(root,spec)
        checks={g:comparison(results['tail800'][g]['summary'],results['old800'][g]['summary']) for g in GROUPS}
        write(root/'tail800_comparison.json',checks)
        if not all(r['pass_all'] for r in checks.values()):
            spec=dict(name='tail_tuned1600',loss_contract=dict(PROJECTION_FY_LOSS,regularizer_weight=0.),
                output_contract=REACHABLE_TAIL_OUTPUT,lr=1e-4,clip=1.,updates=1600)
            write(root/'next_hypothesis.json',dict(observed=checks,hypothesis='Large normalized tail residual regularization and larger FY update can conflict with real execution/Core; remove non-equivalence penalty and reduce optimizer step',
                changes=spec,not_a_single_variable_FY_claim=True,comparison_budget_difference='1600 candidate versus 800 old: exposure difference explicit'))
            results[spec['name']]=train_candidate(root,spec);specs.append(spec)
        candidates=specs[1:]
        def rank(spec):
            values=[comparison(results[spec['name']][g]['summary'],results['old800'][g]['summary']) for g in GROUPS]
            return (not all(v['pass_all'] for v in values),max(v['dev_exec_ratio']+v['core_exec_ratio'] for v in values))
        chosen=min(candidates,key=rank)
        comparisons={g:comparison(results[chosen['name']][g]['summary'],results['old800'][g]['summary']) for g in GROUPS}
        write(root/'adaptive_selection.json',dict(chosen=chosen,comparisons=comparisons,paired_probe_passed=all(r['pass_all'] for r in comparisons.values()),
            next_stage='Full mixed-data conditional adaptation from protected original trained pair; disposable weights never deployed',
            candidate_status='research candidate, not replacement of protected original model',results=results,
            reason='Capacity bottleneck demonstrated; broader legal mixture tests whether fixed-six conflict generalizes; first gate failure is not terminal'))
        write(root/'adaptive_probe_complete.json',dict(complete=True,utc=time.time()))
    finally:
        (root/'STOP_MONITOR').touch()
        if monitor is not None:
            try:monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        write(root/'adaptive_probe_exit.json',dict(utc=time.time(),monitor_exited=monitor is None or monitor.poll() is not None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);probe_run(args.root.resolve())
