"""Evidence for a registered frozen-front/rear-only complete policy comparison."""
import argparse
import collections
import copy
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tools.tight_frozen_frontend import PROJECT, read, sources, write
from visuotactile.compact import sha256


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v
                    for k,v in r.items()} for r in rows)


def paths(config):
    c = read(config)
    root, parent, history = (PROJECT / c[k] for k in ('root','parent','history'))
    public = PROJECT/'research_docs/evidence'/root.name
    public.mkdir(parents=True, exist_ok=True)
    return c,root,parent,history,public


def training_complete(root):
    pointer=root/'training_attempt.json'
    attempt=Path(read(pointer)['resource_root']) if pointer.exists() else root
    assert (attempt/'project_complete.json').exists()
    assert read(attempt/'project_exit.json')['children_reaped']
    assert read(root/'B/backend/cache_complete.json')['complete']


def evaluation_root(root):
    pointer=root/'evaluation_attempt.json'
    return Path(read(pointer)['resource_root']) if pointer.exists() else root/'evaluation'


def evaluation_waves(root):
    pointer=read(root/'evaluation_attempt.json')
    current=sorted(evaluation_root(root).glob('wave*'))
    return [Path(p) for p in pointer.get('included_prior_waves',[])]+current


def evaluation(config):
    from tools.tight_fresh_rollouts import completed_pair
    c,root,parent,history,public = paths(config)
    training_complete(root)
    pairs = {name:completed_pair(p,'B') for name,p in
             [('B72',parent),('B130_joint',history),('B130_frozen',root)]}
    assert pairs['B72']['frontend_sha256'] == pairs['B130_frozen']['frontend_sha256']
    prior=c.get('included_prior_waves',[])
    assert len(prior)==c.get('evaluation_start_wave',0)
    for previous in prior:
        previous=Path(previous)
        assert read(previous/'result.json')['software_complete']
        assert read(previous/'plan.json')['pairs']==pairs,'Do not reuse a wave from different model identities'
    # Reuse the exact registered spec dictionaries, including their native RNG.
    old = read(history.parent/'capability/plan.json')
    specs = {x['spec']['seed']:x for x in old['cases'] if x['role']=='A'}
    session = read(root/'session.json')
    dest = root/c.get('evaluation_directory','evaluation'); dest.mkdir()
    session.update(resource_root=str(dest),owned_worker_roots=[str(dest)])
    write(dest/'session.json',session)
    if c.get('reclaim_unused_cuda'):
        write(dest/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
    stages=[]
    seed_groups=list(zip(c['training_seeds'],c['development_seeds'])) if c['evaluation_lanes']==6 else [(seed,) for pair in zip(c['training_seeds'],c['development_seeds']) for seed in pair]
    assert c['evaluation_lanes'] in (3,6)
    for part,seeds in enumerate(seed_groups):
        if part<c.get('evaluation_start_wave',0):continue
        wave=dest/f'wave{part:02d}';wave.mkdir()
        p=copy.deepcopy(old);p.update(pairs=pairs,cases=[],resource_root=str(dest),source_sha256=sources())
        for seed in seeds:
            roles=list(pairs)
            # Deterministic lane-order rotation, no result-dependent selection.
            roles=roles[part%3:]+roles[:part%3]
            for role in roles:
                x=copy.deepcopy(specs[seed])
                x.update(role=role,name=role+'_'+x['spec']['identity'],
                         video=seed in (28817004,28826004))
                p['cases'].append(x)
        p['batch']['lanes']=c['evaluation_lanes']
        assert len(p['cases'])==c['evaluation_lanes']
        p['evaluation_scope']=dict(protocol='Three B complete policies, same-wave family groups, paired observable initial states',
            source='Existing training8/development8 registration, no new initial families',
            inference_rng='NativeController per-lane sampler_seed7093; reset seed=family+500000, event_seed retained',
            limitation='Opened development; not hidden PhysX-state exact replay or independent final test',test_used=False)
        write(wave/'session.json',session);write(wave/'plan.json',p)
        stages.append(dict(entry='evaluate',args=['--output',str(wave)]))
    job=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(dest),stages=stages)
    job_path=PROJECT/c['job_directory']/c.get('evaluation_job','evaluate.json')
    write(job_path,job)
    write(root/'evaluation_attempt.json',dict(resource_root=str(dest),job=str(job_path),waves=len(stages),included_prior_waves=c.get('included_prior_waves',[])))
    write(public/(dest.name+'_registration.json'),dict(pairs=pairs,config=c,job=job,
        historical_specs_sha256=sha256(history.parent/'capability/plan.json')))
    print(job_path)


@torch.no_grad()
def predictions(model,r):
    from visuotactile.native_data import residual_batch
    from visuotactile.position_control import reachable_position_bounds
    result=collections.defaultdict(list);state=None
    for start in range(0,len(r['robot']),32):
        stop=min(start+32,len(r['robot']))
        inputs=residual_batch([r],start,stop,'cpu')
        raw,state=model(inputs,state)
        base=r['base'][None,start:stop];previous=inputs['robot'][...,58:87]
        chain=model.compose_chain(base,raw,r['gate'][None,start:stop],r['limits'],previous=previous)
        final=chain['u_exec']
        zero,_=model.compose(base,torch.zeros_like(raw),r['gate'][None,start:stop],r['limits'],previous=previous)
        bounds=reachable_position_bounds(previous,r['limits'],model.execution_adapter)
        result['dp'].append(base[0]);result['anchor'].append(zero[0]);result['final'].append(final[0])
        result['correction'].append((chain['u_pre']-zero)[0]);result['span'].append((bounds[...,1]-bounds[...,0])[0].clamp_min(1e-6))
    return {k:torch.cat(v) for k,v in result.items()}


def offline(config):
    from tools.tight_representation_probe import source
    from visuotactile.models.native_dp import load_residual
    from visuotactile.geometry_sampling import STAGES
    c,root,parent,history,public=paths(config)
    training_complete(root)
    data=read(root/'B/state_data/data.json')
    parent_ids=set(read(parent/'B/state_data/data.json')['splits']['train'])
    original=set(data['normalization_train_ids']);validation=set(data['splits']['validation'])
    names=['B72','B130_joint','B130_frozen']
    roots=[parent,history,root]
    models={name:load_residual(torch.load(p/'B/backend/B/training/best.pt',map_location='cpu',weights_only=False)).eval()
            for name,p in zip(names,roots)}
    # The72 policy and frozen-front candidate share these exact nominal inputs,
    # including all58 trajectories absent from72 training. Joint uses its own DP.
    caches={'B72':root/'B/backend/B/nominal','B130_frozen':root/'B/backend/B/nominal',
            'B130_joint':history/'B/backend/B/nominal'}
    indices={name:read(p/'index.json') for name,p in caches.items()}
    rows=[];deltas=[];began=time.time()
    for n,i in enumerate(data['splits']['train']+data['splits']['validation']):
        s=source(data['entries'][i]);truth=torch.from_numpy(s['command'])
        cohort='validation6' if i in validation else 'original30' if i in original else 'old42' if i in parent_ids else 'new58'
        masks={'legal':s['legal'],'first24':s['onset']}
        masks.update({stage:s['legal']&(s['phase']==k) for k,stage in enumerate(STAGES)})
        outputs={}
        records={}
        for name in names:
            p=caches[name]/(i+'.pt');assert sha256(p)==indices[name]['episodes'][i]['sha256']
            r=torch.load(p,map_location='cpu',weights_only=False)
            assert np.array_equal(r['supervision_mask'].numpy(),s['legal'])
            assert torch.equal(r['target'][r['supervision_mask']],truth[r['supervision_mask']])
            records[name]=r;out=predictions(models[name],r);outputs[name]=out
            for stage,mask in masks.items():
                if not mask.any():continue
                for component in ('dp','anchor','correction','final'):
                    target=truth-out['anchor'] if component=='correction' else truth
                    err=(out[component]-target)[mask];span=out['span'][mask]
                    rows.append(dict(model=name,identity=i,family=data['entries'][i]['family'],cohort=cohort,
                        stage=stage,component=component,ticks=len(err),
                        arm_squared_sum=float(err[:,:7].square().sum()),hand_squared_sum=float(err[:,7:].square().sum()),
                        arm_span_squared_sum=float((err[:,:7]/span[:,:7]).square().sum()),
                        hand_span_squared_sum=float((err[:,7:]/span[:,7:]).square().sum())))
        for name in names[1:]:
            for key in ('dp','anchor','correction','final'):
                delta=outputs[name][key]-outputs['B72'][key]
                deltas.append(dict(model=name,identity=i,component=key,max_rad=float(delta.abs().max()),
                                   rms_rad=float(delta.square().mean().sqrt())))
        assert torch.equal(outputs['B72']['dp'],outputs['B130_frozen']['dp'])
        assert torch.equal(outputs['B72']['anchor'],outputs['B130_frozen']['anchor'])
        if n%20==0:print('offline',n,round(time.time()-began,1),flush=True)
    table(public/'offline_per_episode.csv',rows);table(public/'same_input_changes.csv',deltas)
    agg=collections.defaultdict(collections.Counter)
    for r in rows:
        agg[r['model'],r['cohort'],r['stage'],r['component']].update({k:r[k] for k in
            ('ticks','arm_squared_sum','hand_squared_sum','arm_span_squared_sum','hand_span_squared_sum')})
    summary=[]
    for (name,cohort,stage,component),v in agg.items():
        summary.append(dict(model=name,cohort=cohort,stage=stage,component=component,ticks=v['ticks'],
            arm_rmse_rad=(v['arm_squared_sum']/v['ticks']/7)**.5,
            hand_rmse_rad=(v['hand_squared_sum']/v['ticks']/22)**.5,
            arm_span_rmse=(v['arm_span_squared_sum']/v['ticks']/7)**.5,
            hand_span_rmse=(v['hand_span_squared_sum']/v['ticks']/22)**.5))
    table(public/'offline_summary.csv',summary)
    write(public/'offline_scope.json',dict(episodes=136,seconds=time.time()-began,
        raw_inputs='Same recorded RGB-D/robot/tactile histories, each policy recomputes own hidden states; matched full nominal caches.',
        target='Actual expert target_after_rad, only lawful expert mask. No expert counterfactuals for student prefix.',
        anchor='Zero-rear composed command: reachable nominal center. Correction is preclip rear proposal minus that center, compared against expert minus center; final error is measured after actual execution constraints.',
        frozen_dp_exact=True,frozen_anchor_exact=True,
        new58='Not used to train72; previously used for joint130 and now frozen130; training fit, not generalization.'))


def training(config):
    c,root,parent,history,public=paths(config);training_complete(root)
    folder=root/'B/backend/B/training';selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    initial=torch.load(parent/'B/backend/B/training/best.pt',map_location='cpu',weights_only=False)
    done=read(folder/'finished.json')
    assert done['updates']==c['historical_selected_updates']==4839 and done['target_reached']
    assert done['base_optimizer_updates']==0 and selected['configuration']['base_sha256']==sha256(parent/'B/training/best.pt')
    original=set(read(root/'B/state_data/data.json')['normalization_train_ids'])
    assert all(v==(45 if i in original else 15) for i,v in last['episode_counts'].items())
    parameters=[]
    for key,v in selected['model'].items():
        diff=v-initial['model'][key]
        parameters.append(dict(parameter=key,elements=v.numel(),max_change=float(diff.abs().max()),rms_change=float(diff.float().square().mean().sqrt())))
    table(public/'rear_parameter_changes.csv',parameters)
    exposure=[dict(identity=i,original=i in original,actual_visits=last['episode_counts'][i],selected_visits=v)
              for i,v in selected['episode_counts'].items()]
    table(public/'selected_exposure.csv',exposure)
    write(public/'training_result.json',dict(actual_updates=last['updates'],selected_updates=selected['updates'],selected_epoch=selected['epoch'],
        selected_supervised_ticks=selected['supervised_ticks'],actual_supervised_ticks=last['supervised_ticks'],
        frontend_sha256=sha256(root/'B/training/best.pt'),backend_sha256=sha256(folder/'best.pt'),
        parent_backend_sha256=sha256(parent/'B/backend/B/training/best.pt'),
        optimizer='Fresh AdamW, initial lr1e-4; same original adaptive validation plateau rule, no EMA',
        selected_optimizer_lrs=[g['lr'] for g in selected['optimizer']['param_groups']],
        selected_optimizer_steps=sorted({int(x['step']) for x in selected['optimizer']['state'].values()}),
        selected_exposure=exposure,finished=done,base_optimizer_updates=0))
    dest=public/'training';dest.mkdir(exist_ok=True)
    for name in ('configuration.json','finished.json','selected.json','validation.json','curves.png','latest_fit_initial.json'):
        if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
    for name in ('registration.json','training_attempt.json'):
        if (root/name).exists():shutil.copyfile(root/name,public/name)
    shutil.copyfile(root/'B/backend/cache_complete.json',public/'cache_complete.json')
    shutil.copyfile(root/'B/backend/B/offline/result.json',public/'selected_offline.json')
    shutil.copyfile(root/'B/backend/B/offline/latest_fit_selected.json',public/'latest_fit_selected.json')
    # Historical and frozen curves share the same independent validation6.
    curves=[]
    for model,p in [('B130_joint',history/'B/backend/B/training'),('B130_frozen',folder)]:
        for r in read(p/'validation.json'):
            curves.append(dict(model=model,epoch=r['epoch'],updates=r['update'],score=r['selection_score'],
                train2=r['train']['all']['normalized_equal_mse'],learning_rates=r.get('learning_rates')))
    table(public/'validation_curves.csv',curves)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for model in ('B130_joint','B130_frozen'):
        rr=[r for r in curves if r['model']==model]
        ax.plot([r['updates'] for r in rr],[r['score'] for r in rr],marker='.',label=model)
    ax.axvline(4839,color='gray',linestyle='--',label='Registered selected-history budget')
    ax.set(xlabel='New rear optimizer updates',ylabel='Validation6 equal normalized MSE');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(public/'validation_curves.png',dpi=150);plt.close(fig)
    print('training published',selected['updates'],selected['epoch'])


def report(config):
    from scripts.tight_layout_20261116.phase_report import summarize
    c,root,parent,history,public=paths(config)
    evaluation_dir=evaluation_root(root)
    wave_count=read(evaluation_dir/'project_complete.json')['stages']
    waves=evaluation_waves(root)
    assert wave_count==len(list(evaluation_dir.glob('wave*')))
    assert len(waves)*c['evaluation_lanes']==48
    rows=[];pairing=[]
    for d in waves:
        dest=public/'evaluation'/d.name;dest.mkdir(parents=True,exist_ok=True)
        summarize(d,dest/'summary.json')
        for r in read(dest/'summary.json')['rows']:
            r['cohort']='training' if r['seed'] in c['training_seeds'] else 'development';rows.append(r)
        pairing.extend(read(d/'paired_initial_check.json')['rows'])
        for name in ('plan.json','result.json','paired_initial_check.json','videos.json','wave_profiles.json'):
            shutil.copyfile(d/name,dest/name)
    assert len(rows)==48 and len({(r['role'],r['seed']) for r in rows})==48
    counts=[];changes=[]
    for co in ('training','development'):
        for name in ('B72','B130_joint','B130_frozen'):
            rr=[r for r in rows if r['cohort']==co and r['role']==name]
            counts.append(dict(cohort=co,model=name,n=len(rr),legal=sum(r['legal'] for r in rr),
                success=sum(r['success'] for r in rr),hold60=sum(r['hold60']['sustained'] for r in rr),
                hold120=sum(r['hold120']['sustained'] for r in rr),first_subgoal=sum(r['native_subgoals']>0 for r in rr),
                near_goal30mm=sum(r['student_near_goal']['minimum_keypoint_distance_m']<.03 for r in rr)))
    for seed in c['training_seeds']+c['development_seeds']:
        by={r['role']:r for r in rows if r['seed']==seed};a=by['B72']
        for name in ('B130_joint','B130_frozen'):
            b=by[name];changes.append(dict(seed=seed,cohort=a['cohort'],model=name,
                baseline_success=a['success'],candidate_success=b['success'],
                retained=bool(a['success'] and b['success']),lost=bool(a['success'] and not b['success']),
                gained=bool(not a['success'] and b['success']),baseline_hold60=a['hold60']['sustained'],candidate_hold60=b['hold60']['sustained']))
    table(public/'per_case.csv',rows);table(public/'counts.csv',counts);table(public/'paired_changes.csv',changes)
    write(public/'results.json',dict(counts=counts,changes=changes,initial_pairing=pairing,
        scope='48 complete autonomous runs; same-wave three-way observable-state pairing. Development already opened; small sample, no significance or exact hidden replay claim.'))
    print(json.dumps(counts))


def closure(config):
    import os
    c,root,parent,history,public=paths(config)
    training_complete(root)
    evaluation_dir=evaluation_root(root)
    wave_count=read(evaluation_dir/'project_complete.json')['stages']
    waves=evaluation_waves(root)
    assert wave_count==len(list(evaluation_dir.glob('wave*')))
    assert len(waves)*c['evaluation_lanes']==48
    assert read(evaluation_dir/'project_exit.json')['children_reaped']
    protected=read(public/'admission.json')['protected']+read(public/'additional_protected.json')['files']
    for r in protected:
        r['closure_sha256']=sha256(r['path'])
        assert r['closure_sha256']==r['sha256']
    data=read(root/'B/state_data/data.json')
    assert len(data['splits']['train'])==130
    assert all(Path(e['path']).is_file() for e in data['entries'].values())
    source=read(root/'registration.json')['source_sha256']
    changes=[p for p,h in source.items() if sha256(PROJECT/p)!=h]
    assert set(changes)<= {'tools/tight_frozen_frontend.py','visuotactile/state_resources.py'}
    resources=[]
    for label,d in [('cache_and_rejected_cli',root),('training',root/'training_attempt02'),('evaluation_vram_interrupted',root/'evaluation'),('evaluation_disk_interrupted',root/'evaluation_attempt02'),('evaluation_allocator_interrupted',root/'evaluation_attempt03'),('evaluation',evaluation_dir)]:
        rr=[json.loads(l) for l in (d/'resources/resources.jsonl').read_text().splitlines()]
        duration=area=0.;gaps=[]
        for a,b in zip(rr,rr[1:]):
            dt=b['utc_epoch']-a['utc_epoch'];duration+=dt
            if dt>4 or a['gpu_util_percent'] is None:gaps.append(dt)
            else:area+=dt*a['gpu_util_percent']
        summary=dict(stage=label,samples=len(rr),seconds=duration,
            gpu_card_time_mean_percent=area/duration if duration else None,
            max_complete60s_percent=max((x['rolling_60s_percent'] for x in rr if x.get('rolling_60s_percent') is not None),default=None),
            telemetry_gaps=gaps,
            min_gpu_free_mib=min(x['gpu_free_mib'] for x in rr if x.get('gpu_free_mib') is not None),
            min_ram_available_gib=min(x['ram_available_bytes']/2**30 for x in rr),
            policy='full_gpu_v1; no50percent cap, preserve4GiB VRAM/10GiB RAM/12GiB disk')
        resources.append(summary)
        dest=public/'closure'/label;dest.mkdir(parents=True,exist_ok=True)
        for name in ('project_job_started.json','project_complete.json','project_failure.json','project_exit.json','session.json'):
            if (d/name).exists():shutil.copyfile(d/name,dest/name)
        table(dest/'resources.csv',rr)
    wave_sources=[]
    for wave in waves:
        assert read(wave/'paired_initial_check.json')['passed']
        pid=read(wave/'run_started.json')['pid'];assert not (Path('/proc')/str(pid)).exists()
        differences={p:dict(registered=h,current=sha256(PROJECT/p)) for p,h in read(wave/'plan.json')['source_sha256'].items() if sha256(PROJECT/p)!=h}
        assert set(differences)<= {'tools/tight_frozen_comparison.py'},differences
        wave_sources.append(dict(wave=str(wave),diagnostic_only_changes=differences))
    # Report matching owned Python workers, excluding this closure process.
    running=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        try:
            argv=(d/'cmdline').read_bytes().split(b'\0')
            if argv and b'python' in argv[0] and str(root).encode() in b' '.join(argv):running.append(int(d.name))
        except OSError:pass
    assert not running,running
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    write(public/'closure.json',dict(protected=protected,data_entries_existing=len(data['entries']),
        source_changed_registration_only=changes,wave_sources=wave_sources,resources=resources,owned_root_workers_remaining=running,
        new_local_regular_file_bytes=size,free_disk_bytes=shutil.disk_usage(root).free,
        hardware_scope='Disk files, host RAM availability and GPU VRAM reported separately. Whole-card utilization includes unrelated jobs.',
        utc=time.time(),elapsed_from_registration_s=time.time()-read(root/'session.json')['started_utc']))
    print('closed',len(protected),'protected files;48cases; no owned workers')


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['evaluation','offline','report','training','closure']);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a.config)


if __name__=='__main__':
    main()
