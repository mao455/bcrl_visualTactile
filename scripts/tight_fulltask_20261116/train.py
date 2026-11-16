"""One shared-data, equal-update A/B refresh, then full native task evaluation."""
import copy,os,sys,time,traceback
from pathlib import Path
from scripts.tight_fulltask_20261116.run import ROOT,OLD,TREE,DIAG,DEADLINE,execute,read,atomic_json,monitor_start
from visuotactile.compact import sha256
OUT=ROOT/'update'

def eval_prepare(directory,pairs):
    from visuotactile.narrow_data import direct_configuration
    template=copy.deepcopy(read(DIAG/'candidates/plan.json'));session=read(ROOT/'session.json')
    template.update(pairs=pairs,resource_root=str(ROOT.resolve()),cases=[],utc=time.time(),
        source_sha256={str(p):sha256(p) for folder in ['tools','visuotactile','scripts/tight_fulltask_20261116'] for p in Path(folder).rglob('*.py')})
    template['batch'].update(lanes=8,seed=28600200)
    template['evaluation_scope']=dict(protocol='Full native natural termination for every legal case; no grasp censor',
        seeds=read(ROOT/'registration.json')['evaluation_seeds'],split='Previously observed development, not final test; no training admission')
    for seed in template['evaluation_scope']['seeds']:
        for role in 'AB':
            spec=dict(identity=f'fulltask_development_{seed}',family=f'fresh_development_{seed}',split='development',seed=seed,event_seed=seed+100000,
                cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(session['domain'],seed))
            template['cases'].append(dict(name=role+'_'+spec['identity'],role=role,spec=spec,video=True))
    directory.mkdir();atomic_json(directory/'plan.json',template);atomic_json(directory/'session.json',session)

def run():
    from tools.tight_fresh_cycle import configure_frontend_sampling,backend_chain,parallel_admission
    from visuotactile.trajectory_training import profile,budget
    from visuotactile.latest_takeover_sampling import sampling_plan,SAMPLING_VERSION
    from concurrent.futures import ThreadPoolExecutor
    assert read(ROOT/'collection/result.json')['software_complete']
    entries=read(ROOT/'collection/correction_entries.json');assert entries,'No successful correction; retain videos and diagnose rather than train empty pool'
    # Separate, immutable incremental budget from the preliminary collection registration.
    epochs=2;rear_epochs=2
    atomic_json(ROOT/'training_registration.json',dict(dp_epochs=epochs,rear_epochs=rear_epochs,latest_repeat=12,
        started=time.time(),deadline=DEADLINE,reason='Measured previous DP ~90s/epoch; equal bounded full-pool updates with time reserved for full native evaluation',
        source_initialization=read(ROOT/'registration.json')['optimizer'],new_successes=len(entries)))
    execute('aggregate',['tools.tight_fresh_dagger','aggregate','--root',ROOT,'--collection',ROOT/'collection','--output',OUT])
    for name in ['training_sampling.json','frozen_visual_io.json','backend_phase_weighting.json']:
        atomic_json(OUT/name,read(OLD/name))
    atomic_json(OUT/'backend_episode_sampling.json',dict(version=SAMPLING_VERSION,latest_repeat=12))
    configure_frontend_sampling(OUT,12)
    planned={}
    for role in 'AB':
        data=read(OUT/role/'state_data/data.json');continuation=read(OUT/role/'continuation.json')
        _,m=sampling_plan(data,continuation['sampling_contract'])
        planned[role]=dict(**budget(sum(m.values()),epochs),data_sha256=sha256(OUT/role/'state_data/data.json'),episode_multiplicities=m)
    atomic_json(OUT/'full_trajectory_batch.json',dict(profile=profile(),roles=planned))
    execute('visual_cache',['tools.tight_fresh_dagger','cache','--root',ROOT,'--output',OUT,'--reuse-index',OLD/'A/cache_index.json','--payload-audit-memo'])
    for role in 'AB':
        execute('frontend_'+role,['tools.tight_fresh_train','train','--root',OUT,'--role',role,'--epochs',epochs,'--offline-anchor-batch',32,'--full-trajectory-batch'])
    def adapter(name,args,completion,**kwargs):
        execute(name,args);assert completion.exists()
    def backend(role):
        backend_chain(adapter,OUT,role,rear_epochs,continue_rear=True,offline_anchor_batch=32,
            frozen_visual_io=read(OUT/'frozen_visual_io.json'),backend_phase_weighting=read(OUT/'backend_phase_weighting.json'))
    # Actual independent cached plans, both new DPs frozen before any rear update.
    admission=parallel_admission(read(ROOT/'shared_resources.json'),time.time());atomic_json(ROOT/'rear_admission.json',admission)
    if admission['admitted']:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(backend,role) for role in 'AB']
            for f in futures:f.result()
    else:
        for role in 'AB':backend(role)
    from tools.tight_fresh_rollouts import completed_pair
    eval_prepare(ROOT/'updated_evaluation',{r:completed_pair(OUT,r) for r in 'AB'})
    execute('updated_evaluation',['scripts.tight_fulltask_20261116.video_eval','--output',ROOT/'updated_evaluation'])
    atomic_json(ROOT/'complete.json',dict(complete=True,utc=time.time(),deadline=DEADLINE,new_successes=len(entries)))

def main():
    assert (ROOT/'collection_owner_complete.json').exists()
    # Previous phase is finished; preserve its monitor receipt and append telemetry.
    (ROOT/'STOP_MONITOR').rename(ROOT/'STOP_MONITOR.collection')
    (ROOT/'monitor.log').rename(ROOT/'monitor.collection.log')
    monitor=monitor_start()
    try:
        time.sleep(2)
        eval_prepare(ROOT/'baseline_evaluation',read(DIAG/'candidates/plan.json')['pairs'])
        execute('baseline_evaluation',['scripts.tight_fulltask_20261116.video_eval','--output',ROOT/'baseline_evaluation'])
        run()
    finally:
        (ROOT/'STOP_MONITOR').touch();monitor.wait(timeout=20)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(ROOT/'training_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
