"""Register sufficient-supervision continuations in the canonical project."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

from tools.tight_frozen_frontend import PROJECT, read, write, sources
from visuotactile.compact import sha256


def prepare(config):
    import torch
    import h5py
    from visuotactile.effective_exposure import profile, Ledger, dp_ticks
    from visuotactile.anchor_sampling import coverage_sampling_profile, sample_training_anchors, supervision_rows
    from visuotactile.native_data import visual_records
    from visuotactile.bounded_extension import exposure50_profile, split_audit
    from visuotactile.trajectory_training import budget
    c=read(config);root=PROJECT/c['root'];history=PROJECT/c['history'];parent=PROJECT/c['parent']
    if root.exists():raise FileExistsError('Preserve registered run identity')
    root.mkdir(parents=True)
    public=PROJECT/'research_docs/evidence'/root.name;public.mkdir(parents=True,exist_ok=True)
    now=time.time();session=read(history/'session.json')
    session.update(started_utc=now,deadline=now+c['hours']*3600,resource_root=str(root),owned_worker_roots=[str(root)],
        authorization='A/B own72 parents; new >=50 distinct optimizer-step exposures per legal tick, >=50 visits; no DAgger/encoder',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    for k in ('training_deadline','cache_deadline','offline_deadline'):session[k]=session['deadline']
    write(root/'session.json',session)
    for name in ('metadata.json','gate.json','frozen_visual_io.json','frozen_visual_hotset.json','backend_episode_sampling.json','cuda_allocator_policy.json'):
        write(root/name,read(history/name))
    sampling=read(history/'training_sampling.json');sampling['anchors']=coverage_sampling_profile(sampling['anchors']['seed'])
    write(root/'training_sampling.json',sampling);write(root/'effective_exposure.json',profile())
    write(root/'bounded_extension.json',dict(policy=exposure50_profile()))
    registration=read(history/'full_trajectory_batch.json');protected={};audit={};stages=[]
    for role in 'AB':
        child=root/role;child.mkdir()
        for name in ('state_data','cache_index.json'):(child/name).symlink_to(history/role/name)
        continuation=read(history/role/'continuation.json')
        assert Path(continuation['path'])==parent/role/'training/best.pt'
        for key,hashkey in [('path','sha256'),('previous_backend','previous_backend_sha256')]:
            assert sha256(continuation[key])==continuation[hashkey];protected[continuation[key]]=continuation[hashkey]
        write(child/'continuation.json',continuation)
        data=read(child/'state_data/data.json');ids=data['splits']['train'];assert len(ids)==130 and len(data['normalization_train_ids'])==30
        assert data['supervision']==('q_reached' if role=='A' else 'u_expert')
        split=split_audit(data)
        records=visual_records(child/'state_data',ids,'G00',target_window_contract=continuation['target_window_contract'])
        rows=[]
        for i in ids:
            r=records[i];mask,_=supervision_rows(r)
            a,receipt=sample_training_anchors(r,sampling['anchors'],identity=i,epoch=1,visit_index=1)
            covered=dp_ticks(r,a);assert set(covered)==set(mask.nonzero().flatten().tolist())
            assert torch.isfinite(r['clean_sample'][mask]).all()
            rows.append(dict(identity=i,history_ticks=len(mask),legal_ticks=int(mask.sum()),excluded_ticks=(~mask).nonzero().flatten().tolist(),
                excluded_reason='stored expert ownership/transition legality mask; no counterfactual expert labels',
                selected_anchors=a.tolist(),covered_legal_ticks=covered,added_anchors=receipt['added_coverage_anchors']))
        for name in ('data.json','statistics.pt','numeric.pt'):
            p=child/'state_data'/name;protected[str(p.resolve())]=sha256(p)
        audit[role]=dict(split=split,trajectories=rows,legal_ticks=sum(r['legal_ticks'] for r in rows),
            zero_coverage_ticks=0,checkpoint=continuation,statistics_unchanged=True)
        registration['roles'][role].update(budget(190,c['epochs']))
        stages.append(dict(entry='frontend',args=['--root',str(root),'--role',role,'--epochs',str(c['epochs']),
            '--full-trajectory-batch','--offline-anchor-batch','64']))
    # Inspect raw labels and current immutable dependencies before any optimizer update.
    data=read(root/'B/state_data/data.json')
    raw=[]
    for identity,e in data['entries'].items():
        with h5py.File(e['path'],'r') as f:
            g=f['episodes'][e['episode']];assert 'obs' in g and 'frames' in g
        raw.append(dict(identity=identity,path=e['path'],bytes=Path(e['path']).stat().st_size))
    index=read(root/'A/cache_index.json')
    for row in index['entries'].values():assert Path(row['path']).is_file()
    protected[str(PROJECT/'outputs/visuotactile/data/close_v4/manifest.json')]=sha256(PROJECT/'outputs/visuotactile/data/close_v4/manifest.json')
    for p in (session['expert_checkpoint'],session['dino_checkpoint']):protected[p]=sha256(p)
    write(root/'full_trajectory_batch.json',registration)
    for role in 'AB':
        backend=root/role/'backend'
        stages.extend([
            dict(entry='backend_prepare',args=['--root',str(root),'--role',role,'--epochs',str(c['epochs']),
                '--continue-rear','--offline-anchor-batch','64','--frozen-visual-io-json',str(root/'frozen_visual_io.json')]),
            dict(entry='backend_cache',args=['--root',str(backend)]),
            dict(entry='backend_train',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'training')]),
            dict(entry='backend_offline',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'offline')])])
    job=dict(version='tight_project_job_v1',operation='train',resource_root=str(root),stages=stages)
    dest=PROJECT/c['job_directory'];dest.mkdir(parents=True,exist_ok=True);write(dest/'train.json',job)
    write(public/'registration.json',dict(config=c,session=session,protected=protected,source_sha256=sources(),
        optimizer='Own72 selected DP EMA/rear weights; fresh AdamW, DP fresh EMA; same run recovery restores complete optimizer/RNG/exposure',
        selection='60 complete decks, no early stop before60; validation minimum only among actual >=50 per-tick-step AND visit eligible checkpoints; ties earliest',
        learning_rates='DP3e-5, rear1e-4; one .5 reduction on eight non-improving validations after40',
        geometry='Original A legal_tick_half_phase_v1 objective and weighting unchanged; window cover expands eligible sampled population',
        storage_free_gib=shutil.disk_usage(root).free/2**30,raw=raw,preflight=audit))
    write(public/'protected.json',protected);write(root/'registration.json',read(public/'registration.json'))
    print(json.dumps(dict(root=str(root),job=str(dest/'train.json'),legal_ticks={k:v['legal_ticks'] for k,v in audit.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare']);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();prepare(a.config)

if __name__=='__main__':main()
