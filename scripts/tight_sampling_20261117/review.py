"""Read-only matched-budget review; no optimizer, inference, or simulation."""
import collections, gzip, json, shutil, sys, time
from pathlib import Path
import h5py
import numpy as np
import torch
from .control import ROOT, OUT, ON, PARENT, read
from scripts.tight_layout_20261116.publication import write, table, sha
from scripts.tight_balance_20261117.pair_diagnosis import first_run
EVAL=ROOT/'evaluation_recovery01' if (ROOT/'evaluation_recovery01').exists() else ROOT

def exposure():
    from visuotactile.anchor_sampling import supervision_rows
    from visuotactile.target_windows import window_mask
    records=torch.load(ROOT/'A/state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
    totals=[];decks={};starts={}
    for label,root in [('onset',ON),('ordinary',ROOT)]:
        counts=collections.Counter();deck=[];anchors_by_visit={}
        for line in (root/'A/training/learning.jsonl').open():
            row=json.loads(line)
            if row['updates']>494:break
            for v in row['visits']:
                identity=v['episode'];r=records[identity];mask,seg=supervision_rows(r)
                start=int(mask.nonzero()[0]);original=start==0
                a=torch.tensor(v['anchor_sampling']['selected_anchors'])
                valid=window_mask(mask,seg,a,24,'recovery_onset_commit12_v1')
                ticks=(a[:,None]+torch.arange(24)[None])[valid];n=len(ticks)
                assert n==v['target_ticks']
                early=int((ticks<150).sum()) if original else 0
                onset=int((ticks<start+24).sum()) if not original else 0
                counts.update(visits=1,windows=len(a),slots=n,original_early_slots=early,onset24_slots=onset,
                    original_visits=int(original),original_objective_mass=v['loss_weight']*original,
                    original_early_base_mass=v['loss_weight']*early/n,onset24_base_mass=v['loss_weight']*onset/n,
                    total_objective_mass=v['loss_weight'])
                key=(row['epoch'],identity,v['anchor_sampling']['visit_index'])
                deck.append(key);anchors_by_visit[key]=(original,a.tolist())
        decks[label]=deck;starts[label]=anchors_by_visit
        totals.append(dict(label=label,**counts,original_early_base_fraction=counts['original_early_base_mass']/counts['total_objective_mass'],
            onset24_base_fraction=counts['onset24_base_mass']/counts['total_objective_mass']))
    assert decks['onset']==decks['ordinary']
    original=[k for k,v in starts['onset'].items() if v[0]]
    assert all(starts['onset'][k]==starts['ordinary'][k] for k in original)
    result=dict(rows=totals,episode_order_equal=True,original_anchor_visits_exact=len(original),
        changed_correction_visits=sum(starts['onset'][k]!=starts['ordinary'][k] for k in decks['onset']),
        interpretation='Actual first494 update logs, equal per-trajectory objectives. Base coefficient excludes geometry terms and is not gradient norm; slot counts are overlapping exposures, not independent states.')
    write(OUT/'exposure.json',result);table(OUT/'exposure.csv',totals)

def training():
    reg=read(ROOT/'registration.json');models=[];curves=[];performance=[]
    for label,root in [('baseline72',PARENT),('onset76',ON),('ordinary76',ROOT)]:
        dp=root/'A/training/best.pt';rear=root/'A/backend/A/training/best.pt'
        a=torch.load(dp,map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(rear,map_location='cpu',weights_only=False,mmap=True)
        assert b['configuration']['base_sha256']==sha(dp)
        models.append(dict(label=label,root=str(root),dp_sha256=sha(dp),backend_sha256=sha(rear),dp_updates=a['updates'],backend_updates=b['updates'],
            dp_parent=a['configuration']['continuation'],backend_parent=b['continuation_from'],
            data_sha256=a['configuration']['data_sha256'],statistics_sha256=a['configuration']['statistics_sha256']))
    assert models[1]['dp_updates']==models[2]['dp_updates']==494
    assert models[1]['backend_updates']==models[2]['backend_updates']==369
    for key in ['dp_parent','backend_parent','data_sha256','statistics_sha256']:
        assert models[1][key]==models[2][key],key
    configs={label:read(root/'A/training/configuration.json') for label,root in [('onset',ON),('ordinary',ROOT)]}
    differences={k:{label:c.get(k) for label,c in configs.items()} for k in set().union(*configs.values()) if configs['onset'].get(k)!=configs['ordinary'].get(k)}
    write(OUT/'model_match.json',dict(models=models,dp_config_differences=differences,selection_limitation=reg['selection_limitation']))
    for component,folder in [('DP',ROOT/'A/training'),('Transformer',ROOT/'A/backend/A/training')]:
        done=read(folder/'finished.json')
        if component=='DP':
            rows=[json.loads(l) for l in (folder/'validation.jsonl').open()]
            train=sum(r['training_epoch_io']['wall_seconds'] for r in rows)
            validation=sum(r['validation']['offline_planning']['seconds'] for r in rows)
            diagnostic=sum(r['train']['offline_planning']['seconds'] for r in rows)
            wall=done['wall_seconds'];cache=done['frozen_visual_hotset']
            performance.append(dict(component=component,updates=done['updates'],stage_seconds=wall,training_epoch_seconds=train,
                validation_seconds=validation,train_diagnostic_seconds=diagnostic,updates_per_training_second=done['updates']/train,
                read_bytes=done['stage_process_io']['read_bytes'],hotset_fill_seconds=cache['first_fill_seconds'],hotset_resident_bytes=cache['resident_bytes'],
                peak_cuda_allocated_bytes=done['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=done['peak_cuda_reserved_bytes']))
            for r in rows:curves.append(dict(component=component,epoch=r['epoch'],updates=r['updates'],validation=r['validation']['selection'],loss=r['loss']))
            logs=['learning.jsonl','validation.jsonl']
        else:
            performance.append(dict(component=component,updates=done['updates'],stage_seconds=done['wall_s'],training_epoch_seconds=done['training_wall_s'],setup_seconds=done['setup_wall_s']))
            for r in read(folder/'validation.json'):curves.append(dict(component=component,epoch=r['epoch'],updates=r['update'],validation=r['validation']['all']['normalized_equal_mse']))
            logs=['loss.jsonl','validation.json']
        for name in logs:
            dest=OUT/'curves'/f'{component}_{name}.gz';dest.parent.mkdir(exist_ok=True)
            with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0) as dst:
                if name=='learning.jsonl':
                    for line in (folder/name).open():
                        row=json.loads(line)
                        for v in row['visits']:v.pop('geometry',None)
                        dst.write((json.dumps(row,separators=(',',':'))+'\n').encode())
                else:
                    with (folder/name).open('rb') as src:shutil.copyfileobj(src,dst)
    table(OUT/'performance.csv',performance);table(OUT/'training_curve.csv',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for ax,component in zip(axes,['DP','Transformer']):
        rr=[r for r in curves if r['component']==component];ax.plot([r['updates'] for r in rr],[r['validation'] for r in rr],marker='o');ax.set_title(component+' ordinary76');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Validation MSE');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training_curve.png',dpi=130);plt.close(fig)

def evaluation():
    from scripts.tight_layout_20261116.phase_report import summarize
    from scripts.tight_balance_20261117.diagnose import phases
    rows=[];signals={};profiles=[]
    for part in range(4):
        folder=EVAL/f'comparison{part:02d}';summary=OUT/f'comparison{part:02d}.json'
        summarize(folder,summary);s=read(summary);profiles.append(s.get('wave_profiles'))
        for row in s['rows']:
            row['hold120_achieved']=row['hold120']['sustained']
            rows.append(row)
        plan=read(folder/'plan.json')
        for c in plan['cases']:
            p=folder/c['name'];z=dict(np.load(p/'trace.npz'))
            with h5py.File(p/'numeric.h5') as f:
                t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:];pose[:,:3]-=t['diagnostics_before/base_pose_w'][:,:3]
                z.update(pose=pose,phase=phases(t),keypoint=t['native_keypoints_max_dist'][:].reshape(-1))
            signals[c['spec']['seed'],c['role']]=z
    table(OUT/'per_initial_state.csv',rows)
    totals={r:dict(count=sum(x['role']==r for x in rows),success=sum(x['success'] for x in rows if x['role']==r),
        hold120=sum(x['hold120_achieved'] for x in rows if x['role']==r),subgoal=sum(bool(x['native_subgoals']) for x in rows if x['role']==r)) for r in ['A_baseline','A_onset','A_ordinary']}
    pairs=[];trace_rows=[]
    for seed in range(28824000,28824008):
        a=signals[seed,'A_baseline']
        for role in ['A_onset','A_ordinary']:
            b=signals[seed,role];n=min(len(a['pose']),len(b['pose']))
            pos=np.linalg.norm(a['pose'][:n,:3]-b['pose'][:n,:3],axis=-1)
            qa=a['pose'][:n,3:].astype(float);qb=b['pose'][:n,3:].astype(float)
            qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True)
            angle=2*np.arccos(np.clip(abs((qa*qb).sum(-1)),0,1))
            tick=first_run((pos>.01)|(angle>.15))
            pairs.append(dict(seed=seed,candidate=role,first_sustained_geometry_divergence_tick=tick,
                baseline_phase_at_divergence=int(a['phase'][tick]) if tick is not None else None,candidate_phase_at_divergence=int(b['phase'][tick]) if tick is not None else None,
                **{k+'_tick0_rms':float(np.sqrt(np.mean((a[k][0].astype(float)-b[k][0])**2))) for k in ['nominal','residual_proposed_rad','command']}))
        for role in ['A_baseline','A_onset','A_ordinary']:
            z=signals[seed,role]
            for tick in range(len(z['pose'])):
                trace_rows.append(dict(seed=seed,role=role,tick=tick,object_z=float(z['pose'][tick,2]),keypoint_m=float(z['keypoint'][tick]),phase=int(z['phase'][tick])))
    table(OUT/'paired_deltas.csv',pairs);table(OUT/'trajectory_curves.csv',trace_rows)
    result=dict(counts=totals,rows=rows,paired_deltas=pairs,paired_initial_pass=True,
        divergence_definition='First12 consecutive ticks with base-relative object displacement>1cm OR normalized quaternion angle>0.15rad. Descriptive, not causal or strict funnel.',
        limitations='Opened development only, 8 starts/one training seed. Same-wave observable initial states matched, not hidden PhysX or future event streams. Six-lane results must not be pooled with historical eight-lane results.')
    write(OUT/'results.json',result);print(json.dumps(totals,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(8,2,figsize=(11,15))
    for i,seed in enumerate(range(28824000,28824008)):
        for role in ['A_baseline','A_onset','A_ordinary']:
            z=signals[seed,role];tt=np.arange(len(z['pose']))/60
            axes[i,0].plot(tt,z['pose'][:,2],label=role);axes[i,1].plot(tt,z['keypoint'],label=role)
        axes[i,0].set_ylabel(str(seed))
        for ax in axes[i]:ax.grid(alpha=.2)
    axes[0,0].set_title('Object height / m');axes[0,1].set_title('Native keypoint distance / m');axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Control time / s')
    fig.tight_layout();fig.savefig(OUT/'paired_traces.png',dpi=120);plt.close(fig)

def export():
    inventory=[]
    paths=[p for p in ROOT.glob('*.json')]+list((ROOT/'logs').glob('*.json'))
    for rel in ['A/continuation.json','A/state_data/data.json','A/training/configuration.json','A/training/finished.json','A/backend/configuration.json','A/backend/base_identities.json','A/backend/cache_complete.json','A/backend/A/nominal/index.json','A/backend/A/training/configuration.json','A/backend/A/training/finished.json','A/backend/A/training/selected.json','A/backend/A/offline/result.json']:
        paths.append(ROOT/rel)
    for part in range(4):
        paths.extend((EVAL/f'comparison{part:02d}').glob('*.json'))
    if EVAL!=ROOT:
        paths.extend(EVAL.glob('*.json'));paths.extend((EVAL/'logs').glob('*.json'))
    paths.extend((ROOT/'ordinary_replay').glob('*.json'))
    for folder in [ROOT,EVAL,ROOT/'ordinary_replay']:
        paths.extend((folder/'resources').glob('resources.json'))
    for p in paths:
        if not p.exists():continue
        q=OUT/'configuration'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        inventory.append(dict(source=str(p),published=str(q),sha256=sha(q),bytes=q.stat().st_size))
    timings=[dict(stage=p.name.removesuffix('_complete.json'),**read(p)) for folder in {ROOT,EVAL} for p in sorted((folder/'logs').glob('*_complete.json'))]
    table(OUT/'stage_times.csv',timings)
    stages=[]
    for part in range(4):
        folder=EVAL/f'comparison{part:02d}'
        wave=read(folder/'wave_profiles.json')[0]
        stages.append(dict(wave=part,worker_wall_seconds=read(EVAL/'logs'/f'eval{part}_complete.json')['seconds'],
            kit_started_since_worker_seconds=read(folder/'kit_started.json')['utc']-read(folder/'run_started.json')['utc'],
            environment_initialization_seconds=read(folder/'initialization.json')['seconds'],
            wave_seconds=wave['seconds'],controller_seconds=wave['controller_s'],environment_seconds=wave['environment_s'],
            render_seconds_nested_in_environment=wave['render_reuse']['render_seconds'],record_submit_seconds=wave['record_submit_s']))
    table(OUT/'evaluation_stage_times.csv',stages)
    write(OUT/'export_inventory.json',inventory)

def media():
    import cv2, subprocess, imageio_ffmpeg
    from PIL import Image,ImageDraw
    directory=OUT/'media';directory.mkdir(exist_ok=True);rows=[];panels=[]
    # Preregistered illustrative original A success, independent of new outcome.
    folder=EVAL/'comparison01';plan=read(folder/'plan.json')
    for c in plan['cases']:
        if c['spec']['seed']!=28824002:continue
        source=folder/c['name']/'autonomous.mp4';dest=directory/(c['role']+'_28824002.mp4')
        shutil.copyfile(source,dest)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(dest),'-f','null','-'],check=True)
        cap=cv2.VideoCapture(str(dest));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));result=read(folder/c['name']/'result.json')
        assert frames==result['steps'];thumbs=[]
        ticks=sorted(set(min(frames-1,t) for t in [60,84,120,180,300,frames-1]))
        for tick in ticks:
            cap.set(cv2.CAP_PROP_POS_FRAMES,tick);ok,frame=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((672,378))
            panel=Image.new('RGB',(672,im.height+24),'white');panel.paste(im,(0,24))
            ImageDraw.Draw(panel).text((4,4),f"{c['role']} seed28824002 tick{tick} success={result['success']}",fill='black');thumbs.append(panel)
        cap.release();panels.append(thumbs)
        rows.append(dict(role=c['role'],source=str(source),published=str(dest),sha256=sha(dest),frames=frames,decoded_all=True,success=result['success'],ticks=ticks))
    canvas=Image.new('RGB',(max(sum(x.width for x in row) for row in panels),sum(max(x.height for x in row) for row in panels)),'white');y=0
    for row in panels:
        x=0
        for p in row:canvas.paste(p,(x,y));x+=p.width
        y+=max(p.height for p in row)
    canvas.save(OUT/'video_review.jpg',quality=88)
    write(OUT/'media.json',dict(videos=rows,selection='Preregistered seed28824002, all three full policies; viewing only, all24 counted in results.'))

def closure():
    reg=read(ROOT/'registration.json');checks={p:sha(Path(p))==h for p,h in reg['protected'].items()}
    assert all(checks.values());assert read(ROOT/'owner_exit.json')['children_reaped']
    assert read(EVAL/'complete.json')['complete'] and not (EVAL/'failure.json').exists()
    assert read(EVAL/'owner_exit.json')['children_reaped']
    replay=ROOT/'ordinary_replay'
    assert read(replay/'complete.json')['complete'] and read(replay/'owner_exit.json')['children_reaped']
    assert not (replay/'failure.json').exists()
    # Source hashes are those actually admitted before worker launch.
    source={p:sha(Path(p))==h for p,h in read(ROOT/'prepared.json')['source_sha256'].items()}
    assert all(source.values())
    seen=set();allocated=0
    for p in ROOT.rglob('*'):
        if p.is_symlink() or not p.is_file():continue
        st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:seen.add(key);allocated+=st.st_blocks*512
    assert allocated<4*1024**3 and shutil.disk_usage(ROOT).free>8*1024**3
    assert read(replay/'owner_exit.json')['utc']<reg['deadline']
    model_checks={}
    for model in read(OUT/'model_match.json')['models']:
        folder=Path(model['root'])
        model_checks[model['label']]=sha(folder/'A/training/best.pt')==model['dp_sha256'] and sha(folder/'A/backend/A/training/best.pt')==model['backend_sha256']
    assert all(model_checks.values())
    write(OUT/'closure.json',dict(protected=checks,model_pairs_match=model_checks,source_hashes_pass=all(source.values()),source_files=len(source),
        owner_exit=read(ROOT/'owner_exit.json'),owned_output_allocated_bytes=allocated,output_cap_bytes=4*1024**3,
        disk_free_bytes=shutil.disk_usage(ROOT).free,deadline=reg['deadline'],completed_utc=read(EVAL/'complete.json')['utc'],
        elapsed_since_owner_start_s=read(replay/'owner_exit.json')['utc']-read(ROOT/'owner_started.json')['utc'],
        replay_owner_exit=read(replay/'owner_exit.json'),
        evaluation_owner_exit=read(EVAL/'owner_exit.json'),failed_initial_attempt=read(ROOT/'failure.json') if (ROOT/'failure.json').exists() else None,
        no_dagger=True,no_B_training=True,no_model_promotion=True,utc=time.time()))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for name in sys.argv[1:] or ['training','exposure','evaluation','export']:globals()[name]()
