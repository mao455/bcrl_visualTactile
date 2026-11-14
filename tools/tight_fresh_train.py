"""Fresh full-episode training and measured continuous-vision acceptance.

Episode visits follow the registered schedule; learned hidden state is never cached.
"""
import argparse,copy,json,time
import hashlib,os,uuid
from contextlib import nullcontext
from pathlib import Path
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.checkpointing import save,training_state,restore_training
from visuotactile.native_data import visual_records,Episode
from visuotactile.models.native_dp import NativePolicy,encode_segment
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.position_control import continuous_image_contract,bounded_targets
from visuotactile.continuous_vision import encode_dense
from visuotactile.anchor_sampling import labels
from visuotactile.latest_takeover_sampling import sampling_plan,episode_deck,validate_sampling_exposure
from visuotactile.episode_io import EpisodePrefetch, normalize_io, io_snapshot, io_delta


def read(p):return json.loads(Path(p).read_text())


def inputs(root,role):
    data_root=root/role/'state_data';data=read(data_root/'data.json');ids=data['splits']['train']+data['splits']['validation']
    records=visual_records(data_root,ids,'G00',device='cuda:0')
    continuation=root/role/'continuation.json'
    if continuation.exists():
        from visuotactile.target_windows import with_target_window
        records=with_target_window(records,read(continuation).get('target_window_contract'))
    return data_root,data,records,torch.load(data_root/'statistics.pt',weights_only=False,map_location='cpu')


def new_model(role,statistics):
    torch.manual_seed(2026111401)
    model=NativePolicy('I-HistoryDP',24,modality='G00')
    model.contract=continuous_image_contract('q_reached' if role=='A' else 'u_expert')
    for k,v in statistics.items():getattr(model,k).copy_(v)
    return model.cuda()


def supervised_conditions(model,record,*,episode=None,selected_anchors=None):
    """All frames update history; only actual legal target windows train DP."""
    if selected_anchors is not None:
        from visuotactile.anchor_sampling import validate_training_anchors
        selected_anchors=validate_training_anchors(record,selected_anchors)
        return encode_dense(model,record,episode=episode,selected_anchors=selected_anchors)
    c,a=(encode_dense(model,record) if episode is None else encode_dense(model,record,episode=episode))
    from visuotactile.anchor_sampling import planning_anchors
    permitted=planning_anchors(record).to(a.device)
    keep=torch.isin(a,permitted)
    if not keep.any():raise ValueError('No legal expert planning windows')
    return c[keep],a[keep]


def bind_geometry_supervision(objective, records):
    """Bind the same legal target windows and check every episode before updates."""
    from visuotactile.anchor_sampling import planning_anchors
    from visuotactile.anchor_continue import geometry_selection
    objective.supervision_records = records
    report = {}
    for identity, record in records.items():
        anchors = planning_anchors(record)
        _, valid = labels(record, anchors, 24)
        selected, row = geometry_selection(objective, [(identity, int(t)) for t in anchors],
            {identity: len(record['time'])}, 8091)
        committed = valid.detach().cpu() & (torch.arange(24)[None] < 12)
        if (selected & ~committed).any() or row['eligible_commit_count'] != int(committed.sum()):
            raise ValueError('Geometry and supervised commit population differ: ' + identity)
        report[identity] = row
    return report


@torch.no_grad()
def gate(root):
    from visuotactile.precision import state_fp32
    state_fp32()
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.native_data import depth_grid
    import h5py
    data_root,data,records,stats=inputs(root,'A');i=data['splits']['train'][0];r=records[i];ep=Episode(data_root,i,'I-HistoryDP')
    m=new_model('A',stats).eval();ep.record.update(robot=r['robot'],time=r['time'])
    torch.cuda.synchronize();t=time.perf_counter();seq,a,_=encode_segment(m,ep,0,25);torch.cuda.synchronize();serial=time.perf_counter()-t
    t=time.perf_counter();batch,b=encode_dense(m,r,episode=ep,stop=25);torch.cuda.synchronize();batched=time.perf_counter()-t
    err=float((seq-batch).abs().max())
    if err>2e-5:raise ValueError('Real-input batch/online visual parity: '+str(err))
    dino=NativeDinoSpatial().cuda().eval();checks=[]
    with h5py.File(data['entries'][i]['path']) as f:
        g=f['episodes'][data['entries'][i]['episode']]
        for tick in [0,1,11,12,24]:
            p=dino(torch.from_numpy(g['frames/rgb_u8'][tick]).cuda()).cpu()
            difference=float((p-ep.perception['patches'][tick]).abs().max())
            if difference:raise ValueError('Frozen visual cache is not exact at '+str(tick))
            dep=depth_grid(torch.from_numpy(g['frames/depth_m'][tick]),torch.from_numpy(g['frames/depth_valid'][tick]))
            if not torch.equal(dep,ep.perception['depth'][tick]):raise ValueError('Depth cache differs')
            checks.append(dict(tick=tick,patch_max_abs=difference))
    atomic_json(root/'gate.json',dict(passed=True,identity=i,frames=25,plan_ticks=a.tolist(),
        real_batch_online_max_abs=err,serial_encoder_seconds=serial,batch_encoder_seconds=batched,
        exact_raw_cache=checks,intermediate_frame_causality='tests/test_continuous_vision.py',
        limitations='Untrained encoder transport/causality only, not task success or learned visual dependence',utc=time.time()))
    print(json.dumps(read(root/'gate.json')),flush=True)


@torch.no_grad()
def predict_on_native_clock(model, conditions, anchors):
    """Keep the online DDIM noise stream across unsupervised student prefixes.

    DDIM eta=0 consumes one initial Hx29 noise tensor per 5Hz replan. Skipped
    loss anchors still consume that draw, without unnecessary denoising.
    """
    ticks=anchors.detach().cpu().tolist()
    if (len(ticks)!=len(conditions) or not ticks or ticks!=sorted(set(ticks))
            or any(t<0 or t%12 for t in ticks)):
        raise ValueError('Expected increasing supervised anchors on the native planning clock')
    generator=torch.Generator(device=conditions.device).manual_seed(7091)
    cursor=0;plans=[]
    for index,tick in enumerate(ticks):
        while cursor<tick:
            torch.randn((1,model.horizon,29),device=conditions.device,generator=generator)
            cursor+=12
        plans.append(model.predict(conditions[index:index+1],generator,scheduler_index_mode='host'))
        cursor+=12
    return torch.cat(plans)


@torch.no_grad()
def metrics(model,records,identities,*,offline_planning=None,check=None,frozen_visual_io=None):
    from visuotactile.frozen_visual_io import validate_profile as validate_visual_io
    frozen_visual_io=validate_visual_io(frozen_visual_io)
    if offline_planning is not None:
        from visuotactile.offline_planning import validate_profile,predict_contiguous
        offline_planning=validate_profile(offline_planning)
    started=time.perf_counter() if offline_planning is not None else None
    model.eval();errors=[];normalized=[];rows=[]
    for i in identities:
        rec=records[i]
        if offline_planning is not None:
            if check is not None:check()
            if model.action_scale.is_cuda:torch.cuda.synchronize(model.action_scale.device)
            began=time.perf_counter()
        visual_receipt=None
        if frozen_visual_io is None:c,a=supervised_conditions(model,rec)
        else:
            episode=Episode(rec['source_root'],i,model.group,frozen_visual_io=frozen_visual_io)
            try:
                c,a=supervised_conditions(model,rec,episode=episode)
                visual_receipt=episode.frozen_visual_io_receipt
            finally:del episode
        if offline_planning is None:pred=predict_on_native_clock(model,c,a)
        else:
            if c.is_cuda:torch.cuda.synchronize(c.device)
            encoding_seconds=time.perf_counter()-began;began=time.perf_counter();trace={}
            pred=predict_contiguous(model,c,a,configuration=offline_planning,check=check,receipt=trace)
            if c.is_cuda:torch.cuda.synchronize(c.device)
            planning_seconds=time.perf_counter()-began
        pred=bounded_targets(pred.flatten(0,1),rec['joint_limits']).reshape_as(pred)
        truth,valid=labels(rec,a,24);mask=valid[:,:12];e=(pred[:,:12]-truth[:,:12])[mask]
        errors.append(e.cpu());normalized.append((e*model.action_scale).cpu())
        rows.append(dict(identity=i,arm_rmse=float(e[:,:7].square().mean().sqrt()),hand_rmse=float(e[:,7:].square().mean().sqrt())))
        if visual_receipt is not None:rows[-1]['frozen_visual_io']=visual_receipt
        if offline_planning is not None:
            rows[-1]['offline_planning']=dict(trace,full_history_frames=len(rec['time']),
                encoding_seconds=encoding_seconds,planning_seconds=planning_seconds)
    e=torch.cat(errors);n=torch.cat(normalized)
    result=dict(arm_rmse_rad=float(e[:,:7].square().mean().sqrt()),hand_rmse_rad=float(e[:,7:].square().mean().sqrt()),
        selection=float(.5*n[:,:7].square().mean()+.5*n[:,7:].square().mean()),episodes=rows,
        noise_schedule='seed7091 advances at every native5Hz planning tick, including unsupervised prefixes',
        scope='teacher-forced full causal prefix; DDIM16 commit12; no autonomy claim')
    if offline_planning is not None:
        result['offline_planning']=dict(profile=offline_planning,seconds=time.perf_counter()-started,
            timing_scope='Complete metrics wall; per-episode encoding and DDIM timings synchronize CUDA; concurrent-worker contention remains',
            selection_scope='Same labels and metric reduction; batch kernel rounding can change near-tied selected epochs')
    return result


def fresh_objective(root,role,data,records,*,geometry_sampling_profile=None):
    """Shared full fresh action objective for training and disposable acceptance."""
    from visuotactile.state_image_loss import GeometryObjective,PALM_LOSS_ID,geometry_loss_context
    meta=read(root/'metadata.json');objective=GeometryObjective(meta,mode=PALM_LOSS_ID)
    objective.weights=dict(position=.3342396121814294,rotation=.029946394754032265)
    geometry=None;proof=None
    if role=='A':
        base=Path('outputs/visuotactile/run')
        acceptance=base/'tight_transformer_20261110_02/geometry_acceptance.json'
        live=base/'tight_transformer_20261110_01/physics_gate_meshfix/geometry_live.json'
        a=read(acceptance);inventory=read(live)['inventory']
        objective.enable_hand_geometry(inventory,a,loss_kind='hand_surface_soft_hinge_v1',penetration_tolerance_m=.001,contact_tolerance_m=.002)
        objective.weights['hand_geometry']=a['hand_geometry_weight']
        objective.loss_context={i:geometry_loss_context(data['entries'][i],**(dict(sampling_profile=geometry_sampling_profile)
            if geometry_sampling_profile is not None and i in data['splits']['train'] else {})) for i in records}
        proof=bind_geometry_supervision(objective,records)
        if geometry_sampling_profile is not None:objective.geometry_sampling_profile=geometry_sampling_profile
        geometry=dict(acceptance_sha256=sha256(acceptance),live_sha256=sha256(live),
            loss_kind='hand_surface_soft_hinge_v1',weight=a['hand_geometry_weight'],hyperparameter_source='prior accepted train-only physical geometry calibration; no test tuning')
    return objective,geometry,proof


def reconcile_learning_log(out,*,durable_update,completed_epochs,episodes_per_epoch,checkpoint_sha256=None,
        trajectory_batch_profile=None):
    """Archive actual discarded work before keeping only checkpointed update rows.

    Full-epoch checkpoints contain the optimizer, RNG and exposure ledger. A
    crash-tail update happened, but its optimizer state is absent from last.pt;
    preserve its raw log separately and never count it in the restored model.
    """
    from visuotactile.trajectory_training import validate_profile
    batch_profile=validate_profile(trajectory_batch_profile)
    if type(episodes_per_epoch) is not int or episodes_per_epoch<1:
        raise ValueError('Exact completed-epoch recovery cursor required')
    updates_per_epoch=(episodes_per_epoch+1)//2 if batch_profile else episodes_per_epoch
    if (type(durable_update) is not int or type(completed_epochs) is not int
            or type(episodes_per_epoch) is not int or episodes_per_epoch < 1
            or completed_epochs < 0 or durable_update != completed_epochs*updates_per_epoch):
        raise ValueError('Exact completed-epoch recovery cursor required')
    out=Path(out);path=out/'learning.jsonl'
    if not path.exists():
        if durable_update:raise ValueError('Durable checkpoint learning log is missing')
        return None
    raw=path.read_bytes();lines=raw.splitlines(keepends=True);kept=[];abandoned=[];seen=[];tail_updates=[]
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if index!=len(lines)-1:raise ValueError('Corrupt nonterminal learning log row')
            abandoned.append(line);continue
        update=row.get('updates')
        if type(update) is not int or update<1 or row.get('epoch')!=(update-1)//updates_per_epoch+1:
            raise ValueError('Learning log update/epoch identity differs')
        if batch_profile:
            from visuotactile.training_receipts import BATCH_ROW_VERSION
            batch_index=(update-1)%updates_per_epoch+1;start=(batch_index-1)*2+1
            expected=list(range(start,min(start+2,episodes_per_epoch+1)))
            if (row.get('version')!=BATCH_ROW_VERSION or row.get('full_trajectory_batch')!=batch_profile or
                    row.get('batch_index')!=batch_index or row.get('episodes_per_epoch')!=episodes_per_epoch or
                    [v['episode_index'] for v in row.get('visits',[])]!=expected):
                raise ValueError('B2 learning batch cursor differs from registered deck')
        elif 'visits' in row or 'full_trajectory_batch' in row:
            raise ValueError('B2 learning log cannot resume as single-trajectory training')
        elif ('episode_index' in row and row['episode_index']!=(update-1)%episodes_per_epoch+1
                or 'episodes_per_epoch' in row and row['episodes_per_epoch']!=episodes_per_epoch):
            raise ValueError('Learning log episode cursor differs from registered deck length')
        if update<=durable_update:
            if abandoned:raise ValueError('Durable learning row appears after a discarded tail')
            seen.append(update);kept.append(line)
        else:abandoned.append(line);tail_updates.append(update)
    if seen!=list(range(1,durable_update+1)):
        raise ValueError('Learning log lacks unique complete durable update sequence')
    if batch_profile:
        from visuotactile.training_receipts import learning_counters
        from visuotactile.trajectory_training import validate_deck_receipts
        durable_rows=[json.loads(line) for line in kept]
        validate_deck_receipts(out,durable_rows)
        counters=learning_counters(durable_rows)
        if counters['visits_seen']!=completed_epochs*episodes_per_epoch:
            raise ValueError('B2 durable visit count differs from completed deck')
    if not abandoned:return None
    identity=uuid.uuid4().hex
    archive=out/('learning.abandoned.'+identity+'.jsonl')
    with archive.open('xb') as stream:
        stream.write(b''.join(abandoned));stream.flush();os.fsync(stream.fileno())
    # Never truncate before the complete original tail is safely archived.
    temporary=out/('learning.reconciled.'+identity+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(b''.join(kept));stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    report=dict(version='fresh_epoch_log_recovery_v1',durable_updates=durable_update,
        completed_epochs=completed_epochs,episodes_per_epoch=episodes_per_epoch,checkpoint_sha256=checkpoint_sha256,
        original_log_sha256=hashlib.sha256(raw).hexdigest(),retained_log_sha256=sha256(path),
        abandoned_path=str(archive.resolve()),abandoned_sha256=sha256(archive),
        abandoned_rows=len(abandoned),known_discarded_update_ids=tail_updates,
        discarded_updates_in_restored_model=0,utc=time.time(),
        scope='Abandoned rows preserve real work whose optimizer/RNG state was not checkpointed; excluded from the restored model learning dose.')
    if batch_profile:report.update(version='fresh_batch_epoch_log_recovery_v1',full_trajectory_batch=batch_profile,
        durable_episode_visits=completed_epochs*episodes_per_epoch,optimizer_updates_per_epoch=updates_per_epoch)
    atomic_json(out/('recovery.'+identity+'.json'),report)
    return report


def train(root,role,epochs,*,offline_anchor_batch=None,trajectory_batch_profile=None):
    from visuotactile.trajectory_training import validate_registration
    validate_registration(root,trajectory_batch_profile)
    options={}
    if trajectory_batch_profile is not None:
        from visuotactile.trajectory_training import validate_profile,reject_incompatible
        options['trajectory_batch_profile']=validate_profile(trajectory_batch_profile)
        reject_incompatible(root,role)
    if offline_anchor_batch is not None:
        from visuotactile.offline_planning import profile
        options['offline_planning']=profile(offline_anchor_batch)
    pairing_path=root/'paired_episode_io.json'
    if not pairing_path.exists():
        return _train(root,role,epochs,**options)
    from visuotactile.paired_episode_io import PairWorker
    with PairWorker(root,role,epochs) as pairing:
        return _train(root,role,epochs,pairing=pairing,**options)


def _train(root,role,epochs,*,pairing=None,offline_planning=None,trajectory_batch_profile=None):
    from visuotactile.anchor_continue import projection_batch
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32
    from visuotactile.trajectory_training import validate_registration,validate_exposure as validate_batch_exposure
    batch_profile=validate_registration(root,trajectory_batch_profile)
    from visuotactile.frozen_visual_io import load_profile as load_visual_io
    visual_io=load_visual_io(root)
    if visual_io is not None and batch_profile is None:
        raise ValueError('Formal eager visual I/O currently requires the explicit B2 frontend path')
    if offline_planning is not None:
        from visuotactile.offline_planning import validate_profile
        offline_planning=validate_profile(offline_planning)
    if not read(root/'gate.json')['passed'] or not read(root/role/'cache_index.json')['complete']:raise ValueError('Gate incomplete')
    state_fp32();session=read(root/'session.json');lease=lease_for(root,deadline=session['deadline'])
    out=root/role/'training';out.mkdir(exist_ok=True)
    continuation=read(root/role/'continuation.json') if (root/role/'continuation.json').exists() else None
    io_path=root/role/'episode_io.json'
    io_config=normalize_io(read(io_path)) if io_path.exists() else None
    from visuotactile.visual_aux_training import (load_auxiliary_spec, attach_auxiliary_head,
        optimizer_groups, prepare_pose_contexts, supervised_visual_conditions, auxiliary_episode_loss,
        auxiliary_metrics, empty_exposure, add_exposure, validate_exposure)
    from visuotactile.visual_pose_aux import clip_auxiliary_gradients
    auxiliary=load_auxiliary_spec(root,role,continuation)
    if batch_profile:
        from visuotactile.trajectory_training import reject_incompatible
        reject_incompatible(root,role,pairing=pairing,io_config=io_config,auxiliary=auxiliary)
    pose_contexts=pose_reports=None;pose_exposure=empty_exposure() if auxiliary else None
    with lease.block():
        dr,data,records,stats=inputs(root,role)
        sampling,multiplicities=sampling_plan(data,continuation.get('sampling_contract') if continuation else None)
        if auxiliary and sampling and sampling['latest_repeat']!=1:
            raise ValueError('Repeated latest sampling with auxiliary exposure accounting is not implemented')
        episodes_per_epoch=sum(multiplicities.values())
        sampling_exposure=dict.fromkeys(data['splits']['train'],0) if sampling or batch_profile else None
        model=new_model(role,stats)
        if continuation:
            path=Path(continuation['path'])
            if sha256(path)!=continuation['sha256'] or sha256(dr/'statistics.pt')!=continuation['statistics_sha256']:
                raise ValueError('Frozen frontend continuation or affine changed')
            source=torch.load(path,map_location='cpu',weights_only=False)
            if source['contract']!=model.contract:raise ValueError('Continuation frontend contract differs')
            model.load_state_dict(source['ema'],strict=True)
        if auxiliary:
            attach_auxiliary_head(model,auxiliary)
            pose_contexts,pose_reports=prepare_pose_contexts(data,records)
            atomic_json(out/'visual_auxiliary_labels.json',dict(spec=auxiliary,episodes=pose_reports,
                scope='Current loss-only pose targets; original legal action masks are unchanged'))
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        learning_rate=continuation['learning_rate'] if continuation else 1e-4
        opt=torch.optim.AdamW(optimizer_groups(model) if auxiliary else model.parameters(),
            lr=learning_rate,betas=(.95,.999),weight_decay=1e-6)
    from visuotactile.fresh_visit_sampling import load_profile,Coverage,visit as sample_visit
    visit_profile=load_profile(root)
    if visit_profile and auxiliary:raise ValueError('Random visit training and visual auxiliary must use separate explicit implementations')
    coverage=Coverage() if visit_profile else None
    objective,geometry,proof=fresh_objective(root,role,data,records,
        **(dict(geometry_sampling_profile=visit_profile['geometry']) if visit_profile and role=='A' else {}))
    if proof is not None:
        atomic_json(out/'geometry_supervision.json',dict(passed=True,episodes=proof,
            scope='Same legal expert window masks for base, palm and sampled hand geometry; before optimizer updates'))
    generator=torch.Generator().manual_seed(2026111402);noise=torch.Generator(device='cuda').manual_seed(2026111403)
    config=dict(version='fresh30_continuous_episode_v1',data_sha256=sha256(dr/'data.json'),statistics_sha256=sha256(dr/'statistics.pt'),
        dino_sha256=session['dino_sha256'],cache_index_sha256=sha256(root/role/'cache_index.json'),
        initialization=continuation['initialization'] if continuation else 'random identical shared parameter seed; fresh AdamW/EMA',learning_rate=learning_rate,weight_decay=1e-6,
        history='full actual episode prefix at current weights; every frame; full episode gradients',
        exposure='each episode and every legal planning anchor once per epoch; loss equal episode weight',
        validation='all six fixed families DDIM16 commit12; fixed train2 diagnostics',hand_geometry=geometry,
        normalization_identity=model.contract['normalization_identity'],input_builder='canonical_joint_state_v1',
        source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools'] for p in sorted(Path(folder).rglob('*.py'))})
    if visit_profile:
        config.update(training_sampling=visit_profile,training_sampling_sha256=sha256(root/'training_sampling.json'),
            exposure='Full real history once per visit; random legal starts with original anchor-count budget; actual visits, optimizer/EMA updates and distinct times recorded separately.')
    if continuation:config['continuation']=continuation
    if visual_io is not None:
        config.update(frozen_visual_io=visual_io,frozen_visual_io_sha256=sha256(root/'frozen_visual_io.json'))
    if offline_planning is not None:config['offline_planning']=offline_planning
    if io_config is not None:
        config['episode_io']=io_config
        config['episode_io_configuration_sha256']=sha256(io_path)
    if pairing is not None:
        config['paired_episode_io']=dict(version=pairing.config['version'],
            registration_sha256=pairing.registration_sha256,
            scope='Only inter-process I/O scheduling; each role keeps its own full episode updates and RNG')
    if auxiliary:
        config.update(version='fresh30_continuous_episode_aux_v1',visual_auxiliary=auxiliary,
            auxiliary_label_report_sha256=sha256(out/'visual_auxiliary_labels.json'),
            auxiliary_gradient_clip='policy_excludes_head_norm1; head_norm1_separate; equal AdamW hyperparameters')
    if sampling:
        config.update(version=config['version'].removesuffix('_v1')+'_latest_repeat_v1',
            sampling_contract=sampling,episode_multiplicities=multiplicities,episodes_per_epoch=episodes_per_epoch,
            exposure='original and historical episodes once; latest takeover episodes at registered repeat; '
                'each visit recomputes full real history at current weights and retains all legal anchor masks; '
                'equal loss weight per visit; repeated visits are reuse, not independent data')
    if batch_profile:
        from visuotactile.trajectory_training import budget
        config.update(version='fresh30_continuous_full_trajectory_batch_v1',full_trajectory_batch=batch_profile,
            **budget(episodes_per_epoch,epochs),episode_multiplicities=multiplicities,
            exposure='Same registered deck visits; one AdamW/EMA step per adjacent B2 batch, singleton tail retained. '
                'Mean per-trajectory loss within actual batch; singleton coefficient1 versus paired coefficient1/2. '
                'No automatic epoch, learning-rate or EMA adjustment; not equivalent to sequential episode updates.')
        registration=root/'full_trajectory_batch.json'
        if registration.exists():
            expected=dict(**budget(episodes_per_epoch,epochs),data_sha256=config['data_sha256'],episode_multiplicities=multiplicities)
            registered=read(registration)
            if registered['profile']!=batch_profile or registered['roles'][role]!=expected:
                raise ValueError('B2 planned visit/update budget differs from cycle registration')
    begin_epoch=updates=anchors_seen=target_ticks=visits_seen=0;best=float('inf');history=[]
    if (out/'last.pt').exists():
        old=torch.load(out/'last.pt',weights_only=False,map_location='cpu')
        if old['configuration']!=config:raise ValueError('Attempt restore source/config changed; use explicit new continuation identity')
        restore_training(old,model,averaged,opt,ema,generator,noise)
        if visit_profile:coverage=Coverage(old['visit_sampling_coverage'])
        begin_epoch=old['epoch'];updates=old['updates'];best=old['best'];history=old['history'];anchors_seen=old['anchors_seen'];target_ticks=old['target_ticks']
        if batch_profile:
            visits_seen=old['visits_seen'];sampling_exposure=old['sampling_exposure']
            validate_batch_exposure(sampling_exposure,multiplicities,completed_epochs=begin_epoch,updates=updates,visits_seen=visits_seen)
            if old['ema_optimization_step']!=updates:raise ValueError('B2 EMA step count differs from optimizer count')
        elif sampling:
            sampling_exposure=old.get('sampling_exposure')
            validate_sampling_exposure(sampling_exposure,multiplicities,completed_epochs=begin_epoch,updates=updates)
        if auxiliary:
            if 'auxiliary_exposure' not in old:raise ValueError('Missing complete auxiliary exposure state')
            pose_exposure=old['auxiliary_exposure']
            validate_exposure(pose_exposure,pose_reports,data['splits']['train'],data['normalization_train_ids'],
                completed_epochs=begin_epoch,updates=updates)
    else:
        if (out/'configuration.json').exists() and read(out/'configuration.json')!=config:
            raise ValueError('Existing attempt configuration changed before first checkpoint')
        atomic_json(out/'configuration.json',config)
    reconcile_learning_log(out,durable_update=updates,completed_epochs=begin_epoch,
        episodes_per_epoch=episodes_per_epoch,
        checkpoint_sha256=sha256(out/'last.pt') if (out/'last.pt').exists() else None,
        trajectory_batch_profile=batch_profile)
    if batch_profile:
        from collections import Counter
        from visuotactile.training_receipts import learning_counters,iter_visits
        rows=[json.loads(line) for line in (out/'learning.jsonl').read_text().splitlines()] if (out/'learning.jsonl').exists() else []
        if learning_counters(rows)!=dict(updates=updates,anchors_seen=anchors_seen,target_ticks=target_ticks,visits_seen=visits_seen):
            raise ValueError('B2 durable learning counters differ from checkpoint')
        counts=Counter(v['episode'] for v in iter_visits(rows))
        if any(counts[i]!=sampling_exposure[i] for i in sampling_exposure) or set(counts)-set(sampling_exposure):
            raise ValueError('B2 durable per-episode visits differ from checkpoint')
    def checkpoint(epoch,name='last.pt'):
        if auxiliary:
            validate_exposure(pose_exposure,pose_reports,data['splits']['train'],data['normalization_train_ids'],
                completed_epochs=epoch,updates=updates)
        extras=dict(auxiliary_exposure=pose_exposure) if auxiliary else {}
        if visit_profile:extras['visit_sampling_coverage']=coverage.state_dict()
        if batch_profile:
            validate_batch_exposure(sampling_exposure,multiplicities,completed_epochs=epoch,updates=updates,visits_seen=visits_seen)
            if ema.optimization_step!=updates:raise ValueError('B2 EMA/optimizer steps differ')
            extras.update(sampling_exposure=dict(sampling_exposure),visits_seen=visits_seen)
        elif sampling:
            validate_sampling_exposure(sampling_exposure,multiplicities,completed_epochs=epoch,updates=updates)
            extras['sampling_exposure']=dict(sampling_exposure)
        save(out/name,training_state(model,averaged,opt,ema,generator,noise,kind='dp',group='I-HistoryDP',
            updates=updates,epoch=epoch,configuration=config,best=best,history=history,anchors_seen=anchors_seen,target_ticks=target_ticks,
            train_ids=data['splits']['train'],validation_ids=data['splits']['validation'],**extras))
    started=time.perf_counter()
    def evaluate(identities):
        io_options={} if visual_io is None else dict(frozen_visual_io=visual_io)
        if offline_planning is None:return metrics(averaged,records,identities,**io_options)
        return metrics(averaged,records,identities,offline_planning=offline_planning,check=lease.check,**io_options)
    if continuation and not (out/'initial_metrics.json').exists():
        with lease.block():
            initial=dict(validation=evaluate(data['splits']['validation']),
                train=evaluate(data['splits']['train'][:2]),
                correction=evaluate(data['aggregation']['new_training_ids']))
            if auxiliary:
                initial['visual_auxiliary']=dict(
                    validation=auxiliary_metrics(averaged,records,data['splits']['validation'],pose_contexts,auxiliary),
                    train=auxiliary_metrics(averaged,records,data['splits']['train'][:2],pose_contexts,auxiliary))
        atomic_json(out/'initial_metrics.json',initial)
    for epoch in range(begin_epoch,epochs):
        order=episode_deck(data['splits']['train'],multiplicities,generator);losses=[]
        if pairing is not None:pairing.begin_epoch(epoch+1,order)
        epoch_io_before=io_snapshot();epoch_began=time.perf_counter();epoch_started_utc=time.time()
        prefetch=(EpisodePrefetch(order,lambda identity:Episode(dr,identity,'I-HistoryDP'),io_config)
                  if io_config is not None else None)
        try:
            if batch_profile:
                from visuotactile.trajectory_training import batches,execute_update
                from visuotactile.training_receipts import BATCH_ROW_VERSION
                deck_receipt=dict(epoch=epoch+1,order=order,
                    sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest())
                deck_path=out/('deck_%03d.json'%(epoch+1))
                if deck_path.exists():
                    if read(deck_path)!=deck_receipt:raise ValueError('B2 restored registered deck changed')
                else:atomic_json(deck_path,deck_receipt)
                visit_losses=[]
                for batch_index,visits in enumerate(batches(order),1):
                    began=time.perf_counter();io_before=io_snapshot()
                    identities=[identity for _,identity in visits]
                    with lease.block():
                        # No previous batch survives into the next load, including
                        # partial loads or failed updates. B2 owns only these visits.
                        episodes=[]
                        try:
                            for identity in identities:
                                episodes.append(Episode(dr,identity,'I-HistoryDP',
                                    **({} if visual_io is None else dict(frozen_visual_io=visual_io))))
                            visit_options={}
                            if visit_profile:
                                selections=[sample_visit(records[identity],visit_profile,identity=identity,epoch=epoch+1,visit_index=index)
                                    for index,identity in visits]
                                visit_options=dict(selected_anchors=[v[0] for v in selections],
                                    anchor_receipts=[v[1] for v in selections],geometry_generators=[v[2] for v in selections],
                                    geometry_sampling_profile=visit_profile['geometry'])
                            result=execute_update(model,opt,ema,objective,identities,
                                [records[identity] for identity in identities],episodes,noise,check=lease.check,**visit_options)
                            if visual_io is not None:
                                for visit_row,episode in zip(result['visits'],episodes):
                                    visit_row['frozen_visual_io']=episode.frozen_visual_io_receipt
                                del episode
                        finally:
                            episodes.clear();del episodes
                        updates+=1
                        for (index,identity),visit_row in zip(visits,result['visits']):
                            if visit_profile:coverage.add(identity,visit_row['anchor_sampling'],visit_row.get('geometry_sampling'))
                            sampling_exposure[identity]+=1;visits_seen+=1
                            visit_row.update(episode_index=index,episode_visit=sampling_exposure[identity])
                            anchors_seen+=visit_row['anchors'];target_ticks+=visit_row['target_ticks']
                            visit_losses.append(visit_row['loss'])
                        losses.append(result['loss']);torch.cuda.synchronize()
                    row=dict(version=BATCH_ROW_VERSION,full_trajectory_batch=batch_profile,
                        epoch=epoch+1,batch_index=batch_index,updates=updates,visits_seen=visits_seen,
                        visits=result['visits'],episodes_per_epoch=episodes_per_epoch,
                        optimizer_updates_per_epoch=config['optimizer_updates_per_epoch'],
                        loss=result['loss'],gradient_norm=result['gradient_norm'],
                        anchors_seen=anchors_seen,target_ticks=target_ticks,
                        visual_frames=sum(v['visual_frames'] for v in result['visits']),
                        deck_sha256=deck_receipt['sha256'],seconds=time.perf_counter()-began,utc=time.time(),
                        process_io=io_delta(io_before,io_snapshot()))
                    atomic_json(out/'progress.json',row)
                    with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            else:
                for order_index,i in enumerate(order):
                    rec=records[i];began=time.perf_counter()
                    io_before=io_snapshot()
                    pair_ready_wait=(pairing.before_visit(epoch+1,order_index,i) if pairing is not None else None)
                    visit=(prefetch.visit(order_index) if prefetch is not None else nullcontext((None,None)))
                    with visit as (episode,episode_io),lease.block():
                        model.train();opt.zero_grad(set_to_none=True)
                        anchor_receipt=geometry_generator=None
                        if visit_profile:
                            selected_anchors,anchor_receipt,geometry_generator=sample_visit(rec,visit_profile,identity=i,epoch=epoch+1,visit_index=order_index+1)
                            cond,a=supervised_conditions(model,rec,episode=episode,selected_anchors=selected_anchors)
                        elif auxiliary:cond,a,visual=supervised_visual_conditions(model,rec,episode=episode)
                        else:cond,a=supervised_conditions(model,rec,episode=episode)
                        truth,valid=labels(rec,a,24)
                        selected=[(i,int(t)) for t in a]
                        context=projection_batch(objective,selected,'cuda:0',**(dict(geometry_generator=geometry_generator,
                            sampling_profile=visit_profile['geometry']) if visit_profile and role=='A' else {}))
                        loss,parts,_=objective.forward(model,cond,truth,valid,noise,geometry_context=context)
                        if auxiliary:
                            weighted_pose,pose_parts=auxiliary_episode_loss(model,rec,visual,pose_contexts[i],auxiliary)
                            parts=dict(parts,auxiliary_position=pose_parts['position'],auxiliary_orientation6=pose_parts['orientation6'],
                                auxiliary_weighted=weighted_pose)
                            loss=loss+weighted_pose
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fresh loss')
                        loss.backward()
                        norm=(clip_auxiliary_gradients(model,model.visual_pose_head)['policy_norm'] if auxiliary
                            else torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                        opt.step();ema.step(model);updates+=1;anchors_seen+=len(a);target_ticks+=int(valid.sum())
                        if visit_profile:coverage.add(i,anchor_receipt,context.get('selection_report') if context else None)
                        if sampling:sampling_exposure[i]+=1
                        if auxiliary:add_exposure(pose_exposure,i,pose_reports[i],data['normalization_train_ids'])
                        losses.append(float(loss));torch.cuda.synchronize()
                    row=dict(epoch=epoch+1,episode=i,episode_index=order_index+1,updates=updates,loss=losses[-1],
                        parts={k:float(v.detach()) for k,v in parts.items()},anchors_seen=anchors_seen,target_ticks=target_ticks,
                        visual_frames=len(rec['time']),seconds=time.perf_counter()-began,utc=time.time())
                    if visit_profile:
                        row.update(anchor_sampling=anchor_receipt,
                            geometry_sampling=context.get('selection_report') if context else None)
                    row['process_io']=io_delta(io_before,io_snapshot())
                    if episode_io is not None:row['episode_io']=episode_io
                    if pairing is not None:row['paired_episode_io_ready_wait_s']=pair_ready_wait
                    if auxiliary:row['auxiliary_exposure']=pose_reports[i]['perception_exposure']
                    if sampling:row.update(episode_visit=sampling_exposure[i],episodes_per_epoch=episodes_per_epoch)
                    atomic_json(out/'progress.json',row)
                    with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if pairing is not None:pairing.after_visit(updates)
        finally:
            if prefetch is not None:prefetch.close()
        training_epoch_io=dict(process_io=io_delta(epoch_io_before,io_snapshot()),
            wall_seconds=time.perf_counter()-epoch_began,started_utc=epoch_started_utc,ended_utc=time.time(),
            prefetch=prefetch.summary() if prefetch is not None else None,
            scope='Whole training epoch before validation; process I/O includes overlapping preload and OS cache sharing; no per-worker disk attribution')
        if pairing is not None:training_epoch_io['paired_episode_io']=pairing.summary(epoch=epoch+1)
        with lease.block():
            val=evaluate(data['splits']['validation']);tr=evaluate(data['splits']['train'][:2])
            pose_metrics_report=(dict(validation=auxiliary_metrics(averaged,records,data['splits']['validation'],pose_contexts,auxiliary),
                train=auxiliary_metrics(averaged,records,data['splits']['train'][:2],pose_contexts,auxiliary)) if auxiliary else None)
        report=dict(epoch=epoch+1,updates=updates,loss=sum(losses)/len(losses),train=tr,validation=val,
            seconds=time.perf_counter()-started,anchors_seen=anchors_seen,target_ticks=target_ticks,utc=time.time())
        report['training_epoch_io']=training_epoch_io
        if visit_profile:
            report['visit_sampling_coverage']=coverage.summary()
            atomic_json(out/'sampling_coverage.json',dict(profile=visit_profile,summary=coverage.summary(),episodes=coverage.state_dict()))
        if batch_profile:report.update(visits_seen=visits_seen,sampling_exposure=dict(sampling_exposure),
            full_trajectory_batch=batch_profile,loss_per_optimizer_batch_mean=report['loss'],
            loss_per_episode_visit_mean=sum(visit_losses)/len(visit_losses),
            loss_scope='Batch loss is the mean of actual one/two visit losses; visit mean is a separate descriptive statistic.')
        if auxiliary:
            report.update(visual_auxiliary=pose_metrics_report,auxiliary_exposure=copy.deepcopy(pose_exposure))
        if sampling:report['sampling_exposure']=dict(sampling_exposure)
        if continuation and epoch+1==epochs:
            with lease.block():report['correction']=evaluate(data['aggregation']['new_training_ids'])
        history.append(report);improved=val['selection']<best;best=min(best,val['selection'])
        checkpoint(epoch+1)
        if improved:checkpoint(epoch+1,'best.pt')
        atomic_json(out/'validation.json',report)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(dict(role=role,epoch=epoch+1,updates=updates,loss=report['loss'],train=tr['selection'],validation=val['selection'])),flush=True)
    finished=dict(epochs=epochs,updates=updates,anchors_seen=anchors_seen,target_ticks=target_ticks,
        checkpoint_sha256=sha256(out/'best.pt'),complete=True,wall_seconds=time.perf_counter()-started)
    if visit_profile:finished.update(training_sampling=visit_profile,visit_sampling_coverage=coverage.summary())
    if auxiliary:finished['auxiliary_exposure']=pose_exposure
    if sampling:finished.update(sampling_contract=sampling,episode_multiplicities=multiplicities,
        episodes_per_epoch=episodes_per_epoch,sampling_exposure=sampling_exposure)
    if batch_profile:finished.update(full_trajectory_batch=batch_profile,visits_seen=visits_seen,
        ema_updates=ema.optimization_step,sampling_exposure=dict(sampling_exposure),episode_multiplicities=multiplicities,
        **budget(episodes_per_epoch,epochs))
    if pairing is not None:finished['paired_episode_io']=pairing.summary()
    if offline_planning is not None:finished['offline_planning']=offline_planning
    atomic_json(out/'finished.json',finished)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gate','train']);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--role',choices=['A','B']);p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--offline-anchor-batch',type=int,help='Explicit offline metrics DDIM anchor batch; training updates and online planning unchanged')
    p.add_argument('--full-trajectory-batch',action='store_true',help='Explicit B2 training method: same visits, one optimizer/EMA update per actual batch; incompatible with paired I/O and prefetch')
    a=p.parse_args();root=a.root.resolve()
    if a.offline_anchor_batch is not None and (a.stage!='train' or not 1<=a.offline_anchor_batch<=128):
        p.error('Offline anchor batching requires train stage and an explicit size1..128')
    if a.full_trajectory_batch and (a.stage!='train' or a.role is None):p.error('B2 requires a train stage and explicit role')
    if a.stage=='gate':gate(root)
    elif a.full_trajectory_batch:
        from visuotactile.trajectory_training import profile
        train(root,a.role,a.epochs,offline_anchor_batch=a.offline_anchor_batch,trajectory_batch_profile=profile())
    elif a.offline_anchor_batch is None:train(root,a.role,a.epochs)
    else:train(root,a.role,a.epochs,offline_anchor_batch=a.offline_anchor_batch)

if __name__=='__main__':main()
