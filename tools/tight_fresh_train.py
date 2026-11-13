"""Fresh full-episode training and measured continuous-vision acceptance.

Episode balance is exact per epoch; no past learned hidden state is cached.
"""
import argparse,copy,json,time
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


def read(p):return json.loads(Path(p).read_text())


def inputs(root,role):
    data_root=root/role/'state_data';data=read(data_root/'data.json');ids=data['splits']['train']+data['splits']['validation']
    records=visual_records(data_root,ids,'G00',device='cuda:0')
    return data_root,data,records,torch.load(data_root/'statistics.pt',weights_only=False,map_location='cpu')


def new_model(role,statistics):
    torch.manual_seed(2026111401)
    model=NativePolicy('I-HistoryDP',24,modality='G00')
    model.contract=continuous_image_contract('q_reached' if role=='A' else 'u_expert')
    for k,v in statistics.items():getattr(model,k).copy_(v)
    return model.cuda()


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
def metrics(model,records,identities):
    model.eval();errors=[];normalized=[];rows=[]
    for i in identities:
        rec=records[i];c,a=encode_dense(model,rec)
        # Online draws one H24 noise tensor per replan. Keep that RNG order.
        g=torch.Generator(device='cuda').manual_seed(7091)
        plans=[]
        for k in range(len(c)):plans.append(model.predict(c[k:k+1],g,scheduler_index_mode='host'))
        pred=torch.cat(plans)
        pred=bounded_targets(pred.flatten(0,1),rec['joint_limits']).reshape_as(pred)
        truth,valid=labels(rec,a,24);mask=valid[:,:12];e=(pred[:,:12]-truth[:,:12])[mask]
        errors.append(e.cpu());normalized.append((e*model.action_scale).cpu())
        rows.append(dict(identity=i,arm_rmse=float(e[:,:7].square().mean().sqrt()),hand_rmse=float(e[:,7:].square().mean().sqrt())))
    e=torch.cat(errors);n=torch.cat(normalized)
    return dict(arm_rmse_rad=float(e[:,:7].square().mean().sqrt()),hand_rmse_rad=float(e[:,7:].square().mean().sqrt()),
        selection=float(.5*n[:,:7].square().mean()+.5*n[:,7:].square().mean()),episodes=rows,
        scope='teacher-forced full causal prefix; DDIM16 commit12; no autonomy claim')


def train(root,role,epochs):
    from visuotactile.state_image_loss import GeometryObjective,PALM_LOSS_ID,geometry_loss_context
    from visuotactile.anchor_continue import projection_batch
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32
    if not read(root/'gate.json')['passed'] or not read(root/role/'cache_index.json')['complete']:raise ValueError('Gate incomplete')
    state_fp32();session=read(root/'session.json');lease=lease_for(root,deadline=session['deadline'])
    out=root/role/'training';out.mkdir(exist_ok=True)
    with lease.block():
        dr,data,records,stats=inputs(root,role);model=new_model(role,stats)
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    meta=read(root/'metadata.json');objective=GeometryObjective(meta,mode=PALM_LOSS_ID)
    objective.weights=dict(position=.3342396121814294,rotation=.029946394754032265)
    geometry=None
    if role=='A':
        base=Path('outputs/visuotactile/run')
        acceptance=base/'tight_transformer_20261110_02/geometry_acceptance.json'
        live=base/'tight_transformer_20261110_01/physics_gate_meshfix/geometry_live.json'
        a=read(acceptance);inventory=read(live)['inventory']
        objective.enable_hand_geometry(inventory,a,loss_kind='hand_surface_soft_hinge_v1',penetration_tolerance_m=.001,contact_tolerance_m=.002)
        objective.weights['hand_geometry']=a['hand_geometry_weight']
        objective.loss_context={i:geometry_loss_context(data['entries'][i]) for i in records}
        geometry=dict(acceptance_sha256=sha256(acceptance),live_sha256=sha256(live),
            loss_kind='hand_surface_soft_hinge_v1',weight=a['hand_geometry_weight'],hyperparameter_source='prior accepted train-only physical geometry calibration; no test tuning')
    generator=torch.Generator().manual_seed(2026111402);noise=torch.Generator(device='cuda').manual_seed(2026111403)
    config=dict(version='fresh30_continuous_episode_v1',data_sha256=sha256(dr/'data.json'),statistics_sha256=sha256(dr/'statistics.pt'),
        dino_sha256=session['dino_sha256'],cache_index_sha256=sha256(root/role/'cache_index.json'),
        initialization='random identical shared parameter seed; fresh AdamW/EMA',learning_rate=1e-4,weight_decay=1e-6,
        history='full actual episode prefix at current weights; every frame; full episode gradients',
        exposure='each episode and every legal planning anchor once per epoch; loss equal episode weight',
        validation='all six fixed families DDIM16 commit12; fixed train2 diagnostics',hand_geometry=geometry,
        normalization_identity=model.contract['normalization_identity'],input_builder='canonical_joint_state_v1',
        source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools'] for p in sorted(Path(folder).rglob('*.py'))})
    begin_epoch=updates=anchors_seen=target_ticks=0;best=float('inf');history=[]
    if (out/'last.pt').exists():
        old=torch.load(out/'last.pt',weights_only=False,map_location='cpu')
        if old['configuration']!=config:raise ValueError('Attempt restore source/config changed; use explicit new continuation identity')
        restore_training(old,model,averaged,opt,ema,generator,noise)
        begin_epoch=old['epoch'];updates=old['updates'];best=old['best'];history=old['history'];anchors_seen=old['anchors_seen'];target_ticks=old['target_ticks']
    else:atomic_json(out/'configuration.json',config)
    def checkpoint(epoch,name='last.pt'):
        save(out/name,training_state(model,averaged,opt,ema,generator,noise,kind='dp',group='I-HistoryDP',
            updates=updates,epoch=epoch,configuration=config,best=best,history=history,anchors_seen=anchors_seen,target_ticks=target_ticks,
            train_ids=data['splits']['train'],validation_ids=data['splits']['validation']))
    started=time.perf_counter()
    for epoch in range(begin_epoch,epochs):
        order=torch.randperm(len(data['splits']['train']),generator=generator).tolist();losses=[]
        for order_index,j in enumerate(order):
            i=data['splits']['train'][j];rec=records[i];began=time.perf_counter()
            with lease.block():
                model.train();opt.zero_grad(set_to_none=True)
                cond,a=encode_dense(model,rec);truth,valid=labels(rec,a,24)
                selected=[(i,int(t)) for t in a]
                context=projection_batch(objective,selected,'cuda:0')
                loss,parts,_=objective.forward(model,cond,truth,valid,noise,geometry_context=context)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fresh loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();ema.step(model);updates+=1;anchors_seen+=len(a);target_ticks+=int(valid.sum())
                losses.append(float(loss));torch.cuda.synchronize()
            row=dict(epoch=epoch+1,episode=i,episode_index=order_index+1,updates=updates,loss=losses[-1],
                parts={k:float(v.detach()) for k,v in parts.items()},anchors_seen=anchors_seen,target_ticks=target_ticks,
                visual_frames=len(rec['time']),seconds=time.perf_counter()-began,utc=time.time())
            atomic_json(out/'progress.json',row)
            with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        with lease.block():
            val=metrics(averaged,records,data['splits']['validation']);tr=metrics(averaged,records,data['splits']['train'][:2])
        report=dict(epoch=epoch+1,updates=updates,loss=sum(losses)/len(losses),train=tr,validation=val,
            seconds=time.perf_counter()-started,anchors_seen=anchors_seen,target_ticks=target_ticks,utc=time.time())
        history.append(report);improved=val['selection']<best;best=min(best,val['selection'])
        checkpoint(epoch+1)
        if improved:checkpoint(epoch+1,'best.pt')
        atomic_json(out/'validation.json',report)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(dict(role=role,epoch=epoch+1,updates=updates,loss=report['loss'],train=tr['selection'],validation=val['selection'])),flush=True)
    atomic_json(out/'finished.json',dict(epochs=epochs,updates=updates,anchors_seen=anchors_seen,target_ticks=target_ticks,
        checkpoint_sha256=sha256(out/'best.pt'),complete=True,wall_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gate','train']);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--role',choices=['A','B']);p.add_argument('--epochs',type=int,default=5)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='gate':gate(root)
    else:train(root,a.role,a.epochs)

if __name__=='__main__':main()
