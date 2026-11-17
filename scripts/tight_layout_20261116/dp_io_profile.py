"""Bounded same-state DP I/O comparison; invokes the existing training transaction.

No selected-model promotion, backend, simulation, or new data. Fine timing is
restricted to three registered updates; all other updates retain normal kernels.
"""
import argparse,copy,hashlib,json,os,resource,time,threading,subprocess
from pathlib import Path
from contextlib import contextmanager
import torch
from tools import tight_fresh_train as train
from visuotactile.checkpointing import restore_training,training_state,save
from visuotactile.episode_io import io_snapshot,io_delta,available_memory
from visuotactile.compact import sha256
from visuotactile.frozen_visual_io import mapped_profile
from visuotactile.native_data import Episode
from visuotactile.precision import state_fp32
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.latest_takeover_sampling import episode_deck
from visuotactile.fresh_visit_sampling import load_profile,visit
from visuotactile.trajectory_training import batches,execute_update


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sync():torch.cuda.synchronize()
def digest(value):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,torch.Tensor):
   h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());walk(v[k])
  elif isinstance(v,(tuple,list)):
   for x in v:walk(x)
  else:h.update(repr(v).encode())
 walk(value);return h.hexdigest()

@contextmanager
def detailed(objective,enabled):
 """Synchronized, nested inclusive timings. Do not sum nested regions."""
 rows={};patches=[]
 def patch(obj,name,label,condition=lambda *a,**k:True):
  old=getattr(obj,name)
  def wrapped(*a,**kw):
   if not condition(*a,**kw):return old(*a,**kw)
   sync();start=time.perf_counter();before=io_snapshot()
   result=old(*a,**kw);sync()
   row=rows.setdefault(label,dict(seconds=0.,calls=0,read_bytes=0))
   row['seconds']+=time.perf_counter()-start;row['calls']+=1;row['read_bytes']+=io_delta(before,io_snapshot())['read_bytes']
   return result
  setattr(obj,name,wrapped);patches.append((obj,name,old))
 if enabled:
  import visuotactile.trajectory_batch as tb
  import visuotactile.anchor_continue as ac
  patch(tb,'supervised_trajectories','history_forward_including_H2D')
  patch(ac,'projection_batch','geometry_context')
  patch(tb,'trajectory_objective','denoiser_and_losses')
  patch(objective,'components','loss_components_including_geometry')
  patch(objective.hand_geometry,'per_configuration','hand_geometry_forward')
  patch(torch.Tensor,'backward','backward')
  patch(torch.optim.AdamW,'step','optimizer')
  patch(EMAModel,'step','ema')
  patch(torch.Tensor,'to','pageable_H2D_including_faults',lambda x,*a,**k:x.device.type=='cpu' and ((a and str(a[0]).startswith('cuda')) or str(k.get('device','')).startswith('cuda')))
 try:yield rows
 finally:
  for obj,name,old in reversed(patches):setattr(obj,name,old)


def main(args):
 if not 1<=args.epochs<=12:raise ValueError('Short diagnostic is bounded to at most 336 updates')
 root=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 admission_start=time.time();idle_since=None
 while args.wait_idle:
  result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  active=[int(v) for v in result.stdout.split() if v.isdigit() and int(v)!=os.getpid()]
  now=time.time();idle_since=(idle_since or now) if not active else None
  dump(out/'admission.json',dict(started_utc=admission_start,utc=now,other_gpu_processes=len(active),idle_since=idle_since,seconds=now-admission_start))
  if idle_since and now-idle_since>=30:break
  if now-admission_start>1200:raise TimeoutError('Bounded admission wait expired; no optimizer update')
  time.sleep(5)
 torch.set_num_threads(2);state_fp32();start=time.perf_counter();start_utc=time.time()
 telemetry=[];stop=threading.Event()
 def monitor():
  while not stop.is_set():
   r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
   status=dict(x.split(':',1) for x in Path('/proc/self/status').read_text().splitlines() if ':' in x)
   telemetry.append(dict(utc=time.time(),available=available_memory(),rss=int(status['VmRSS'].split()[0])*1024,gpu=r.stdout.strip(),returncode=r.returncode))
   stop.wait(1)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  dr,data,records,stats=train.inputs(root,'A')
  if len(data['splits']['train'])!=56:raise ValueError('This registered comparison requires the frozen56 data')
  config=train.read(root/'A/training/configuration.json')
  checkpoint=root/'A/training/last.pt';source_hash=sha256(checkpoint)
  model=train.new_model('A',stats);averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
  opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
  deck=torch.Generator();noise=torch.Generator(device='cuda');source=torch.load(checkpoint,map_location='cpu',weights_only=False)
  restore_training(source,model,averaged,opt,ema,deck,noise)
  initial=digest({k:source[k] for k in ['model','ema','optimizer','schedule_rng','noise_rng','torch_rng','cuda_rng']})
  start_epoch=source['epoch'];start_updates=source['updates'];del source
  profile=load_profile(root);objective,_,proof=train.fresh_objective(root,'A',data,records,geometry_sampling_profile=profile['geometry'])
  # File-specific advisory eviction, never global drop_caches. No other Tight
  # worker is active. This is a request, not proof of fully cold storage.
  files=[dr/'perception'/(i+'.pt') for i in records]
  inventory=[dict(identity=i,path=str(p.resolve()),bytes=p.stat().st_size) for i,p in zip(records,files)]
  for p in files:
   fd=os.open(p,os.O_RDONLY)
   try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
  loader=None
  if args.variant=='hotset':
   from visuotactile.frozen_visual_io import FrozenVisualHotset
   loader=FrozenVisualHotset(dr,data['splits']['validation']+data['splits']['train'],max_bytes=4*1024**3)
  original_episode=train.Episode
  if loader is not None:train.Episode=lambda r,i,g,**kw:loader(i)
  rows=[];epochs=[];setup=time.perf_counter()-start;stage_start=time.perf_counter();stage_io=io_snapshot()
  dump(out/'registration.json',dict(source=str(root),source_checkpoint_sha256=source_hash,initial_state_sha256=initial,source_epoch=start_epoch,source_updates=start_updates,variant=args.variant,epochs=args.epochs,updates=args.epochs*28,lr=[g['lr'] for g in opt.param_groups],data_sha256=sha256(dr/'data.json'),statistics_sha256=sha256(dr/'statistics.pt'),cache_index_sha256=sha256(root/'A/cache_index.json'),sampling=profile,config=config,files=inventory,cold='fresh process; file-specific POSIX_FADV_DONTNEED on the 62 inactive source shards; advisory, not global cache flush',setup_seconds=setup,started_utc=start_utc))
  for e in range(start_epoch+1,start_epoch+args.epochs+1):
   epoch_start=time.perf_counter();epoch_io=io_snapshot();order=episode_deck(data['splits']['train'],dict.fromkeys(data['splits']['train'],1),deck)
   for idx,visits in enumerate(batches(order),1):
    if available_memory()<12*1024**3:raise RuntimeError('RAM reserve; bounded diagnostic stopped')
    if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('VRAM reserve; bounded diagnostic stopped')
    sync();began=time.perf_counter();update_utc=time.time();before=io_snapshot();ids=[i for _,i in visits]
    episodes=[loader(i) if loader else Episode(dr,i,'I-HistoryDP',frozen_visual_io=mapped_profile()) for i in ids]
    sync();load_s=time.perf_counter()-began;load_io=io_delta(before,io_snapshot());sample_start=time.perf_counter()
    selections=[visit(records[i],profile,identity=i,epoch=e,visit_index=v) for v,i in visits]
    sync();sample_s=time.perf_counter()-sample_start
    sequence=len(rows)+1
    with detailed(objective,sequence in (1,2,29)) as timing:
     result=execute_update(model,opt,ema,objective,ids,[records[i] for i in ids],episodes,noise,selected_anchors=[s[0] for s in selections],anchor_receipts=[s[1] for s in selections],geometry_generators=[s[2] for s in selections],geometry_sampling_profile=profile['geometry'])
    sync();elapsed=time.perf_counter()-began
    row=dict(started_utc=update_utc,ended_utc=time.time(),update=sequence,epoch=e,batch=idx,ids=ids,seconds=elapsed,load_seconds=load_s,sampling_seconds=sample_s,load_io=load_io,io=io_delta(before,io_snapshot()),detail=timing,loss=result['loss'],gradient_norm=result['gradient_norm'],visits=result['visits'])
    if sequence in (1,29,args.epochs*28):row['gradient_sha256']=digest({n:p.grad for n,p in model.named_parameters() if p.grad is not None})
    if sequence==1 and args.save_gradient:
     torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None},out/'first_gradients.pt')
    rows.append(row)
    with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    del episodes,result
   train_s=time.perf_counter()-epoch_start;train_io=io_delta(epoch_io,io_snapshot());vstart=time.perf_counter();vbefore=io_snapshot()
   val=train.metrics(averaged,records,data['splits']['validation'],offline_planning=config['offline_planning'],frozen_visual_io=mapped_profile())
   tr=train.metrics(averaged,records,data['splits']['train'][:2],offline_planning=config['offline_planning'],frozen_visual_io=mapped_profile())
   sync();validation_s=time.perf_counter()-vstart;validation_io=io_delta(vbefore,io_snapshot());sstart=time.perf_counter()
   payload=training_state(model,averaged,opt,ema,deck,noise,kind='diagnostic_only',epoch=e,updates=start_updates+len(rows),configuration=config)
   save(out/'last.pt',payload);save_s=time.perf_counter()-sstart;del payload
   er=dict(epoch=e,train_seconds=train_s,train_io=train_io,validation_seconds=validation_s,validation_io=validation_io,save_seconds=save_s,validation=val,train=tr)
   epochs.append(er);dump(out/'epochs.json',epochs);print(json.dumps({k:v for k,v in er.items() if k not in ('validation','train')}),flush=True)
  sync();stage_s=time.perf_counter()-stage_start;total_io=io_delta(stage_io,io_snapshot())
  if loader is not None and args.save_gradient:dump(out/'cached_tensor_audit.json',loader.audit_cached())
  final=digest(dict(model=model.state_dict(),ema=averaged.state_dict(),optimizer=opt.state_dict(),deck=deck.get_state(),noise=noise.get_state()))
  result=dict(complete=True,variant=args.variant,updates=len(rows),stage_seconds=stage_s,setup_seconds=setup,process_seconds=time.perf_counter()-start,process_io=total_io,final_state_sha256=final,initial_state_sha256=initial,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),cache=None if loader is None else loader.summary(),started_utc=start_utc,ended_utc=time.time(),limitations='224-update A diagnostic, not full long training or autonomous performance. No scheduler transition in this inherited constant-LR short window. Detailed sync overhead in updates1/2/29 in both arms. Page faults and pageable H2D overlap cannot be completely separated.')
  dump(out/'result.json',result);train.Episode=original_episode;print(json.dumps(result),flush=True)
 finally:
  stop.set();thread.join();dump(out/'telemetry.json',telemetry)

def numeric_difference(left,right):
 """CPU/mmap comparison; describe residuals without claiming bitwise parity."""
 import math
 rows=[];metadata=[]
 def walk(a,b,path):
  if isinstance(a,torch.Tensor):
   if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:
    raise ValueError('State tensor structure changed: '+path)
   x=a.reshape(-1);y=b.reshape(-1);maximum=ss=reference=0.;exact=True
   for first in range(0,len(x),262144):
    u=x[first:first+262144];v=y[first:first+262144]
    exact=exact and torch.equal(u,v)
    u=u.double();v=v.double();d=u-v
    maximum=max(maximum,float(d.abs().max()) if d.numel() else 0.)
    ss+=float(d.square().sum());reference+=float(u.square().sum())
   rows.append(dict(path=path,elements=a.numel(),exact=exact,max_abs=maximum,
    squared_difference=ss,squared_reference=reference))
  elif isinstance(a,dict):
   if a.keys()!=b.keys():raise ValueError('State keys changed: '+path)
   for k in a:walk(a[k],b[k],path+'/'+str(k))
  elif isinstance(a,(list,tuple)):
   if type(a)!=type(b) or len(a)!=len(b):raise ValueError('State sequence changed: '+path)
   for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
  elif a!=b:metadata.append(dict(path=path,left=a,right=b))
 walk(left,right,'')
 ss=sum(r['squared_difference'] for r in rows);ref=sum(r['squared_reference'] for r in rows)
 return dict(tensors=len(rows),elements=sum(r['elements'] for r in rows),
  exact=all(r['exact'] for r in rows) and not metadata,
  max_abs=max((r['max_abs'] for r in rows),default=0.),relative_l2=math.sqrt(ss/max(ref,1e-300)),
  metadata_differences=metadata,largest_residuals=sorted(rows,key=lambda r:r['max_abs'],reverse=True)[:8])


def summarize(root, output):
 """Export original lightweight receipts and both registered comparisons."""
 import gzip,statistics,shutil
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 runs={};loaded={}
 def read(path):return json.loads(path.read_text())
 def distribution(values):
  x=sorted(values)
  return dict(count=len(x),mean=statistics.mean(x),median=statistics.median(x),p90=x[int(.9*(len(x)-1))],p95=x[int(.95*(len(x)-1))],maximum=max(x))
 names=('baseline_recovery','hotset','hotset_reverse','baseline_reverse')
 for name in names:
  directory=root/name;result=read(directory/'result.json');assert result['complete']
  rows=[json.loads(x) for x in (directory/'updates.jsonl').read_text().splitlines()]
  epochs=read(directory/'epochs.json');telemetry=read(directory/'telemetry.json')
  runs[name]=dict(result,update_distribution=distribution([x['seconds'] for x in rows]),
   steady_update_distribution=distribution([x['seconds'] for x in rows[28:] if not x['detail']]),
   first_epoch=epochs[0],train_seconds=sum(x['train_seconds'] for x in epochs),
   validation_seconds=sum(x['validation_seconds'] for x in epochs),save_seconds=sum(x['save_seconds'] for x in epochs),
   train_physical_bytes=sum(x['train_io']['read_bytes'] for x in epochs),
   validation_physical_bytes=sum(x['validation_io']['read_bytes'] for x in epochs),
   minimum_available_ram=min(x['available'] for x in telemetry),
   minimum_gpu_free_mib=min(float(x['gpu'].split(',')[2]) for x in telemetry if x['returncode']==0),
   telemetry_samples=len(telemetry),telemetry_max_gap=max(b['utc']-a['utc'] for a,b in zip(telemetry,telemetry[1:])),
   visits=sum(len(x['visits']) for x in rows),unique_trajectories=len({i for x in rows for i in x['ids']}),
   anchors=sum(v['anchors'] for x in rows for v in x['visits']),
   expert_target_slots=sum(v['target_ticks'] for x in rows for v in x['visits']),
   real_visual_frames=sum(v['visual_frames'] for x in rows for v in x['visits']))
  loaded[name]=(rows,epochs)
  dest=output/name;dest.mkdir(exist_ok=True)
  for file in ('registration.json','result.json','epochs.json','updates.jsonl','telemetry.json','admission.json','cached_tensor_audit.json'):
   if not (directory/file).exists():continue
   payload=(directory/file).read_bytes()
   if len(payload)>100000:(dest/(file+'.gz')).write_bytes(gzip.compress(payload,mtime=0))
   else:(dest/file).write_bytes(payload)
 from visuotactile.native_data import visual_records
 from visuotactile.anchor_sampling import labels
 registration=read(root/'baseline_recovery/registration.json')
 source=Path(registration['source']);dr=source/'A/state_data';data=read(dr/'data.json')
 contract=read(source/'A/continuation.json').get('target_window_contract')
 records=visual_records(dr,data['splits']['train'],'G00',target_window_contract=contract)
 supervision=dict(numeric_sha256=sha256(dr/'numeric.pt'),contract=contract,
  reconstruction='CPU labels/masks reconstructed from immutable source and actual sampling receipts, not a separate runtime tensor capture',runs={})
 for name,(rows,_) in loaded.items():
  h=hashlib.sha256();slots=0
  for row in rows:
   for v in row['visits']:
    anchors=torch.tensor(v['anchor_sampling']['selected_anchors'],dtype=torch.long)
    truth,mask=labels(records[v['episode']],anchors,24)
    assert int(mask.sum())==v['target_ticks']
    h.update(digest(dict(identity=v['episode'],anchors=anchors,truth=truth,mask=mask)).encode())
    slots+=int(mask.sum())
  supervision['runs'][name]=dict(labels_and_masks_sha256=h.hexdigest(),valid_target_slots=slots)
 dump(output/'supervision_reconstruction.json',supervision)
 comparisons={}
 for key,an,bn,count in [('main','baseline_recovery','hotset',224),('reverse','baseline_reverse','hotset_reverse',56)]:
  a,ae=loaded[an];b,be=loaded[bn];assert len(a)==len(b)==count
  receipt_keys=('episode','anchors','target_ticks','visual_frames','loss_weight','anchor_sampling','geometry_sampling')
  comparison=dict(same_initial_state=runs[an]['initial_state_sha256']==runs[bn]['initial_state_sha256'],
   bitwise_final_state=runs[an]['final_state_sha256']==runs[bn]['final_state_sha256'],
   same_order=all(x['ids']==y['ids'] and x['epoch']==y['epoch'] for x,y in zip(a,b)),
   same_sampling_exposure=all(all(v[k]==w[k] for k in receipt_keys) for x,y in zip(a,b) for v,w in zip(x['visits'],y['visits'])),
   max_loss_abs=max(abs(x['loss']-y['loss']) for x,y in zip(a,b)),
   max_gradient_norm_abs=max(abs(x['gradient_norm']-y['gradient_norm']) for x,y in zip(a,b)),
   gradient_hashes=[dict(update=x['update'],equal=x['gradient_sha256']==y['gradient_sha256'],baseline=x['gradient_sha256'],hotset=y['gradient_sha256']) for x,y in zip(a,b) if 'gradient_sha256' in x],
   max_validation_selection_abs=max(abs(x['validation']['selection']-y['validation']['selection']) for x,y in zip(ae,be)))
  for field in ('stage_seconds','train_seconds','validation_seconds','save_seconds','train_physical_bytes','validation_physical_bytes'):
   av=runs[an][field];bv=runs[bn][field]
   comparison[field]=dict(baseline=av,hotset=bv,reduction_fraction=1-bv/av)
  left=torch.load(root/an/'last.pt',map_location='cpu',mmap=True,weights_only=False)
  right=torch.load(root/bn/'last.pt',map_location='cpu',mmap=True,weights_only=False)
  comparison['final_numeric']={k:numeric_difference(left[k],right[k]) for k in ('model','ema','optimizer','schedule_rng','noise_rng','torch_rng','cuda_rng')}
  for field in ('exp_avg','exp_avg_sq','step'):
   comparison['final_numeric']['optimizer_'+field]=numeric_difference(
    {k:v[field] for k,v in left['optimizer']['state'].items()},
    {k:v[field] for k,v in right['optimizer']['state'].items()})
  comparison['same_reconstructed_labels_masks']=supervision['runs'][an]==supervision['runs'][bn]
  comparison['final_metadata_equal']=all(left[k]==right[k] for k in ('epoch','updates','configuration','contract','ema_optimization_step'))
  del left,right
  if key=='reverse':
   left=torch.load(root/an/'first_gradients.pt',map_location='cpu',mmap=True,weights_only=False)
   right=torch.load(root/bn/'first_gradients.pt',map_location='cpu',mmap=True,weights_only=False)
   comparison['first_gradient_numeric']=numeric_difference(left,right);del left,right
  comparison['interpretation']='Exact input/order/exposure and RNG checks are separate from floating point residuals. No bitwise equivalence claimed; numerical residuals are descriptive, no post-hoc tolerance promoted to preregistered acceptance.'
  comparisons[key]=comparison
 report=dict(runs=runs,comparisons=comparisons,historical=read(root/'historical.json'),
  limitation='One A224 main comparison plus preregistered56 reverse-order check. Serial physical machine with variable storage service time. Cold means advisory eviction and measured reads; no global flush. Not a117min extrapolation or autonomous result.')
 dump(output/'summary.json',report);dump(root/'summary.json',report)
 interrupted=root/'baseline';dest=output/'interrupted';dest.mkdir(exist_ok=True)
 for file in ('updates.jsonl','epochs.json','telemetry.json','interruption.log'):
  payload=(interrupted/file).read_bytes();(dest/(file+'.gz')).write_bytes(gzip.compress(payload,mtime=0))
 for file in ('historical.json','order_check_registration.json','disk_pressure_probe.json','disk_probe.log'):
  if file.endswith('.log'):(output/(file+'.gz')).write_bytes(gzip.compress((root/file).read_bytes(),mtime=0))
  else:shutil.copyfile(root/file,output/file)
 print(json.dumps({k:{f:v for f,v in c.items() if f!='final_numeric'} for k,c in comparisons.items()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['baseline','hotset'],required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--wait-idle',action='store_true');p.add_argument('--save-gradient',action='store_true')
 main(p.parse_args())
