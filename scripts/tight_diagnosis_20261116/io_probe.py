"""Finite train-only I/O and identical disposable B2-update comparison; no checkpoint saved."""
import copy,json,time,sys,resource,os,gc,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from tools import tight_fresh_train as trainer
from visuotactile.native_data import Episode,visual_records
from visuotactile.frozen_visual_io import profile,mapped_profile
from visuotactile.fresh_visit_sampling import load_profile,visit
from visuotactile.trajectory_training import execute_update
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.episode_io import io_snapshot,io_delta,available_memory
from visuotactile.precision import state_fp32
R=Path('outputs/visuotactile/run/tight_fresh_20261116_04/round_19').resolve();parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cold',action='store_true');args=parser.parse_args();O=args.output
assert not O.exists();torch.set_num_threads(2);state_fp32();began=time.time()
def check():
 assert time.time()-began<600,'bounded probe deadline'
 assert available_memory()>10*2**30,'RAM reserve'
 if torch.cuda.is_initialized():assert torch.cuda.mem_get_info()[0]>4*2**30,'VRAM reserve'
def emit(rows,**kw):O.write_text(json.dumps(dict(rows=rows,**kw),indent=2)+'\n')
# Exact formerly-stalled train records, not a small convenient synthetic fixture.
ids=['r08_B_expanded_session06_new16_s28400010_contact0','r14_A_expanded_session20261115_04_balanced14_s29000105_held0']
dr=R/'B/state_data';data=json.loads((dr/'data.json').read_text());assert set(ids)<=set(data['splits']['train'])
records=visual_records(dr,ids,'G00',device='cuda:0');stats=torch.load(dr/'statistics.pt',weights_only=False);saved=torch.load(R/'B/training/best.pt',weights_only=False,map_location='cpu',mmap=True)
vp=load_profile(R);objective,_,_=trainer.fresh_objective(R,'B',data,records)
rows=[];reference=None
# ABBA order. Optional per-file DONTNEED only on these closed train shards; no global cache drop.
for repetition,mode in enumerate(['eager','mapped','mapped','eager']):
 check();model=trainer.new_model('B',stats);model.load_state_dict(saved['ema']);model.train();average=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(average,power=.75)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
 torch.manual_seed(90231);torch.cuda.manual_seed_all(90231);noise=torch.Generator(device='cuda').manual_seed(90231)
 selections=[visit(records[i],vp,identity=i,epoch=1,visit_index=j+1) for j,i in enumerate(ids)]
 opts=dict(selected_anchors=[v[0] for v in selections],anchor_receipts=[v[1] for v in selections],geometry_generators=[v[2] for v in selections],geometry_sampling_profile=vp['geometry'])
 if args.cold:
  for identity in ids:
   with (dr/'perception'/(identity+'.pt')).open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 before=io_snapshot();start=time.perf_counter();episodes=[Episode(dr,i,'I-HistoryDP',frozen_visual_io=profile() if mode=='eager' else mapped_profile()) for i in ids];loaded=time.perf_counter()-start
 torch.cuda.synchronize();t=time.perf_counter();result=execute_update(model,opt,ema,objective,ids,[records[i] for i in ids],episodes,noise,check=check,**opts);torch.cuda.synchronize();compute=time.perf_counter()-t
 tensors={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng=torch.cuda.get_rng_state().clone()
 if reference is None:reference=(tensors,rng,result['loss']);difference=0.
 else:
  difference=max(float((tensors[k]-v).abs().max()) for k,v in reference[0].items())
 rows.append(dict(mode=mode,repetition=repetition,identities=ids,file_bytes=sum((dr/'perception'/(i+'.pt')).stat().st_size for i in ids),load_s=loaded,update_s=compute,total_s=loaded+compute,loss=result['loss'],parameter_max_abs_difference=difference,cuda_rng_equal=torch.equal(reference[1],rng),loss_abs_difference=abs(reference[2]-result['loss']),process_io=io_delta(before,io_snapshot()),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
 emit(rows,complete=False);print(json.dumps(rows[-1]),flush=True)
 del episodes,tensors,model,average,ema,opt;gc.collect();torch.cuda.empty_cache()
emit(rows,complete=True,seconds=time.time()-began,formal_updates=0,disposable_updates=4,source=str(R),cold_file_advice=args.cold,scope='B2 real complete histories/forward/backward/Adam/EMA; equal initial checkpoint/RNG/anchors. ABBA warm uncontrolled OS residency, no full-cycle claim; no simulation/test/checkpoint write.')
