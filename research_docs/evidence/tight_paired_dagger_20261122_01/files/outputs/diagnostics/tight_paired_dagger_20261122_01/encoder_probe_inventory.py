"""CPU-only metadata/phase inventory; no model forwards, fitting, GPU or Kit."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import json,time,hashlib
from pathlib import Path
import h5py,numpy as np,torch
from scripts.tight_balance_20261117.diagnose import phases,STAGES
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS')
RUN=BASE/'outputs/visuotactile/run'
AUDIT=BASE/'outputs/diagnostics/tight_paired_dagger_20261122_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    start=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    old=[]
    for p in sorted((RUN/'tight_representation_audit_20261119_01').glob('features_*.json')):
        d=json.loads(p.read_text());fam=sorted({x['family'] for x in d['episodes']})
        old.append(dict(path=str(p),sha256=sha(p),families=fam,dimensions=d['dimensions'],episodes=len(d['episodes']),samples=sum(x['count'] for x in d['episodes']),eligible_untouched_readout_families=0))
    dev=RUN/'tight_entry_phase_20261121_01/development_20261122_01';numeric=[]
    for p in sorted(dev.glob('wave*/*/numeric.h5')):
        with h5py.File(p) as f:
            datasets=[];f.visititems(lambda n,v:datasets.append(n) if isinstance(v,h5py.Dataset) else None)
            image=[n for n in datasets if any(w in n.lower() for w in ['rgb','depth','patch','visual','feature'])]
            numeric.append(dict(path=str(p),image_or_feature_datasets=image,steps=int(f['episode/obs/time_s'].shape[0])))
    fork=RUN/'tight_fork_diagnosis_20261117_01_recovery01';short=[]
    for p in sorted((fork/'capture').glob('*/numeric.h5')):
        with h5py.File(p) as f:
            ph=phases(f['episode/transition'])
            short.append(dict(path=str(p),family=p.parent.name.rsplit('_',1)[-1],ticks=len(ph),phase_counts={s:int((ph==j).sum()) for j,s in enumerate(STAGES)}))
    cache=fork/'inputs/000.pt';x=torch.load(cache,map_location='cpu',weights_only=False)
    shapes={'local_dino':list(x['features'].shape),'depth_m':list(x['packet']['depth_m'].shape),'depth_valid':list(x['packet']['depth_valid'].shape),'aggregate_per_controller':[list(c['visual_read'].shape) for c in x['controllers']]}
    del x
    result=dict(status='not_run_no_qualified_near_hole_cache',elapsed_inventory_seconds=time.time()-start,cpu_threads=2,cuda_initialized=torch.cuda.is_initialized(),gpu_or_kit_started=False,policy_updates=0,readout_fits=0,
        restrictions=dict(no_sealed64=True,no_new_training24=True,no_historical_test_observations_or_results=True,no_frozen_source_edits=True),
        historical_readout_cache=old,entry_development=dict(numeric_rollouts=len(numeric),image_or_feature_rollouts=sum(bool(x['image_or_feature_datasets']) for x in numeric),rollouts=numeric),
        short_development_cache=dict(path=str(fork),independent_families=sorted({x['family'] for x in short}),cached_ticks=121,lanes=8,example_path=str(cache),example_sha256=sha(cache),shapes=shapes,rollouts=short,held_proxy_ticks=sum(sum(x['phase_counts'][s] for s in STAGES[2:5]) for x in short),near_hole_ticks=sum(x['phase_counts']['held_near_goal60'] for x in short)),
        decision='No fit: historical 36 families already participated in readouts; independent96-rollout development has no RGB-D/feature payload; 2-family121-tick cached prefix has both features but zero near-hole frames and only33 held-proxy ticks. It cannot evaluate the requested independent-family stable-hold/near-hole comparison.',
        label_limitations='Historical32-value readout labels are object-root/palm/hole relative poses, not calibrated peg-tip lateral offsets or peg-axis/hole-axis angular error. They must not be relabeled as requested insertion geometry.',
        future_requirement='Record existing aggregate512, local DINO patches and depth+valid with synchronized actual sensor inputs in separate predeclared development families under native60Hz, covering stable hold and near hole; bind calibrated peg-tip/hole-axis labels offline only. No four-model retraining is authorized or started.',
        inventory_source_sha256=sha(Path(__file__)))
    (AUDIT/'encoder_probe_feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','elapsed_inventory_seconds','cpu_threads','cuda_initialized','readout_fits']}));print('held',result['short_development_cache']['held_proxy_ticks'],'near',result['short_development_cache']['near_hole_ticks'])
if __name__=='__main__':main()
