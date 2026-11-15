"""Prepare a new, bounded throughput attempt; this never launches or retries."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.tight_fresh_expansion import prepare
from tools.tight_fresh_batch_eval import prepare as prepare_probe
from visuotactile.training_resources import atomic_json
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
SESSION=BASE/'tight_fresh_20261115_04';SOURCE=SESSION/'throughput_source15'
BUNDLE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(__doc__);p.add_argument('--name',required=True);p.add_argument('--lanes',type=int,default=2)
p.add_argument('--probe',choices=['standard','tiled']);p.add_argument('--ticks',type=int,default=36)
p.add_argument('--legacy',action='store_true');a=p.parse_args()
if Path(a.name).name!=a.name:raise ValueError('Local distinct attempt name required')
backend=json.loads((BUNDLE.parent/'tight_fresh_20261115_05/collection_backend.json').read_text())
if a.probe:
 if a.probe=='tiled':backend.update(version='tight_execution_backend_v3',camera_backend='tiled',tiled_fixed_camera_usd_sync=True)
 prepare_probe(SOURCE,SESSION/a.name,a.lanes,29510000,neural_batch=True,max_control_ticks=a.ticks,roles=['A'],
     execution_backend=backend,vision_microbatch=min(8,a.lanes))
else:
 prepare(SOURCE,SESSION/a.name,session_root=SESSION,registry_id=a.name,seeds=[],
  partition_manifest=BASE/'tight_state_20261107_01/composition_06/state_data/data.json',
  frozen_families=BASE/'tight_state_20261107_01/composition_06/frozen_families.json',
  schedules=['grasp_early'],lanes=a.lanes,event_seed=29105000,round_index=15,
  revisit_families=['expanded_session20261115_03_prefetch13_s29000067','expanded_session20261115_03_prefetch13_s29000091'],
  execution_backend=backend,vision_microbatch=2,verification_workers=2,verification_max_pending=4,
  capture_audit_memo=True,queued_gpu=not a.legacy)
