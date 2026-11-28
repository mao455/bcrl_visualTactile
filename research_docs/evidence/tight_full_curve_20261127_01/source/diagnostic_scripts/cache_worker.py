from pathlib import Path
import argparse,gc,json,sys,time
import torch
from visuotactile.training_resources import atomic_json
read=lambda p:json.loads(p.read_text())
def work(root):
 from visuotactile.native_data import cache_visual_pair
 from visuotactile.frozen_payload_audit import FrozenPayloadAudit
 from visuotactile.precision import state_fp32
 from tools.tight_paired_prepare import exposure_preflight
 state_fp32();torch.set_num_threads(2);audit=FrozenPayloadAudit();reuse=root/'reusable_visual_index.json'
 for condition in ['plus_corrected24','old_only']:
  dest=root/condition
  # The active cache monitor is shared by the sequential cache consumers only.
  if condition=='old_only':
   link=dest/'shared_resources.json';link.unlink(missing_ok=True);link.symlink_to(root/'plus_corrected24/shared_resources.json')
  for role in 'AB':
   cache_visual_pair(dest/role,reuse_index=reuse,payload_audit=audit)
   combined=read(reuse);combined['entries'].update(read(dest/role/'cache_index.json')['entries']);atomic_json(reuse,combined)
   gc.collect();torch.cuda.empty_cache()
  for role in 'AB':
   import shutil
   shutil.copyfile(dest/f'paired_group_deck_{role}_60.json',dest/f'paired_group_deck_{role}.json')
  exposure_preflight(dest)
 atomic_json(root/'visual_cache_complete.json',dict(utc=time.time(),audit=audit.statistics(),old_features_reused=True))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['curve_cache']=('prepare','outputs.diagnostics.tight_full_curve_20261127_01.cache_worker',[]);sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);work(p.parse_args().root)
