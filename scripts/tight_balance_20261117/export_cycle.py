"""Export the closed current cycle only, with literal configs and measured timing."""
import gzip,json,shutil
from pathlib import Path
from scripts.tight_balance_20261117.offline_phases import ROOT,OUT
from scripts.tight_layout_20261116.publication import write,table,sha
from scripts.tight_layout_20261116.learning_report import summarize

def read(p):return json.loads(p.read_text())
def main():
 assert read(OUT/'completed.json')['complete']
 inventory=[];curves=[];performance=[]
 def copy(p,rel):
  q=OUT/'configuration'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  inventory.append(dict(source=str(p),published=str(q),sha256=sha(q),bytes=q.stat().st_size))
 for name in ['registration.json','session.json','prepared.json','training_sampling.json','full_trajectory_batch.json','bounded_extension.json','frozen_visual_hotset.json','frozen_visual_io.json','complete.json','owner_exit.json','recovery.json']:
  copy(ROOT/name,Path(name))
 for role in 'AB':
  for rel in ['continuation.json','state_data/data.json','cache_index.json','training/configuration.json','training/finished.json','backend/configuration.json','backend/base_identities.json','backend/'+role+'/nominal/index.json','backend/'+role+'/training/configuration.json','backend/'+role+'/training/finished.json','backend/'+role+'/training/selected.json']:
   p=ROOT/role/rel
   if p.exists():copy(p,Path(role)/rel)
  for component,folder in [('DP',ROOT/role/'training'),('Transformer',ROOT/role/'backend'/role/'training')]:
   done=read(folder/'finished.json')
   if component=='DP':
    rows=[json.loads(l) for l in (folder/'validation.jsonl').open()]
    train=sum(r['training_epoch_io']['wall_seconds'] for r in rows)
    validation=sum(r['validation']['offline_planning']['seconds'] for r in rows)
    diagnostic=sum(r['train']['offline_planning']['seconds'] for r in rows)
    wall=done['wall_seconds'];cache=done['frozen_visual_hotset']
    performance.append(dict(role=role,component=component,updates=done['updates'],stage_seconds=wall,training_epoch_seconds=train,validation_seconds=validation,train_diagnostic_seconds=diagnostic,remaining_stage_seconds=wall-train-validation-diagnostic,updates_per_stage_second=done['updates']/wall,updates_per_training_epoch_second=done['updates']/train,stage_read_bytes=done['stage_process_io']['read_bytes'],training_epoch_read_bytes=sum(r['training_epoch_io']['process_io']['read_bytes'] for r in rows),hotset_resident_bytes=cache['resident_bytes'],hotset_fill_seconds=cache['first_fill_seconds'],hotset_hits=cache['hits'],hotset_misses=cache['misses'],hotset_admission_fallbacks=cache['admission_fallbacks'],avoided_logical_payload_bytes=cache['avoided_logical_payload_bytes'],peak_cuda_allocated_bytes=done['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=done['peak_cuda_reserved_bytes']))
    for r in rows:curves.append(dict(role=role,component=component,epoch=r['epoch'],updates=r['updates'],selection=r['validation']['selection'],loss=r['loss'],training_epoch_seconds=r['training_epoch_io']['wall_seconds'],training_epoch_read_bytes=r['training_epoch_io']['process_io']['read_bytes']))
    logs=['learning.jsonl','validation.jsonl']
   else:
    wall=done['wall_s'];performance.append(dict(role=role,component=component,updates=done['updates'],stage_seconds=wall,training_epoch_seconds=done['training_wall_s'],setup_seconds=done['setup_wall_s'],updates_per_stage_second=done['updates']/wall,supervised_ticks=done['supervised_ticks']))
    logs=['loss.jsonl','validation.json']
    if (folder/'loss.uncommitted.jsonl').exists():logs.append('loss.uncommitted.jsonl')
    if (folder/'restore.json').exists():copy(folder/'restore.json',Path(role)/'backend_restore.json')
   for name in logs:
    p=folder/name;q=OUT/'curves'/f'{role}_{component}_{name}.gz';q.parent.mkdir(exist_ok=True)
    kind='complete compressed original log'
    with gzip.GzipFile(filename=str(q),mode='wb',mtime=0) as dst:
     if component=='DP' and name=='learning.jsonl':
      # Keep every update, action start and sampling receipt. Per-point geometry
      # arrays remain in the unchanged local source, not in the public export.
      kind='all update rows; visits.geometry omitted; sampling receipts retained'
      with p.open() as src:
       for line in src:
        row=json.loads(line)
        for visit in row['visits']:visit.pop('geometry',None)
        dst.write((json.dumps(row,separators=(',',':'))+'\n').encode())
     else:
      with p.open('rb') as src:shutil.copyfileobj(src,dst)
    inventory.append(dict(source=str(p),published=str(q),source_sha256=sha(p),sha256=sha(q),bytes=q.stat().st_size,kind=kind))
 for part in range(4):
  folder=ROOT/f'comparison{part:02d}'/'development'
  for name in ['plan.json','result.json','paired_initial_check.json','wave_profiles.json','eval_startup.json','kit_started.json']:
   if (folder/name).exists():copy(folder/name,Path(f'comparison{part:02d}')/name)
 table(OUT/'performance.csv',performance);table(OUT/'validation_curve.csv',curves)
 write(OUT/'export_inventory.json',inventory)
 summarize(ROOT,OUT/'learning')
 print(json.dumps(performance,indent=2))

if __name__=='__main__':main()
