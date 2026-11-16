"""Summarize closed real recovery suffixes without reading large RGB-D payloads."""
import argparse,json,time
from pathlib import Path
import h5py
import numpy as np
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def summarize(root,output):
 read=lambda p:json.loads(p.read_text())
 collection=root/'collection';plan=read(collection/'plan.json')
 assert read(collection/'result.json')['software_complete']
 entries=read(collection/'correction_entries.json');rows=[]
 for case in plan['cases']:
  folder=collection/case['name'];result=read(folder/'result.json')
  row=dict(case=case['name'],role=case['role'],seed=case['spec']['seed'],legal=result['legal'],
   success=result['success'],steps=result.get('steps'),reasons=result.get('reasons'),accepted=case['name'] in entries)
  if result['legal']:
   with h5py.File(folder/'numeric.h5') as f:
    t=f['episode/transition'];d=t['dagger'];mask=d['expert_label_valid'][:].reshape(-1).astype(bool)
    source=d['control_source'][:].reshape(-1);student=source==0
    assert not mask[student].any(),'Student prefix must never supervise'
    assert np.array_equal(d['q_reached'][:][mask],t['diagnostics_after/joint_position_rad'][:][mask])
    assert np.array_equal(d['u_expert'][:][mask],t['target_after_rad'][:][mask])
    row.update(student_ticks=int(student.sum()),valid_expert_ticks=int(mask.sum()),
     masked_expert_ticks=int(((~student)&~mask).sum()),student_prefix_supervised=False,
     reached_label_matches_real_posttransition=True,command_matches_real_executed_target=True)
   receipt=result['dagger']['trigger_receipt'];row['trigger']={k:receipt.get(k) for k in
    ['reason','actual_scheduled_takeover_tick','first_contact_height_window_end_completed_tick','trigger_completed_tick']}
  rows.append(row)
 data={r:read(root/'learning'/r/'state_data/data.json') for r in 'AB'}
 assert data['A']['splits']==data['B']['splits']
 report=dict(root=str(root),utc=time.time(),rows=rows,registered=len(rows),legal=sum(r['legal'] for r in rows),
  successful_recoveries=sum(r['success'] for r in rows),accepted=len(entries),
  valid_expert_ticks=sum(r.get('valid_expert_ticks',0) for r in rows),
  shared_split_counts={k:len(v) for k,v in data['A']['splits'].items()},
  normalization_count=len(data['A']['normalization_train_ids']),
  source_hashes={name:sha256(collection/name) for name in ['plan.json','result.json','correction_entries.json']},
  paired_initial_check=read(collection/'paired_initial_check.json'),
  collection_owner_exit=read(root/'collection_owner_exit.json'),
  interpretation='Expert recovery is not autonomous insertion. All successful A/B suffixes share one pool; prefixes and queue flush are history only.')
 output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,report)
 print(json.dumps({k:report[k] for k in ['registered','legal','successful_recoveries','accepted','valid_expert_ticks','shared_split_counts']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();summarize(a.root.resolve(),a.output.resolve())
