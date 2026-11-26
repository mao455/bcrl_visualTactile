"""Offline audit of published correction evidence, without Kit or private weights."""
from pathlib import Path
import csv,json,hashlib,argparse
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def yes(value):return value is True or str(value).lower() in ('1','true')
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[2];root=a.bundle or repo/'research_docs/evidence/tight_correction_20261127_01'
 manifest=json.loads((root/'MANIFEST.json').read_text());files=manifest['files']
 for name,row in files.items():
  path=root/name
  assert path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],name
 rows=list(csv.DictReader((root/'audit/all_completed_development_cases.csv').open()));assert len(rows)==64 and all(yes(r['complete']) for r in rows)
 summary=json.loads((root/'audit/primary_results.json').read_text())
 for role,item in summary['policies'].items():
  rr=[r for r in rows if r['role']==role];assert len(rr)==8 and {int(r['seed']) for r in rr}==set(range(28911000,28911008));assert sum(yes(r['success']) for r in rr)==item['strict_success'];assert item['denominator']==item['complete']==8
 assert summary['original64_reused'] is False
 captures=json.loads((root/'audit/current_capture_summary.json').read_text())
 assert captures['A']['legal_ticks']==475 and captures['B']['legal_ticks']==256
 training=json.loads((root/'audit/short_optimization_audit.json').read_text())
 for condition in ('old_only','with_corrections'):
  for role in 'AB':
   item=training[condition][role]
   assert item['frontend_updates']==192 and item['backend_updates']==668
 assert json.loads((root/'audit/targeted_checks.json').read_text())['passed']==98
 for row in rows:
  raw=Path(row['source']).with_name('numeric.h5')
  matches=[p for p in (root/'runs').rglob('core_numeric.npz') if '/'+str(p.parent.relative_to(root/'runs'))+'/' in '/'+str(raw.parent)+'/' ]
  assert len(matches)==1,(raw,matches)
  with np.load(matches[0],allow_pickle=False) as arrays:
   key=arrays['episode/transition/native_keypoints_max_dist']
   before=arrays['episode/transition/diagnostics_before/successes']
   after=arrays['episode/transition/diagnostics_after/successes']
   assert yes(row['ever15mm'])==bool(np.any((before==0)&(key<=np.float32(.015))))
   assert yes(row['native_first_subgoal'])==bool(np.any(after>0))

 retired=list(csv.DictReader((root/'audit/retired_paths.csv').open()));assert len(retired)==30
 assert all(not (repo/r['path']).exists() for r in retired)
 admissions=list((root/'runs').rglob('actual_layout_admission.json'));assert sum(len(json.loads(p.read_text())['rows']) for p in admissions)>=64
 for path in admissions:
  receipt=json.loads(path.read_text())
  for row in receipt['rows']:
   e=row['expected'];t=e['tolerance_m'];actual=row['object_xy_m'];hole=row['hole_xy_m']
   assert max(abs(actual[j]-e['object_xy_m'][j]) for j in (0,1))<=t
   assert max(abs(hole[j]-e['hole_xy_m'][j]) for j in (0,1))<=t
   assert max(abs(actual[j]-e['unshifted_object_xy_m'][j]-e['displacement_xy_m'][j]) for j in (0,1))<=t
 assert (repo/'research_docs/TIGHT_CORRECTION_20261127.md').is_file()
 print(f'PASS: {len(files)} payload hashes;64 completed development cases /8 families;actual-layout receipts;30 retired paths absent. Private model/physics replay is not performed.')
if __name__=='__main__':main()
