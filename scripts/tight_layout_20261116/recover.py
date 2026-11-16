"""Preserve two admitted captures and recover only the unstarted remainder."""
import copy,os,subprocess,time
from pathlib import Path
from scripts.tight_layout_20261116.collect import ROOT,TREE,read
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256

def main():
 old=TREE/'outputs/visuotactile/run/tight_layout10_20261116_01'
 assert ROOT.resolve()!=old.resolve() and not ROOT.exists()
 assert read(old/'collection_owner_exit.json')['children_reaped']
 gate=read(old/'collection_gate_exit.json');assert gate['exit_code']==0 and gate['monitor_alive']
 rows=read(old/'collection_results.json');assert len(rows)==2 and all(r['success'] and r['legal'] for r in rows)
 assert len(list((old/'episodes').iterdir()))==2
 assert 'FileExistsError' in read(old/'collection_failure.json')['error']
 ROOT.mkdir();s=read(old/'session.json');s.update(resource_root=str(ROOT.resolve()),owned_worker_roots=[str(ROOT.resolve())],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip())
 atomic_json(ROOT/'session.json',s)
 plan=read(old/'collection_plan.json');plan['source_sha256']={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')}
 atomic_json(ROOT/'collection_plan.json',plan);atomic_json(ROOT/'collection_results.json',rows)
 atomic_json(ROOT/'collection_gate_exit.json',dict(gate,source_receipt=str((old/'collection_gate_exit.json').resolve())))
 atomic_json(ROOT/'recovery.json',dict(source=str(old.resolve()),source_results_sha256=sha256(old/'collection_results.json'),carried_successes=2,physical_attempts_repeated=0,reason='Second process reused exclusive startup diagnostics path; failed before Kit startup. Give each worker a fresh diagnostics directory.',original_session_sha256=sha256(old/'session.json'),all_deadlines_unchanged=True,utc=time.time()))
if __name__=='__main__':main()
