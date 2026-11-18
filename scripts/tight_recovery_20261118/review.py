"""Reuse artifact analyzers with explicit new identities and output destinations."""
import sys,copy
from scripts.tight_recovery_20261118.run import ROOT,DATA,OUT,PARENT,COLLECTION,read,write

def collect():
 from scripts.tight_layout_20261116.phase_report import summarize
 import shutil
 d=ROOT/COLLECTION;out=OUT/'collection';out.mkdir(exist_ok=True)
 summarize(d,out/'summary.json')
 for name in ['plan.json','result.json','correction_entries.json','paired_initial_check.json','initialization.json','wave_profiles.json','owner_exit.json','owner_complete.json']:
  shutil.copyfile(d/name,out/name)


def learning():
 from scripts.tight_cycle_20261118 import review as r
 r.ROOT=ROOT;r.DATA=DATA;r.OUT=OUT;r.learning()
 for p in (OUT/'learning').rglob('*'):
  if p.suffix in ('.json','.csv'):
   p.write_text(p.read_text().replace('old_pool_unseen4','prior105_unseen33'))
 import shutil
 for name in ['training_sampling.json','backend_episode_sampling.json','frozen_visual_hotset.json','frozen_visual_io.json','bounded_extension.json','full_trajectory_batch.json']:
  shutil.copyfile(ROOT/'learning'/name,OUT/'learning'/name)


def evaluation():
 from scripts.tight_cycle_20261118 import review as r,paired as p
 for m in [r,p]:m.ROOT=ROOT;m.OUT=OUT
 r.evaluation(parts=4);p.main(parts=4)

def offline():
 from scripts.tight_cycle_20261118 import offline as o
 o.ROOT=ROOT;o.OUT=OUT;o.run()
 # Existing implementation's labels are historical counts, corrected explicitly.
 import json
 data=read(ROOT/'learning/A/state_data/data.json');n=len(data['splits']['train']);new=n-72
 for p in (OUT/'offline').iterdir():
  if p.suffix not in ('.json','.csv'):continue
  text=p.read_text().replace('candidate105',f'candidate{n}').replace('new_to_parent33',f'new_to_parent{new}').replace('legal105',f'legal{n}');name=p.name.replace('candidate105',f'candidate{n}');(p.parent/name).write_text(text)
  if name!=p.name:p.unlink()
 v=read(OUT/'offline/identity.json');v.update(candidate_trajectories=n,new_to_parent=new);write(OUT/'offline/identity.json',v)

def evaluation_prepare(part):
 from scripts.tight_cycle_20261118 import evaluate as e,cycle
 e.ROOT=ROOT;e.OUT=OUT
 # Existing opened development protocol and training diagnostic16 families.
 if not (ROOT/'capability').exists():
  (ROOT/'capability').mkdir();write(ROOT/'capability/plan.json',read(cycle.ROOT/'capability/plan.json'))
  reg=read(cycle.ROOT/'registration.json');write(ROOT/'registration.json',dict(training_seeds=reg['training_seeds'],new_development_seeds=reg['new_development_seeds']))
 e.prepare(part,parts=4)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
