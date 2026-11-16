"""Small metadata-only ancestry/identity check; no held-out trajectory reads."""
import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.tight_budget_20261116.run import ROOT,read
from visuotactile.training_expansion import configuration_digest
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
v=read(ROOT/'A/state_data/data.json');bindings=v['training_expansion_registries'];registries=[];families={}
for binding in bindings:
 assert sha256(binding['path'])==binding['sha256']
 registry=read(binding['path']);registries.append(registry)
 for name,row in registry['families'].items():
  if name in families:assert families[name]==row
  families[name]=row
original=read(registries[0]['original_frozen_families']['path'])
for row in original:families[row['family']]=row
train={v['entries'][i]['family'] for i in v['splits']['train']};val={v['entries'][i]['family'] for i in v['splits']['validation']}
assert not train&val
th={configuration_digest(families[i]['narrow_reset']) for i in train};vh={configuration_digest(families[i]['narrow_reset']) for i in val}
assert not th&vh
assert not {families[i]['seed'] for i in train}&{families[i]['seed'] for i in val}
assert all(families[i]['split']=='train' for i in train) and all(families[i]['split']=='validation' for i in val)
atomic_json(ROOT/'split_identity_evidence.json',dict(complete=True,registries=len(bindings),train_families=len(train),validation_families=sorted(val),exact_reset_configuration_intersection=[],seed_intersection=[],
 current_entries_grouped_by_original_or_registered_revisited_family=True,held_out_raw_trajectories_opened=0,
 limitations='All are deliberately in the same millimetre-scale task domain. Family/seed/reset identity separation does not mean large geometric separation or wide physical generalization.'))
print(ROOT/'split_identity_evidence.json')
