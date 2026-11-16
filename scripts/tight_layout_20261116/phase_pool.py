"""Freeze a union of reviewed closed collection waves, retaining every source path."""
import argparse,copy,json,time
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from scripts.tight_layout_20261116.dagger import prepare_learning,source_hashes

def prepare(source,output,session_root,collections):
 read=lambda p:json.loads(p.read_text())
 assert not output.exists() and read(source/'complete.json')['complete']
 cases=[];results=[];entries={};pairs=None;bindings=[];pair_rows=[];wall=0
 for folder in collections:
  assert read(folder.parent/'collection_owner_exit.json')['children_reaped']
  complete=read(folder/'result.json');assert complete['software_complete'] and complete['worker_error'] is None
  plan=read(folder/'plan.json');paired=read(folder/'paired_initial_check.json');assert paired['passed']
  if pairs is None:pairs=plan['pairs'];template=copy.deepcopy(plan)
  assert pairs==plan['pairs'],'Union must use the same frozen full students'
  added=read(folder/'correction_entries.json');assert not set(added)&set(entries);entries.update(added)
  for case in plan['cases']:
   assert case['name'] not in {c['name'] for c in cases}
   cases.append(case)
  results.extend(read(folder/'results.json'));pair_rows.extend(paired['rows']);wall+=complete['elapsed_seconds']
  bindings.append(dict(root=str(folder),plan_sha256=sha256(folder/'plan.json'),entries_sha256=sha256(folder/'correction_entries.json'),result_sha256=sha256(folder/'result.json')))
 assert len(entries)>=2,'Insufficient useful new suffixes; review before training'
 output.mkdir();collection=output/'collection';collection.mkdir()
 for folder in collections:
  for case in read(folder/'plan.json')['cases']:(collection/case['name']).symlink_to(folder/case['name'],target_is_directory=True)
 template.update(cases=cases,source_sha256=source_hashes(),closed_collection_union=bindings,
  scope='Manifest-only union, no new rollouts and no deleted failures; aggregate all admitted A/B suffixes once')
 atomic_json(collection/'plan.json',template);atomic_json(collection/'correction_entries.json',entries)
 atomic_json(collection/'results.json',results)
 atomic_json(collection/'result.json',dict(software_complete=True,worker_error=None,cases=len(cases),elapsed_seconds=wall,manifest_union_only=True))
 atomic_json(collection/'paired_initial_check.json',dict(passed=True,rows=pair_rows,source_collections=bindings))
 session=read(session_root/'session.json');session.update(resource_root=str(output),owned_worker_roots=[str(output)])
 atomic_json(output/'session.json',session)
 atomic_json(output/'prepared.json',dict(source=str(source),source_sha256=source_hashes(),union=bindings,utc=time.time()))
 atomic_json(output/'collection_owner_exit.json',dict(children_reaped=True,manifest_union_only=True,source_collections=bindings))
 target=prepare_learning(output);print(target)
 return target
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--session-root',type=Path,required=True);p.add_argument('--collections',type=Path,nargs='+',required=True);a=p.parse_args()
 prepare(a.source.resolve(),a.output.resolve(),a.session_root.resolve(),[p.resolve() for p in a.collections])
