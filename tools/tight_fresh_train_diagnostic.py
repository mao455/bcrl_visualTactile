"""Register full autonomous diagnostics on original training reset families.

These are separate from development performance and never count as new labels.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import h5py
from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def prepare(root,output,count):
    if output.exists():raise FileExistsError('Preserve previous diagnostic identity')
    if not 1<=count<=4:raise ValueError('Bounded original-training diagnostic required')
    session=read(root/'session.json');data=read(root/'A/state_data/data.json')
    if time.time()>=session['deadline']:raise TimeoutError('Original deadline expired')
    pairs={role:completed_pair(root,role) for role in ('A','B')}
    originals=data['normalization_train_ids'];cases=[];sources={}
    if len(originals)!=30 or len(set(originals))!=30:
        raise ValueError('Expected the unchanged original30 training identities')
    heldout=set(data['splits']['validation']+data['splits'].get('test',[]))
    heldout_families={data['entries'][i]['family'] for i in heldout}
    selected_families=[data['entries'][i]['family'] for i in originals[:count]]
    if len(set(selected_families))!=count or set(selected_families)&heldout_families:
        raise ValueError('Training diagnostic families must be unique and independent of heldout families')
    for index,identity in enumerate(originals[:count]):
        entry=data['entries'][identity]
        if (identity not in data['splits']['train'] or identity in heldout
                or entry.get('trajectory_kind')=='expert_takeover_v1'):
            raise ValueError('Expected an original training demonstration')
        digest=sha256(entry['path'])
        if digest!=entry['source_sha256']:raise ValueError('Original successful demonstration changed')
        with h5py.File(entry['path']) as stream:recorded=json.loads(stream.attrs['result_json'])
        if not recorded['success'] or not recorded['legal']:raise ValueError('Original source is not a legal success')
        sources[identity]=dict(source_sha256=digest,source_family=entry['family'],seed=recorded['seed'],
            event_seed=recorded['event_seed'],narrow_reset=recorded['narrow_reset'])
        spec=dict(identity='train_diagnostic_'+identity,family=entry['family'],split='train',cohort='fresh',
            diagnostic_cohort='original_training_diagnostic',
            source_identity=identity,seed=recorded['seed'],event_seed=recorded['event_seed'],
            narrow_reset=recorded['narrow_reset'],collision_gate='live_collision_mesh_bounds_v1')
        for role in (('A','B') if index%2==0 else ('B','A')):
            cases.append(dict(name=role+'_'+spec['identity'],role=role,spec=spec,video=index==0))
    output.mkdir(parents=True)
    atomic_json(output/'session.json',session)
    atomic_json(output/'plan.json',dict(kind='autonomous',pairs=pairs,cases=cases,
        resource_root=str(root),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ('tools','visuotactile') for p in sorted(Path(folder).rglob('*.py'))},
        inference_scheduler_index_mode='host',test_used=False,original_sources=sources,
        cohort='First original training families only; complete autonomous student with no expert. Separate training diagnostic, not development performance or exact simulator replay. Same recorded seed/reset does not restore hidden PhysX state.',utc=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=2)
    a=parser.parse_args();prepare(a.root.resolve(),a.output.resolve(),a.count)
