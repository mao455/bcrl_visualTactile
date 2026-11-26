"""Register a four-case grasp diagnostic on the first two original train families."""
import argparse
import json
from pathlib import Path
import h5py
from tools.tight_fresh_batch_run import prepare
from visuotactile.training_resources import atomic_json


def register(root,output,backend):
    data=json.loads((root/'A/state_data/data.json').read_text())
    ids=data['normalization_train_ids'][:2]
    assert len(ids)==2 and set(ids)<=set(data['splits']['train'])
    plan=prepare(root,output,lanes=4,count=2,seed=29600000,
        execution_backend=json.loads(backend.read_text()),vision_microbatch=4,queued_gpu=True)
    cases=[]
    for identity in ids:
        entry=data['entries'][identity]
        with h5py.File(entry['path']) as stream:
            result=json.loads(stream.attrs['result_json'])
        assert result['success'] and result['legal']
        for role in 'AB':
            name='original_'+role+'_'+identity
            spec=dict(identity=name,family=entry['family'],split='train',cohort='fresh',source_identity=identity,
                seed=result['seed'],event_seed=result['event_seed'],narrow_reset=result['narrow_reset'],
                collision_gate='live_collision_mesh_bounds_v1')
            cases.append(dict(name=name,role=role,spec=spec,video=False,grasp_evaluation_ticks=300))
    plan.update(cases=cases,cohort='Original train2 families, full DP+Transformer, grasp-stage diagnostic only',
        evaluation_scope=dict(grasp_horizon=300,full_task_per_role=0),
        original_sources={i:dict(path=data['entries'][i]['path'],declared_source_sha256=data['entries'][i]['source_sha256']) for i in ids},
        initial_state_caveat='Registered reset/seed reused; hidden PhysX state is not an exact replay',test_used=False)
    atomic_json(output/'plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('root','output','backend'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();register(a.root.resolve(),a.output.resolve(),a.backend.resolve())
