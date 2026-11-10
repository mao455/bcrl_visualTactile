"""Formal test registration must retain every fixed case and model identity."""
import copy
import hashlib
import json
from pathlib import Path

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def plan_function():
    from tools.tight_finish import formal_test_plan
    return formal_test_plan


def fixture(tmp_path):
    source = tmp_path/'source'; output = tmp_path/'formal/evaluation_plan.json'
    ids = [f'case_{i:02d}' for i in range(22)]; entries = {}
    reset = dict(fixed_start_pose=[0., 0., .63, 1., 0., 0., 0.], table_reset_z=.38,
        table_reset_z_range=0., reset_dof_pos_random_interval_arm=.0075,
        reset_dof_pos_random_interval_fingers=.0075, reset_dof_vel_random_interval=.03,
        hole_x_range=[.04, .04], hole_y_range=[-.03, -.03])
    for i, identity in enumerate(ids):
        row = source/'captures'/identity
        entries[identity] = dict(path=str(row/'demonstration/data.h5'), family=f'family_{i}')
        write(row/'started.json', dict(spec=dict(identity=identity, family=f'family_{i}', split='test', cohort='fresh',
            seed=100+i, event_seed=1000+i, collision_gate='live_collision_mesh_bounds_v1', narrow_reset=reset,
            speed_scheme=dict(arm=.5, hand=1.), alpha=None)))
        write(row/'initial_observable.json', dict(object_pose_w=[reset['fixed_start_pose']]))
        write(row/'initial_collision.json', dict(legal=True))
        # Neither trajectories nor result labels are present: planning must not need them.
    write(source/'state_data/data.json', dict(splits=dict(train=['train'], validation=['validation'], test=ids), entries=entries))
    cp=tmp_path/'selected.pt'; cp.write_bytes(b'selected model, not loaded during planning')
    rules=tmp_path/'evaluation_rules.json';write(rules, dict(attempts=22, replacements=False))
    write(output.parent/'model_freeze.json',dict(frozen=True,
        models={'G00':dict(checkpoint=str(cp),sha256=sha(cp)), 'G01':dict(checkpoint=str(cp),sha256=sha(cp))},
        evaluation_rules=str(rules),evaluation_rules_sha256=sha(rules), source_data_sha256=sha(source/'state_data/data.json')))
    return source,output,ids,cp


def test_all_fixed_cases_same_original_seeds_and_no_collection_speed(plan_function,tmp_path):
    source,out,ids,_=fixture(tmp_path)
    rows=plan_function(source,out)
    assert [x['identity'] for x in rows] == ids
    assert [x['spec']['seed'] for x in rows] == list(range(100,122))
    assert [x['spec']['event_seed'] for x in rows] == list(range(1000,1022))
    assert sum(x['record_video'] for x in rows) == 2
    assert all(x['spec']['cohort']=='formal_test' and x['spec']['split']=='test' for x in rows)
    assert all('speed_scheme' not in x['spec'] and 'alpha' not in x['spec'] for x in rows)
    receipt=json.loads(out.with_name('evaluation_plan_provenance.json').read_text())
    assert receipt['attempts_per_group']==22 and receipt['test_future_label_reads']==0
    assert receipt['groups']==['G00','G01'] and not receipt['replacement_seeds_allowed']
    with pytest.raises(ValueError,match='already exists'):plan_function(source,out)


@pytest.mark.parametrize('failure',['missing_freeze','changed_checkpoint','changed_rules','wrong_spec','missing_case'])
def test_registration_refuses_incomplete_or_drifting_inputs_without_partial_plan(plan_function,tmp_path,failure):
    source,out,ids,cp=fixture(tmp_path)
    if failure=='missing_freeze':(out.parent/'model_freeze.json').unlink()
    elif failure=='changed_checkpoint':cp.write_bytes(b'changed')
    elif failure=='changed_rules':(tmp_path/'evaluation_rules.json').write_text('{}')
    elif failure=='wrong_spec':
        p=source/'captures'/ids[0]/'started.json';v=json.loads(p.read_text());v['spec']['identity']='wrong';write(p,v)
    else:(source/'captures'/ids[-1]/'started.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):plan_function(source,out)
    assert not out.exists()
