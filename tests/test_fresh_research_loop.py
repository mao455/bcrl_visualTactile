"""CPU-only owner boundaries; child work is replaced, never a simulator launch."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from tools import tight_fresh_research_loop as loop


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def registration(tmp_path,monkeypatch):
    from visuotactile.anchor_sampling import sampling_profile
    from visuotactile.geometry_sampling import profile
    from visuotactile.fresh_visit_sampling import VERSION
    from visuotactile.execution_backend import DEFAULT
    root=tmp_path/'session';tree=tmp_path/'frozen';ledger=root/'research_loop'
    write(root/'session.json',dict(deadline=time.time()+43200,resource_policy='full_gpu_v1',
        control_hz=60,planning_hz=5,visual_stride=1,reserve_bytes=60*1024**3))
    write(root/'gate.json',dict(passed=True));write(root/'shared_resources.json',{})
    for role in ('A','B'):
        write(root/role/'state_data/data.json',dict(splits=dict(train=['initial'])))
        write(root/role/'cache_index.json',dict(complete=True))
        write(root/role/'training/best.pt',{})
        write(root/role/'backend'/role/'training/best.pt',{})
    (tree/'.venv_isaacsim/bin').mkdir(parents=True)
    (tree/'.venv_isaacsim/bin/python').symlink_to(sys.executable)
    config=loop.example_configuration();config['rounds']=config['rounds'][:2]
    values=dict(partition_manifest={},frozen_families=[],collection_backend_json=dict(DEFAULT,physics_device='cuda:0'),
        evaluation_backend_json=DEFAULT,training_sampling_json=dict(version=VERSION,anchors=sampling_profile(),geometry=profile()),
        fallback_manifest={role:dict(root='/protected/'+role,frontend_sha256='a'*64,backend_sha256='b'*64) for role in ('A','B')})
    for key,value in values.items():
        path=tmp_path/(key+'.json');write(path,value);config['inputs'][key]=str(path)
    monkeypatch.setattr(loop.owner,'source_files',lambda tree:{'tools/tight_fresh_research_loop.py':loop.sha256(loop.__file__)})
    monkeypatch.setattr(loop.subprocess,'check_output',lambda *args,**kwargs:'a'*40+'\n')
    monkeypatch.setattr(loop.shutil,'disk_usage',lambda path:SimpleNamespace(free=200*1024**3))
    loop.prepare(root,ledger,tree,config)
    return ledger/'registration.json'


def install_stages(monkeypatch,fail=None):
    calls=[]
    def stage(name,command,context,deadline,stops):
        calls.append((name,command,deadline,stops))
        if name==fail:return dict(complete=False,returncode=1,stop_reason=None)
        def value(flag):return Path(command[command.index(flag)+1])
        if name=='source_prepare':
            write(value('--output')/'prepared.json',dict(complete=True))
            write(value('--output')/'source_selection.json',dict(previous=str(value('--data-source'))))
            write(value('--output')/'A/cache_index.json',dict(complete=True))
        elif name=='collection_prepare':write(value('--output')/'plan.json',dict(registered=True))
        elif name=='collect_cycle':
            inner=value('--registration');registered=loop.read(inner)
            write(inner.parent/'owner_finished.json',dict(software_complete=True,reason='complete'))
            output=Path(registered['output'])
            write(output.with_name(output.name+'_stages')/'finished.json',dict(software_complete=True))
        elif name=='summary':write(value('--output'),dict(student_success=0,expert_recoveries=8))
        else:raise AssertionError(name)
        return dict(complete=True,returncode=0,stop_reason=None)
    monkeypatch.setattr(loop.owner,'run_stage',stage)
    return calls


def test_two_round_dependency_handoff_and_no_repeat(registration,monkeypatch):
    config=loop.read(registration);config['configuration']['bounds']['max_seconds']=12000;write(registration,config)
    calls=install_stages(monkeypatch)
    r=loop.run(registration)
    assert r['software_complete'] and r['completed_rounds']==2 and not r['policy_adopted']
    assert [v[0] for v in calls]==['collection_prepare','collect_cycle','summary','source_prepare','collection_prepare','collect_cycle','summary']
    cfg=loop.read(registration);root=Path(cfg['session_root'])
    source=calls[3][1]
    assert source[source.index('--data-source')+1]==str(root/'round_10')
    assert source.count('--role-source')==2 and 'A='+str(root/'round_10') in source
    assert source.count('--role-evaluation')==2 and 'B='+str(root/'round_10/development') in source
    assert calls[4][1][calls[4][1].index('--source')+1]==str(root/'collection_source_11')
    for index in (10,11):
        inner=loop.read(registration.parent/f'round_{index}/collect_cycle/registration.json')
        command=inner['cycle_command']
        assert '--full-trajectory-batch' in command and '--prefetch-mib' not in command and '--paired-episode-io' not in command
        assert '--continue-rear' in command
        expected_source=root if index==10 else root/'collection_source_11'
        assert command[command.index('--reuse-index')+1]==str(expected_source/'A/cache_index.json')
        assert inner['source_visual_index_binding']['sha256']==loop.sha256(expected_source/'A/cache_index.json')
        assert command[command.index('--count')+1]=='8' and command[command.index('--seed')+1]=='28500000'
        assert command[command.index('--backend-json')+1]==cfg['configuration']['inputs']['evaluation_backend_json']
        assert inner['minimum_cycle_remaining_seconds']==7200
        assert inner['deadline']==cfg['deadline']
        assert inner['requested_collection_max_seconds']==7200 and 60<inner['collection_max_seconds']<4800
        assert inner['loop_work_deadline']==r['work_deadline']
    for name,command,deadline,stops in calls:
        assert deadline==r['work_deadline']<cfg['deadline']
        assert root/'STOP' in stops and registration.parent/'STOP' in stops
        if name=='collection_prepare':
            assert command[command.index('--backend-json')+1]==cfg['configuration']['inputs']['collection_backend_json']
            assert command[command.index('--verification-max-pending')+1]=='16'
    assert loop.read(registration.parent/'owner_finished.json')==r
    with pytest.raises(ValueError,match='restart'):loop.run(registration)


@pytest.mark.parametrize('failure',['collection_prepare','collect_cycle','summary','source_prepare'])
def test_failure_preserves_previous_results_and_stops_without_retry(registration,monkeypatch,failure):
    calls=install_stages(monkeypatch,fail=failure);r=loop.run(registration)
    assert not r['software_complete'] and calls[-1][0]==failure
    assert sum(name==failure for name,*_ in calls)==1
    assert r['completed_rounds']==(1 if failure=='source_prepare' else 0)
    assert r['rounds'][-1]['failed_stage']['returncode']==1
    assert not (Path(loop.read(registration)['session_root'])/'new_family_11').exists()
    assert (registration.parent/'owner_finished.json').is_file()


@pytest.mark.parametrize('boundary',['space','time','stop','input_drift','duplicate_seed'])
def test_admission_or_binding_stops_before_any_child(registration,monkeypatch,boundary):
    calls=install_stages(monkeypatch);r=loop.read(registration)
    if boundary=='space':monkeypatch.setattr(loop.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    elif boundary=='time':monkeypatch.setattr(loop.time,'time',lambda:r['deadline']-5000)
    elif boundary=='stop':(registration.parent/'STOP').touch()
    elif boundary=='input_drift':write(Path(r['configuration']['inputs']['training_sampling_json']),dict(changed=True))
    else:
        r['configuration']['rounds'][1]['seeds'][0]=r['configuration']['rounds'][0]['seeds'][0]
        write(registration,r)
    if boundary in ('input_drift','duplicate_seed'):
        with pytest.raises(ValueError):loop.run(registration)
    else:
        result=loop.run(registration)
        assert result['reason']==dict(space='insufficient_space',time='insufficient_remaining_time',stop='requested_stop')[boundary]
        assert not result['software_complete'] and result['completed_rounds']==0
    assert not calls


def test_summary_separates_expert_success_and_student_phase_denominators(tmp_path,monkeypatch):
    from tools import tight_fresh_stage_audit
    collection=tmp_path/'collection';candidate=tmp_path/'round';output=tmp_path/'summary.json'
    write(collection/'owner_finished.json',dict(software_complete=True))
    write(candidate.with_name('round_stages')/'finished.json',dict(software_complete=True))
    rows=[];cases=[]
    for i in range(3):
        result=dict(identity=f'case{i}',family=f'f{i}',legal=i<2,success=i==0,accepted_for_training=i==0,
            native_lift=True,native_subgoals=2,dagger=dict(expert_ticks=100 if i<2 else 0,
            expert_takeover_success=i==0,expert_valid_ticks=98 if i<2 else 0,student_autonomous_success=False))
        rows.append(dict(case=f'case{i}',role='A',result=result))
        cases.append(dict(name=f'case{i}',expansion_schedule='fixed80',dagger={}))
    write(collection/'plan.json',dict(cases=cases));write(collection/'results.json',rows)
    write(collection/'correction_entries.json',dict(case0={}))
    observed=[]
    def phase(path,result):
        observed.append(path)
        return {},dict(student_native_lift=False,student_settled_reference_available=result['identity']=='case0',
            first_student_stable=None,student_first_contact_height_window_end_tick=70 if result['identity']=='case0' else None,
            authority=dict(student_near_goal_first_tick=None,student_max_subgoals=0))
    monkeypatch.setattr(tight_fresh_stage_audit,'audit_numeric',phase)
    dev=[dict(case='devA',role='A',result=dict(identity='devA',legal=True,success=False,native_lift=True,native_subgoals=0,native_alignment=False)),
         dict(case='devB',role='B',result=dict(identity='devB',legal=False,success=False))]
    write(candidate/'development/plan.json',dict(cases=[{},{}]));write(candidate/'development/results.json',dev)
    write(candidate/'development/result.json',dict(software_complete=True))
    monkeypatch.setattr(loop,'development_windows',lambda path:dict(reference_available=True,held=True,old_stable=False,native_near=False))
    value=loop.summarize_round(collection,candidate,output);total=value['collection']['total']
    assert total['attempts']==3 and total['legal']==2 and total['accepted']==1 and total['expert_recoveries']==1
    assert total['valid_expert_ticks']==196 and total['accepted_valid_expert_ticks']==98
    assert total['student_native_lift']==total['student_native_subgoal']==total['student_autonomous_success']==0
    assert total['student_window_available']==total['student_held']==1
    assert len(observed)==2 and value['development']['A']['held']==1 and value['development']['B']['illegal']==1
    assert value['rgb_opened']==value['weights_opened']==0
    with pytest.raises(ValueError,match='Preserve'):loop.summarize_round(collection,candidate,output)
