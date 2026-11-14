"""Real tiny data/cache bindings plus native-clock HDF5, without CUDA or Kit."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest
import torch

from tools import tight_fresh_collection_source as view
from visuotactile.compact import sha256
from test_fresh_followup_session import source, data_pool, alternate_policy, write, change
from test_training_expansion import setup_registry


def complete_evaluation(directory, *, illegal=False, batch=False):
    plan = view.read(directory/'plan.json')
    for case in plan['cases']:case['spec']['split']='development'
    if illegal:
        case = copy.deepcopy(plan['cases'][0]); case['name'] = 'A_rejected'; case['spec'].update(identity='rejected', collision_gate='gate_v1')
        plan['cases'].append(case)
    if batch:
        from visuotactile.execution_backend import DEFAULT
        plan['batch'] = dict(version='whole_wave_complete_student_v1', neural_batch=True, lanes=8, seed=900, vision_microbatch=8)
        plan['execution_backend'] = DEFAULT
        write(directory/'execution_backend_observed.json', dict(requested=dict(configuration=DEFAULT,num_envs=8),
            physics_device='cpu', camera_backend='standard', num_envs=8, physics_dt=1/120, control_dt=1/60))
        write(directory/'initialization.json',dict(lanes=8,execution_backend=DEFAULT,vision_microbatch=8))
    rows = []; counters = {'A':0, 'B':0}
    for case in plan['cases']:
        path = directory/case['name']; pair = plan['pairs'][case['role']]
        started = dict(spec=case['spec'],checkpoint=pair['backend'])
        result = dict(case['spec'],legal=True,software_complete=True,success=False,steps=2,residual_ticks=2,
            checkpoint_sha256=pair['backend_sha256'],residual_version=pair['residual_version'],
            simulation_step_hz=60,physics_hz=120,plan_hz=5,expert_shadow_computed=False,
            action_control_mode='absolute_joint_position_target_rad', native_inference=dict(summary={
                k:dict(wall_s=dict(count=1 if k=='ddim_plan' else 2)) for k in
                ['dino_rgb','learned_visual_and_history','residual_forward_compose','ddim_plan']}))
        if case['name'] == 'A_rejected':
            result = dict(case['spec'],legal=False,software_complete=True,not_started=True,success=False)
            write(path/'initial_collision.json',dict(legal=False,version='gate_v1'))
            write(path/'initial_observable.json',dict(real='observed'))
        else:
            path.mkdir(parents=True,exist_ok=True)
            with h5py.File(path/'numeric.h5','w') as f:
                f['episode/transition/target_after_rad'] = np.zeros((2,29))
                f['episode/obs/actuator_target_rad'] = np.zeros((2,29))
                times=np.arange(2)/60
                f['episode/obs/time_s'] = times
                f['episode/obs/camera_time_s'] = np.repeat(times[:,None],3,axis=1)
                f['episode/obs/camera_frame'] = np.repeat(np.arange(2)[:,None],3,axis=1)
                f['episode/obs/camera_valid'] = np.ones((2,3),dtype=bool)
                f['episode/transition/time_before_s'] = times
                f['episode/transition/time_after_s'] = times+1/60
        if batch:
            wave = 0 if case['role']=='A' else 1
            execution=dict(wave=wave,lane=counters[case['role']],lanes=8,vision_microbatch=8,
                physics_seed=900+wave,event_seed=100900+wave)
            counters[case['role']] += 1
            started['batch_execution']=execution; result['batch_execution']=execution
            if result['legal']:result['complete_student_policy']=pair
            write(path/'metadata.json',dict(policy_checkpoint_sha256=pair['backend_sha256'],
                batch_execution=execution,execution_backend=plan['execution_backend']))
        write(path/'started.json',started); write(path/'result.json',result)
        rows.append(dict(case=case['name'],role=case['role'],result=result))
    write(directory/'plan.json',plan)
    change(directory/'run_started.json',lambda v:v.update(plan_sha256=sha256(directory/'plan.json')))
    write(directory/'results.json',rows)
    write(directory/'result.json',dict(software_complete=True,worker_error=None,cases=len(rows)))


@pytest.fixture
def inputs(source, alternate_policy, monkeypatch):
    x, newer = source, alternate_policy
    active=x.old.parent/'active'; active.mkdir()
    session=copy.deepcopy(x.session)
    session.update(deadline=time.time()+300,cache_deadline=time.time()+200,training_deadline=time.time()+250,
        resource_root=str(active),owned_worker_roots=[str(active)])
    write(active/'session.json',session)
    for name in ('metadata.json','gate.json'):shutil.copyfile(x.root/name,active/name)
    write(active/'shared_resources.json',dict(utc_epoch=time.time()))
    for root in [x.root,newer.root]:
        for role in ('A','B'):
            write(root/role/'training/finished.json',dict(complete=True,checkpoint_sha256=sha256(root/role/'training/best.pt')))
        complete_evaluation(root/'development',illegal=True,batch=True)
    monkeypatch.setattr(view,'completed_pair',newer.completed)
    return dict(session_root=active,data_source=newer.root,output=active/'selected_source',
        role_sources={'A':x.root,'B':newer.root},role_evaluations={'A':x.evaluation,'B':newer.root/'development'})


def test_mixed_same_session_view_keeps_real_policy_cache_and_independent_larger_pool(inputs):
    before={str(p):sha256(p) for root in set(inputs['role_sources'].values()) for p in root.rglob('*') if p.is_file()}
    result=view.prepare(**inputs);out=inputs['output'];session=inputs['session_root']
    assert result['version']==view.VERSION and result['deadline']==view.read(session/'session.json')['deadline']
    assert (out/'session.json').read_bytes()==(session/'session.json').read_bytes()
    assert (out/'shared_resources.json').resolve()==session/'shared_resources.json'
    assert not (out/'monitor.json').exists()
    assert result['heldout_training_payloads_opened']==0
    assert result['development_numeric_audited'] is True
    assert 'heldout_trajectories_opened' not in result
    for role in ('A','B'):
        original=inputs['role_sources'][role]
        assert (out/role/'state_data').resolve()==inputs['data_source']/role/'state_data'
        assert (out/role/f'backend/{role}/state_data').resolve()==original/role/'state_data'
        assert (out/role/f'backend/{role}/nominal').resolve()==original/role/f'backend/{role}/nominal'
        assert (out/role/'training/finished.json').resolve()==original/role/'training/finished.json'
        assert view.read(out/role/'training/finished.json')['checkpoint_sha256']==result['selected_pairs'][role]['frontend_sha256']
        assert (out/role/'session.json').read_bytes()==(session/'session.json').read_bytes()
        saved=torch.load(out/role/'training/best.pt',weights_only=False)
        assert saved['configuration']['data_sha256']==sha256(original/role/'state_data/data.json')
    assert result['data_pool']['roles']['A']['added_training_ids']==['new']
    assert result['data_pool']['roles']['B']['added_training_ids']==[]
    assert result['evaluations']['A']['counts']=={'A':{'legal':1,'illegal':1},'B':{'legal':1,'illegal':0}}
    assert before=={str(p):sha256(p) for root in set(inputs['role_sources'].values()) for p in root.rglob('*') if p.is_file()}
    view.validate_binding(view.binding_for(out),out)
    with pytest.raises(FileExistsError):view.prepare(**inputs)


def test_candidate_a_and_original_b_are_independently_resolved(inputs):
    old=inputs['role_sources']['A'];pool=inputs['data_source']
    inputs['role_sources']={'A':pool,'B':old}
    inputs['role_evaluations']={'A':pool/'development','B':old/'development'}
    result=view.prepare(**inputs)
    assert Path(result['selected_pairs']['A']['frontend']).is_relative_to(pool)
    assert Path(result['selected_pairs']['B']['frontend']).is_relative_to(old)
    assert result['data_pool']['roles']['A']['added_training_ids']==[]
    assert result['data_pool']['roles']['B']['added_training_ids']==['new']


@pytest.mark.parametrize('fault',['deadline','stop','incomplete_front','wrong_front_sha','wrong_pair_eval',
    'incomplete_eval','clock','no_legal','illegal_actions','batch_lane','pool_reorder','pool_affine','pool_numeric','source_drift'])
def test_rejects_incomplete_drift_or_unbound_sources_before_output(inputs,monkeypatch,fault):
    session=inputs['session_root'];root=inputs['role_sources']['A'];evaluation=inputs['role_evaluations']['A'];pool=inputs['data_source']
    if fault=='deadline':change(session/'session.json',lambda v:v.update(deadline=time.time()-1))
    elif fault=='stop':(session/'STOP').touch()
    elif fault=='incomplete_front':change(root/'A/training/finished.json',lambda v:v.update(complete=False))
    elif fault=='wrong_front_sha':change(root/'A/training/finished.json',lambda v:v.update(checkpoint_sha256='bad'))
    elif fault=='wrong_pair_eval':inputs['role_evaluations']['A']=pool/'development'
    elif fault=='incomplete_eval':change(evaluation/'result.json',lambda v:v.update(software_complete=False))
    elif fault=='clock':
        with h5py.File(evaluation/'A_trial/numeric.h5','r+') as f:f['episode/obs/camera_frame'][1]=0
    elif fault=='no_legal':
        plan=view.read(evaluation/'plan.json');plan['cases']=[r for r in plan['cases'] if r['name']!='A_trial'];write(evaluation/'plan.json',plan)
        rows=[r for r in view.read(evaluation/'results.json') if r['case']!='A_trial'];write(evaluation/'results.json',rows)
        change(evaluation/'result.json',lambda v:v.update(cases=2))
        change(evaluation/'run_started.json',lambda v:v.update(plan_sha256=sha256(evaluation/'plan.json')))
        # One remaining A illegal slot must have lane0 under the new complete registration.
        for name in ['metadata.json','started.json','result.json']:
            change(evaluation/'A_rejected'/name,lambda v:v['batch_execution'].update(lane=0))
        rows[1]['result']=view.read(evaluation/'A_rejected/result.json');write(evaluation/'results.json',rows)
    elif fault=='illegal_actions':
        with h5py.File(evaluation/'A_rejected/numeric.h5','w') as f:f['episode/transition/step_id']=[0]
    elif fault=='batch_lane':
        for name in ['metadata.json','started.json','result.json']:
            change(evaluation/'A_rejected'/name,lambda v:v['batch_execution'].update(lane=3))
        rows=view.read(evaluation/'results.json');rows[-1]['result']=view.read(evaluation/'A_rejected/result.json');write(evaluation/'results.json',rows)
    elif fault=='pool_reorder':change(pool/'A/state_data/data.json',lambda v:v['splits']['train'].reverse())
    elif fault=='pool_affine':(pool/'A/state_data/statistics.pt').write_bytes(b'refit')
    elif fault=='pool_numeric':
        p=pool/'A/state_data/numeric.pt';x=torch.load(p,weights_only=False);x['records']['train']['leak']=torch.ones(1);torch.save(x,p)
    elif fault=='source_drift':
        audit=view.audit_data_extension
        def change_after(*args,**kwargs):
            result=audit(*args,**kwargs);change(session/'gate.json',lambda v:v.update(changed=True));return result
        monkeypatch.setattr(view,'audit_data_extension',change_after)
    with pytest.raises((ValueError,KeyError)):view.prepare(**inputs)
    assert not inputs['output'].exists()


def fake_process(proc,pid,start,args,root):
    d=proc/str(pid);d.mkdir(parents=True)
    fields=['S','1',str(pid),str(pid)]+['0']*15+[str(start)]
    d.joinpath('stat').write_text(str(pid)+' (worker) '+' '.join(fields))
    d.joinpath('cmdline').write_bytes(('\0'.join(args)+'\0').encode())
    d.joinpath('cwd').symlink_to(root)


def test_owner_checks_use_start_identity_and_leave_independent_readers_and_monitor(tmp_path):
    proc=tmp_path/'proc';root=tmp_path/'source';root.mkdir();receipt=tmp_path/'writer.json'
    fake_process(proc,100,123,['python','-m','tools.tight_fresh_train','train','--root',str(root)],tmp_path)
    write(receipt,dict(pid=100,proc_start_ticks=123))
    with pytest.raises(ValueError,match='still live'):view.assert_owners_exited([receipt],{root},proc=proc)
    write(receipt,dict(pid=100,proc_start_ticks=122))
    assert view.assert_owners_exited([receipt],{root},proc=proc)[0]['status']=='pid_reused'
    fake_process(proc,101,124,['python','-m','tools.tight_fresh_stage_fit','--root',str(root)],tmp_path)
    fake_process(proc,102,125,['python','-m','visuotactile.state_resources','monitor','--root',str(root)],tmp_path)
    assert view.assert_owners_exited([],{root},proc=proc)==[]
    write(receipt,dict(pid=100))
    with pytest.raises(ValueError,match='still names'):view.assert_owners_exited([receipt],{root},proc=proc)
    (proc/'100/cmdline').write_bytes(b'python\0unrelated.py\0')
    assert view.assert_owners_exited([receipt],{root},proc=proc)[0]['status']=='legacy_pid_no_longer_bound_to_source'


def test_worker_group_start_ticks_disambiguate_legacy_pid_receipt_for_new_reader(tmp_path):
    proc=tmp_path/'proc';root=tmp_path/'source';root.mkdir()
    fake_process(proc,100,456,['python','-m','tools.tight_fresh_stage_fit','--root',str(root)],tmp_path)
    legacy=tmp_path/'run_started.json';group=tmp_path/'owner_worker_started.json'
    write(legacy,dict(pid=100))
    write(group,dict(worker_pid=100,worker_group=dict(pid=100,start_ticks=123)))
    assert {r['status'] for r in view.assert_owners_exited([legacy,group],{root},proc=proc)}=={'pid_reused'}


def test_live_bound_writer_rejected_but_unrelated_reader_does_not_block(inputs):
    root=inputs['role_sources']['A'];stages=root.with_name(root.name+'_stages')
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)','--root',str(root)],start_new_session=True)
    try:
        from tools.tight_batch_owner import process_identity
        write(stages/'frontend_A_worker.json',dict(pid=worker.pid,proc_start_ticks=process_identity(worker.pid)['start_ticks']))
        with pytest.raises(ValueError,match='still live'):view.prepare(**inputs)
        (stages/'frontend_A_worker.json').unlink()
        # No writer receipt binds this separate read-only process; no blanket source-root quiescence.
        view.prepare(**inputs)
    finally:
        worker.terminate();worker.wait(timeout=5)


@pytest.mark.parametrize('fault',['receipt','link','session','data','plan_pairs','missing_plan_binding'])
def test_collection_binding_detects_drift_and_cannot_be_dropped(inputs,fault):
    from visuotactile.training_expansion import validate_collection_plan,COLLECTION_CONTRACT
    result=view.prepare(**inputs);out=inputs['output'];binding=view.binding_for(out)
    plan=dict(stage_takeover_contract=COLLECTION_CONTRACT,test_used=False,source_selection=binding,
        source_root=str(out),resource_root=str(inputs['session_root']),pairs=result['selected_pairs'],
        source_session_sha256=result['session']['sha256'],source_data_sha256={r:sha256(out/r/'state_data/data.json') for r in ('A','B')})
    view.validate_binding(binding,out,plan=plan)
    if fault=='receipt':change(out/view.RECEIPT,lambda v:v.update(deadline=v['deadline']+1))
    elif fault=='link':
        (out/'A/state_data').unlink();(out/'A/state_data').symlink_to(inputs['role_sources']['A']/'A/state_data')
    elif fault=='session':change(out/'session.json',lambda v:v.update(deadline=v['deadline']+100))
    elif fault=='data':change(inputs['data_source']/'A/state_data/data.json',lambda v:v.update(changed=True))
    elif fault=='plan_pairs':plan['pairs']=dict(plan['pairs'],A=plan['pairs']['B'])
    elif fault=='missing_plan_binding':plan.pop('source_selection')
    with pytest.raises(ValueError):validate_collection_plan(plan)


def test_cli_requires_explicit_role_sources_and_does_not_create_a_new_hours_argument(monkeypatch,tmp_path):
    captured={}
    monkeypatch.setattr(view,'prepare',lambda *args,**kwargs:captured.update(args=args,kwargs=kwargs) or dict(view_root='view',deadline=5,selected_pairs={}))
    monkeypatch.setattr(sys,'argv',['tool','--session-root',str(tmp_path),'--data-source',str(tmp_path/'pool'),
        '--output',str(tmp_path/'view'),'--role-source','A=/real05','--role-source','B=/real06',
        '--role-evaluation','A=/eval','--role-evaluation','B=/eval'])
    view.main()
    assert captured['kwargs']['role_sources']=={'A':Path('/real05'),'B':Path('/real06')}
    assert 'hours' not in captured['kwargs']
    with pytest.raises(ValueError):view.role_paths(['A=/one','A=/two'])


@pytest.mark.parametrize('explicit',[False,True])
def test_formal_expansion_registers_optional_receipt_and_revalidates_in_run_and_aggregate(setup_registry,monkeypatch,explicit):
    from tools import tight_fresh_expansion as expansion
    from tools.tight_fresh_batch_run import validate_takeover_plan
    from visuotactile.training_expansion import validate_collection_plan
    x=setup_registry
    write(x['source']/'session.json',dict(deadline=time.time()+300,domain=x['domain'],expert_sha256='expert'))
    monkeypatch.setattr(expansion,'completed_pair',lambda root,role:dict(role=role))
    binding=dict(version=view.VERSION,path='bound_source_receipt',sha256='fixture') if explicit else None
    monkeypatch.setattr(view,'binding_for',lambda source:binding)
    calls=[]
    def validate(received,source,*,plan):
        assert received==binding and Path(source)==x['source']
        assert plan['pairs']=={'A':{'role':'A'},'B':{'role':'B'}}
        calls.append(plan['source_data_sha256'])
    monkeypatch.setattr(view,'validate_binding',validate)
    plan=expansion.prepare(x['source'],x['root']/'collection',session_root=x['source'],registry_id='selection_fixture',
        seeds=[6000],partition_manifest=x['partition'],frozen_families=x['families'],
        schedules=['held60'],lanes=8,event_seed=8000,round_index=9)
    assert ('source_selection' in plan)==explicit
    validate_takeover_plan(plan);validate_collection_plan(plan)
    assert len(calls)==(3 if explicit else 0)
