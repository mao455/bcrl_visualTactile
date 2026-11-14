"""Finite gate ownership and real numeric recording, without Kit/CUDA."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import pytest
import torch

from tools import tight_inactive_render_gate as gate


class Environment:
    def __init__(self):
        self.research_cameras=[SimpleNamespace(render_product_paths=[f'/view{v}/lane{i}' for i in range(16)],
            _rep_registry={'rgb':[object() for _ in range(16)]}) for v in range(3)]
        self.fault_tick=None;self.reset()
    def reset(self):self.tick=0;self.update()
    def update(self):
        self.research_packet=dict(image=torch.zeros(16,3,4,2,2),camera_valid=torch.ones(16,3,dtype=torch.bool),
            camera_time_s=torch.full((16,3),self.tick/60,dtype=torch.float64),
            time_s=torch.full((16,),self.tick/60,dtype=torch.float64),camera_frame=torch.full((16,3),self.tick,dtype=torch.long),
            q=torch.full((16,29),float(self.tick)))
    def _diagnostics(self):return dict(q=self.research_packet['q'])
    def step(self,command):
        assert command.shape==(16,29)
        self.tick+=1;self.last_transition=dict(executed=command.clone(),step=torch.full((16,),self.tick))
        self.update();terminated=torch.zeros(16,dtype=torch.bool)
        if self.fault_tick==self.tick:terminated[0]=True
        return None,None,terminated,torch.zeros_like(terminated),{}


class Controller:
    def __init__(self):self.visual_frames=[0]*16;self.ack=[0]*16
    def act(self,packet,diagnostics,active):
        for i,on in enumerate(active):
            if on:
                assert self.ack[i]==self.visual_frames[i]
                self.visual_frames[i]+=1
        return diagnostics['q'].clone()
    def observe_execution(self,transition,active):
        for i,on in enumerate(active):
            if on:self.ack[i]+=1
    def inference_metrics(self,lane):
        count=self.visual_frames[lane]
        return dict(rows=[dict(stage=stage) for _ in range(count) for stage in
            ('learned_visual_and_history','robot_history_input','residual_forward_compose')]+
            [dict(stage='ddim_plan') for _ in range((count+11)//12)])
    def vision_execution_report(self):return dict(real_lane_observations=sum(self.visual_frames),profile=None)


class Guard:
    def __init__(self):self.active=[False]*16;self.enabled=[True]*16;self.events=[];self.last=None
    def begin_wave(self,wave):
        assert not any(self.active)
        self.active=[True]*16;self.enabled=[True]*16;self.last=None;self.events.append(('begin',wave))
    def observe(self,packet,*,control_tick):
        assert self.last is None and control_tick==0 or self.last+1==control_tick
        self.last=control_tick;self.events.append(('observe',control_tick))
        return packet,dict(active_lanes=[i for i,a in enumerate(self.active) if a],enabled_products=3*sum(self.enabled))
    def retire(self,lanes,*,reason,control_tick):
        assert control_tick==self.last and reason in ('diagnostic_censor','initial_illegal')
        for lane in lanes:assert self.active[lane];self.active[lane]=False;self.enabled[lane]=False
        self.events.append(('retire',control_tick,list(lanes)))
    def snapshot(self):return dict(active=self.active.copy(),products=[dict(enabled=x) for x in self.enabled])
    def mask_inactive(self,packet):return packet


@pytest.mark.parametrize('arm',gate.ARMS)
def test_two_real_recorded_waves_restore_and_keep_active_histories(tmp_path,arm):
    directory=tmp_path/arm;directory.mkdir();env=Environment();guard=Guard();sync=[]
    plan=dict(resource_root=str(tmp_path),work_deadline=time.time()+60)
    for wave,role in enumerate(('A','B')):
        guard.begin_wave(wave);assert all(guard.enabled)
        env.reset();controller=Controller()
        result=gate.run_wave(env,SimpleNamespace(block=nullcontext),guard,controller,env,directory/role,
            plan,arm,wave,synchronize=lambda:sync.append(True))
        assert result['counts']==[36]+[24]*7+[12]*8==controller.ack==controller.visual_frames
        assert result['software_complete'] and result['success'] is None and not result['accepted_for_training']
        observed=[r['renderer']['enabled_products'] for r in result['rows']]
        assert observed==([48]*12+[24]*12+[3]*12 if arm=='inactive' else [48]*36)
        assert [r['active_before'] for r in result['timing']]==[16]*12+[8]*12+[1]*12
        assert not any(guard.active)
        for lane,count in enumerate(result['counts']):
            with h5py.File(directory/role/f'lane_{lane:02d}_numeric.h5') as stream:
                assert len(stream['episode/obs/q'])==count
                assert stream.attrs['diagnostic_only']
        assert json.loads((directory/role/'result.json').read_text())['counts']==controller.ack
    assert len(sync)==2*36*2
    assert [event[1] for event in guard.events if event[0]=='observe']==list(range(36))*2


@pytest.mark.parametrize('arm',gate.ARMS)
def test_illegal_slots_have_zero_control_history_and_no_seed_replacement(tmp_path,arm):
    directory=tmp_path/arm;directory.mkdir();env=Environment();guard=Guard();guard.begin_wave(0)
    legal=[i not in (3,10) for i in range(16)];controller=Controller()
    result=gate.run_wave(env,SimpleNamespace(block=nullcontext),guard,controller,env,directory/'A',
        dict(resource_root=str(tmp_path),work_deadline=time.time()+60),arm,0,synchronize=lambda:None,legal=legal)
    assert result['counts']==[v if legal[i] else 0 for i,v in enumerate([36]+[24]*7+[12]*8)]
    assert controller.ack[3]==controller.ack[10]==0
    observed=[r['renderer']['enabled_products'] for r in result['rows']]
    assert observed==([48]+[42]*11+[21]*12+[3]*12 if arm=='inactive' else [48]*36)
    assert result['rows'][0]['renderer']['active_lanes']==list(range(16))
    assert result['rows'][0]['renderer']['consumed_active_lanes']==[i for i,a in enumerate(legal) if a]
    for lane in (3,10):
        with h5py.File(directory/'A'/f'lane_{lane:02d}_numeric.h5') as stream:assert len(stream['episode'])==0


def test_illegal_final_lane_fails_before_control_and_keeps_identity(tmp_path):
    env=Environment();guard=Guard();guard.begin_wave(0);legal=[False]+[True]*15
    with pytest.raises(ValueError,match='Lane0'):
        gate.run_wave(env,SimpleNamespace(block=nullcontext),guard,Controller(),env,tmp_path/'A',
            dict(resource_root=str(tmp_path),work_deadline=time.time()+60),'inactive',0,synchronize=lambda:None,legal=legal)
    assert env.tick==0


def test_prepare_and_source_binding_keep_finite_scope_and_original_session_deadline(tmp_path,monkeypatch):
    root=tmp_path/'session';root.mkdir();source=tmp_path/'source';source.mkdir()
    (root/'session.json').write_text(json.dumps(dict(deadline=time.time()+500)))
    for folder in ('tools','visuotactile'):
        (tmp_path/folder).mkdir();(tmp_path/folder/'frozen.py').write_text('# frozen\n')
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(gate.subprocess,'check_output',lambda *a,**k:'commit')
    pair={}
    for key in ('frontend','backend'):
        path=source/(key+'.pt');path.write_bytes(b'fixture')
        pair[key]=str(path);pair[key+'_sha256']=gate.sha256(path)
    monkeypatch.setattr(gate,'completed_pair',lambda *a:dict(pair))
    cases=[dict(role=role,spec=dict(split='train',family=f'train_{i}',narrow_reset={})) for i in range(16) for role in ('A','B')]
    case_plan=tmp_path/'cases.json';case_plan.write_text(json.dumps(dict(kind='takeover',test_used=False,cases=cases,
        execution_backend={},batch=dict(seed=8000))))
    output=tmp_path/'gate'
    plan=gate.prepare(root,source,case_plan,output,max_seconds=600)
    assert plan['deadline']==json.loads((root/'session.json').read_text())['deadline']
    assert plan['work_deadline']==plan['deadline']-30 and plan['ticks_per_wave']==36
    assert plan['accepted_for_training'] is False and plan['test_used'] is False
    gate.verify_plan(plan,tmp_path)
    with pytest.raises(FileExistsError):gate.prepare(root,source,case_plan,output)
    (source/'frontend.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='student changed'):gate.verify_plan(plan,tmp_path)


def test_natural_termination_is_preserved_not_replaced_or_published(tmp_path,monkeypatch):
    directory=tmp_path/'inactive';directory.mkdir();env=Environment();env.fault_tick=3;guard=Guard();guard.begin_wave(0)
    import visuotactile.camera_failure_diagnostics as diagnostic
    captured=[];monkeypatch.setattr(diagnostic,'save_failure',lambda *a,**kw:captured.append(kw['control_tick']))
    with pytest.raises(ValueError,match='Native termination'):
        gate.run_wave(env,SimpleNamespace(block=nullcontext),guard,Controller(),env,directory/'A',
            dict(resource_root=str(tmp_path),work_deadline=time.time()+60),'inactive',0,synchronize=lambda:None)
    assert not (directory/'A/result.json').exists()
    failure=json.loads((directory/'A/failure.json').read_text())
    assert failure['completed_ticks']==2 and failure['counts']==[3]*16 and captured==[2]
    for lane in range(16):
        with h5py.File(directory/'A'/f'lane_{lane:02d}_numeric.h5') as stream:
            assert len(stream['episode/transition/step'])==3


@pytest.mark.parametrize('where',['output','session','deadline'])
def test_deadline_and_both_stops_are_checked(tmp_path,where):
    output=tmp_path/'out';root=tmp_path/'root';output.mkdir();root.mkdir()
    plan=dict(resource_root=str(root),work_deadline=time.time()+10)
    if where=='deadline':plan['work_deadline']=time.time()-1
    else:((output if where=='output' else root)/'STOP').touch()
    with pytest.raises((RuntimeError,TimeoutError)):gate.check_bounds(plan,output)


@pytest.mark.parametrize('fault',['exit','stop','stop_cleanup'])
def test_owner_fails_closed_cleans_group_and_never_launches_second_arm(tmp_path,monkeypatch,fault):
    from tools import tight_batch_owner as ownership,tight_fresh_eval_recovery as recovery
    output=tmp_path/'gate';output.mkdir();root=tmp_path/'root';root.mkdir()
    plan=dict(resource_root=str(root),work_deadline=time.time()+100,deadline=time.time()+130,startup_seconds=10)
    (output/'plan.json').write_text(json.dumps(plan));calls=[];cleaned=[]
    monkeypatch.setattr(gate,'verify_plan',lambda *a:None)
    def launch(command,**kwargs):
        calls.append(command)
        if fault!='exit':(output/'STOP').touch()
        return SimpleNamespace(pid=os.getpid(),returncode=1 if fault=='exit' else None,poll=lambda:1 if fault=='exit' else None)
    monkeypatch.setattr(gate.subprocess,'Popen',launch)
    monkeypatch.setattr(ownership,'bind_worker_group',lambda pid:dict(pid=pid,creation_bound=True))
    monkeypatch.setattr(ownership,'cleanup_owned_group',lambda binding:cleaned.append(binding) or dict(live_descendants_exited=True))
    def stop(process):
        if fault=='stop_cleanup':raise RuntimeError('direct stop failed')
    monkeypatch.setattr(recovery,'stop_owned',stop)
    with pytest.raises(RuntimeError,match='gate failed'):gate.owner(output,tmp_path)
    assert len(calls)==len(cleaned)==1 and not (output/'inactive').exists()
    report=json.loads((output/'result.json').read_text())
    assert not report['software_complete'] and report['error'] and report['success'] is None
    cleanup=json.loads((output/'ordinary/owner_group_cleanup.json').read_text())
    assert cleanup['live_descendants_exited'] and bool(cleanup['stop_error'])==(fault=='stop_cleanup')
    with pytest.raises(FileExistsError):
        (output/'STOP').unlink(missing_ok=True)
        gate.owner(output,tmp_path)


@pytest.mark.parametrize('same_legal',[False,True])
def test_owner_comparison_requires_actual_legal_cohort_and_both_closed_waves(tmp_path,monkeypatch,same_legal):
    from tools import tight_batch_owner as ownership,tight_fresh_eval_recovery as recovery
    output=tmp_path/'gate';output.mkdir();root=tmp_path/'root';root.mkdir()
    (output/'plan.json').write_text(json.dumps(dict(resource_root=str(root),work_deadline=time.time()+100,
        deadline=time.time()+130,startup_seconds=10)))
    calls=[];cleaned=[]
    monkeypatch.setattr(gate,'verify_plan',lambda *a:None)
    def launch(command,**kwargs):
        calls.append(command);arm=command[-1];directory=output/arm;rows=[]
        for role in ('A','B'):
            (directory/role).mkdir();(directory/role/'result.json').write_text(json.dumps(dict(software_complete=True)))
            legal=[True]*16
            if not same_legal and arm=='inactive':legal[4]=False
            rows.append(dict(role=role,legal=legal,counts=[v if legal[i] else 0 for i,v in enumerate([36]+[24]*7+[12]*8)],
                result_sha256=gate.sha256(directory/role/'result.json')))
        (directory/'result.json').write_text(json.dumps(dict(software_complete=True,waves=rows)))
        return SimpleNamespace(pid=os.getpid(),returncode=0,poll=lambda:0)
    monkeypatch.setattr(gate.subprocess,'Popen',launch)
    monkeypatch.setattr(ownership,'bind_worker_group',lambda pid:dict(pid=pid))
    monkeypatch.setattr(ownership,'cleanup_owned_group',lambda binding:cleaned.append(binding) or dict(live_descendants_exited=True))
    monkeypatch.setattr(recovery,'stop_owned',lambda *a:None)
    result=gate.owner(output,tmp_path)
    assert result['software_complete'] and len(calls)==len(cleaned)==2
    assert all(row['timing_comparable']==same_legal for row in result['comparison'])
    assert result['success'] is None and result['accepted_for_training'] is False
