"""Real bridge and recorders around fake physics; isolate terminal/label clocks."""
import contextlib
import copy
import json
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
import torch

from test_batched_position_bridge import environment, native_pipeline
from visuotactile.batched_rollout import lane_metadata
from visuotactile.position_control import ExpertPositionBridge, apply_position_targets


def metadata(count):
    return dict(joint_limits_rad=[[[-2.,2.]]*29 for _ in range(count)],
        env_origins_w=[[float(lane)*2,0.,0.] for lane in range(count)],
        cameras=[dict(id=f'view_{view}',pos_env=[[lane,view,1.] for lane in range(count)],
            quat_ros_wxyz=[[1.,lane*.01,0.,0.] for lane in range(count)],
            intrinsics=[[[lane+1.,0.,0.],[0.,view+1.,0.],[0.,0.,1.]] for lane in range(count)])
            for view in range(3)],
        camera_calibration_bindings=[dict(camera_paths=[f'/env_{lane}/camera_{view}' for lane in range(count)],
            render_products=[f'/diagnostic_{lane}_{view}' for lane in range(count)],
            image_render_products=[f'/shared_tiled_{view}'],calibration_resolution=[1,1]) for view in range(3)])


def test_lane_metadata_binds_own_pose_intrinsics_and_tiled_calibration_without_mutation():
    source=metadata(8);original=copy.deepcopy(source)
    case=dict(spec=dict(family='f',split='train',source_identity='train_f'))
    for lane in (0,3,7):
        selected=lane_metadata(source,lane,8,case,dict(backend_sha256='b'),None,dict(lane=lane))
        for view in range(3):
            camera=selected['cameras'][view];binding=selected['camera_calibration_bindings'][view]
            assert camera['pos_env']==[[lane,view,1.]]
            assert camera['intrinsics'][0][0][0]==lane+1
            assert binding['camera_paths']==[f'/env_{lane}/camera_{view}']
            assert binding['render_products']==[f'/diagnostic_{lane}_{view}']
            assert binding['image_render_products']==[f'/shared_tiled_{view}']
    assert source==original
    source['cameras'][1]['intrinsics']=source['cameras'][1]['intrinsics'][:2]
    with pytest.raises(ValueError,match='intrinsics'):
        lane_metadata(source,0,8,case,dict(backend_sha256='b'),None,{})


@pytest.mark.parametrize('microbatch',[None,8])
def test_actual_wave_stops_each_lane_once_preserves_history_and_masks_delayed_labels(tmp_path,monkeypatch,microbatch):
    import visuotactile.batched_rollout as rollout
    import visuotactile.batched_position_bridge as bridges
    import visuotactile.narrow_data as narrow
    import visuotactile.live_collision as collision
    import visuotactile.takeover_trigger as trigger_module
    import visuotactile.async_recording as recording
    from visuotactile.takeover_trigger import CONTACT_HEIGHT_VERSION
    env=environment(3);env.research_step=0;ends=[3,8,999]
    env.env_max_goals=torch.ones(3,dtype=torch.long);env.max_episode_length=10
    env.scene=NS(env_origins=torch.tensor([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]]))
    env.cfg.to_dict=lambda: {};env.research_metadata=lambda: metadata(3)
    env.clear_recorded_events=lambda: None
    def diagnostics():
        pose=torch.tensor([[lane*2.,0.,.5,1.,0.,0.,0.] for lane in range(3)])
        return dict(joint_position_rad=env._prev_targets[:,env._perm_lab_to_canon]*.8,
            joint_velocity_rad_s=torch.zeros(3,29),object_pose_w=pose,palm_body_pose_w=pose.clone(),
            finger_position_w=torch.zeros(3,5,3),lifted_object=torch.zeros(3,dtype=torch.bool),
            successes=torch.zeros(3),retract_succeeded=torch.zeros(3,dtype=torch.bool))
    def packet(tick):
        image=torch.stack([torch.full((3,4,2,2),float(lane*50+tick))/255 for lane in range(3)])
        return dict(image=image,depth_m=image[:,:,3:].clone(),depth_valid=torch.ones(3,3,1,2,2,dtype=torch.bool),
            camera_time_s=torch.full((3,3),tick/60.,dtype=torch.float64),camera_frame=torch.full((3,3),tick),
            camera_valid=torch.ones(3,3,dtype=torch.bool),time_s=torch.full((3,),tick/60.,dtype=torch.float64),
            episode_id=torch.arange(3)+7,step_id=torch.full((3,),tick),
            actuator_target_rad=env._prev_targets[:,env._perm_lab_to_canon].clone())
    env._diagnostics=diagnostics;env.research_packet=packet(0)
    env.reset=lambda seed=None:(dict(policy=torch.zeros(3,3)),{})
    def step(proposal):
        tick=env.research_step;before=env._prev_targets[:,env._perm_lab_to_canon].clone()
        applied,_=apply_position_targets(env,proposal);bridge=env._dagger_position_bridge
        done=torch.tensor([tick+1==end for end in ends]);terminated=done.clone()
        env.last_transition=dict(episode_id=torch.arange(3)+7,step_id=torch.full((3,),tick),
            time_before_s=torch.full((3,),tick/60.,dtype=torch.float64),
            time_after_s=torch.full((3,),(tick+1)/60.,dtype=torch.float64),
            target_before_rad=before,target_after_rad=applied.clone(),position_requested_rad=bridge.requested.clone(),
            position_applied_rad=applied.clone(),dagger=copy.deepcopy(bridge.trace),
            control_trace=dict(delay_index=torch.zeros(3,dtype=torch.long)),diagnostics_after=diagnostics(),
            actuator_substep_target_rad=applied[:,None].repeat(1,2,1),
            contact_substeps_after_w_N=torch.zeros(3,2,5,3),object_contact_force_w=torch.zeros(3,2,5,3),
            applied_wrench=dict(force_w_N=torch.zeros(3,2,3),torque_w_Nm=torch.zeros(3,2,3)),
            termination_reasons=dict(max_successes=done,timeout=torch.zeros(3,dtype=torch.bool)),
            native_alignment_condition=torch.zeros(3,dtype=torch.bool))
        env.research_step+=1
        # Simulate native automatic reset of terminal lane targets. The next
        # wave step must hold these targets, without feeding reset frames to it.
        env._prev_targets[done]=-.15;env._cur_targets[done]=-.15
        env.research_packet=packet(env.research_step)
        return dict(policy=torch.full((3,3),float(env.research_step))),None,terminated,torch.zeros_like(done),{}
    env.step=step
    class Expert(torch.nn.Module):
        def forward(self,actor):
            return dict(mus=torch.stack([torch.full((29,),lane*.1+float(actor[lane,0])*.03) for lane in range(3)]))
    player=NS(model=Expert(),intr_reward_coef_embd=torch.zeros(3,1),reset=lambda:None)
    expert_calls=[]
    def get_action(actor,is_deterministic):
        expert_calls.append(actor[:,0].tolist());return player.model(actor)['mus'].clamp(-1,1)
    player.get_action=get_action
    students=[]
    class Student:
        def __init__(self,lane):
            self.lane=lane;self.base=NS(command=None);self.seen=[];self.issued=[]
            self.action_scale_cpu=torch.ones(29);self.residual_scales_cpu=torch.full((29,),.05)
        def act(self,p):
            self.seen.append((int(p['step_id'][self.lane]),float(p['image'][self.lane,0,0,0,0]),
                None if self.base.command is None else self.base.command.clone()))
            self.nominal=torch.full((1,29),.1+self.lane*.03+int(p['step_id'][self.lane])*.001)
            self.delta=torch.full((1,29),.01);self.command=self.nominal+self.delta
            self.plan=self.nominal[:,None].repeat(1,24,1);self.contact_gate=False
            self.residual_diagnostics=dict(proposed_rad=self.delta,applied_rad=self.delta,
                clip_mask=torch.zeros(1,29,dtype=torch.bool),tanh_fraction=self.delta,contact_valid=torch.tensor(False))
            return self.command
        def observe_execution(self,tr):
            assert torch.equal(self.base.command,tr['position_requested_rad'])
            self.issued.append(self.base.command.clone())
        def inference_metrics(self):
            return dict(summary={},pending_cuda_events=0,timing_scope='test',wall_scope='test',excludes=[])
    class Batch:
        def __init__(self,*args,**kwargs):
            assert kwargs['vision_microbatch']==(2 if microbatch is None else microbatch)
            self.lanes=[Student(i) for i in range(3)];students.extend(self.lanes);self.visual_frames=[0]*3
        def act(self,p,d,active):
            result=p['actuator_target_rad'].clone()
            for lane,running in enumerate(active):
                if running:result[lane:lane+1]=self.lanes[lane].act(p);self.visual_frames[lane]+=1
            return result
        def vision_execution_report(self):
            return dict(real_lanes=sum(self.visual_frames),consumed_observations_per_lane=self.visual_frames)
    original_bridge=bridges.BatchedExpertPositionBridge;created=[]
    class CPUBridge(original_bridge):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,pipeline=native_pipeline(),**kwargs);created.append(self)
    class Trigger:
        def __init__(self,configuration,fallback):self.ticks=[];self.scheduled=None
        def observe(self,tick,pose,palm,force):
            self.ticks.append(tick);lane=round(float(pose[0])/2)
            if tick==lane:
                self.scheduled=tick+1;return self.scheduled
        def receipt(self):return dict(observed_ticks=self.ticks,actual_scheduled_takeover_tick=self.scheduled)
    original_async=recording.AsyncLaneRecorder.finish_async;wait_steps=[]
    def asynchronous(self,result):
        ticket=original_async(self,result)
        class WaitAfterAllControl:
            def result(self):
                wait_steps.append(env.research_step);assert env.research_step==8
                return ticket.result()
        return WaitAfterAllControl()
    monkeypatch.setattr(recording.AsyncLaneRecorder,'finish_async',asynchronous)
    monkeypatch.setattr(rollout,'BatchedResidualController',Batch)
    monkeypatch.setattr(bridges,'BatchedExpertPositionBridge',CPUBridge)
    monkeypatch.setattr(narrow,'configure_direct_reset_lanes',lambda *args:None)
    monkeypatch.setattr(collision,'clearance',lambda env,lane:dict(legal=True))
    monkeypatch.setattr(trigger_module,'make_trigger',Trigger)
    source=tmp_path/'data.json';source.write_text(json.dumps(dict(splits=dict(train=['original_0','original_1']))))
    cases=[]
    for lane in range(2):
        identity=f'case_{lane}'
        cases.append(dict(name=identity,role='A',source_data=str(source),spec=dict(identity=identity,
            family=f'family_{lane}',split='train',cohort='dagger_train',source_identity=f'original_{lane}',
            seed=lane,event_seed=20+lane,narrow_reset={},takeover_tick=10**9),
            dagger=dict(version=ExpertPositionBridge.version,expert_speed=ExpertPositionBridge.speed_scheme,
                trigger=dict(version=CONTACT_HEIGHT_VERSION,delay_ticks=0))))
    pair=dict(frontend='frontend',frontend_sha256='f',backend='backend',backend_sha256='b')
    saved=dict(residual_version='test',configuration={})
    plan=dict(kind='takeover',batch=dict(seed=23,writer_bytes=32<<20),expert_sha256='e')
    if microbatch is not None:plan['batch']['vision_microbatch']=microbatch
    finished,profile=rollout.run_wave(env,player,NS(block=contextlib.nullcontext),tmp_path,cases,pair,saved,None,plan,0)
    assert [r['steps'] for _,r in finished]==[3,8]
    assert len(expert_calls)==8 and [len(s.seen) for s in students]==[3,8,0]
    assert wait_steps==[8,8] and [b.tick for b in created[0].bridges]==[3,8,0]
    assert profile['control_ticks']==11
    assert profile['vision_execution']['consumed_observations_per_lane']==[3,8,0]
    for lane,count in enumerate((3,8)):
        result=finished[lane][1];directory=tmp_path/f'case_{lane}'
        assert result['dagger']['expert_valid_ticks']==(0 if lane==0 else 3)
        assert result['dagger']['recovery_coverage']['switch_tick']==lane+1
        assert result['dagger']['trigger_receipt']['observed_ticks']==list(range(lane+1))
        assert (directory/'capture_complete.json').exists() and (directory/'execution_complete.json').exists()
        with h5py.File(directory/'demonstration/data.h5') as f:
            g=f[f'episodes/case_{lane}'];d=g['transition/dagger']
            assert g.attrs['length']==count
            assert d['control_source'][:].tolist()==[int(t>=lane+1) for t in range(count)]
            assert d['expert_label_valid'][:].tolist()==[t>=lane+4 for t in range(count)]
            assert np.array_equal(d['q_reached'][:],g['transition/diagnostics_after/joint_position_rad'][:])
            assert d['bridge_active'][:].all()
            assert json.loads(f.attrs['metadata_json'])['cameras'][0]['pos_env']==[[lane,0,1.]]
        for tick,_,history in students[lane].seen[1:]:
            assert torch.equal(history,students[lane].issued[tick-1])
    assert not hasattr(env,'_dagger_position_bridge')


def test_owner_census_separates_execution_verification_and_admission(tmp_path):
    from tools.tight_batch_owner import census
    plan=dict(kind='takeover',cases=[dict(name='a',role='A')]);directory=tmp_path/'a';directory.mkdir()
    (directory/'execution_complete.json').write_text(json.dumps(dict(software_complete=True)))
    assert census(tmp_path,plan)[0]['state']=='execution_complete_pending_verification'
    (directory/'result.json').write_text(json.dumps(dict(software_complete=True,legal=True)))
    assert census(tmp_path,plan)[0]['state']=='execution_complete_pending_admission'
    (directory/'result.json').write_text(json.dumps(dict(software_complete=True,legal=True,accepted_for_training=False)))
    assert census(tmp_path,plan)[0]['state']=='complete'


def test_admission_persists_status_without_destroying_audited_result_bytes(tmp_path,monkeypatch):
    from tools.tight_fresh_batch_run import admit_capture
    import tools.tight_fresh_report as reports
    import tools.tight_fresh_stage_takeover as stages
    import visuotactile.state_benchmark as benchmark
    from visuotactile.compact import sha256
    from visuotactile.training_resources import atomic_json
    from tools.tight_fresh_stage_takeover import CONTACT_HEIGHT_CONTRACT
    from visuotactile.takeover_trigger import CONTACT_HEIGHT_VERSION
    directory=tmp_path/'case';directory.mkdir()
    trigger=dict(version=CONTACT_HEIGHT_VERSION,delay_ticks=0)
    result=dict(software_complete=True,legal=True,success=False,dagger=dict(expert_valid_ticks=0,expert_ticks=0,
        configuration=dict(trigger=trigger)))
    atomic_json(directory/'result.json',result);original=(directory/'result.json').read_bytes()
    atomic_json(tmp_path/'plan.json',dict(kind='takeover'))
    monkeypatch.setattr(benchmark,'audit_dagger_capture',lambda path,pair:dict(passed=True,result_sha256=sha256(path/'result.json')))
    monkeypatch.setattr(reports,'audit_trigger_case',lambda *args:dict(passed=True))
    monkeypatch.setattr(stages,'classify',lambda *args:'ended_without_student_contact_height_window')
    admit_capture(tmp_path,dict(name='case',spec={},dagger=dict(trigger=trigger)),result,{},
        dict(round_index=7,stage_takeover_contract=CONTACT_HEIGHT_CONTRACT),{})
    audit=json.loads((directory/'capture_audit.json').read_text())
    assert (directory/'result_before_admission.json').read_bytes()==original
    assert sha256(directory/audit['result_binding_path'])==audit['result_sha256']
    assert not json.loads((directory/'result.json').read_text())['accepted_for_training']
    assert json.loads((tmp_path/'correction_entries.json').read_text())=={}
