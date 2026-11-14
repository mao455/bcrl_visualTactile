"""CPU-only component registration, full native histories, and finite owner."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from tools import tight_fresh_frontend_component_gate as gate
from test_batched_frontend_diagnostic import setup, build, packet
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def registered_fixture():
    # Real public preregistration, no raw held-out data or CUDA.
    tree = Path(gate.__file__).resolve().parents[1]
    original = gate.read(tree/'research_docs/evidence/tight_fresh_20261114/session06_preregistration/initial_families.json')
    cases = [dict(role='B', spec=dict(identity='development_'+str(row['seed']),
        family=row['family'], split='development', seed=row['seed'], event_seed=row['seed']+100000,
        narrow_reset=copy.deepcopy(row['narrow_reset']))) for row in original['development']]
    dev = dict(kind='autonomous', test_used=False, cases=cases, execution_backend=gate.BACKEND,
        batch=dict(lanes=8, vision_microbatch=8), inference_scheduler_index_mode='host')
    return original, dev


def test_registration_reuses_original_eight_configs_exactly():
    original, dev = registered_fixture()
    selected = gate.registered_cases(dev, original, original['domain'])
    assert [x['seed'] for x in selected] == list(range(28500000,28500008))
    changed = copy.deepcopy(dev); changed['cases'][3]['spec']['narrow_reset']['table_reset_z'] += 1e-12
    with pytest.raises(ValueError, match='initial configuration'): gate.registered_cases(changed, original, original['domain'])
    changed = copy.deepcopy(dev); changed['cases'].pop()
    with pytest.raises(ValueError, match='Exactly'): gate.registered_cases(changed, original, original['domain'])


def train_data():
    ids = ['train_%02d'%i for i in range(30)]
    return dict(normalization_train_ids=ids, splits=dict(train=ids, validation=['validation'], test=[]),
        entries={i:dict(family=i, length=100) for i in ids+['validation']})


def test_actual_split_schema_missing_entry_split_is_authoritative_train_only():
    data = train_data()
    assert gate.selected_train(data) == data['splits']['train'][:8]
    changed = copy.deepcopy(data); changed['splits']['validation'].append('train_00')
    with pytest.raises(ValueError, match='disjoint'): gate.selected_train(changed)
    changed = copy.deepcopy(data); changed['entries']['train_00']['family'] = 'validation'
    with pytest.raises(ValueError, match='Only original'): gate.selected_train(changed)
    changed = copy.deepcopy(data); changed['entries']['train_00']['trajectory_kind'] = 'expert_takeover_v1'
    with pytest.raises(ValueError, match='Only original'): gate.selected_train(changed)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    from tools import tight_fresh_rollouts
    from visuotactile import batched_frontend_diagnostic
    root = tmp_path/'round'; resource = tmp_path/'resource'; root.mkdir(); resource.mkdir()
    original, dev = registered_fixture()
    for key in ('expert','dino'):
        (resource/(key+'.pt')).write_bytes((key+' bytes').encode())
    session = dict(deadline=time.time()+3600, precision_policy='unified_fp32', camera_content_validation=True,
        domain=original['domain'])
    for key in ('expert','dino'):
        session.update({key+'_checkpoint':str(resource/(key+'.pt')), key+'_sha256':gate.sha256(resource/(key+'.pt'))})
    write(resource/'session.json', session)
    front = root/'B/training/best.pt'; front.parent.mkdir(parents=True); front.write_bytes(b'front selected')
    rear = root/'B/backend/B/training/best.pt'; rear.parent.mkdir(parents=True)
    torch.save(dict(configuration=dict(output_contract=REACHABLE_CENTERED_OUTPUT)), rear)
    pair = dict(frontend=str(front), frontend_sha256=gate.sha256(front), backend=str(rear), backend_sha256=gate.sha256(rear),
        supervision='u_expert', modality='G00', execution_adapter=REACHABLE_POSITION_EXECUTION)
    monkeypatch.setattr(tight_fresh_rollouts, 'completed_pair', lambda *args:pair)
    monkeypatch.setattr(batched_frontend_diagnostic, 'selected_frontend', lambda *args:(
        dict(configuration=dict(data_sha256=gate.sha256(root/'B/state_data/data.json'))),dict(selected_epoch=5)))
    dev['pairs'] = dict(B=pair)
    write(root/'development/plan.json',dev); write(root/'development/result.json',dict(software_complete=True))
    for lane,case in enumerate(dev['cases']):
        spec=case['spec']
        write(root/'development'/('B_'+spec['identity'])/'result.json',dict(software_complete=True,
            complete_student_policy=pair,seed=spec['seed'],narrow_reset=spec['narrow_reset'],
            batch_execution=dict(lane=lane,lanes=8,vision_microbatch=8,wave=1,physics_seed=28500001,event_seed=28600001)))
    for path in (front.parent/'finished.json', rear.parent/'finished.json', root/'B/backend/B/offline/result.json'): write(path,{})
    data = train_data()
    for i in data['splits']['train']:
        raw = root/(i+'.h5'); raw.write_bytes(('original raw '+i).encode())
        data['entries'][i].update(path=str(raw), source_sha256=gate.sha256(raw))
    write(root/'B/state_data/data.json',data)
    return root, resource, pair


def test_prepare_cpu_binds_best_split_session_two_component_identities(artifacts,tmp_path):
    root, resource, pair = artifacts; output = tmp_path/'prepare'
    plan = gate.prepare(root,resource,output)
    assert plan['cuda_initialized'] is False and not torch.cuda.is_initialized()
    assert plan['stages'] == ['admission','full_B','projected_front_B']
    assert plan['pair'] == pair and plan['max_seconds'] == 1800 and plan['selected_frontend']['selected_epoch'] == 5
    assert plan['batch_seed']==28500001 and plan['batch_event_seed']==28600001
    assert plan['identities']['full_B']['complete_student_pair'] is True
    pure = plan['identities']['projected_front_B']
    assert pure['complete_student_pair'] is False and pure['rear_network_loaded'] is False and 'backend' not in pure
    assert plan['train_ids'] == train_data()['normalization_train_ids'][:8]
    assert plan['original_session_deadline'] == gate.read(resource/'session.json')['deadline']
    assert 'validation' not in ''.join(plan['bound_files'])
    gate.verify(plan,Path(plan['source_tree']))
    with pytest.raises(FileExistsError): gate.prepare(root,resource,output)
    changed = copy.deepcopy(plan); changed['source_sha256']['tools/tight_fresh_frontend_component_gate.py'] = 'bad'
    with pytest.raises(ValueError, match='source bytes'): gate.verify(changed,Path(plan['source_tree']))


def test_actual_batch_seed_comes_from_all_eight_old_b_receipts(artifacts):
    root,_,pair=artifacts; original,dev=registered_fixture()
    cases=gate.registered_cases(dev,original,original['domain'])
    batch,receipts=gate.source_batch_receipts(root,cases,pair)
    assert batch['physics_seed']==28500001 and len(receipts)==8 and 'lane' not in batch
    path=root/'development/B_development_28500007/result.json'
    changed=gate.read(path); changed['batch_execution']['event_seed']=28600000; write(path,changed)
    with pytest.raises(ValueError,match='wave1 native RNG'): gate.source_batch_receipts(root,cases,pair)


@pytest.mark.parametrize('seconds',[True,0,59,1801,None])
def test_invalid_owner_bound_does_not_create_output(tmp_path,seconds):
    with pytest.raises(ValueError, match='Finite'): gate.prepare(tmp_path,tmp_path,tmp_path/'output',max_seconds=seconds)
    assert not (tmp_path/'output').exists()


def test_original_local_session_bytes_and_deadline_are_preserved(artifacts,tmp_path):
    root, resource, _ = artifacts; plan = gate.prepare(root,resource,tmp_path/'prepared')
    arm = tmp_path/'arm'; arm.mkdir(); gate.stage_session(plan,arm)
    assert (arm/'session.json').read_bytes() == (resource/'session.json').read_bytes()
    with pytest.raises(FileExistsError): gate.stage_session(plan,arm)
    (resource/'session.json').write_text((resource/'session.json').read_text()+' ')
    other = tmp_path/'other'; other.mkdir()
    with pytest.raises(ValueError, match='SHA256'): gate.stage_session(plan,other)
    assert not (other/'session.json').exists()


class SameInputFull:
    """Real NativeController histories; independent real zero compose implementation."""
    def __init__(self,x):
        from visuotactile.models.native_dp import TargetResidual
        self.inner = build(x,'projection')
        residual = TargetResidual.__new__(TargetResidual); torch.nn.Module.__init__(residual)
        residual.output_contract = REACHABLE_CENTERED_OUTPUT
        residual.execution_adapter = REACHABLE_POSITION_EXECUTION
        residual.loss_contract = None; residual.action_scale = torch.ones(29)
        self.lanes = [SimpleNamespace(base=base,residual=residual,device='cpu',nominal=None) for base in self.inner.lanes]
        self.visual_frames = self.inner.visual_frames

    def act(self,p,d,active):
        out = self.inner.act(p,d,active)
        for lane,row in enumerate(self.lanes): row.nominal = self.inner.last_nominal[lane]
        return out

    def observe_execution(self,tr,active):
        self.inner._pending_commands = tr['position_requested_rad'].clone()
        self.inner.observe_execution(tr,active)

    def vision_execution_report(self): return self.inner.execution_report()['vision']


def samples():
    result = [[] for _ in range(3)]
    for tick in range(25):
        p,d = packet(tick,torch.ones(3,29)*((tick-1)*.0001))
        p['episode_id'] = torch.ones(3,dtype=torch.long); p['step_id'] = torch.full((3,),tick)
        target = torch.ones(3,29)*(tick*.0001)
        for lane in range(3):
            result[lane].append(({k:v[lane:lane+1] for k,v in p.items()},
                {k:d[k][lane:lane+1] for k in ('joint_position_rad','joint_velocity_rad_s')},target[lane:lane+1]))
    return result


def test_same_input_gate_real_native_history_25_ticks_and_recorded_ack(setup,monkeypatch):
    monkeypatch.setattr(gate,'LANES',3)
    full, front = SameInputFull(setup), build(setup,'projection')
    progress = []
    result = gate.run_same_input(full,front,samples(),lease=SimpleNamespace(block=lambda:nullcontext()),cuda=False,progress=progress.append)
    assert result['passed'] and len(result['rows']) == len(progress) == 25
    assert front.executed_ticks == [25]*3 and front.visual_frames == [25]*3
    assert [len(x) for x in front.plan_records] == [3]*3
    assert all(row['teacher_forced_recorded_target'] for row in result['rows'])
    for lane in range(3):
        assert torch.equal(front.lanes[lane].position_previous[0], torch.full((29,),.0024))
        assert torch.equal(front.lanes[lane].generator.get_state(),full.lanes[lane].base.generator.get_state())


def test_same_input_gate_rejects_plan_drift_and_never_admits_partial(setup,monkeypatch):
    monkeypatch.setattr(gate,'LANES',3)
    full, front = SameInputFull(setup), build(setup,'projection')
    original = full.act
    def altered(*args):
        result = original(*args); full.lanes[0].base.plan += .01; return result
    full.act = altered
    with pytest.raises(ValueError, match='numerical parity'): gate.run_same_input(full,front,samples(),lease=SimpleNamespace(block=lambda:nullcontext()),cuda=False)
    assert front.executed_ticks == [0]*3


def owner_fixture(tmp_path,monkeypatch):
    from tools import tight_batch_owner, tight_fresh_eval_recovery
    output = tmp_path/'owner'; output.mkdir(); resource = tmp_path/'resource'; resource.mkdir()
    write(resource/'session.json',dict(deadline=time.time()+3000))
    plan = dict(resource_root=str(resource),session_sha256=gate.sha256(resource/'session.json'),
        original_session_deadline=time.time()+3000,max_seconds=60)
    write(output/'plan.json',plan); monkeypatch.setattr(gate,'verify',lambda *a,**kw:None)
    events = []
    monkeypatch.setattr(tight_batch_owner,'bind_worker_group',lambda pid:dict(pid=pid))
    monkeypatch.setattr(tight_batch_owner,'cleanup_owned_group',lambda b:events.append('cleanup') or dict(live_descendants_exited=True))
    monkeypatch.setattr(tight_fresh_eval_recovery,'stop_owned',lambda p:events.append('stop'))
    return output, events


def test_failed_admission_never_launches_kit_and_owner_does_not_retry(tmp_path,monkeypatch):
    output, events = owner_fixture(tmp_path,monkeypatch)
    class Process:
        pid=123; returncode=1
        def __init__(self,cmd,**kwargs):
            assert kwargs['start_new_session'] is True
            assert (output/'admission/session.json').is_file()
            events.append(cmd[-1])
        def poll(self): return self.returncode
    monkeypatch.setattr(gate.subprocess,'Popen',Process)
    with pytest.raises(ValueError,match='Bounded'): gate.owner(output,tmp_path)
    assert events == ['admission','stop','cleanup']
    assert not (output/'full_B').exists() and not gate.read(output/'owner_finished.json')['complete']
    with pytest.raises(FileExistsError): gate.owner(output,tmp_path)
    assert events == ['admission','stop','cleanup']


def test_owner_hard_deadline_cleans_up_hung_own_worker(tmp_path,monkeypatch):
    output, events = owner_fixture(tmp_path,monkeypatch)
    class Process:
        pid=123; returncode=None
        def __init__(self,*args,**kwargs): events.append('launch')
        def poll(self): return None
    monkeypatch.setattr(gate.subprocess,'Popen',Process); monkeypatch.setattr(gate.time,'sleep',lambda n:None)
    checks = []
    def stop(*args):
        checks.append(1)
        if len(checks) == 3: raise TimeoutError('hard bound')
    monkeypatch.setattr(gate,'check_bounds',stop)
    with pytest.raises(ValueError,match='Bounded'): gate.owner(output,tmp_path)
    assert events == ['launch','stop','cleanup']
    result = gate.read(output/'owner_finished.json'); started = gate.read(output/'owner_started.json')
    assert 'TimeoutError' in result['error'] and result['automatic_retry'] is False
    assert started['deadline']-started['utc'] == 60 and started['deadline']-started['work_deadline'] == 30


def test_successful_owner_launches_exactly_three_independent_stages(tmp_path,monkeypatch):
    output, events = owner_fixture(tmp_path,monkeypatch)
    class Process:
        pid=123; returncode=0
        def __init__(self,cmd,**kwargs):
            stage = cmd[-1]; events.append(stage)
            write(output/stage/'result.json',dict(software_complete=True,passed=True))
        def poll(self): return 0
    monkeypatch.setattr(gate.subprocess,'Popen',Process)
    monkeypatch.setattr(gate,'initial_comparison',lambda p:dict(all_observables_equal=True))
    gate.owner(output,tmp_path)
    assert events == [x for stage in gate.STAGES for x in (stage,'stop','cleanup')]
    assert gate.read(output/'owner_finished.json')['complete'] is True
    assert gate.read(output/'result.json')['accepted_training_trajectories'] == 0


def test_clock_counts_allow_zero_rear_only_for_explicit_pure_arm():
    rows = [dict(stage=name) for name in ('robot_history_input','learned_visual_and_history','ddim_plan')]
    controller = SimpleNamespace(visual_frames=[1],inference_metrics=lambda i:dict(rows=rows))
    _, value = gate.counts(controller,0,1,'projected_front_B')
    assert value['backend_updates'] == 0
    with pytest.raises(ValueError,match='full real-frame'): gate.counts(controller,0,1,'full_B')


def test_initial_observable_difference_is_disclosed_not_hidden_state_claim(tmp_path):
    for stage, offset in [('full_B',0.),('projected_front_B',1e-7)]:
        write(tmp_path/stage/'initial_observable.json',dict(q=[[offset]*29]*8,object=[[0.]*7]*8))
        write(tmp_path/stage/'native_bound.json',dict(native_max_episode_length=600,collector_max_control_ticks=1803))
    result = gate.initial_comparison(tmp_path)
    assert result['all_observables_equal'] is False and result['fields']['q']['maximum_absolute_difference'] == 1e-7
    write(tmp_path/'projected_front_B/native_bound.json',dict(native_max_episode_length=601,collector_max_control_ticks=1806))
    with pytest.raises(ValueError,match='different native'): gate.initial_comparison(tmp_path)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('stage', ['full_B', 'projected_front_B'])
def test_real_simulation_loop_preserves_invalid_lane_and_acknowledges_terminal_steps(tmp_path,monkeypatch,fail,stage):
    """Exercise the actual runner/clock/HDF path with a deterministic CPU plant."""
    from tools import tight_state
    from visuotactile import environment_rng, narrow_data, live_collision
    from visuotactile.models import native_vision
    class Env:
        num_envs=8; physics_dt=1/120; step_dt=1/60; max_episode_length=2
        env_max_goals=torch.ones(8,dtype=torch.long)
        cfg=SimpleNamespace(to_dict=lambda:{})
        def __init__(self): self.tick=0; self.commands=[]
        def clear_recorded_events(self): pass
        def research_metadata(self): return dict(action_control_mode=self.action_control_mode)
        def _diagnostics(self):
            return dict(joint_position_rad=torch.zeros(8,29), joint_velocity_rad_s=torch.zeros(8,29),
                lifted_object=torch.zeros(8,dtype=torch.bool), successes=torch.zeros(8,dtype=torch.long))
        @property
        def research_packet(self):
            stamp=torch.full((8,),self.tick/60,dtype=torch.float64)
            return dict(image=torch.zeros(8,3,4,2,2),depth_valid=torch.ones(8,3,1,2,2,dtype=torch.bool),
                camera_valid=torch.ones(8,3,dtype=torch.bool), camera_time_s=stamp[:,None].repeat(1,3),
                camera_frame=torch.full((8,3),self.tick+1), time_s=stamp,
                episode_id=torch.ones(8,dtype=torch.long),step_id=torch.full((8,),self.tick),actuator_target_rad=torch.zeros(8,29))
        def step(self,command):
            if fail: raise RuntimeError('finite simulated software failure')
            before=self.research_packet; self.commands.append(command.clone()); self.tick+=1
            done=torch.zeros(8,dtype=torch.bool); done[1 if self.tick==1 else 2:]=True
            self.last_transition=dict(position_requested_rad=command.clone(),position_applied_rad=command.clone(),
                step_id=before['step_id'],episode_id=before['episode_id'],target_after_rad=command.clone(),
                time_before_s=before['time_s'],time_after_s=torch.full((8,),self.tick/60,dtype=torch.float64),
                diagnostics_after=self._diagnostics(),native_alignment_condition=torch.zeros(8,dtype=torch.bool),
                termination_reasons=dict(max_successes=torch.zeros(8,dtype=torch.bool),timeout=done))
            return None,None,done,torch.zeros_like(done),{}
    class Controller:
        def __init__(self):
            self.visual_frames=[0]*8;self.acks=[];self.last_nominal=[torch.zeros(1,29) for _ in range(8)]
            self.lanes=[SimpleNamespace(plan=torch.zeros(1,24,29),nominal=torch.zeros(1,29),
                base=SimpleNamespace(plan=torch.zeros(1,24,29))) for _ in range(8)]
            self.plan_records=[[] for _ in range(8)]
        def act(self,p,d,active):
            self.pending=list(active)
            for lane,running in enumerate(active): self.visual_frames[lane]+=int(running)
            return torch.zeros(8,29)
        def observe_execution(self,tr,active):
            assert active==self.pending
            self.acks.append(list(active))
        def inference_metrics(self,lane):
            n=self.visual_frames[lane]
            return dict(rows=[dict(stage=name) for _ in range(n) for name in ('robot_history_input','learned_visual_and_history')]
                +[dict(stage='ddim_plan') for _ in range((n+11)//12)]
                +([dict(stage='residual_forward_compose') for _ in range(n)] if stage=='full_B' else []))
        def vision_execution_report(self): return dict(observations=self.visual_frames)
        def execution_report(self):
            return dict(all_submitted_acknowledged=True,rear_forward_ticks=0,
                lanes=[dict(acknowledged_control_ticks=n) for n in self.visual_frames])
    env=Env(); controller=Controller(); lease=SimpleNamespace(block=lambda **kwargs:nullcontext())
    monkeypatch.setattr(tight_state,'environment',lambda *args,**kwargs:(None,env,None,lease))
    monkeypatch.setattr(narrow_data,'configure_direct_reset_lanes',lambda *args:None)
    monkeypatch.setattr(live_collision,'clearance',lambda e,lane:dict(legal=lane!=0))
    monkeypatch.setattr(native_vision,'NativeDinoSpatial',lambda:SimpleNamespace(cuda=lambda:SimpleNamespace(eval=lambda:None)))
    monkeypatch.setattr(environment_rng,'IsolatedEnvironment',lambda e,s:SimpleNamespace(
        reset=lambda seed:None,begin_events=lambda seed:None,step=e.step))
    monkeypatch.setattr(gate,'create_controllers',lambda *args,**kwargs:dict(front=controller,full=controller))
    plan=dict(execution_backend=gate.BACKEND,batch_seed=gate.SEED,batch_event_seed=gate.SEED+100000,
        cases=[dict(identity='case%d'%i,narrow_reset={}) for i in range(8)],
        identities=dict(projected_front_B=dict(complete_student_pair=False),full_B=dict(complete_student_pair=True)))
    if fail:
        with pytest.raises(RuntimeError,match='finite simulated software failure'):
            gate.simulate(plan,tmp_path,lease,stage)
        partial=gate.read(tmp_path/'partial_result.json')
        assert partial['software_complete'] is False and partial['steps']==[0]*8 and partial['automatic_retry'] is False
        assert not (tmp_path/'result.json').exists() and controller.acks==[]
        import h5py
        with h5py.File(tmp_path/'case1/numeric.h5') as f: assert len(f['episode'])==0
        return
    gate.simulate(plan,tmp_path,lease,stage)
    result=gate.read(tmp_path/'result.json')
    assert result['software_complete'] and result['complete_student_pair'] is (stage=='full_B')
    assert result['results'][0]['legal'] is False and result['results'][0]['steps']==0
    assert controller.acks==[[False]+[True]*7]
    assert all(row['steps']==1 for row in result['results'][1:])
    assert all(row['success'] is False and row['accepted_for_training'] is False for row in result['results'])
    assert result['diagnostic_only'] is True and result['autonomous_evaluation_eligible'] is False
    assert all(row['diagnostic_only'] is True and row['autonomous_evaluation_eligible'] is False
        and row['formal_evaluation_eligible'] is False for row in result['results'])
    import h5py
    with h5py.File(tmp_path/'case1/numeric.h5') as f:
        assert f['episode/transition/step_id'][:].tolist()==[0]
        assert f['episode/component/current_plan_rad'].shape==(1,24,29)
    assert not torch.cuda.is_initialized()
