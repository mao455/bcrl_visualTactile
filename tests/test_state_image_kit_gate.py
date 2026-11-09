import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch







def test_restricted_helper_fds_remain_visible_without_fabricated_zero_counts(monkeypatch):
    from pathlib import Path
    import os
    from tools.runtime import proc
    original=Path.iterdir
    def denied(path):
        if path==Path('/proc',str(os.getpid()),'fd'):
            raise PermissionError(13,'Permission denied')
        return original(path)
    monkeypatch.setattr(Path,'iterdir',denied)
    row=proc(os.getpid())
    assert row['pid']==os.getpid() and row['start_ticks']>0
    assert row['fds'] is None and row['watches'] is None
    assert row['telemetry_missing']==['fd_directory_permission_denied']


@pytest.mark.parametrize('modality', ['G00', 'G10'])
def test_actual_trial_routes_only_current_joints_to_visual_policy(tmp_path, monkeypatch, modality):
    """Exercise the runner call site that failed before the first real action."""
    from visuotactile import state_benchmark as runner, native_controller
    from visuotactile.models import native_vision
    from visuotactile.state_history import controller_proprio

    class ReachedPolicy(Exception):
        pass

    q=torch.arange(29, dtype=torch.float32)[None]/100
    qd=-q
    packet={'episode_id':torch.tensor([1]), 'step_id':torch.tensor([0])}
    observed={'joint_position_rad':q, 'joint_velocity_rad_s':qd,
        'object_pose_w':object(), 'palm_body_pose_w':object(), 'successes':object()}

    class Controller:
        def __init__(self, saved, metadata, dino=None):
            self.model=SimpleNamespace(group=saved['group'], modality=saved['modality'])

        def act(self, current, joints=None):
            assert current is packet
            assert set(joints)=={'joint_position_rad', 'joint_velocity_rad_s'}
            assert joints['joint_position_rad'] is q and joints['joint_velocity_rad_s'] is qd
            assert torch.equal(controller_proprio(current,joints,None,'canonical_joint_state_v1'),
                torch.cat((q[0],qd[0])))
            raise ReachedPolicy

    class Dino:
        def to(self, device):return self
        def eval(self):return self

    env=SimpleNamespace(research_packet=packet, _diagnostics=lambda:observed,
        research_metadata=lambda:{}, cfg=SimpleNamespace(to_dict=lambda:{}),
        scene=SimpleNamespace(env_origins=torch.zeros(1,3)), env_max_goals=torch.tensor([1]),
        max_episode_length=601)
    player=SimpleNamespace(reset=lambda:None, intr_reward_coef_embd=torch.zeros(1,1),
        get_action=lambda *args,**kw:torch.zeros(1,29))
    monkeypatch.setattr(runner,'IsolatedEnvironment',lambda *args:SimpleNamespace(stream=SimpleNamespace(state={})))
    monkeypatch.setattr(runner,'prepare',lambda *args:({'policy':torch.zeros(1,141)},{'legal':True}))
    monkeypatch.setattr(runner,'environment_identity',lambda env:({},'unchanged'))
    monkeypatch.setattr(runner,'export_initial',lambda *args:{})
    monkeypatch.setattr(native_controller,'NativeController',Controller)
    monkeypatch.setattr(native_vision,'NativeDinoSpatial',Dino)
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'identity only')
    with pytest.raises(ReachedPolicy):
        runner.trial(env,SimpleNamespace(block=nullcontext),tmp_path/'trial',
            dict(seed=1,identity='case',cohort='development'),{},player,str(checkpoint),
            saved=dict(group='I-HistoryDP',modality=modality),position=True)


@pytest.mark.parametrize('group', ['I-HistoryDP', 'I-OfficialDP'])
def test_legacy_image_runner_does_not_request_diagnostics(group):
    from visuotactile.state_benchmark import native_policy_action
    packet=object();command=object()
    def act(current):
        assert current is packet
        return command
    def forbidden():raise AssertionError('Legacy image contract must not read diagnostics')
    controller=SimpleNamespace(model=SimpleNamespace(group=group,modality=None),act=act)
    assert native_policy_action(controller,packet,forbidden) is command


def test_native_state_runner_retains_existing_diagnostics_route():
    from visuotactile.state_benchmark import native_policy_action
    packet=object();observed=object()
    controller=SimpleNamespace(model=SimpleNamespace(group='S-NativeDP',modality=None),
        act=lambda current,diagnostics:(current,diagnostics))
    assert native_policy_action(controller,packet,lambda:observed)==(packet,observed)


@pytest.mark.parametrize('budget', ['remaining', 'expired', 'reset'])
def test_evaluation_resume_never_trains_or_resets_deadline(tmp_path, monkeypatch, budget):
    import time
    from tools import tight_finish as owner
    original=tmp_path/'original';original.mkdir()
    root=tmp_path/'resume';root.mkdir()
    deadline=time.time()+60 if budget!='expired' else time.time()-1
    owner.write(original/'session.json',dict(evaluation_deadline=deadline))
    owner.write(original/'status.json',dict(owned_workers_exited=True))
    owner.write(original/'evaluation_plan.json',[])
    owner.write(root/'evaluation_plan.json',[])
    owner.write(root/'recovery.json',dict(passed=True,original_completed_trials=0,
        original_root=str(original),protected_sha256={}))
    owner.write(root/'configuration.json',dict(vision_pair={'groups':['G00','G10']}))
    owner.write(root/'budget.json',{})
    owner.write(root/'session.json',dict(evaluation_deadline=deadline+(60 if budget=='reset' else 0),
        deadline=time.time()+3600,source_head='training_commit'))
    (root/'state_data').mkdir();owner.write(root/'state_data/data.json',{})
    monkeypatch.setattr(owner.subprocess,'check_output',lambda args,**kw:'evaluation_commit\n' if args[1]=='rev-parse' else '')
    launched=[]
    class Child:
        pid=123;returncode=0
        def poll(self):return 0
        def wait(self,timeout=None):return 0
    def spawn(args,**kw):
        launched.append(args)
        if 'visuotactile.state_resources' in args:
            owner.write(root/'shared_resources.json',dict(utc_epoch=time.time()))
        elif 'vision_evaluate' in args:
            owner.write(root/'evaluation_complete.json',dict(complete=True))
        else:raise AssertionError('Unexpected training/cache/offline process: '+str(args))
        return Child()
    monkeypatch.setattr(owner.subprocess,'Popen',spawn)
    reports=[];monkeypatch.setattr(owner,'vision_report',lambda path:reports.append(path))
    if budget!='remaining':
        with pytest.raises(ValueError,match='budget exhausted|cannot reset'):owner.vision_resume(root)
        assert not launched and not (root/'launch.json').exists()
    else:
        owner.vision_resume(root)
        assert len(launched)==2 and reports==[root]
        session=owner.read(root/'session.json')
        assert session['evaluation_deadline']==deadline and session['training_source_head']=='training_commit'
        assert session['source_head']=='evaluation_commit'
        assert owner.read(root/'status.json')['training_repeated'] is False
