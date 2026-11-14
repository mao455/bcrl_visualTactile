from contextlib import nullcontext
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from tools import tight_fresh_batch_gate as gate
from visuotactile.models.native_vision import EXACT_TAIL_PROFILE, vision_lane_budget


class FakeDino:
    def __init__(self):
        self.backbone=SimpleNamespace(forward_features=lambda images:images[:,0])

    def forward_lanes(self,rgb,*,vision_microbatch,vision_tail_profile=None):
        count=len(rgb)
        self.last_lane_execution=vision_lane_budget(count,vision_microbatch,vision_tail_profile=vision_tail_profile)
        images=3*(count if vision_tail_profile else vision_microbatch)
        self.backbone.forward_features(torch.zeros(images,1))
        return rgb[:,0,0,0,0]


class Batch:
    """Stateful CPU fixture: resets, recurrence, sampler and source acknowledgements."""
    def __init__(self,exact=False,defect=None):
        self.exact,self.defect=exact,defect
        self.dino=gate.KernelObservedDino(FakeDino())
        self.visual_frames=[0]*8
        self.lanes=[SimpleNamespace(base=SimpleNamespace(generator=torch.Generator().manual_seed(7093)),state=None) for _ in range(8)]
        self.vision_work=vision_lane_budget(0,8,vision_tail_profile=EXACT_TAIL_PROFILE if exact else None)
        self.calls=0

    def act(self,packet,diagnostics,active):
        ids=[i for i,a in enumerate(active) if a]
        profile=EXACT_TAIL_PROFILE if self.exact else None
        features=self.dino.forward_lanes(packet['image'][ids,:,:3],vision_microbatch=8,vision_tail_profile=profile)
        work=vision_lane_budget(len(ids),8,vision_tail_profile=profile)
        for key in ('real_lanes','padding_lanes','real_images','processed_images','forward_features_calls'):
            self.vision_work[key]+=work[key]
        if profile:
            for size,count in work['kernel_image_batch_counts'].items():
                h=self.vision_work['kernel_image_batch_counts'];h[size]=h.get(size,0)+count
        self.calls+=1
        command=packet['actuator_target_rad'].clone()
        for index,lane in enumerate(ids):
            controller=self.lanes[lane];base=controller.base
            tick=int(packet['step_id'][lane]);episode=int(packet['episode_id'][lane])
            if episode!=getattr(base,'episode',None):
                assert tick==0
                base.episode=episode;base.started=packet['time_s'][lane]
                base.memory=SimpleNamespace(robot=[],frames=[],native=torch.zeros(1),visual=torch.zeros(1),next_tick=0)
                base.position_previous=None
            memory=base.memory
            assert memory.next_tick==tick
            memory.robot=(memory.robot+[torch.tensor([float(tick)])])[-13:]
            memory.frames=(memory.frames+[features[index:index+1]])[-3:]
            memory.native=memory.native+features[index]/100
            memory.visual=memory.visual+features[index]/200
            memory.next_tick=tick+1;base.next_step=tick+1
            if tick%12==0:
                base.plan=torch.rand(1,24,29,generator=base.generator)/10
            controller.plan=base.plan
            controller.command=base.plan[:,tick%12]+memory.native/100
            if self.defect=='target' and self.calls==9:controller.command+=1e-3
            controller.nominal=controller.command.clone();controller.delta=torch.zeros(1,29)
            controller.previous_nominal=controller.nominal.clone()
            valid=torch.arange(16)>=max(0,16-(tick+1))
            controller.state=dict(history_valid=valid[None],history=memory.native.clone())
            if self.defect=='hidden' and self.calls==9:controller.state['history']+=1
            if self.defect=='history_count' and self.calls==9:memory.next_tick+=1
            controller.visual=features[index:index+1];controller.contact_gate=False;controller.residual_diagnostics={}
            base.last_robot=torch.zeros(178);base.last_condition=torch.zeros(1,512)
            base.last_packet={key:value[lane:lane+1] for key,value in packet.items()};base.command=controller.command
            command[lane:lane+1]=controller.command
            self.visual_frames[lane]+=1
        if self.defect=='inactive' and self.calls==9:self.lanes[7].base.memory.native+=1
        if self.defect=='shared_input' and self.calls==9:packet['image'].add_(1)
        return command

    def observe_execution(self,transition,active):
        for lane,running in enumerate(active):
            if running:
                base=self.lanes[lane].base
                base.position_previous=(base.command[0].clone(),transition['position_applied_rad'][lane].clone(),
                    base.last_packet['time_s'][0].clone())
                if self.defect=='ack' and self.calls==9:base.position_previous[1].add_(1)

    def vision_execution_report(self):
        report=copy.deepcopy(self.vision_work)
        if self.defect=='padding':report['padding_lanes']=1
        report['controller_calls']=self.calls
        report['consumed_observations_per_lane']=self.visual_frames.copy()
        return report


def sources():
    lanes=[]
    for lane in range(8):
        rows=[]
        for tick in range(25):
            rows.append((dict(episode_id=torch.tensor([lane+1]),step_id=torch.tensor([tick]),
                time_s=torch.tensor([tick/60]),image=torch.full((1,3,4,1,1),(lane+tick)/100),
                actuator_target_rad=torch.full((1,29),float(tick)/100)),{},torch.full((1,29),float(tick+1)/100)))
        lanes.append(rows)
    return lanes


def run(exact):
    return gate.run_tail_sequence(Batch(),exact,sources(),lease=SimpleNamespace(block=nullcontext),cuda=False)


@pytest.mark.parametrize('exact_first',[False,True])
def test_complete_tail_sequence_history_reset_inactive_and_actual_shapes(exact_first):
    result=gate.run_tail_sequence(Batch(),Batch(True),sources(),lease=SimpleNamespace(block=nullcontext),
        cuda=False,exact_first=exact_first)
    assert result['passed'],result['failure']
    assert result['consumed_observations_per_lane']==[25,20,20,20,20,14,14,8]
    assert result['actual_source_acknowledgements_per_lane']==result['consumed_observations_per_lane']
    assert result['rows'][13]['history'][0]['next_step']==1
    assert result['rows'][13]['history'][1]['next_step']==14
    assert result['rows'][-1]['active']==[True]+[False]*7
    reports=result['vision_execution']
    assert reports['fixed']['observed_backbone_kernel_image_batch_counts']=={'24':25}
    assert reports['exact']['observed_backbone_kernel_image_batch_counts']=={'24':8,'21':6,'15':6,'3':5}
    assert reports['exact']['padding_lanes']==0 and reports['fixed']['padding_lanes']==59
    assert reports['exact']['processed_images']==423 and reports['fixed']['processed_images']==600


@pytest.mark.parametrize('defect,reason',[
    ('target','target parity'),('hidden','history numerical'),('history_count','real frame'),
    ('inactive','Inactive lane'),('ack','acknowledgement'),('padding','padded'),('shared_input','mutated')])
def test_tail_gate_rejects_real_boundary_errors(defect,reason):
    result=run(Batch(True,defect))
    assert not result['passed'] and reason in result['failure']['message']


def test_observer_restores_backbone_even_when_forward_raises():
    model=FakeDino();original=model.backbone.forward_features
    def fail(*args,**kwargs):
        model.backbone.forward_features(torch.zeros(3,1))
        raise RuntimeError('injected')
    model.forward_lanes=fail
    observed=gate.KernelObservedDino(model)
    with pytest.raises(RuntimeError,match='injected'):observed.forward_lanes(None)
    assert model.backbone.forward_features is original
    assert observed.kernel_image_batch_counts=={'3':1}


def test_numeric_history_tolerance_is_fixed_and_discrete_exact():
    a={'x':torch.tensor([1.]),'mask':torch.tensor([True])}
    assert gate.numeric_history_comparison(a,dict(a,x=torch.tensor([1.+1e-5])))['atol']==2e-5
    for b in (dict(a,x=torch.tensor([1.1])),dict(a,mask=torch.tensor([False])),dict(a,x=torch.tensor([float('nan')]))):
        with pytest.raises(ValueError):gate.numeric_history_comparison(a,b)


def test_source_selection_train_only_and_cpu_cannot_pass_live_gate(tmp_path,monkeypatch):
    from tools import tight_fresh_rollouts as rollouts
    from visuotactile import state_resources
    root=tmp_path/'session';root.mkdir();(root/'session.json').write_text(json.dumps({'deadline':1.}))
    ids=[f'train_{i:02}' for i in range(30)]
    data=dict(normalization_train_ids=ids,splits=dict(train=ids,validation=['v'],test=['t']),
        entries={i:dict(episode=i,family=i) for i in ids})
    for role in ('A','B'):
        directory=root/role/'state_data';directory.mkdir(parents=True)
        (directory/'data.json').write_text(json.dumps(data))
    monkeypatch.setattr(rollouts,'completed_pair',lambda *a:{'backend':'unused'})
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext,wait_s=0.))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    output=tmp_path/'gate'
    with pytest.raises(RuntimeError,match='did not pass'):gate.exact_tail_gate(root,root,output)
    assert json.loads((output/'result.json').read_text())['passed'] is False
    registration=json.loads((output/'registration.json').read_text())
    assert registration['selected_training_ids']=={'A':ids[:8],'B':ids[:8]}
    assert registration['test_used'] is False
    data['splits']['validation'].append(ids[0])
    (root/'A/state_data/data.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='held-out'):gate.exact_tail_gate(root,root,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('pass_sequence',[False,True])
def test_orchestration_creates_role_receipts_and_preserves_failure(tmp_path,monkeypatch,pass_sequence):
    from tools import tight_fresh_rollouts as rollouts,tight_fresh_forensics_gate as forensics
    from visuotactile import state_resources,batched_controller
    from visuotactile.models import native_vision
    root=tmp_path/'session';root.mkdir();(root/'session.json').write_text(json.dumps({'deadline':1.}))
    ids=[f'train_{i:02}' for i in range(30)]
    data=dict(normalization_train_ids=ids,splits=dict(train=ids,validation=['v'],test=['t']),
        entries={i:dict(episode=i,family=i) for i in ids})
    for role in ('A','B'):
        directory=root/role/'state_data';directory.mkdir(parents=True)
        (directory/'data.json').write_text(json.dumps(data))
    monkeypatch.setattr(rollouts,'completed_pair',lambda root,role:dict(backend=role))
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext,wait_s=0.))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:{})
    fake=SimpleNamespace();fake.cuda=lambda:fake;fake.eval=lambda:fake
    monkeypatch.setattr(native_vision,'NativeDinoSpatial',lambda:fake)
    monkeypatch.setattr(forensics,'load_source',lambda entry,identity,ticks:(
        dict(joint_limits_rad=[[[0,1]]*29],action_control_mode='absolute_joint_position_target_rad'),[],dict(identity=identity)))
    constructors=[]
    def controller(*args,**kwargs):constructors.append(kwargs);return SimpleNamespace()
    monkeypatch.setattr(batched_controller,'BatchedResidualController',controller)
    def sequence(*args,**kwargs):
        assert kwargs['cuda'] is True
        return dict(passed=pass_sequence,max_abs_command_rad=0.,failure=None if pass_sequence else dict(message='injected'))
    monkeypatch.setattr(gate,'run_tail_sequence',sequence)
    output=tmp_path/'gate'
    if pass_sequence:result=gate.exact_tail_gate(root,root,output)
    else:
        with pytest.raises(RuntimeError,match='did not pass'):gate.exact_tail_gate(root,root,output)
        result=json.loads((output/'result.json').read_text())
    assert result['passed'] is pass_sequence
    for role in (('A','B') if pass_sequence else ('A',)):
        receipt=json.loads((output/role/'parity.json').read_text())
        assert receipt['passed'] is pass_sequence and receipt['role']==role
    assert all(c['neural_batch'] is True and c['vision_microbatch']==8 for c in constructors)
    assert 'vision_tail_profile' not in constructors[0] and constructors[1]['vision_tail_profile']==EXACT_TAIL_PROFILE
