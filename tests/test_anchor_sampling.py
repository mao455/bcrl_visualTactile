import copy
import pytest
import torch

from visuotactile.anchor_sampling import AnchorSampler,weighted_parts,continuation_lr
from visuotactile.dp_objectives import masked_objectives


def test_episode_diversity_anchor_coverage_and_exact_restart():
    lengths={str(i):119+i%80 for i in range(320)}
    g=torch.Generator().manual_seed(19);a=AnchorSampler(lengths,g)
    first=[a.draw() for _ in range(5)]
    assert len({i for batch in first for i,t in batch})==320
    assert all(len({i for i,t in batch})==64 and all(t%12==0 and t<lengths[i] for i,t in batch) for batch in first)
    state=copy.deepcopy(a.state_dict());rng=g.get_state()
    expected=[a.draw() for _ in range(90)]
    b=AnchorSampler(lengths,torch.Generator());b.generator.set_state(rng);b.load_state_dict(state)
    assert [b.draw() for _ in range(90)]==expected
    assert a.coverage()==b.coverage()
    assert a.coverage()['unique_anchors']==sum((n+11)//12 for n in lengths.values())
    assert a.coverage()['per_episode_selection_min']==a.coverage()['per_episode_selection_max']==19


def test_accumulation_matches_global_masked_loss_and_gradient_at_tails():
    torch.manual_seed(12);prediction=torch.randn(8,24,29,requires_grad=True);truth=torch.randn_like(prediction)
    valid=torch.arange(24)[None]<torch.tensor([24,24,13,5,1,24,8,19])[:,None]
    weights={'temporal':.17}
    def parts(p,y,m):
        return dict(base=masked_objectives(p,y,m)['B'],temporal=masked_objectives(p[:,1:]-p[:,:-1],y[:,1:]-y[:,:-1],m[:,1:]&m[:,:-1])['B'])
    values=parts(prediction,truth,valid);full=values['base']+.17*values['temporal'];full.backward()
    gradient=prediction.grad.clone();p=prediction.detach().clone().requires_grad_()
    totals=(int(valid.sum()),int((valid[:,1:]&valid[:,:-1]).sum()))
    summed=0
    for lo,hi in ((0,3),(3,5),(5,8)):
        loss,_=weighted_parts(parts(p[lo:hi],truth[lo:hi],valid[lo:hi]),valid[lo:hi],totals,weights)
        loss.backward();summed+=float(loss.detach())
    assert summed==pytest.approx(float(full.detach()),rel=2e-7)
    torch.testing.assert_close(p.grad,gradient,atol=2e-9,rtol=2e-6)
    assert torch.count_nonzero(p.grad[~valid])==0


def test_scheduler_is_nonzero_continuous_and_does_not_restart_at_milestones():
    values=[continuation_lr(i,23918) for i in (1,199,200,201,2250,6584,10918,15251,23918)]
    assert values[0]==values[-1]==3e-6
    assert values[2]==pytest.approx(3e-5,rel=1e-14) and all(x>0 for x in values)
    assert abs(values[2]-values[3])<1e-10
    assert all(b<a for a,b in zip(values[2:],values[3:]))
    with pytest.raises(ValueError):continuation_lr(0,23918)


def test_coverage_does_not_call_sampled_windows_full_segment_or_episode_epochs():
    sampler=AnchorSampler({str(i):120 for i in range(320)},torch.Generator().manual_seed(2),32)
    sampler.draw();c=sampler.coverage()
    assert c['sampled_anchors']==32 and c['anchor_equivalent_epochs']==.01
    assert c['episode_selection_passes']==.1 and c['full_segment_traversals']==0
    assert c['gradient_ticks']<=32*96


def test_status_uses_actual_progress_not_stale_zero(tmp_path):
    from tools.anchor_continue_queue import Owner
    from tools.state_image_queue import write,read
    write(tmp_path/'session.json',{})
    write(tmp_path/'status.json',dict(state='running',training_updates=0))
    write(tmp_path/'state_train/progress.json',dict(update=10050,continuation_update=50))
    owner=Owner(tmp_path);owner.status(state='supervised_ready',pid=None)
    saved=read(tmp_path/'status.json')
    assert 'training_updates' not in saved and saved['pid'] is None
    assert saved['model_progress']['state']['update']==10050
    assert saved['model_progress']['image']['continuation_update']==0


def test_committed_worker_result_recovers_without_new_process(tmp_path,monkeypatch):
    from tools.anchor_continue_queue import Owner
    from tools.state_image_queue import write,read
    write(tmp_path/'session.json',{});write(tmp_path/'status.json',{})
    output=tmp_path/'stages/state_contract/attempt_1/output';result=dict(passed=True,optimizer_updates=0)
    write(output/'result.json',result)
    previous=dict(name='state_contract',valid=True,resources_released=True,output=str(output),result=result)
    write(tmp_path/'attempts.json',[previous])
    def unexpected(*a,**kw):raise AssertionError('Must not start duplicate worker')
    monkeypatch.setattr('subprocess.Popen',unexpected)
    assert Owner(tmp_path).stage('state_contract','unused',[])==previous
    assert read(tmp_path/'stages/state_contract/committed.json')==previous
