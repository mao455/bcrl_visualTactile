import copy
import pytest
import torch

from visuotactile.recurrent_execution import profile, validate, load, sequences
from visuotactile.trajectory_batch import encode_trajectories
from test_trajectory_batch import model, episode


def test_packed_unequal_histories_keep_outputs_gradients_and_rng():
    torch.set_num_threads(2)
    torch.manual_seed(817)
    cell=torch.nn.GRUCell(7,11).double()
    packed_cell=copy.deepcopy(cell)
    inputs=[torch.randn(n,7,dtype=torch.float64,requires_grad=True) for n in [13,5,17]]
    other=[x.detach().clone().requires_grad_(True) for x in inputs]
    original=[]
    for x in inputs:
        h=None;rows=[]
        for row in x:
            h=cell(row[None],h);rows.append(h)
        original.append(torch.cat(rows))
    rng=torch.get_rng_state().clone()
    packed=sequences(packed_cell,other)
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in zip(original,packed):torch.testing.assert_close(a,b,atol=2e-14,rtol=2e-14)
    sum(x.square().mean() for x in original).backward()
    sum(x.square().mean() for x in packed).backward()
    for a,b in zip(cell.parameters(),packed_cell.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-14,rtol=2e-14)
    for a,b in zip(inputs,other):torch.testing.assert_close(a.grad,b.grad,atol=2e-14,rtol=2e-14)
    assert len(packed[1])==5  # No padded recurrent transitions become real states.


def test_complete_visual_training_path_and_one_adam_update_match(model):
    pairs=[episode(13,351),episode(25,352)];records,episodes=zip(*pairs)
    original=copy.deepcopy(model)
    model._offline_recurrent_execution=profile()
    assert set(original.state_dict())==set(model.state_dict())
    results=[];gradients=[]
    for m in [original,model]:
        m.train();opt=torch.optim.AdamW(m.parameters(),lr=1.5e-5,betas=(.95,.999),weight_decay=1e-6)
        values=encode_trajectories(m,records,episodes)
        sum(c.square().mean() for c,_ in values).div(2).backward()
        gradients.append({n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None})
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        results.append([(c.detach(),a) for c,a in values])
    for (a,ai),(b,bi) in zip(*results):
        assert torch.equal(ai,bi)
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    assert gradients[0].keys()==gradients[1].keys()
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=2e-5,rtol=2e-5)
    for a,b in zip(original.parameters(),model.parameters()):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


def test_packed_histories_remain_causal_and_independent(model):
    pairs=[episode(25,61),episode(13,62)];records,episodes=zip(*pairs)
    model._offline_recurrent_execution=profile();model.eval()
    with torch.no_grad():
        before=encode_trajectories(model,records,episodes)
        episodes[0].perception['patches'][13:].add_(.4)
        episodes[1].perception['patches'].add_(.8)
        after=encode_trajectories(model,records,episodes)
    torch.testing.assert_close(before[0][0][:2],after[0][0][:2],atol=1e-7,rtol=0)
    assert not torch.allclose(before[0][0][-1],after[0][0][-1],atol=1e-7,rtol=0)
    assert not torch.allclose(before[1][0],after[1][0],atol=1e-7,rtol=0)


def test_explicit_profile_is_required(tmp_path):
    assert validate(None) is None and load(tmp_path) is None
    assert validate(profile())==profile()
    with pytest.raises(ValueError):validate(dict(profile(),padding='run_padding'))
    with pytest.raises(ValueError):sequences(torch.nn.GRUCell(2,3),[])
