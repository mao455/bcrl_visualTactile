import copy
import torch
from tools.state_mechanism_audit import WindowSampler
from visuotactile.anchor_sampling import labels,mask_totals
from visuotactile.position_control import overlap_transition,recorded_target_plan


def test_tiny_sampler_uses_multiple_windows_and_covers_tail_without_duplication():
    g=torch.Generator().manual_seed(31);s=WindowSampler({'one':300},g,16)
    a,b=s.draw(),s.draw()
    assert len(a)==16 and len(b)==9 and len(set(a+b))==25
    assert {t for _,t in a+b}==set(range(0,300,12))
    assert set(i for i,_ in a+b)=={'one'}
    saved=copy.deepcopy(s.state_dict());rng=g.get_state();expected=s.draw()
    other=WindowSampler({'one':300},torch.Generator(),16);other.load_state_dict(saved);other.generator.set_state(rng)
    assert other.draw()==expected


def test_mask_counts_are_targets_and_pairs_not_window_count():
    rec=dict(time=torch.arange(25),command=torch.arange(25*29).reshape(25,29).float())
    y,m=labels(rec,torch.tensor([0,12,24]),24)
    assert m.sum(1).tolist()==[24,13,1]
    assert mask_totals({'x':25},[('x',0),('x',12),('x',24)])==(38,35)
    assert torch.equal(y[2,0],rec['command'][-1]) and torch.count_nonzero(y[2,1:])==0


def test_oracle_chunk_commit_does_not_shift_or_repeat_expert_targets():
    target=torch.arange(185*29).reshape(185,29)
    issued=[]
    for t in range(185):
        if t%12==0:plan=recorded_target_plan(target,t)[0]
        issued.append(plan[t%12])
    assert torch.equal(torch.stack(issued),target)
    # C4 transition aligns future ticks12:16, without holding past tick11.
    before=target[:24].float()[None];current=target[12:36].float()[None]+4
    out=overlap_transition(before,current,12,4)
    assert torch.equal(out[0,:4]-target[12:16],torch.arange(1,5)[:,None].expand(4,29))
    assert torch.equal(out[:,4:],current[:,4:])
