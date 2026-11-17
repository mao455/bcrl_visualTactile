import torch
from visuotactile.anchor_sampling import grasp_sampling_profile,onset_sampling_profile,sample_training_anchors,labels
from visuotactile.fresh_visit_sampling import visit

def record(n=300,first=0):
 return dict(time=torch.arange(n)/60,command=torch.zeros(n,29),supervision_mask=torch.arange(n)>=first,control_segment=torch.zeros(n,dtype=torch.long))

def test_original_anchors_exactly_preserved():
 r=record()
 for epoch in range(1,8):
  a,_=sample_training_anchors(r,grasp_sampling_profile(),identity='original',epoch=epoch,visit_index=1)
  b,_=sample_training_anchors(r,onset_sampling_profile(),identity='original',epoch=epoch,visit_index=1)
  assert torch.equal(a,b)

def test_onset_legal_unique_and_suffix_retained():
 r=record(first=203)
 for epoch in range(1,8):
  a,x=sample_training_anchors(r,onset_sampling_profile(),identity='late',epoch=epoch,visit_index=1)
  assert (a>=203).all() and len(a)==len(a.unique())
  assert x['priority_selected']==(len(a)+1)//2
  assert (a>=227).any()
  _,mask=labels(r,a,24)
  assert mask.any()

def test_short_suffix_without_replacement():
 r=record(n=210,first=203)
 a,x=sample_training_anchors(r,onset_sampling_profile(),identity='short',epoch=1,visit_index=1)
 assert (a>=203).all() and (a<210).all() and len(a)==len(a.unique())
