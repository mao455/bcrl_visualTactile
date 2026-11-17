import unittest
from types import SimpleNamespace
import torch
from .replay import acknowledge

class ActualHistory(unittest.TestCase):
 def test_shadow_output_never_becomes_real_history(self):
  c=SimpleNamespace(base=SimpleNamespace(command=torch.full((1,29),99.)))
  requested=torch.arange(29,dtype=torch.float32);applied=requested+.01;stamp=torch.tensor(1.2,dtype=torch.float64)
  acknowledge(c,requested,applied,stamp)
  requested.zero_();applied.zero_();stamp.zero_()
  r,a,t=c.base.position_previous
  self.assertTrue(torch.equal(r,torch.arange(29,dtype=torch.float32)))
  self.assertTrue(torch.equal(a,torch.arange(29,dtype=torch.float32)+.01))
  self.assertEqual(t.item(),1.2)
  self.assertTrue(torch.equal(c.base.command,torch.full((1,29),99.)))

if __name__=='__main__':unittest.main()
