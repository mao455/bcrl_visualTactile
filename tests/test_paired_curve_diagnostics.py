"""CPU checks: diagnostics consume detached real predictions without training effects."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from visuotactile.paired_curve_diagnostics import Recorder,profile,validate,load,export_dp_visit_losses


class CurveChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.data=dict(entries={'a':dict(family='train_a'),'b':dict(family='validation_b')},
            splits=dict(train=['a'],validation=['b'],test=[]))
        self.labels={'a':torch.tensor([0,1,2,4]),'b':torch.tensor([2,4,5,6])}
        self.recorder=Recorder(self.root,self.data,component='Transformer',role='A',action_scale=torch.ones(29),labels=self.labels)
    def tearDown(self):self.temporary.cleanup()

    def test_absent_default_and_exact_optin(self):
        self.assertIsNone(load(self.root));self.assertIsNone(validate(None))
        (self.root/'family_phase_diagnostics.json').write_text(json.dumps(profile()))
        self.assertEqual(load(self.root),profile())
        broken=profile();broken['affects_sampling']=True
        with self.assertRaises(ValueError):validate(broken)

    def test_detached_training_phase_sums_legal_only_and_no_rng(self):
        before=torch.get_rng_state().clone()
        p=torch.ones(1,4,29,requires_grad=True);target=torch.zeros_like(p)
        self.recorder.residual_chunk(['a'],0,p,target,torch.tensor([[False,True,True,False]]))
        self.assertIsNone(p.grad)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(self.recorder.log.exists())
        self.recorder.commit(1,1)
        rows=self.recorder.export(1)
        all_row=next(r for r in rows if r['family']=='__all__' and r['phase']=='__all__')
        self.assertEqual(all_row['samples'],2);self.assertEqual(all_row['arm_rmse_rad'],1.)
        self.assertEqual(all_row['normalized_equal_mse'],1.)
        self.assertEqual({r['phase'] for r in rows if r['phase']!='__all__'}, {'contact_not_held','held_far'})

    def test_successful_commit_and_resume_latest_event_wins(self):
        self.recorder.observe('a',torch.tensor([0]),torch.ones(1,29));self.recorder.commit(1,1)
        self.recorder.observe('a',torch.tensor([0]),torch.full((1,29),2.));self.recorder.commit(1,1)
        self.recorder.observe('a',torch.tensor([1]),torch.full((1,29),99.));self.recorder.commit(2,2)
        rows=self.recorder.export(1)
        row=next(r for r in rows if r['family']=='__all__' and r['phase']=='__all__')
        self.assertEqual(row['samples'],1);self.assertEqual(row['arm_rmse_rad'],2.)
        self.assertEqual({r['epoch'] for r in rows},{1})

    def test_validation_is_existing_prediction_no_forwards(self):
        class Bomb:
            def __call__(self,*a,**k):raise AssertionError('Extra forward')
        record=dict(target=torch.zeros(4,29),supervision_mask=torch.tensor([False,True,True,False]))
        prediction=torch.ones(4,29)
        self.recorder.residual_evaluation(1,1,'validation',['b'],{'b':record},[prediction])
        rows=self.recorder.export(1,plot=True)
        self.assertTrue((self.root/'family_phase_curves.png').exists())
        self.assertEqual(next(r for r in rows if r['family']=='__all__' and r['phase']=='__all__')['samples'],2)

    def test_dp_uses_actual_anchors_H24_validity_not_padding(self):
        class Model:
            horizon=24;action_scale=torch.ones(29);action_offset=torch.zeros(29)
        valid=torch.zeros(1,24,dtype=torch.bool);valid[0,:2]=True
        values=[dict(anchors=torch.tensor([1]),truth=torch.zeros(1,24,29),valid=valid)]
        pred=torch.full((1,24,29),3.,requires_grad=True)
        dp=Recorder(self.root/'dp',self.data,component='DP',role='A',action_scale=Model.action_scale,labels=self.labels)
        dp.dp_training(2,7,['a'],values,pred,Model)
        row=next(r for r in dp.export(7) if r['family']=='__all__' and r['phase']=='__all__')
        self.assertEqual(row['samples'],2);self.assertEqual(row['arm_rmse_rad'],3.)
        self.assertEqual(row['target'],'q_reached');self.assertIsNone(pred.grad)

    def test_existing_visit_losses_no_fake_phase_decomposition(self):
        (self.root/'learning.jsonl').write_text(json.dumps(dict(epoch=1,visits=[dict(episode='a',loss=2.),dict(episode='a',loss=4.)]))+'\n')
        rows=export_dp_visit_losses(self.root,self.data)
        self.assertEqual(rows[0]['mean_actual_objective_loss'],3.)
        self.assertEqual(rows[0]['phase'],'whole_episode_mixed_phases')


if __name__=='__main__':unittest.main()


def test_optin_actual_DP_AdamW_EMA_noise_and_forwards_unchanged(tmp_path,monkeypatch):
    from test_full_trajectory_training import TinyNative,Objective,make_records,install_small_vision
    from test_paired_episode_io import equal_tree
    from visuotactile.trajectory_training import execute_update
    from visuotactile.models.vendor.ema_model import EMAModel
    install_small_vision(monkeypatch);torch.manual_seed(42)
    records,episodes=make_records();model=TinyNative();reference=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.001);refopt=torch.optim.AdamW(reference.parameters(),lr=.001)
    averaged=copy.deepcopy(model);refavg=copy.deepcopy(reference)
    ema=EMAModel(averaged,power=.75);refema=EMAModel(refavg,power=.75)
    noise=torch.Generator().manual_seed(22);refnoise=torch.Generator().manual_seed(22)
    ids=['x','y'];data=dict(entries={i:dict(family='family_'+i) for i in ids},splits=dict(train=ids,validation=[],test=[]))
    recorder=Recorder(tmp_path,data,component='DP',role='B',action_scale=model.action_scale,
        labels={i:torch.zeros(len(records[i]['time']),dtype=torch.long) for i in ids})
    calls=[];refcalls=[]
    hook=model.denoiser.register_forward_pre_hook(lambda *a:calls.append(1))
    refhook=reference.denoiser.register_forward_pre_hook(lambda *a:refcalls.append(1))
    before=torch.get_rng_state().clone()
    actual=execute_update(model,opt,ema,Objective(),ids,[records[i] for i in ids],[episodes[i] for i in ids],noise,
        diagnostics=recorder,diagnostic_epoch=1,diagnostic_update=1)
    assert torch.equal(before,torch.get_rng_state())
    expected=execute_update(reference,refopt,refema,Objective(),ids,[records[i] for i in ids],[episodes[i] for i in ids],refnoise)
    hook.remove();refhook.remove()
    assert calls==refcalls==[1]
    assert actual==expected
    equal_tree(model.state_dict(),reference.state_dict());equal_tree(opt.state_dict(),refopt.state_dict())
    equal_tree(averaged.state_dict(),refavg.state_dict());assert torch.equal(noise.get_state(),refnoise.get_state())
    assert recorder.export(1)
