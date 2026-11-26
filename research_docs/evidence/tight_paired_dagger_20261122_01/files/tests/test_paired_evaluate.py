"""Meaningful protocol checks without Kit, GPU, weights, or new real plans."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import tight_paired_evaluate as evaluation


class PairedEvaluationTests(unittest.TestCase):
    def test_wilson_zero_does_not_prove_zero_and_full_denominator(self):
        self.assertEqual(evaluation.wilson(0,64)[0],0)
        self.assertGreater(evaluation.wilson(0,64)[1],.05)
        self.assertLess(evaluation.wilson(32,64)[0],.5)
        self.assertGreater(evaluation.wilson(32,64)[1],.5)

    def test_paired_contrast_retained_lost_and_gained(self):
        result=evaluation.contrast([1,1,0,0],[1,0,1,0])
        self.assertEqual([result[k] for k in ('both_success','gained','lost','both_failure')],[1]*4)
        self.assertEqual(result['paired_difference_pp'],0)
        self.assertLess(result['paired_bootstrap95_pp'][0],0)
        self.assertGreater(result['paired_bootstrap95_pp'][1],0)
        self.assertEqual(evaluation.contrast([1]*64,[0]*64)['paired_bootstrap95_pp'],[100,100])

    def test_held12_requires_lift_and_both_physics_substeps(self):
        obj=np.zeros((80,7));obj[:,2]=.63;obj[52:,2]+=.03
        force=np.zeros((80,2,5,3));force[:,:,:2,2]=.11
        key=np.full(80,.02);goal=np.zeros(80)
        value=evaluation.geometry_metrics(obj,force,key,goal)
        self.assertTrue(value['stable_hold']);self.assertEqual(value['first_stable_hold_tick'],63)
        force[60,1,1,2]=0
        self.assertEqual(evaluation.geometry_metrics(obj,force,key,goal)['first_stable_hold_tick'],72)
        force[:]=0
        self.assertFalse(evaluation.geometry_metrics(obj,force,key,goal)['stable_hold'])

    def test_first_gate_distinguishes_gate_exit_and_later_native_goal(self):
        obj=np.zeros((80,7));force=np.zeros((80,2,5,3));key=np.full(80,.03);goal=np.zeros(80)
        key[52:56]=.014;key[68:]=.006;goal[65:]=1
        value=evaluation.geometry_metrics(obj,force,key,goal)
        self.assertTrue(value['first_15mm_gate']);self.assertEqual(value['first_15mm_gate_tick'],52)
        self.assertTrue(value['left_15mm_gate_after_entry'])
        goal[:]=1
        self.assertFalse(evaluation.geometry_metrics(obj,force,key,goal)['first_15mm_gate'])

    def test_illegal_or_missing_numeric_retains_conservative_funnel(self):
        with tempfile.TemporaryDirectory() as td:
            value=evaluation.numeric_metrics(Path(td))
            self.assertFalse(value['geometry_observed']);self.assertFalse(value['stable_hold'])
            self.assertFalse(value['first_15mm_gate'])

    def test_wave_rejects_tampering_and_expert_privileges(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);wave=root/'wave00';wave.mkdir()
            pairs={r:{} for r in evaluation.ROLES}
            plan=dict(resource_root=str(root),evaluation_contract=evaluation.CONTRACT,kind='autonomous',
                expert_permission=False,pairs=pairs,batch=dict(lanes=4,vision_microbatch=2),record_sampler_rng=True,inference_scheduler_index_mode='host',cases=[dict(role=r,spec=dict(seed=1)) for r in evaluation.ROLES])
            def install():
                (wave/'plan.json').write_text(json.dumps(plan))
                freeze=dict(version=evaluation.CONTRACT,pairs=pairs,wave_plan_sha256={str(wave):evaluation.sha256(wave/'plan.json')})
                (root/'registration.json').write_text(json.dumps(freeze))
            install();evaluation.validate_wave(wave)
            plan['cases'][0]['dagger']={'takeover':True};install()
            with self.assertRaisesRegex(ValueError,'No expert'):evaluation.validate_wave(wave)
            del plan['cases'][0]['dagger'];install()
            plan['expert_permission']=True;(wave/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'binding'):evaluation.validate_wave(wave)

    def test_original_cohort_seal_and_cross_split_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cohorts=dict(train=[dict(seed=1,family='train1',narrow_reset={'x':1})],
                development=[dict(seed=2,family='dev2',narrow_reset={'x':2})],
                test=[dict(seed=3,family='test3',narrow_reset={'x':3})])
            path=root/'cohorts.json';path.write_text(json.dumps(cohorts))
            (root/'registration.json').write_text(json.dumps(dict(cohorts_sha256=evaluation.sha256(path))))
            evaluation.validate_cohorts(root)
            cohorts['test'][0]['seed']=4;path.write_text(json.dumps(cohorts))
            with self.assertRaisesRegex(ValueError,'SHA'):evaluation.validate_cohorts(root)
            (root/'registration.json').unlink()
            cohorts['test'][0]['narrow_reset']={'x':1};path.write_text(json.dumps(cohorts))
            with self.assertRaisesRegex(ValueError,'overlap'):evaluation.validate_cohorts(root)

    def test_no_expert_network_or_action_and_scope_restored(self):
        import torch
        from visuotactile import expert,queued_rollout
        original_factory=expert.create_expert; original_actor=queued_rollout.expert_actor_input
        cuda_was_initialized=torch.cuda.is_initialized()
        with tempfile.TemporaryDirectory() as td:
            with evaluation.no_expert_scope(Path(td),4):
                player,_=expert.create_expert(None,'not_loaded')
                player.reset();self.assertFalse(player.is_rnn)
                with self.assertRaisesRegex(RuntimeError,'forbidden'):player.get_action()
                actor=queued_rollout.expert_actor_input(None,player)
                self.assertEqual(tuple(actor.shape),(4,1))
                self.assertFalse(evaluation.read(Path(td)/'expert_disabled.json')['network_loaded'])
            self.assertIs(expert.create_expert,original_factory)
            self.assertIs(queued_rollout.expert_actor_input,original_actor)
        self.assertEqual(torch.cuda.is_initialized(),cuda_was_initialized)

    def test_fake_freeze_is_balanced_without_preparing_real_cohort(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fake=root/'model';fake.write_text('model')
            digest=evaluation.sha256(fake)
            pairs={r:dict(version='complete_student_policy_v1',controller='ResidualController',
                supervision='q_reached' if r.startswith('A') else 'u_expert',
                frontend=str(fake),frontend_sha256=digest,backend=str(fake),backend_sha256=digest) for r in evaluation.ROLES}
            session=dict(dino_checkpoint=str(fake),dino_sha256=digest,deadline=evaluation.time.time()+600,layout={},visual_arithmetic={},expert_sha256='no_network')
            (root/'session.json').write_text(json.dumps(session));(root/'parent_pairs.json').write_text('{}')
            cohort=dict(development=[dict(seed=s,family=str(s),narrow_reset={'fixed_start_pose':[s]}) for s in range(28911000,28911008)])
            (root/'cohorts.json').write_text(json.dumps(cohort))
            template=dict(batch={},execution_backend={})
            def fake_session_at(path):
                (path/'session.json').write_text(json.dumps(session));return session
            with patch.object(evaluation,'session_at',fake_session_at):
                result=evaluation.prepare('development',root=root,pairs=pairs,template=template,job_path=root/'job.json')
            self.assertEqual(result['episodes'],32)
            summary=evaluation.summarize(Path(result['output']))
            self.assertTrue(summary['full_denominator_preserved'])
            self.assertEqual(summary['episodes'],32)
            self.assertEqual(summary['absolute_success'][0]['missing'],8)
            self.assertEqual(summary['absolute_success'][0]['n'],8)
            self.assertEqual(summary['absolute_success'][0]['success'],0)
            wave=Path(result['output'])/'wave00'
            plan=json.loads((wave/'plan.json').read_text());case=plan['cases'][0]
            directory=wave/case['name'];directory.mkdir()
            (directory/'result.json').write_text(json.dumps(dict(legal=True,software_complete=True,success=True)))
            summary=evaluation.summarize(Path(result['output']))
            self.assertEqual(summary['absolute_success'][0]['success'],0)
            self.assertEqual(summary['absolute_success'][0]['protocol_unverified'],1)
            self.assertFalse(summary['primary_protocol_verified'])
            positions={r:[0]*4 for r in evaluation.ROLES}
            for path in sorted((root/'evaluation/development').glob('wave*/plan.json')):
                plan=json.loads(path.read_text());self.assertFalse(plan['expert_permission'])
                for lane,case in enumerate(plan['cases']):positions[case['role']][lane]+=1
                self.assertEqual(plan['batch']['vision_microbatch'],2)
                self.assertFalse(plan['test_used'])
            self.assertTrue(all(counts==[2]*4 for counts in positions.values()))
            with self.assertRaises(FileExistsError):evaluation.prepare('development',root=root,pairs=pairs,template=template)


if __name__=='__main__':unittest.main()
