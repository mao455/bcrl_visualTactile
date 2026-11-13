"""A real round-shaped fixture exercises distinct supervision, exposure and opportunities."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.tight_ab_fairness import audit


class FairnessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        stats=self.root/'commands.pt';stats.write_bytes(b'shared frozen train-only command affine')
        dp=dict(seed=17,target=514,effective_batch=64,microbatch=16,lr_peak=1e-5,lr_floor=1e-6,
                warmup=100,sampler='stratified',incremental_schedule='fresh',selection='dev',
                diagnostic_cohorts=['core','correction'],dino_sha256='frozen',
                target_window_contract='masked12',training_performance='same',
                sampling_weights=[.5,.2,.3],recovery_prefix=48,correction_dev_ids=['dev'])
        rear=dict(initialization_seed=17,microbatch=4,learning_rate=1e-4,weight_decay=1e-4,
                  maximum_control_tick_epochs=1,minimum_control_tick_epochs=1,validation_epoch_interval=1,
                  selection='same',loss_contract='fy',output_contract='linear',history_contract='past',
                  execution_adapter='shared',tick_window_contract=128,supervision_mask_sha256='mask',
                  sampling_strata={'core':['train']},sampling_weights=[.5,.2,.3],recovery_prefix=48,
                  architecture='transformer',residual_statistics_path=str(stats),
                  residual_statistics_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
                  residual_normalization_identity='u_expert_train166_affine_v1',cache_definition='held',
                  residual_hz=60,horizon=24,commit=12,tbptt_ticks=32,teacher_supervision='u_expert')
        plan=[dict(identity='case',spec=dict(identity='case',seed=18,narrow_reset={'x':.1}))]
        for group,label,front_stats in [('QG01','q_reached','reached'),('G01','u_expert','command')]:
            self.save(group+'/dp/training/configuration.json',dict(dp,teacher_supervision=label,statistics_sha256=front_stats))
            cache = dict(base_sha256=group, data_sha256=group+'data', statistics_sha256=front_stats,
                sampler={'steps':16,'seed':7091,'commit':12}, inputs='held causal plan',
                supervision_mask_sha256='mask', residual_statistics_sha256=rear['residual_statistics_sha256'],
                residual_normalization_identity=rear['residual_normalization_identity'])
            cache_name = group+'/backend/'+group+'/nominal/index.json'
            self.save(cache_name,cache)
            self.save(group+'/backend/'+group+'/training/configuration.json',dict(rear,statistics_sha256=front_stats,
                base_sha256=group, data_sha256=group+'data', base_checkpoint=str(self.root/group/'best.pt'),
                cache_index_sha256=hashlib.sha256((self.root/cache_name).read_bytes()).hexdigest()))
            self.save(group+'/dp/state_data/data.json',dict(splits={'train':['train'],'validation':['dev']},sampling_strata={'core':['train']},snapshot_id='same'))
            self.save(group+'/backend/evaluation_plan.json',plan)
            self.save(group+'/dp/training/finished.json',dict(best_checkpoint=str(self.root/group/'best.pt'),updates=514,supervised_target_ticks=765303,coverage={'train':32},target_reached=True))
            self.save(group+'/backend/'+group+'/training/finished.json',dict(updates=1362,supervised_ticks=193581,coverage={'train':88},target_reached=True))
            self.save(group+'/backend/'+group+'/evaluation_summary.json',dict(completed=1,valid=1,successes=int(group=='G01'),complete=True,software_failures=0,expert_takeover=False,formal_test22_used=False,results=[dict(identity='case',seed=18,narrow_reset={'x':.1})]))
            self.save(group+'/backend/evaluation_complete.json',dict(complete=True))

    def save(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))

    def change(self,name,key,value):
        data=json.loads((self.root/name).read_text());data[key]=value;self.save(name,data)

    def test_distinct_frontends_and_successes_are_legal(self):
        self.assertTrue(audit(self.root)['matched_increment_passed'])
        self.assertTrue(audit(self.root,complete=False)['matched_increment_passed'])

    def frozen_fixture(self):
        frozen = {}
        for group in ('QG01', 'G01'):
            checkpoint = self.root / group / 'best.pt'
            checkpoint.write_bytes(group.encode())
            hashed = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            rear_path = group + '/backend/' + group + '/training/configuration.json'
            rear = json.loads((self.root / rear_path).read_text())
            rear.update(base_sha256=hashed, continuation_from=dict(path=str(checkpoint), sha256=hashed),
                continuation_optimizer_policy='fresh', evaluation_selection_policy='endpoint')
            cache_path = group + '/backend/' + group + '/nominal/index.json'
            self.change(cache_path, 'base_sha256', hashed)
            rear['cache_index_sha256'] = hashlib.sha256((self.root / cache_path).read_bytes()).hexdigest()
            self.save(rear_path, rear)
            frozen[group] = dict(checkpoint=str(checkpoint), sha256=hashed, new_optimizer_updates=0)
        self.save('frozen_frontends.json', frozen)

    def test_frozen_frontend_reports_zero_new_dp_exposure(self):
        self.frozen_fixture()
        result = audit(self.root, frozen_frontend=True)
        self.assertTrue(result['matched_increment_passed'])
        self.assertEqual(result['actual']['G01']['dp_updates'], 0)
        self.assertEqual(result['actual']['QG01']['dp_target_slots'], 0)

    def test_frozen_frontend_checks_actual_checkpoint(self):
        self.frozen_fixture()
        (self.root / 'G01/best.pt').write_bytes(b'drift')
        self.assertIn('G01.frozen_frontend', audit(self.root, frozen_frontend=True)['failed'])

    def test_frozen_prepared_contract_is_not_completed_evidence(self):
        self.frozen_fixture()
        for group in ('QG01', 'G01'):
            directory = self.root / group / 'backend' / group
            (directory / 'training/configuration.json').rename(directory / 'prepared_training_contract.json')
        self.assertTrue(audit(self.root, complete=False, frozen_frontend=True)['matched_increment_passed'])
        with self.assertRaises(FileNotFoundError):
            audit(self.root, complete=True, frozen_frontend=True)

    def test_equal_updates_cannot_hide_unequal_exposure(self):
        self.change('G01/backend/G01/training/finished.json','supervised_ticks',193580)
        self.assertIn('rear_done.supervised_ticks',audit(self.root)['failed'])

    def test_equal_total_cannot_hide_different_coverage(self):
        self.change('G01/dp/training/finished.json','coverage',{'train':31,'other':1})
        self.assertIn('dp_done.coverage',audit(self.root)['failed'])

    def test_actual_eval_seed_must_equal_plan(self):
        self.change('G01/backend/G01/evaluation_summary.json','results',[dict(identity='case',seed=99,narrow_reset={'x':.1})])
        self.assertIn('G01.actual_opportunities',audit(self.root)['failed'])

    def test_missing_completed_evidence_fails(self):
        (self.root/'G01/dp/training/finished.json').unlink()
        with self.assertRaises(FileNotFoundError):audit(self.root)

    def test_semantics_cannot_be_swapped_for_equality(self):
        self.change('QG01/dp/training/configuration.json','teacher_supervision','u_expert')
        self.assertIn('labels.intentional_semantics',audit(self.root)['failed'])

    def test_stale_cache_is_rejected(self):
        self.change('G01/backend/G01/nominal/index.json','base_sha256','wrong-front')
        self.assertIn('G01.cache_manifest_binding',audit(self.root)['failed'])

    def test_command_statistics_contents_are_verified(self):
        (self.root/'commands.pt').write_bytes(b'modified')
        self.assertIn('G01.actual_command_statistics',audit(self.root)['failed'])


if __name__=='__main__':unittest.main()
