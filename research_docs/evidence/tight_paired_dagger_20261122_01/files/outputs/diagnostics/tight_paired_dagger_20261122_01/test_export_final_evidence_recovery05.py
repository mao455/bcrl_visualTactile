"""CPU-only isolated fixture checks; never opens actual experiment payloads."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('export_final',Path(__file__).with_name('export_final_evidence_recovery05.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ExportFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.project=Path(self.temp.name)/'project';self.project.mkdir()
        self.run=self.project/m.RUN_REL;self.audit=self.project/m.AUDIT_REL;self.assembly=self.audit/'assembled'
        self.assembly.mkdir(parents=True);self.dest=self.project/'research_docs/evidence/testfinal'
    def tearDown(self):
        # Immutable output belongs only to this temporary fixture.
        for p in sorted(self.project.rglob('*'),key=lambda x:len(x.parts)):
            if p.is_dir():p.chmod(0o755)
        self.temp.cleanup()
    def write(self,p,d):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d) if not isinstance(d,str) else d)
    def csv(self,p,rows):
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def complete(self):
        self.write(self.project/'tools/base.py','pass\n')
        registration=dict(cohort='test',n=64,seeds=list(range(64)),source_sha256={'tools/base.py':hashlib.sha256((self.project/'tools/base.py').read_bytes()).hexdigest()})
        rp=self.run/'evaluation/test/registration.json';self.write(rp,registration)
        summary=dict(cohort='test',n=64,episodes=256,full_denominator_preserved=True,
            registration_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
            absolute_success=[dict(role=r,n=64,success=0,success_rate=0.) for r in m.ROLES])
        pp=self.run/'evaluation/test/summary/per_initial_state.csv';rr=self.run/'evaluation/test/summary/rollouts.csv';sp=pp.with_name('primary_summary.json')
        self.write(sp,summary)
        self.csv(pp,[dict(seed=i,family='f'+str(i),**{r+'_success':False for r in m.ROLES}) for i in range(64)])
        self.csv(rr,[dict(seed=i,role=r,success=False,software_complete=False) for i in range(64) for r in m.ROLES])
        self.write(self.run/'cycle_logs/summarize_test_exit.json',dict(returncode=0))
        self.write(self.assembly/'DRAFT_REPORT.md','Draft reviewed only by assembler, needs root review')
        self.closed_receipt_fixture()
        self.rebind()
        self.ledger_fixture()
    def closed_receipt_fixture(self):
        self.write(self.audit/'driver_main_recovery05/status.json',dict(stage='main_stages_complete',training_or_evaluation_reruns=0))
        self.write(self.audit/'driver_main_recovery05/storage_gate.json',dict(ready=True))
        self.write(self.audit/'wait_main_recovery05/status.json',dict(stage='reviewed_main_closed',collection_retries=0))
        self.write(self.audit/'wait_main_recovery05/reviewed_main_exit.json',dict(returncode=0,owned_helper_returned_after_cleanup=True))
        for stage in m.RECOVERY05_STAGE_NAMES:
            self.write(self.run/'cycle_logs'/(stage+'_exit.json'),dict(returncode=0))
        for relative in m.CLOSED_OWNER_PATHS:
            owner=self.run/relative
            self.write(owner/'project_job_started.json',dict(job=dict(resource_root=str(owner),stages=[dict(entry='fixture')]),commands=[['fixture_only']]))
            self.write(owner/'project_complete.json',dict(stages=1))
            self.write(owner/'project_exit.json',dict(children_reaped=True))
        observed=dict(passed=47,unverified=1,failed=0,all_verified=False)
        self.write(self.audit/'collection_integrity_recovery05_actual01/integrity.json',
            dict(version='paired_collection_integrity_recovery05_v1',passed=True,opportunities=96,
                paired_opportunities=48,observable_pair_checks=observed,full48_observed_initial_states_verified=False,
                cuda_initialized=False,read_only_inputs=True))
        for relative,cuda in [('collection_integrity_recovery05_actual01/noise_audit.json',False),
                              ('collection_noise_recovery05_actual01/noise_audit.json',True)]:
            self.write(self.audit/relative,dict(version='paired_collection_native_noise_recovery05_v1',
                assignments=96,paired_assignments=48,observable_pair_checks=observed,sampler_seed=7091,
                per_native_plan_draw_shape=[1,24,29],commit_control_ticks=12,cuda_used=cuda,
                rows=[dict(case='case'+str(i)) for i in range(95)],
                unavailable=[dict(case='case95',retained_in_denominator=True)],pairs=[{}]*48,
                cpu_receipt_checks_passed=95,uncertified_execution_endpoints=1,
                exact_cuda_full_endpoint_checks=95 if cuda else 0,
                student_prefix_cuda_reconstructed=95 if cuda else 0,
                observed_takeover_rng_states=0,independent_observed_takeover_rng_claim=False,
                all96_execution_endpoints_certified=False))
        for name in m.RECOVERY05_REVIEW_NAMES:
            self.write(self.audit/name,'# Synthetic review\n' if name.endswith('.md') else dict(status='fixture_only'))
    def ledger_fixture(self):
        self.ledgers=self.audit/'selected_ledgers';actual={};files={}
        for scope in ('common180_parents','learning'):
            actual[scope]={}
            for role in 'AB':
                actual[scope][role]={}
                for component in ('frontend','backend'):
                    episode=dict(family='family',source_sha256='c'*64,complete_history_ticks=5,legal_ticks=[2,3,4],successful_optimizer_steps=[60,60,60],visits=60,mask_sha256=hashlib.sha256(bytes([0,0,1,1,1])).hexdigest())
                    value=dict(scope=scope,role=role,component=component,epoch=60,updates=120,checkpoint_sha256='a'*64,minimum_tick_steps=60,maximum_tick_steps=60,below50_ticks=0,ancestors_counted=False,duplicate_within_step='count_once',episodes={'episode':episode},trajectories=1,legal_ticks=3,native_entry_rows=None,base_sha256='a'*64,data_sha256='d'*64,train_mask_sha256='e'*64)
                    if component=='frontend':value['native_entry_rows']=[dict(identity='episode',native_anchor=0,slot=t,absolute_tick=t,successful_optimizer_steps=60) for t in [2,3,4]]
                    filename=f'{scope}_{role}_{component}.json';self.write(self.ledgers/filename,value)
                    files[filename]=dict(bytes=(self.ledgers/filename).stat().st_size,sha256=hashlib.sha256((self.ledgers/filename).read_bytes()).hexdigest())
                    actual[scope][role][component]=value
        self.write(self.assembly/'actual_exposure_and_age.json',actual)
        summary=dict(status='passed',checkpoint_reads=8,cuda_initialized=False,all_selected_epoch=60,all_legal_ticks_at_least50=True,all_native_entry_slots_at_least50=True,paired_actual_updates_equal=True,own_frontend_rear_mask_and_base_binding=True,files=files)
        self.write(self.ledgers/'selected_ledger_summary.json',summary)
        self.write(self.ledgers/'extraction_status.json',dict(**summary,root=str(self.run),output=str(self.ledgers),raw_data_opened=False,model_forward_calls=0))
    def rebind_ledger(self,filename):
        sp=self.ledgers/'selected_ledger_summary.json';s=json.loads(sp.read_text());p=self.ledgers/filename
        s['files'][filename]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest());self.write(sp,s)
        status=self.ledgers/'extraction_status.json';d=json.loads(status.read_text());d['files']=s['files'];self.write(status,d)
    def rebind(self):
        paths={'final_summary':self.run/'evaluation/test/summary/primary_summary.json',
            'final_registration':self.run/'evaluation/test/registration.json',
            'per_initial_state.csv':self.run/'evaluation/test/summary/per_initial_state.csv',
            'rollouts.csv':self.run/'evaluation/test/summary/rollouts.csv'}
        status=dict(status='draft_ready_for_root_review',inputs={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()})
        self.write(self.assembly/'assembly_status.json',status)
    def call(self,apply=False):
        return m.export(self.project,self.run,self.audit,self.assembly,self.dest,apply=apply,selected_ledgers=getattr(self,'ledgers',None))
    def test_pending_never_creates_evidence_destination(self):
        self.write(self.assembly/'assembly_status.json',{'status':'pending'})
        self.assertEqual(self.call()['status'],'blocked')
        with self.assertRaises(ValueError): self.call(True)
        self.assertFalse((self.project/'research_docs').exists())
    def test_complete_dry_inventory_writes_nothing(self):
        self.complete();d=self.call();self.assertEqual(d['status'],'dry_inventory_ready');self.assertFalse(self.dest.parent.exists())
    def test_actual_final_receipts_all_stage_exits_and_reviews_included(self):
        self.complete();result=self.call(True);names={row['source'] for row in result['files']}
        required={m.AUDIT_REL+'/'+name for name in m.RECOVERY05_AUDIT_RECEIPTS+m.RECOVERY05_REVIEW_NAMES}
        required|={m.RUN_REL+'/cycle_logs/'+name+'_exit.json' for name in m.RECOVERY05_STAGE_NAMES}
        required|={m.RUN_REL+'/'+owner+'/'+name for owner in m.CLOSED_OWNER_PATHS
                   for name in ('project_job_started.json','project_complete.json','project_exit.json')}
        self.assertTrue(required<=names,required-names)
        self.assertEqual(result['gate']['actual_recovery05_closure']['status'],'closed')
        self.assertNotIn(m.RUN_REL+'/cycle_logs/storage_gate_exit.json',names)
    def test_each_required_final_receipt_blocks_dry_and_apply_before_mkdir(self):
        self.complete()
        paths=[self.audit/name for name in m.RECOVERY05_AUDIT_RECEIPTS+m.RECOVERY05_REVIEW_NAMES]
        paths += [self.run/'cycle_logs'/(stage+'_exit.json') for stage in m.RECOVERY05_STAGE_NAMES]
        paths += [self.run/owner/name for owner in m.CLOSED_OWNER_PATHS
                  for name in ('project_job_started.json','project_complete.json','project_exit.json')]
        for path in paths:
            with self.subTest(path=str(path.relative_to(self.project))):
                raw=path.read_bytes();path.unlink()
                try:
                    # Required paths must be named in failures, never silently skipped.
                    try:dry=self.call()
                    except ValueError as error:self.assertIn(str(path.relative_to(self.project)),str(error))
                    else:
                        self.assertEqual(dry['status'],'blocked')
                        self.assertIn(str(path.relative_to(self.project)),dry['reason'])
                    with self.assertRaises(ValueError):self.call(True)
                    self.assertFalse(self.dest.parent.exists())
                finally:path.write_bytes(raw)
    def test_unclosed_or_failed_actual_receipts_reject_before_mkdir(self):
        self.complete()
        mutations=[
            ('driver_main_recovery05/status.json',dict(stage='evaluate_test')),
            ('wait_main_recovery05/status.json',dict(stage='reviewed_main')),
            ('wait_main_recovery05/reviewed_main_exit.json',dict(returncode=1)),
            ('wait_main_recovery05/reviewed_main_exit.json',dict(owned_helper_returned_after_cleanup=False)),
            ('driver_main_recovery05/storage_gate.json',dict(ready=False)),
            ('collection_integrity_recovery05_actual01/integrity.json',dict(passed=False)),
            ('collection_integrity_recovery05_actual01/noise_audit.json',dict(cuda_used=True)),
            ('collection_noise_recovery05_actual01/noise_audit.json',dict(cuda_used=False)),
            ('collection_noise_recovery05_actual01/noise_audit.json',dict(exact_cuda_full_endpoint_checks=94)),
            ('collection_noise_recovery05_actual01/noise_audit.json',dict(all96_execution_endpoints_certified=True)),
            ('collection_noise_recovery05_actual01/noise_audit.json',dict(unavailable=[dict(case='other',retained_in_denominator=True)])),
        ]
        for relative,change in mutations:
            path=self.audit/relative;raw=path.read_bytes();value=json.loads(raw);value.update(change);self.write(path,value)
            with self.subTest(path=relative,change=change):
                try:
                    with self.assertRaises(ValueError):self.call(True)
                    self.assertFalse(self.dest.parent.exists())
                finally:path.write_bytes(raw)
        for relative,change in [('cycle_logs/cache_own_observations_exit.json',dict(returncode=1)),
                                ('owners/main_train/project_complete.json',dict(stages=2)),
                                ('evaluation/test/project_exit.json',dict(children_reaped=False))]:
            path=self.run/relative;raw=path.read_bytes();value=json.loads(raw);value.update(change);self.write(path,value)
            with self.subTest(path=relative):
                try:
                    with self.assertRaises(ValueError):self.call(True)
                    self.assertFalse(self.dest.parent.exists())
                finally:path.write_bytes(raw)
        for path in [self.audit/'driver_main_recovery05/failure.json',self.audit/'wait_main_recovery05/failure.json',
                     self.run/'owners/main_train/project_failure.json']:
            self.write(path,dict(error='synthetic failure'))
            with self.subTest(failure=str(path)):
                try:
                    with self.assertRaises(ValueError):self.call(True)
                    self.assertFalse(self.dest.parent.exists())
                finally:path.unlink()
    def test_actual_closure_change_during_preparation_rejected_before_mkdir(self):
        self.complete();original=m.gate;calls=0
        def changing_gate(*args):
            nonlocal calls
            calls+=1
            if calls==2:
                path=self.audit/'driver_main_recovery05/status.json';value=json.loads(path.read_text())
                value['synthetic_change_during_preparation']=True;self.write(path,value)
            return original(*args)
        with mock.patch.object(m,'gate',side_effect=changing_gate):
            with self.assertRaisesRegex(ValueError,'Final gate changed'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_conditional_aggregate_exit_included_only_when_present(self):
        self.complete();relative='wait_main_recovery05/aggregate_and_publish_once_exit.json'
        before={row['source'] for row in self.call()['files']}
        self.assertNotIn(m.AUDIT_REL+'/'+relative,before)
        self.write(self.audit/relative,dict(returncode=0,owned_helper_returned_after_cleanup=True))
        after={row['source'] for row in self.call()['files']}
        self.assertIn(m.AUDIT_REL+'/'+relative,after)
    def test_truncated256_denominator_rejected_before_mkdir(self):
        self.complete();p=self.run/'evaluation/test/summary/rollouts.csv';lines=p.read_text().splitlines();p.write_text('\n'.join(lines[:-1])+'\n');self.rebind()
        with self.assertRaisesRegex(ValueError,'256'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_wrong_family_denominator_rejected(self):
        self.complete();p=self.run/'evaluation/test/summary/per_initial_state.csv';s=p.read_text().replace('f63','f62');p.write_text(s);self.rebind()
        self.assertEqual(self.call()['status'],'blocked')
    def test_stale_assembler_input_sha_rejected(self):
        self.complete();p=self.run/'evaluation/test/summary/primary_summary.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_software_failures_remain_full256_outcomes(self):
        self.complete();result=self.call(True);self.assertEqual(result['status'],'export_complete')
        exported=self.dest/'files'/m.RUN_REL/'evaluation/test/summary/rollouts.csv'
        with exported.open() as stream: rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),256);self.assertTrue(all(r['software_complete']=='False' for r in rows))
        self.assertEqual(stat.S_IMODE(exported.stat().st_mode),0o444);self.assertEqual(stat.S_IMODE(self.dest.stat().st_mode),0o555)
        for f in result['files']:
            self.assertEqual(hashlib.sha256((self.dest/f['destination']).read_bytes()).hexdigest(),f['sha256'])
    def test_no_overwrite(self):
        self.complete();self.call(True)
        with self.assertRaisesRegex(ValueError,'already exists'):self.call(True)
    def test_payload_suffixes_and_unsupported_rejected(self):
        for suffix in ('.h5','.pt','.pkl','.npz','.mp4','.txt','.jsonl','.h5.json'):
            with self.subTest(suffix=suffix):
                p=self.project/('payload'+suffix);p.write_text('{}')
                with self.assertRaises(ValueError):m.inventory(self.project,[dict(source=p.name)])
    def test_outside_and_symlink_escape_rejected(self):
        outside=Path(self.temp.name)/'outside.json';outside.write_text('{}');(self.project/'escape.json').symlink_to(outside)
        for name in ('../outside.json','escape.json',str(outside)):
            with self.assertRaises(ValueError):m.inventory(self.project,[dict(source=name)])
    def test_payload_directory_rejected(self):
        p=self.project/'visual/result.json';self.write(p,{})
        with self.assertRaises(ValueError):m.inventory(self.project,[dict(source='visual/result.json')])
    def test_individual_size_cap_rejected(self):
        p=self.project/'large.json'
        with p.open('wb') as f:f.truncate(m.MAX_FILE+1)
        with self.assertRaisesRegex(ValueError,'16MiB'):m.inventory(self.project,[dict(source=p.name)])
    def test_total_size_cap_rejected(self):
        self.write(self.project/'one.json','x'*6);self.write(self.project/'two.json','x'*6)
        with mock.patch.object(m,'MAX_TOTAL',10):
            with self.assertRaisesRegex(ValueError,'256MiB'):m.inventory(self.project,[dict(source='one.json'),dict(source='two.json')])
    def test_frozen_source_hash_required_before_destination(self):
        self.complete();source=self.project/'tools/frozen.py';self.write(source,'pass\n')
        rp=self.run/'evaluation/test/registration.json';d=json.loads(rp.read_text());d['source_sha256']={'tools/frozen.py':'0'*64};self.write(rp,d)
        sp=self.run/'evaluation/test/summary/primary_summary.json';s=json.loads(sp.read_text());s['registration_sha256']=hashlib.sha256(rp.read_bytes()).hexdigest();self.write(sp,s);self.rebind()
        with self.assertRaisesRegex(ValueError,'Frozen source SHA'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_frozen_source_correct_sha_exported(self):
        self.complete();source=self.project/'tools/frozen.py';self.write(source,'pass\n')
        rp=self.run/'evaluation/test/registration.json';d=json.loads(rp.read_text());d['source_sha256']={'tools/frozen.py':hashlib.sha256(source.read_bytes()).hexdigest()};self.write(rp,d)
        sp=self.run/'evaluation/test/summary/primary_summary.json';s=json.loads(sp.read_text());s['registration_sha256']=hashlib.sha256(rp.read_bytes()).hexdigest();self.write(sp,s);self.rebind()
        result=self.call(True);self.assertEqual((self.dest/'files/tools/frozen.py').read_bytes(),source.read_bytes())
        self.assertEqual(next(f for f in result['files'] if f['source']=='tools/frozen.py')['category'],'frozen_production_python')
    def test_registered_case_name_receipt_included(self):
        self.complete();self.write(self.run/'collection/plan.json',dict(cases=[dict(name='A_case')]))
        self.write(self.run/'collection/A_case/result.json',dict(software_complete=False))
        self.write(self.run/'collection/A_unregistered/result.json',dict(software_complete=False))
        result=self.call(True);names={r['source'] for r in result['files']}
        self.assertIn(m.RUN_REL+'/collection/A_case/result.json',names)
        self.assertNotIn(m.RUN_REL+'/collection/A_unregistered/result.json',names)
    def test_real_component_cache_and_deck_basenames(self):
        self.complete()
        names=['common180_parents/A/continuation.json','common180_parents/A/cache_index.json','common180_parents/A/backend/base_identities.json','common180_parents/A/backend/cache_complete.json','common180_parents/A/backend/fixed_epoch_registration.json','common180_parents/A/backend/configuration.json','common180_parents/A/backend/prepared.json','common180_parents/A/training/deck_060.json']
        for name in names:self.write(self.run/name,{})
        result=self.call(True);exported={r['source'] for r in result['files']}
        self.assertTrue(all(m.RUN_REL+'/'+name in exported for name in names))
    def test_wave_registered_cases_name_preserved(self):
        self.complete();wave=self.run/'evaluation/test/wave00'
        self.write(wave/'plan.json',dict(cases=[dict(name='A_new_paired_test_0')]))
        self.write(wave/'A_new_paired_test_0/result.json',dict(software_complete=False))
        rp=self.run/'evaluation/test/registration.json';registration=json.loads(rp.read_text());registration['wave_plan_sha256']={str(wave):hashlib.sha256((wave/'plan.json').read_bytes()).hexdigest()};self.write(rp,registration)
        sp=self.run/'evaluation/test/summary/primary_summary.json';summary=json.loads(sp.read_text());summary['registration_sha256']=hashlib.sha256(rp.read_bytes()).hexdigest();self.write(sp,summary);self.rebind()
        result=self.call(True);exported={r['source'] for r in result['files']}
        self.assertIn(m.RUN_REL+'/evaluation/test/wave00/A_new_paired_test_0/result.json',exported)
    def test_absent_selected_ledger_extraction_blocks_before_mkdir(self):
        self.complete();self.ledgers=None
        with self.assertRaisesRegex(ValueError,'selected-ledgers'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_selected_ledger_identity_bound_to_assembler(self):
        self.complete();p=self.assembly/'actual_exposure_and_age.json';d=json.loads(p.read_text());d['learning']['A']['frontend']['checkpoint_sha256']='b'*64;self.write(p,d)
        with self.assertRaisesRegex(ValueError,'checkpoint differs'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_selected_component_sha_rejected(self):
        self.complete();p=self.ledgers/'learning_A_frontend.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'JSON identity'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_eight_exposure_components_exported(self):
        self.complete();result=self.call(True);records=[r for r in result['files'] if r['category']=='selected_actual_ledger_counts']
        self.assertEqual(len(records),10);self.assertEqual(result['selected_ledger_gate']['components'],8)
    def test_selected_actual_tick_below50_rejected(self):
        self.complete();p=self.ledgers/'learning_A_frontend.json';d=json.loads(p.read_text());d['episodes']['episode']['successful_optimizer_steps'][0]=49;self.write(p,d);self.rebind_ledger(p.name)
        with self.assertRaisesRegex(ValueError,'below50'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_selected_native_entry_population_missing_rejected(self):
        self.complete();p=self.ledgers/'learning_A_frontend.json';d=json.loads(p.read_text());d['native_entry_rows'].pop();self.write(p,d);self.rebind_ledger(p.name)
        with self.assertRaisesRegex(ValueError,'entry population incomplete'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_selected_legal_mask_identity_rejected(self):
        self.complete();p=self.ledgers/'learning_A_frontend.json';d=json.loads(p.read_text());d['episodes']['episode']['mask_sha256']='f'*64;self.write(p,d);self.rebind_ledger(p.name)
        with self.assertRaisesRegex(ValueError,'mask identity'):self.call(True)
        self.assertFalse(self.dest.parent.exists())
    def test_real_extractor_output_schema_integration(self):
        try: import torch
        except ImportError: self.skipTest('torch only needed for isolated extractor integration fixture')
        self.complete()
        path=Path(__file__).with_name('test_extract_selected_ledgers_recovery05.py')
        spec=importlib.util.spec_from_file_location('selected_fixture',path);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            helper=fixture.SelectedLedgers();root,audit,states,loader=helper.fixture(tmp)
            # Align fake fixture paths to canonical project locations, without real checkpoint reads.
            mapped={}
            for key,state in states.items():
                source=Path(key);relative=source.relative_to(root);target=self.run/relative
                self.write(target,source.read_text());mapped[str(target)]=state
            for source in root.rglob('*.json'):
                if source.name=='registration.json':continue
                target=self.run/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
            self.write(self.run/'evaluation/test/registration.json',json.loads((root/'evaluation/test/registration.json').read_text()))
            # Call collect under isolated mocked gate: exactly fake eight selected states only.
            with mock.patch.object(fixture.m,'closure_gate',return_value=dict(status='closed')):
                summary,payloads=fixture.m.collect(self.run,self.audit,loader=lambda p:mapped[str(p)])
            for name,raw in payloads.items():
                target=self.ledgers/name;target.write_bytes(raw)
            self.write(self.ledgers/'extraction_status.json',dict(**summary,root=str(self.run),output=str(self.ledgers),raw_data_opened=False,model_forward_calls=0))
            actual={}
            for scope in fixture.m.SCOPES:
                actual[scope]={}
                for role in 'AB':
                    actual[scope][role]={c:json.loads(payloads[f'{scope}_{role}_{c}.json']) for c in fixture.m.COMPONENTS}
            self.write(self.assembly/'actual_exposure_and_age.json',actual)
            result=m.selected_ledger_gate(self.project,self.audit,self.assembly,self.ledgers)
            self.assertEqual(result['components'],8);self.assertEqual(len(result['files']),10)
    def test_unregistered_file_not_copied(self):
        self.complete();self.write(self.run/'SECRET.json',{});self.write(self.run/'raw.h5','payload');self.write(self.audit/'unapproved.py','pass')
        result=self.call(True);names={r['source'] for r in result['files']}
        self.assertFalse(any('SECRET' in n or 'raw.h5' in n or 'unapproved.py' in n for n in names))

if __name__=='__main__':unittest.main()
