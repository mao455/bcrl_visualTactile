"""Isolated CPU fixtures; no experiment checkpoints, simulator or GPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location('selected_ledgers', Path(__file__).with_name('extract_selected_ledgers_recovery05.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SelectedLedgers(unittest.TestCase):
    def fixture(self, temporary):
        root, audit = Path(temporary)/'run', Path(temporary)/'audit'
        audit.mkdir(); states = {}; parents = {}; ages = {}; evaluated = {}
        for scope in m.SCOPES:
            for role in 'AB':
                child = root/scope/role
                ids = ['full', 'suffix']
                data = dict(splits=dict(train=ids), entries={i:dict(length=6, family='family_'+i, source_sha256='a'*64) for i in ids})
                write(child/'state_data/data.json', data)
                component_hashes = {}
                for component in m.COMPONENTS:
                    directory = m.component_directory(root, scope, role, component)
                    directory.mkdir(parents=True, exist_ok=True)
                    checkpoint = directory/'best.pt'; checkpoint.write_bytes((scope+role+component).encode())
                    component_hashes[component] = m.sha(checkpoint)
                    updates = 600 if component == 'frontend' else 1200
                    write(directory/'deployment.json',dict(epoch=60, update=updates, checkpoint_sha256=m.sha(checkpoint), selection='preregistered_epoch60_only', validation_used_for_selection=False))
                    write(directory/'finished.json', dict(updates=updates))
                    masks = dict(full=torch.tensor([1]*6,dtype=torch.bool), suffix=torch.tensor([0,0,1,1,1,1],dtype=torch.bool))
                    counts = {i:mask.to(torch.int64)*60 for i,mask in masks.items()}
                    state = dict(epoch=60,updates=updates,train_ids=ids,
                        configuration=dict(data_sha256=m.sha(child/'state_data/data.json'),effective_exposure=dict(ancestors_counted=False, duplicate_tick_within_step='count_once')),
                        effective_exposure_state=dict(version='new_optimizer_step_legal_tick_exposure_v1',updates=updates,legal=masks,counts=counts))
                    if component == 'frontend':
                        state['ema_optimization_step'] = updates
                        state['sampling_exposure'] = dict.fromkeys(ids,60)
                        state['entry_exposure_state'] = dict(version='native_entry_commit_optimizer_exposure_v1',updates=updates,
                            entries=dict(suffix=dict(anchor=0,slots=[2,3,4,5],ticks=[2,3,4,5])),counts=dict(suffix={t:60 for t in range(2,6)}))
                    else:
                        state['episode_counts'] = dict.fromkeys(ids,60)
                        state['configuration']['base_sha256'] = component_hashes['frontend']
                    states[str(checkpoint)] = state
                eval_role = role+('_parent' if scope == 'common180_parents' else '_new')
                evaluated[eval_role] = {c+'_sha256':v for c,v in component_hashes.items()}
                if scope == 'common180_parents':
                    parents[role] = evaluated[eval_role]
                    ages[role] = dict(frontend_epochs=60,frontend_updates=600,backend_epochs=60,backend_updates=1200)
        write(root/'parent_pairs.json',parents);write(root/'actual_parent_ages.json',dict(roles=ages))
        write(root/'evaluation/test/registration.json',dict(cohort='test',n=64,pairs=evaluated))
        write(root/'cycle_logs/summarize_test_exit.json',dict(returncode=0))
        write(audit/'driver_main_recovery05/status.json',dict(stage='main_stages_complete'))
        for directory in [root/'owners/main_train',root/'evaluation/development',root/'evaluation/test']:
            write(directory/'project_exit.json',dict(children_reaped=True))
        self.loads = []
        def loader(path):
            self.loads.append(str(path));return copy.deepcopy(states[str(path)])
        return root,audit,states,loader

    def test_missing_main_never_reads_even_completed_parent_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,audit,states,loader = self.fixture(tmp)
            (audit/'driver_main_recovery05/status.json').unlink()
            output=audit/'pending'
            result=m.extract(root,audit,output,loader=loader)
            self.assertEqual(result['status'],'pending');self.assertEqual(self.loads,[])
            self.assertEqual([p.name for p in output.iterdir()],['extraction_status.json'])

    def test_incomplete_or_wrong_final_closure_reads_no_checkpoints(self):
        for change in ['running','summary_failure','wrong_cohort','wrong_denominator','missing_policy','missing_main_owner']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root,audit,_,loader = self.fixture(tmp)
                if change=='running':
                    write(audit/'driver_main_recovery05/status.json',dict(stage='evaluate_test'))
                elif change=='summary_failure':
                    write(root/'cycle_logs/summarize_test_exit.json',dict(returncode=1))
                elif change=='missing_main_owner':
                    (root/'owners/main_train/project_exit.json').unlink()
                else:
                    path=root/'evaluation/test/registration.json';freeze=m.read(path)
                    if change=='wrong_cohort':freeze['cohort']='development'
                    elif change=='wrong_denominator':freeze['n']=63
                    else:freeze['pairs'].pop('B_new')
                    write(path,freeze)
                result=m.extract(root,audit,audit/'blocked',loader=loader)
                self.assertIn(result['status'],['failed','pending'])
                self.assertEqual(self.loads,[])
                self.assertEqual([p.name for p in (audit/'blocked').iterdir()],['extraction_status.json'])

    def test_exports_exact_eight_actual_count_arrays_with_mask_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,audit,states,loader = self.fixture(tmp);output=audit/'final'
            before=torch.cuda.is_initialized()
            result=m.extract(root,audit,output,loader=loader)
            self.assertEqual(result['status'],'passed');self.assertEqual(len(self.loads),8)
            self.assertEqual(len(list(output.glob('*.json'))),10)
            row=json.loads((output/'learning_A_frontend.json').read_text())
            self.assertEqual(row['episodes']['suffix']['legal_ticks'],[2,3,4,5])
            self.assertEqual(row['episodes']['suffix']['successful_optimizer_steps'],[60]*4)
            self.assertEqual(len(row['native_entry_rows']),4)
            self.assertFalse(row['raw_mask_revalidation']);self.assertEqual(torch.cuda.is_initialized(),before)
            summary=json.loads((output/'selected_ledger_summary.json').read_text())
            for name,identity in summary['files'].items():
                self.assertEqual(identity['sha256'],m.sha(output/name));self.assertLessEqual(identity['bytes'],16*1024**2)

    def test_exposure_or_binding_failure_publishes_no_partial_component_evidence(self):
        changes=['below50','illegal_positive','age','entry','base','masks','population','data','ancestors','count_dtype','length','out_of_age','ema_age']
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root,audit,states,loader=self.fixture(tmp)
                key=str(m.component_directory(root,'learning','B','backend')/'best.pt');s=states[key]
                if change=='below50':s['effective_exposure_state']['counts']['suffix'][2]=49
                elif change=='illegal_positive':s['effective_exposure_state']['counts']['suffix'][0]=50
                elif change=='age':s['effective_exposure_state']['updates']-=1
                elif change=='entry':states[str(m.component_directory(root,'learning','B','frontend')/'best.pt')]['entry_exposure_state']['counts']['suffix'][2]=49
                elif change=='base':s['configuration']['base_sha256']='b'*64
                elif change=='masks':
                    s['effective_exposure_state']['legal']['suffix'][1]=True;s['effective_exposure_state']['counts']['suffix'][1]=60
                elif change=='population':s['train_ids']=list(reversed(s['train_ids']))
                elif change=='data':s['configuration']['data_sha256']='c'*64
                elif change=='ancestors':s['configuration']['effective_exposure']['ancestors_counted']=True
                elif change=='count_dtype':s['effective_exposure_state']['counts']['suffix']=s['effective_exposure_state']['counts']['suffix'].float()
                elif change=='length':
                    s['effective_exposure_state']['counts']['suffix']=s['effective_exposure_state']['counts']['suffix'][:-1]
                elif change=='out_of_age':s['effective_exposure_state']['counts']['suffix'][2]=s['updates']+1
                else:states[str(m.component_directory(root,'learning','B','frontend')/'best.pt')]['ema_optimization_step']-=1
                result=m.extract(root,audit,audit/'failed',loader=loader)
                self.assertEqual(result['status'],'failed')
                self.assertEqual([p.name for p in (audit/'failed').iterdir()],['extraction_status.json'])

    def test_unclosed_owner_or_checkpoint_sha_blocks_reads(self):
        for change in ['owner','sha']:
            with tempfile.TemporaryDirectory() as tmp:
                root,audit,states,loader=self.fixture(tmp)
                if change=='owner':write(root/'evaluation/test/project_exit.json',dict(children_reaped=False))
                else:(m.component_directory(root,'common180_parents','A','frontend')/'best.pt').write_bytes(b'altered')
                result=m.extract(root,audit,audit/'failed',loader=loader)
                self.assertEqual(result['status'],'failed');self.assertEqual(self.loads,[])

    def test_mismatched_ab_actual_updates_or_final_evaluated_identity_rejected(self):
        for change in ['updates','evaluated']:
            with tempfile.TemporaryDirectory() as tmp:
                root,audit,states,loader=self.fixture(tmp)
                if change=='updates':
                    d=m.component_directory(root,'learning','B','backend');s=states[str(d/'best.pt')]
                    s['updates']=1201;s['effective_exposure_state']['updates']=1201
                    dep=m.read(d/'deployment.json');dep['update']=1201;write(d/'deployment.json',dep)
                    write(d/'finished.json',dict(updates=1201))
                else:
                    p=root/'evaluation/test/registration.json';f=m.read(p);f['pairs']['B_new']['backend_sha256']='d'*64;write(p,f)
                self.assertEqual(m.extract(root,audit,audit/'failed',loader=loader)['status'],'failed')

    def test_new_audit_output_only_and_size_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,audit,states,loader=self.fixture(tmp)
            for output in [audit,audit.parent/'escape']:
                with self.assertRaises(ValueError):m.extract(root,audit,output,loader=loader)
            output=audit/'new';m.extract(root,audit,output,loader=loader)
            with self.assertRaises(ValueError):m.extract(root,audit,output,loader=loader)
            with self.assertRaises(ValueError):m.encoded({'oversized':'x'*(m.MAX_FILE+1)})

    def test_default_loader_is_explicit_cpu_mmap_without_model_construction(self):
        before=torch.cuda.is_initialized()
        with patch.object(torch,'load',return_value={'fixture':True}) as load:
            self.assertEqual(m.load_checkpoint(Path('/nonexistent_fixture_only.pt')),{'fixture':True})
            load.assert_called_once_with(Path('/nonexistent_fixture_only.pt'),map_location='cpu',weights_only=False,mmap=True)
        self.assertEqual(torch.cuda.is_initialized(),before)

    def test_real_production_ledgers_and_serialization_schema_round_trip(self):
        """Exercise actual counter/save code on tiny CPU fixtures, not real runs."""
        from visuotactile.checkpointing import training_state, cpu_snapshot
        from visuotactile.effective_exposure import Ledger, profile
        from visuotactile.entry_exposure import EntryLedger
        from visuotactile.target_windows import RECOVERY_ONSET_WINDOW

        before = torch.cuda.is_initialized()
        with tempfile.TemporaryDirectory() as tmp:
            root,audit,states,_ = self.fixture(tmp)
            parents = {}; ages = {}; evaluated = {}
            for scope in m.SCOPES:
                for role in 'AB':
                    hashes = {}
                    for component in m.COMPONENTS:
                        directory = m.component_directory(root,scope,role,component)
                        saved = states[str(directory/'best.pt')]
                        records = {
                            identity:dict(time=torch.arange(6),command=torch.zeros(6,29),
                                supervision_mask=mask,control_segment=torch.zeros(6,dtype=torch.long),
                                target_window_contract=RECOVERY_ONSET_WINDOW)
                            for identity,mask in saved['effective_exposure_state']['legal'].items()
                        }
                        ledger = Ledger({i:r['supervision_mask'] for i,r in records.items()})
                        entry_ledger = EntryLedger(records)
                        contributions = [(i,r['supervision_mask'].nonzero().flatten().tolist())
                                         for i,r in records.items()]
                        entry_contributions = entry_ledger.contributions(records,[
                            ('suffix',torch.tensor([0,0],dtype=torch.long))])
                        for step in range(1,61):
                            # Three overlapping contributions remain ONE distinct
                            # successful-step participation in both actual ledgers.
                            ledger.step(step,contributions*3)
                            entry_ledger.step(step,entry_contributions*3)
                        self.assertEqual(ledger.counts['suffix'].tolist(),[0,0,60,60,60,60])
                        self.assertEqual(set(entry_ledger.counts['suffix'].values()),{60})
                        saved.update(updates=60,effective_exposure_state=ledger.state_dict())
                        saved['configuration']['effective_exposure'] = profile()
                        if component == 'frontend':
                            saved.update(ema_optimization_step=60,entry_exposure_state=entry_ledger.state_dict())
                            model = torch.nn.Linear(1,1)
                            optimizer = torch.optim.SGD(model.parameters(),lr=.01)
                            metadata = {k:v for k,v in saved.items() if k!='ema_optimization_step'}
                            saved = training_state(model,model,optimizer,
                                SimpleNamespace(optimization_step=60),torch.Generator(),torch.Generator(),**metadata)
                        else:
                            saved['configuration']['base_sha256'] = hashes['frontend']
                            saved = cpu_snapshot(saved)
                        torch.save(saved,directory/'best.pt')
                        hashes[component] = m.sha(directory/'best.pt')
                        deployment = m.read(directory/'deployment.json')
                        deployment.update(update=60,checkpoint_sha256=hashes[component])
                        write(directory/'deployment.json',deployment)
                        write(directory/'finished.json',dict(updates=60))
                    evaluated[role+('_parent' if scope=='common180_parents' else '_new')] = {
                        c+'_sha256':h for c,h in hashes.items()}
                    if scope == 'common180_parents':
                        parents[role] = evaluated[role+'_parent']
                        ages[role] = dict(frontend_epochs=60,frontend_updates=60,backend_epochs=60,backend_updates=60)
            write(root/'parent_pairs.json',parents)
            write(root/'actual_parent_ages.json',dict(roles=ages))
            write(root/'evaluation/test/registration.json',dict(cohort='test',n=64,pairs=evaluated))
            output = audit/'production_schema_cpu_fixture'
            result = m.extract(root,audit,output)
            self.assertEqual(result['status'],'passed',result.get('error'))
            self.assertEqual(result['checkpoint_reads'],8)
            front = m.read(output/'learning_A_frontend.json')
            self.assertEqual(front['minimum_tick_steps'],60)
            self.assertEqual(front['native_entry_rows'][0]['native_anchor'],0)
            self.assertEqual(front['native_entry_rows'][0]['absolute_tick'],2)
        self.assertEqual(torch.cuda.is_initialized(),before)

    def test_real_native_entry_phase_for_exact_boundary_and_partial_history(self):
        from visuotactile.effective_exposure import Ledger,profile
        from visuotactile.entry_exposure import EntryLedger
        from visuotactile.target_windows import RECOVERY_ONSET_WINDOW

        for first,length in [(12,16),(13,16),(23,25),(24,27),(25,27)]:
            with self.subTest(first=first,length=length),tempfile.TemporaryDirectory() as tmp:
                root,audit,states,_=self.fixture(tmp)
                data_path=root/'learning/A/state_data/data.json';data=m.read(data_path)
                data['entries']['suffix']['length']=length;write(data_path,data)
                saved=states[str(m.component_directory(root,'learning','A','frontend')/'best.pt')]
                masks={'full':torch.ones(6,dtype=torch.bool),'suffix':torch.arange(length)>=first}
                records={i:dict(time=torch.arange(len(mask)),command=torch.zeros(len(mask),29),
                    supervision_mask=mask,control_segment=torch.zeros(len(mask),dtype=torch.long),
                    target_window_contract=RECOVERY_ONSET_WINDOW) for i,mask in masks.items()}
                ledger=Ledger(masks);entry=EntryLedger(records)
                legal=[(i,m.nonzero().flatten().tolist()) for i,m in masks.items()]
                selected=[('suffix',torch.tensor([first//12*12]))]
                for update in range(1,61):
                    ledger.step(update,legal);entry.step(update,entry.contributions(records,selected))
                saved.update(effective_exposure_state=ledger.state_dict(),entry_exposure_state=entry.state_dict(),
                    updates=60,ema_optimization_step=60)
                saved['configuration'].update(data_sha256=m.sha(data_path),effective_exposure=profile())
                directory=m.component_directory(root,'learning','A','frontend')
                deployment=m.read(directory/'deployment.json');deployment['update']=60
                write(directory/'deployment.json',deployment);write(directory/'finished.json',dict(updates=60))
                result=m.selected_component(root,'learning','A','frontend',loader=lambda _:saved)
                ticks=[r['absolute_tick'] for r in result['native_entry_rows']]
                self.assertEqual(ticks,list(range(first,min((first//12+1)*12,length))))
                self.assertTrue(all(r['successful_optimizer_steps']==60 for r in result['native_entry_rows']))


if __name__=='__main__':unittest.main()
