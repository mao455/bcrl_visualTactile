"""Read-only audit of frozen R2 reuse; writes only a new audit receipt."""
from pathlib import Path
import hashlib
import json
import time
import torch

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
AUDIT = PROJECT / 'outputs/diagnostics/tight_entryfix_20261121_01'
OLD = PROJECT / 'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01/round02/learning'
NEW = PROJECT / 'outputs/visuotactile/run/tight_entry_phase_20261121_01'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(p):
    return torch.load(p, map_location='cpu', weights_only=False, mmap=True)


def run():
    result = dict(version='frozen_r2_matched_control_reuse_audit_v1', utc=time.time(),
        old_root=str(OLD), native_entry_root=str(NEW), roles={},
        evidence={name: dict(path=str(AUDIT/name), sha256=sha(AUDIT/name)) for name in
            ['original_control_audit.json', 'visual_cache_verification.json',
             'matched_learning_rates.json', 'matched_backend_learning_rates.json']})
    visual = read(AUDIT/'visual_cache_verification.json')
    assert visual['passed'] and visual['visual_arithmetic']['vision_microbatch'] == 2
    assert all(x['patch_max_abs'] == 0 and x['depth_max_abs_m'] == 0 for x in visual['rows'])
    for role in 'AB':
        roots = [OLD/role, NEW/role]
        configs = [read(x/'training/configuration.json') for x in roots]
        rears = [read(x/f'backend/{role}/training/configuration.json') for x in roots]
        data = [read(x/'state_data/data.json') for x in roots]
        visuals = [read(x/'cache_index.json') for x in roots]
        dpval = [[json.loads(l) for l in (x/'training/validation.jsonl').read_text().splitlines()] for x in roots]
        rearval = [read(x/f'backend/{role}/training/validation.json') for x in roots]
        dpfin = [read(x/'training/finished.json') for x in roots]
        rearfin = [read(x/f'backend/{role}/training/finished.json') for x in roots]
        out = dict(evidence_paths=dict(dp_configuration=[str(x/'training/configuration.json') for x in roots],
            backend_configuration=[str(x/f'backend/{role}/training/configuration.json') for x in roots],
            dp_validation=[str(x/'training/validation.jsonl') for x in roots],
            backend_validation=[str(x/f'backend/{role}/training/validation.json') for x in roots]), checks={})
        checks = out['checks']
        math_sources = [
            'visuotactile/models/native_dp.py', 'visuotactile/models/native_vision.py',
            'visuotactile/dp_objectives.py', 'visuotactile/trajectory_batch.py',
            'visuotactile/trajectory_training.py', 'visuotactile/precision.py',
            'visuotactile/target_windows.py', 'visuotactile/offline_planning.py',
            'visuotactile/position_control.py', 'visuotactile/execution_learning.py',
            'visuotactile/execution_backend.py', 'visuotactile/residual_output.py',
            'visuotactile/residual_tick_window.py',
        ]
        unchanged = {}
        for file in math_sources:
            assert configs[0]['source_sha256'][file] == configs[1]['source_sha256'][file]
            assert rears[0]['source_sha256'][file] == rears[1]['source_sha256'][file]
            assert sha(PROJECT/file) == configs[0]['source_sha256'][file]
            unchanged[file] = configs[0]['source_sha256'][file]
        checks['unchanged_core_source_hashes'] = dict(passed=True, sha256=unchanged,
            changed_helpers_review='git diff 0e914563..90b0a7c -- tools/tight_fresh_train.py tools/tight_fresh_backend.py tools/tight_finish.py visuotactile/anchor_continue.py visuotactile/native_data.py visuotactile/fresh_visit_sampling.py: changes add native entry sampling, observational exposure ledger, explicit actual LR replay and visual identity checks; no model/loss/projection changes.')
        assert data[0] == data[1]
        assert {k:len(v) for k,v in data[0]['splits'].items()} == dict(train=180, validation=6, test=0)
        assert visuals[0]['entries'] == visuals[1]['entries']
        assert len(visuals[0]['entries']) == 186
        assert visuals[1]['visual_arithmetic'] == visual['visual_arithmetic']
        checks['dataset_split_and_visual_payloads'] = dict(passed=True, train=180, validation=6,
            test=0, equal_visual_payload_references=186,
            visual_arithmetic=visuals[1]['visual_arithmetic'],
            arithmetic_verification_scope=visual['limitation'],
            note='Old and new use exactly the same full cache entry paths, SHA declarations, frame counts and timestamp hashes. Sampled arithmetic verification is 672 frames across all 186 episodes; not an all-frame encoder recomputation.')
        for key in ['data_sha256','statistics_sha256','dino_sha256','continuation','bounded_extension',
                    'effective_exposure','hand_geometry','learning_rate','weight_decay','validation',
                    'offline_planning','full_trajectory_batch','sampling_contract','episode_multiplicities',
                    'planned_optimizer_updates','planned_episode_visits','episodes_per_epoch']:
            assert configs[0][key] == configs[1][key], ('dp_config', role, key)
        dp_parent = Path(configs[0]['continuation']['path'])
        assert sha(dp_parent) == configs[0]['continuation']['sha256']
        for x,cfg in zip(roots,configs):
            assert sha(x/'state_data/data.json') == cfg['data_sha256']
            assert sha(x/'state_data/statistics.pt') == cfg['statistics_sha256']
        checks['dp_parents_and_math'] = dict(passed=True, parent=str(dp_parent),
            parent_sha256=configs[0]['continuation']['sha256'],
            actual_new_updates=[x['updates'] for x in dpfin],
            actual_epochs=[x['epochs'] for x in dpfin],
            optimizer='fresh AdamW/EMA from same selected DP EMA parent; no optimizer resume from R1')
        assert [read(roots[0]/f'training/deck_{e:03d}.json') for e in range(1,61)] == [read(roots[1]/f'training/deck_{e:03d}.json') for e in range(1,61)]
        assert [x['learning_rates'] for x in dpval[0]] == [x['learning_rates'] for x in dpval[1]]
        assert [x['selection_eligible'] for x in dpval[0]] == [x['selection_eligible'] for x in dpval[1]]
        eligible = [x['epoch'] for x in dpval[0] if x['selection_eligible']]
        assert eligible == list(range(50,61))
        checks['dp_decks_lr_and_selection'] = dict(passed=True, identical_actual_decks=60,
            rates_by_epoch=[x['learning_rates'][0] for x in dpval[0]],
            eligible_epochs=eligible,
            selection='minimum fixed validation6 DDIM16 commit12 normalized equal Arm/Hand error among eligible epochs',
            extra_entry_gate='Native gate passes at exactly epoch50; it does not change the eligible epoch set compared with old absolute tick gate.')
        same_backend_keys = ['continuation_from','residual_statistics_sha256','residual_normalization_identity',
            'fixed_residual_scales','execution_adapter','loss_contract','history_contract','output_contract',
            'tick_window_contract','learning_rate','weight_decay','microbatch','maximum_control_tick_epochs',
            'minimum_control_tick_epochs','validation_epoch_interval','bounded_extension','effective_exposure',
            'cache_offline_planning_profile','episode_sampling','architecture','loss_id','ema','history',
            'selection','loss_weight_normalization','main_supervision','continuation_audit','teacher_supervision',
            'normalization_train_ids','normalizer_recomputed','supervision_mask_sha256']
        for key in same_backend_keys:
            assert rears[0][key] == rears[1][key], ('rear_config', role, key)
        parent = Path(rears[0]['continuation_from']['path'])
        assert sha(parent) == rears[0]['continuation_from']['sha256']
        assert [x['learning_rates'] for x in rearval[0]] == [x['learning_rates'] for x in rearval[1]]
        assert [x['selection_eligible'] for x in rearval[0]] == [x['selection_eligible'] for x in rearval[1]]
        assert [x['update'] for x in rearval[0]] == [x['update'] for x in rearval[1]]
        for key in ['updates','control_tick_epochs','required_supervised_ticks','supervised_ticks',
                    'unique_supervised_tick_population','supervised_tick_population','coverage']:
            assert rearfin[0][key] == rearfin[1][key], ('rear_finished', role,key)
        checks['backend_parent_math_budget_lr_selection'] = dict(passed=True, parent=str(parent),
            parent_sha256=rears[0]['continuation_from']['sha256'],
            actual_new_updates=[x['updates'] for x in rearfin],
            control_tick_epochs=[x['control_tick_epochs'] for x in rearfin],
            supervised_ticks=[x['supervised_ticks'] for x in rearfin],
            rates_by_epoch=[x['learning_rates'][0] for x in rearval[0]],
            eligible_epochs=[x['epoch'] for x in rearval[0] if x['selection_eligible']],
            selection=rears[0]['selection'])
        out['models'] = {}
        for arm,x,cfg,rear in zip(['original','native_entry'],roots,configs,rears):
            dp_path=x/'training/best.pt'; rear_path=x/f'backend/{role}/training/best.pt'
            dp=checkpoint(dp_path); rear_saved=checkpoint(rear_path)
            dp_sha=sha(dp_path); rear_sha=sha(rear_path)
            index_path=x/f'backend/{role}/nominal/index.json'; index=read(index_path)
            assert index['base_sha256'] == dp_sha == rear['base_sha256']
            assert dp['configuration'] == cfg
            assert rear_saved['configuration'] == rear
            assert dp['train_ids'] == rear_saved['train_ids'] == data[0]['splits']['train']
            assert dp['validation_ids'] == rear_saved['validation_ids'] == data[0]['splits']['validation']
            assert index['sampler'] == dict(steps=16,seed=7091,horizon=24,commit=12,prediction='sample')
            binding=dict(base_checkpoint_sha256=dp_sha,statistics_sha256=cfg['statistics_sha256'],
                data_sha256=cfg['data_sha256'],residual_statistics_sha256=rear['residual_statistics_sha256'])
            if arm=='native_entry':binding['visual_arithmetic_sha256']=rear['visual_arithmetic_sha256']
            assert len(index['episodes']) == 186
            bindings=0
            for identity in index['episodes']:
                payload=checkpoint(x/f'backend/{role}/nominal/{identity}.pt')
                assert payload['cache_identity'] == binding, (role,arm,identity)
                bindings+=1
                del payload
            selected=read(x/f'backend/{role}/training/selected.json')
            assert selected['checkpoint_sha256']==rear_sha and selected['base_checkpoint_sha256']==dp_sha
            best_dp=min((v for v in (dpval[0] if arm=='original' else dpval[1]) if v['selection_eligible']),key=lambda v:v['validation']['selection'])
            assert best_dp['epoch']==dp['epoch'] and best_dp['updates']==dp['updates']
            best_rear=min((v for v in (rearval[0] if arm=='original' else rearval[1]) if v['selection_eligible']),key=lambda v:v['selection_score'])
            assert best_rear['epoch']==rear_saved['epoch'] and best_rear['update']==rear_saved['updates']
            out['models'][arm]=dict(dp=dict(path=str(dp_path),sha256=dp_sha,epoch=dp['epoch'],updates=dp['updates']),
                backend=dict(path=str(rear_path),sha256=rear_sha,epoch=rear_saved['epoch'],updates=rear_saved['updates']),
                nominal_index=dict(path=str(index_path),sha256=sha(index_path),base_sha256=index['base_sha256'],
                    actual_payload_bindings_verified=bindings),selected_minimum_validation_confirmed=True)
            del dp,rear_saved
        out['reusable_as_original_sampler_control']=True
        result['roles'][role]=out
    result.update(passed=True,requires_original_sampler_retraining=False,
        conclusion='Frozen R2 complete A/B pairs can be reused as the original-sampler arm. Actual inputs, parents, budgets, epoch learning rates, eligible epoch sets, selection metrics, backend math and matching DP cache bindings agree; native entry sampler is the intended intervention.',
        limitations=['Historical visual configuration lacks explicit arithmetic identity; this is closed for between-arm training parity by identical full cached payload references and independent sampled microbatch2 verification, not by all-frame encoder regeneration.',
            'The seed alone does not establish identical per-visit diffusion noise after the number of sampled anchors changes. The changed training noise consumption is an unavoidable consequence of this sampler intervention, not an independently controlled identical-noise claim.',
            'This audit establishes training/control comparability; it contains zero new optimizer updates or simulator steps and does not establish entry error or closed-loop improvement.'])
    dest=AUDIT/'r2_control_equivalence_20261122.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(dest),passed=result['passed'],requires_original_sampler_retraining=False,
        roles={r:result['roles'][r]['models'] for r in 'AB'}),ensure_ascii=False))


if __name__=='__main__':
    run()
