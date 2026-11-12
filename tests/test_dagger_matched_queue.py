"""Finite matched continuation: ownership, real exposure, cache and evaluation."""
import copy
import json
from pathlib import Path

import pytest
import torch

from tools import tight_dagger as d
from tools.runtime import read,write,sha


def configuration():
    old=json.loads((Path(__file__).parents[1]/'configs/tight_state.json').read_text())['dagger']
    return d.matched_configuration(old,old['matched_continuation_policy'],Path('/actual/old'))


def test_matched_configuration_preserves_semantics_and_registers_real_exposure():
    cfg=configuration()
    assert cfg['rounds']==2 and cfg['first_round']==1 and cfg['maximum_hours']==24
    assert cfg['frontend']['maximum_updates'] is None
    assert cfg['frontend']['minimum_anchor_epochs']==cfg['frontend']['maximum_anchor_epochs']==6
    assert cfg['backend']['minimum_control_tick_epochs']==cfg['backend']['maximum_control_tick_epochs']==12
    assert cfg['evaluation']['attempts_per_group']==cfg['evaluation']['maximum_attempts_per_group']==10
    assert cfg['groups']['QG01']['supervision']=='q_reached' and cfg['groups']['QG01']['geometry']
    assert cfg['groups']['G01']['supervision']=='u_expert' and not cfg['groups']['G01']['geometry']
    assert not cfg.get('conditional_extension') and not cfg.get('conditional_extension_policy')
    for r in (1,2):
        v=d.round_decision(cfg,{},24*3600,r)
        assert v['attempts_by_group']=={'QG01':48,'G01':48} and not v['unequal_exposure_exploration']
    with pytest.raises(ValueError,match='remaining budget'):d.round_decision(cfg,{},10*3600,1)


def test_paired_opportunities_compare_real_family_seed_switch_and_partition():
    cfg=configuration();data=dict(splits={'train':['a','b']},entries={'a':{'family':'fa'},'b':{'family':'fb'}})
    families=[dict(family='fa',split='train'),dict(family='fb',split='train'),dict(family='test',split='test')]
    plans={g:d.takeover_plan(data,families,dict(cfg['collection'],source_group=g,round_index=1)) for g in cfg['groups']}
    assert d.assert_paired_plans(plans)['attempts_each']==48
    assert len({r['takeover_tick'] for r in plans['QG01']})==8
    assert not ({r['identity'] for r in plans['QG01']} & {r['identity'] for r in plans['G01']})
    plans['G01'][7]['event_seed']+=1
    with pytest.raises(ValueError,match='opportunity'):d.assert_paired_plans(plans)


def test_control_acceptance_is_bound_to_real_source_and_evidence(tmp_path,monkeypatch):
    code=tmp_path/'code';root=tmp_path/'run';source=code/'adapter.py';source.parent.mkdir();source.write_text('accepted')
    evidence=tmp_path/'parity.json';write(evidence,dict(samples=20,passed=True))
    gate=tmp_path/'gate.json';write(gate,dict(passed=True,expert_student_applied_equivalent=True,
        supervision_semantics_preserved=True,sources={'adapter.py':sha(source)},evidence=[dict(path=str(evidence),sha256=sha(evidence))]))
    write(root/'continuation_request.json',dict(control_acceptance=dict(path=str(gate),sha256=sha(gate))))
    monkeypatch.setattr(d,'CODE',code)
    assert d.control_acceptance(root)['passed']
    source.write_text('changed')
    with pytest.raises(ValueError,match='source changed'):d.control_acceptance(root)
    source.write_text('accepted');write(evidence,dict(passed=False))
    with pytest.raises(ValueError,match='evidence changed'):d.control_acceptance(root)


def test_matched_registration_pins_actual_resume_owner_and_preserves_old_deadline(tmp_path,monkeypatch):
    code=tmp_path/'code';old=tmp_path/'old';root=tmp_path/'new';cfg=configuration()
    write(code/'configs/tight_state.json',{'dagger':dict(cfg,matched_continuation_policy=cfg['matched_protocol'])})
    write(old/'configuration.json',cfg);write(old/'session.json',dict(deadline=123456,started=123))
    write(old/'launch.json',dict(pid=12,start_ticks=1));write(old/'resume.json',dict(pid=34,start_ticks=2))
    write(old/'public_git.json',dict(head='old-public',pushed=True))
    write(old/'evaluation_plan.json',[dict(identity=str(i),spec={'seed':i}) for i in range(10)])
    write(root/'continuation_request.json',dict(predecessor=str(old),control_acceptance={}))
    digest=sha(old/'session.json')
    monkeypatch.setattr(d,'CODE',code);monkeypatch.setattr(d,'control_acceptance',lambda root:{'passed':True})
    monkeypatch.setattr(d,'git',lambda *args:'' if args[0]=='status' else 'new-prepared')
    monkeypatch.setattr(d,'proc',lambda pid:dict(pid=pid,start_ticks=2))
    d.matched_prepare(root)
    registration=read(root/'registration.json')
    assert registration['predecessor_launch']['pid']==34 and registration['waiting_deadline']==123456+7200
    assert registration['prepared_source']=='new-prepared' and registration['predecessor_source']=='old-public'
    assert sha(old/'session.json')==digest and not (root/'launch.json').exists()
    assert 'admission' in registration['deadline_rule']
    with pytest.raises(ValueError,match='already registered'):d.matched_prepare(root)


def test_matched_pair_guard_rejects_legacy_adapter_and_unequal_scales():
    from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
    policy=configuration()['matched_protocol']
    pairs={g:dict(execution_adapter=REACHABLE_POSITION_EXECUTION,scales=policy['residual_scales']) for g in ('QG01','G01')}
    d.validate_matched_pairs(pairs,policy)
    pairs['G01']=dict(pairs['G01'],execution_adapter=None)
    with pytest.raises(ValueError,match='execution adapter'):d.validate_matched_pairs(pairs,policy)
    pairs['G01']=dict(pairs['QG01'],scales=[.1]*29)
    with pytest.raises(ValueError,match='bounds differ'):d.validate_matched_pairs(pairs,policy)


def test_endpoint_requires_completed_real_training_and_keeps_validation_best(tmp_path):
    torch.save(dict(updates=6,model={'w':torch.tensor(1.)}),tmp_path/'best.pt')
    torch.save(dict(updates=12,model={'w':torch.tensor(2.)}),tmp_path/'last.pt')
    write(tmp_path/'selected.json',dict(update=6,checkpoint_sha256=sha(tmp_path/'best.pt')))
    write(tmp_path/'finished.json',dict(target_reached=False,new_optimizer_updates=12,updates=12,disposable=False))
    before=sha(tmp_path/'best.pt')
    with pytest.raises(ValueError,match='Incomplete formal exposure'):d.select_exposure_endpoint(tmp_path,'backend')
    assert sha(tmp_path/'best.pt')==before and not (tmp_path/'validation_best.pt').exists()
    done=read(tmp_path/'finished.json');done['target_reached']=True;write(tmp_path/'finished.json',done)
    result=d.select_exposure_endpoint(tmp_path,'backend')
    assert result['update']==12 and sha(tmp_path/'validation_best.pt')==before
    assert sha(tmp_path/'best.pt')==sha(tmp_path/'last.pt')
    assert d.select_exposure_endpoint(tmp_path,'backend')==result


def test_nominal_reuse_rejects_old_front_identity_without_rewriting_actual_history(tmp_path):
    source=tmp_path/'old';source.mkdir();state=tmp_path/'state';state.mkdir()
    write(state/'data.json',dict(splits={'train':['a']}));(state/'statistics.pt').write_bytes(b'stats')
    (source/'a.pt').write_bytes(b'actual issued/applied history and cached old-front prediction')
    index=dict(base_sha256='front',data_sha256=sha(state/'data.json'),statistics_sha256=sha(state/'statistics.pt'),
        residual_statistics_sha256='command-stats',episodes={'a':dict(sha256=sha(source/'a.pt'))})
    write(source/'index.json',index)
    with pytest.raises(ValueError,match='identity differs'):
        d.reuse_nominal_cache(source,tmp_path/'bad',dict(sha256='changed-front'),state,'command-stats')
    dest=tmp_path/'new';d.reuse_nominal_cache(source,dest,dict(sha256='front'),state,'command-stats')
    assert (dest/'a.pt').is_symlink() and sha(dest/'a.pt')==sha(source/'a.pt')
    assert read(dest/'index.json')==index and read(dest/'reuse.json')['recomputed_nominal_episodes']==0


def test_exposure_compares_supervision_not_optimizer_counts(tmp_path):
    for g in ('QG01','G01'):
        write(tmp_path/g/'dp/training/finished.json',dict(target_reached=True,updates=800,supervised_target_ticks=1000,
            coverage=dict(sampled_anchors=51200,anchor_population=8000)))
        write(tmp_path/g/'dp/prepared.json',dict(snapshot_sha256='common'))
        write(tmp_path/g/'backend'/g/'training/finished.json',dict(target_reached=True,updates=1200,
            supervised_ticks=12000,supervised_tick_population=1000))
    assert d.matched_round_exposure(tmp_path,['QG01','G01'])['passed']
    path=tmp_path/'G01/dp/training/finished.json';v=read(path);v['supervised_target_ticks']+=1;write(path,v)
    with pytest.raises(ValueError,match='effective supervision exposure differs'):d.matched_round_exposure(tmp_path,['QG01','G01'])
    assert read(tmp_path/'matched_exposure.json')['passed'] is False


def test_fixed_ten_evaluation_does_not_depend_on_initial_outcomes(tmp_path):
    from test_dagger_evaluation import evaluation,callback,verify,RULES
    from tools import tight_finish as owner
    groups,plan=evaluation(tmp_path,('QG01','G01'));calls=[];rules=dict(RULES,initial_trials=10)
    write(tmp_path/'configuration.json',dict(residual_pair=dict(dagger_evaluation=rules)))
    assert owner.dagger_evaluation_rules(tmp_path,{},dict(cohort='development'))==rules
    result=owner.dagger_evaluation(tmp_path,groups,plan,rules,callback(tmp_path,calls),verify)
    assert result['complete'] and result['completed']==20 and result['unrun']==0
    assert all(read(tmp_path/g/'evaluation_summary.json')['attempts']==10 for g in groups)
    assert read(tmp_path/'dagger_evaluation_freeze.json')['version']=='dagger_development_fixed10_v1'
