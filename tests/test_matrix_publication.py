"""Read-only publication of partial/failed matrices with explicit target semantics."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.tight_diagnostics import matrix_publish


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bytes_at(path, value=b'local binary remains local'):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def metrics(error):
    return [dict(split='validation',phase='all',group=group,motion='all',ticks=12,episodes=1,
        rmse_rad=error,hold_rmse_rad=.2,ee_rmse_mm=error*1000,fingertip_rmse_mm=2.,palm_angle_mean_deg=.3)
        for group in ('arm','hand')]


def stage(root, name, groups, target='u_expert', residual=False, evaluate=True):
    node=root/name
    config=dict(groups=groups,supervision=target,evaluate=evaluate)
    put(node/'session.json',{'residual_pair' if residual else 'vision_pair':config})
    put(node/'status.json',dict(state='FAILED',reason='bounded example'))
    put(node/'evaluation_plan.json',[dict(identity=f'case{i}',spec=dict(seed=i)) for i in range(2)])
    return node


def model(node, group, target='u_expert', residual=False):
    d=node/group
    cp_sha=bytes_at(d/'training/best.pt')
    put(d/'training/configuration.json',dict(supervision=target,selection='fixed validation4',
        **(dict(residual_version='target_residual_transformer60_v1',architecture={'layers':2}) if residual else {})))
    put(d/'training/finished.json',dict(updates=10,stop_reason='fixed_exposure',selection='fixed validation4',
        **(dict(control_tick_population=6,control_tick_epochs=2) if residual else {})))
    put(d/'training/progress.json',dict(update=10,anchor_equivalent_epochs=2))
    put(d/'offline/result.json',dict(complete=True,selected_update=7,checkpoint_sha256=cp_sha,
        metrics=metrics(.02),**(dict(base_metrics=metrics(.03)) if residual else {})))
    if residual:
        put(d/'training/validation.json',[dict(update=10,control_tick_epochs=2,train=dict(all=dict(normalized_equal_mse=.01,
            arm_rmse_rad=.01,hand_rmse_rad=.03)),validation=dict(all=dict(normalized_equal_mse=.02,arm_rmse_rad=.02,hand_rmse_rad=.04)))])
        (d/'training/loss.jsonl').write_text(json.dumps(dict(update=10,supervised_ticks=12,loss=.03))+'\n')
    else:
        (d/'training/validation.jsonl').write_text(json.dumps(dict(update=10,anchor_equivalent_epochs=2,
            metrics=dict(arm_rmse=.02,hand_rmse=.03,selection_score=.02,ee_distance_rmse_mm=2),
            train8=dict(arm_rmse=.01,hand_rmse=.02,selection_score=.01,ee_distance_rmse_mm=1)))+'\n')
        (d/'training/training.jsonl').write_text(json.dumps(dict(update=10,anchor_equivalent_epochs=2,loss=.03))+'\n')
    return cp_sha


def trial(node, group, identity, *, success=False, rejected=False):
    d=node/group/'evaluation'/identity
    row=dict(identity=identity,seed=0,legal=not rejected,software_complete=True,
        success=success,reasons=dict(max_successes=success,timeout=not success),steps=5,
        native_lift=False,native_subgoals=2 if success else 0,native_alignment=success,post_success_actions=0)
    put(d/'result.json',row)
    if not rejected:
        bytes_at(d/'numeric.h5');bytes_at(d/'trace.npz');bytes_at(d/'rollout.mp4')
        put(d/'video_verified.json',dict(decoded_frames=5,frame_rate=60))
    return row


def matrix(tmp_path):
    root=tmp_path/'run';out=tmp_path/'public'
    put(root/'configuration.json',dict(method_matrix=dict(groups=['A','B','C','Q00','Q01']),
        vision_pair=dict(groups=['A','B','C']),residual_pair=dict(groups=[])))
    put(root/'status.json',dict(state='FAILED',owned_workers_exited=True,reason='preserved failure'))
    put(root/'result.json',dict(stages={}))
    command=stage(root,'command',['A','B','C']);model(command,'A')
    rows=[trial(command,'A','case0',success=True),trial(command,'A','case1',rejected=True)]
    put(command/'A/evaluation_summary.json',dict(attempts=2,valid=1,rejected=1,successes=1,results=rows))
    put(command/'B/evaluation/case0/started.json',dict(spec={'seed':0}))
    q=stage(root,'configuration',['Q00'],target='q_reached',evaluate=False);model(q,'Q00',target='q_reached')
    qres=stage(root,'configuration_residual',['Q01'],residual=True);model(qres,'Q01',residual=True)
    (root/'evaluation.log').write_text('\n'.join('PhysX error: setLinearVelocity: Body must be non-kinematic! '+str(i) for i in range(162)))
    source=tmp_path/'retained_source.h5';source.write_bytes(b'must not be read or copied')
    put(command/'manifest.json',dict(episodes=[dict(id='expert',path=str(source),sha256='f'*64)]))
    return root,out,source


def test_partial_matrix_keeps_all_denominators_and_unmeasured_distinct(tmp_path,monkeypatch):
    root,out,source=matrix(tmp_path)
    import torch
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('CPU publisher must not deserialize a policy'))
    original=Path.open
    def protected(path,*args,**kwargs):
        if path==source:pytest.fail('Publisher must use retained manifest hash, not reopen source trajectories')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',protected)
    summary=matrix_publish(root,out)
    groups=summary['stages']['command']['groups']
    assert list(groups)==['A','B','C']
    assert groups['A']['closed_loop']['successes']==1 and groups['A']['closed_loop']['valid']==1
    assert groups['A']['closed_loop']['rejected']==1 and groups['A']['closed_loop']['attempts']==2
    assert groups['B']['closed_loop']['software_incomplete']==1 and groups['B']['closed_loop']['unstarted']==1
    assert groups['B']['closed_loop']['success_rate'] is None
    assert groups['C']['training_state']=='unmeasured' and groups['C']['updates'] is None
    assert groups['C']['closed_loop']['successes'] is None and groups['C']['closed_loop']['success_rate'] is None
    assert summary['status']['state']=='FAILED' and summary['physx_error_lines']==162
    with (out/'physx_errors.csv').open() as f:assert len(list(csv.DictReader(f)))==162
    assert not list(out.rglob('*.pt')) and not list(out.rglob('*.h5')) and not list(out.rglob('*.mp4'))
    assert (out/'training_curves.png').is_file() and (out/'offline_errors.png').is_file()
    local=json.loads((out/'local_artifacts.json').read_text())
    assert local[str(source)]['sha256']=='f'*64 and 'not rehashed' in local[str(source)]['hash_scope']
    assert summary['selection_performed_by_exporter'] is False and summary['inference_or_training_performed'] is False


def test_configuration_and_command_error_targets_remain_separate(tmp_path):
    root,out,_=matrix(tmp_path);matrix_publish(root,out)
    with (out/'offline_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert {r['target'] for r in rows if r['model']=='Q00'}=={'q_reached'}
    assert {r['target'] for r in rows if r['model']=='Q01'}=={'u_expert'}
    nominal=[r for r in rows if r['model']=='Q01' and r['prediction_kind'].startswith('nominal_against')]
    assert nominal and all(r['prediction_kind']=='nominal_against_command_not_configuration_fit' for r in nominal)


def test_formal_outcomes_do_not_change_selected_model_or_training_selection(tmp_path):
    root,out,_=matrix(tmp_path)
    formal=stage(root,'formal_test',['A']);cp_sha=model(formal,'A')
    rows=[trial(formal,'A','case0'),trial(formal,'A','case1')]
    put(formal/'A/evaluation_summary.json',dict(attempts=2,valid=2,rejected=0,successes=0,results=rows))
    put(formal/'status.json',dict(state='COMPLETE'))
    result=matrix_publish(root,out);g=result['stages']['formal_test']['groups']['A']
    assert g['training_role']=='selected_model_reference' and g['selected_checkpoint']['sha256']==cp_sha
    assert g['selection']=='fixed validation4' and g['selected_update']==7
    assert g['closed_loop']['successes']==0 and g['closed_loop']['valid']==2 and g['closed_loop']['success_rate']==0
    with (out/'validation_curves.csv').open() as f:assert all(r['stage']!='formal_test' for r in csv.DictReader(f))


def test_running_export_and_checkpoint_drift_are_not_silent(tmp_path):
    root,out,_=matrix(tmp_path)
    put(root/'status.json',dict(state='RUNNING',owned_workers_exited=False))
    with pytest.raises(ValueError,match='stopped'):matrix_publish(root,out)
    assert not out.exists()
    put(root/'status.json',dict(state='FAILED',owned_workers_exited=True))
    (root/'command/A/training/best.pt').write_bytes(b'changed weights')
    report=matrix_publish(root,out)
    assert any(x['problem']=='offline_selected_checkpoint_SHA_mismatch' for x in report['publication_issues'])


def test_surface_diagnostic_uses_fixed_stride_and_keeps_contact_denominator():
    import torch
    from tools.tight_diagnostics import configuration_geometry_rows
    class Surfaces:
        links=[f'left_{name}_DP' for name in ('index','middle','ring','thumb','pinky')]
        finger_indices=[[i] for i in range(5)]
        def distances(self,q,pose):
            assert not q.requires_grad and len(q)==3
            return q[:,:5]
    pred=torch.full((25,29),.01);truth=pred.clone()
    pred[0,0]=-.002;pred[12,0]=.001;pred[24,0]=.008
    truth[0,0]=truth[12,0]=truth[24,0]=.0005
    contact=torch.zeros(25,5,dtype=torch.bool);contact[:,0]=True
    context=dict(object_pose_base=torch.zeros(25,7),contact_mask=contact,stage=torch.ones(25,dtype=torch.long))
    rows=configuration_geometry_rows(dict(prediction=pred,truth=truth),context,Surfaces(),identity='episode',split='validation',
        penetration_tolerance_m=.001,contact_tolerance_m=.002)
    assert [r['tick'] for r in rows]==[0,12,24]
    assert [r['prediction_kept_required_contacts'] for r in rows]==[0,1,0]
    assert sum(r['required_contacts'] for r in rows)==3
    assert sum(r['reference_reached_kept_required_contacts'] for r in rows)==3
    assert rows[0]['prediction_within_penetration_tolerance'] is False
    assert rows[0]['prediction_maximum_penetration_m']==pytest.approx(.002)


def test_truncated_failure_log_metric_does_not_fabricate_an_update(tmp_path):
    root,out,_=matrix(tmp_path)
    with (root/'command/A/training/training.jsonl').open('a') as f:f.write('{"update":11')
    report=matrix_publish(root,out)
    assert any(r['problem']=='incomplete_or_malformed_metric_line' for r in report['publication_issues'])
    with (out/'training_loss_intervals.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['stage']=='command' and r['model']=='A']
    assert len(rows)==1 and int(rows[0]['update'])==10


def test_geometry_uses_current_owner_selected_gate_not_stale_child_session(tmp_path):
    from tools.tight_diagnostics import accepted_matrix_physics_gate
    owner=tmp_path/'owner'
    put(owner/'session.json',dict(accepted_physics_gate='physics_gate_exitfix'))
    child=dict(matrix_owner=str(owner),accepted_physics_gate='physics_gate_repair')
    assert accepted_matrix_physics_gate(child)==owner/'physics_gate_exitfix'
    put(owner/'session.json',{})
    assert accepted_matrix_physics_gate(child)==owner/'physics_gate_repair'
    assert accepted_matrix_physics_gate(dict(matrix_owner=str(owner)))==owner/'physics_gate'


def test_all_gate_attempts_preserved_without_promoting_functional_success(tmp_path):
    root,out,_=matrix(tmp_path)
    put(root/'session.json',dict(accepted_physics_gate='physics_gate_exitfix'))
    put(root/'physics_gate/result.json',dict(complete=True))
    put(root/'physics_gate/exit.json',dict(returncode=1))
    put(root/'physics_gate/acceptance.json',dict(passed=False,worker_returncode=1))
    put(root/'physics_gate_repair/result.json',dict(complete=True))
    put(root/'physics_gate_repair/geometry_acceptance.json',dict(passed=False,live_collision_parity=False,
        blockers=['cooked hull unavailable'],inventory=dict(cooked_hulls=[])))
    put(root/'physics_gate_repair/teardown_interruption.json',dict(reason='isolated worker shutdown stalled'))
    put(root/'physics_gate_exitfix/result.json',dict(complete=True,software_complete=True))
    put(root/'physics_gate_exitfix/exit.json',dict(returncode=0))
    put(root/'physics_gate_exitfix/acceptance.json',dict(passed=True,worker_returncode=0))
    put(root/'physics_gate_exitfix/geometry_live.json',dict(live_collision_parity=False,inventory=dict(cooked_hulls=[])))
    report=matrix_publish(root,out);gates=report['physics_gate_attempts']
    assert list(gates)==['physics_gate','physics_gate_exitfix','physics_gate_repair']
    assert gates['physics_gate']['state']=='failed' and gates['physics_gate']['functional_complete']
    assert gates['physics_gate_repair']['state']=='functional_complete_without_accepted_exit'
    assert gates['physics_gate_repair']['geometry_passed'] is False
    assert gates['physics_gate_exitfix']['state']=='accepted' and gates['physics_gate_exitfix']['selected_by_session']
    assert report['root_receipts']['physics_gate_repair/teardown_interruption']['reason'].endswith('stalled')
    indexed=report['root_receipts']['physics_gate_exitfix/geometry_live']['artifact']
    assert indexed['exists'] and len(indexed['sha256'])==64
    sources=json.loads((out/'source_artifacts.json').read_text())
    assert str(root/'physics_gate_exitfix/geometry_live.json') in sources
    assert 'inventory' not in report['root_receipts']['physics_gate_exitfix/geometry_live']


def test_unregistered_result_cannot_replace_missing_planned_case_or_change_rate(tmp_path):
    root,out,_=matrix(tmp_path)
    command=root/'command'
    (command/'A/evaluation/case1/result.json').unlink()
    (command/'A/evaluation_summary.json').unlink()
    trial(command,'A','replacement',success=False)
    finished=command/'A/training/finished.json'
    receipt=json.loads(finished.read_text());receipt['target_reached']=False;put(finished,receipt)
    report=matrix_publish(root,out);group=report['stages']['command']['groups']['A'];counts=group['closed_loop']
    assert counts['attempts']==1 and counts['valid']==1 and counts['success_rate']==1
    assert counts['unregistered_attempts']==1 and counts['unstarted']==1
    assert counts['evaluation_state']=='partial' and counts['owner_state']=='FAILED'
    assert group['training_state']=='stopped_before_target'
    with (out/'closed_loop.csv').open() as f:rows=list(csv.DictReader(f))
    replacement=next(r for r in rows if r['identity']=='replacement')
    assert replacement['registered']=='False' and replacement['classification']=='valid'


def test_reused_base_has_zero_new_updates_and_uses_completed_historical_cohort(tmp_path):
    root,out,_=matrix(tmp_path);old=tmp_path/'old';resumed=old/'evaluation_resume_01'
    rows=[trial(resumed,'A','case0'),trial(resumed,'A','case1')]
    put(resumed/'A/evaluation_summary.json',dict(attempts=2,valid=2,rejected=0,successes=0,results=rows))
    put(root/'command/reuse_manifest.json',dict(source_root=str(old),models={},new_training_updates=0,new_evaluation_attempts=0))
    report=matrix_publish(root,out);group=report['stages']['command']['groups']['A']
    assert group['updates']==0 and group['historical_training']['updates']==10
    assert group['training_role']=='reused_selected_model' and group['exposure'] is None
    assert group['closed_loop']['successes']==0 and group['closed_loop']['valid']==2
    assert group['closed_loop']['new_attempts']==0 and group['closed_loop']['historical_attempts']==2
    assert report['stages']['command']['cohort']=='historical_development_reused'
    assert not (out/'command_A_updates.csv.gz').exists()


def test_new_q_evaluation_and_numeric_update_components_are_exported(tmp_path):
    import gzip
    root,out,_=matrix(tmp_path);q=root/'configuration'
    value=json.loads((q/'session.json').read_text());value['vision_pair']['evaluate']=True;put(q/'session.json',value)
    trial(q,'Q00','case0',success=True);trial(q,'Q00','case1')
    update=dict(update=10,loss=.03,anchor_equivalent_epochs=2,hand_geometry_seconds=.01,
        hand_geometry_gain=.5,components=dict(base=.02,hand_geometry=.01),geometry_sampling=dict(selected_rows=2),
        numerical_gradient_norm=1.2,note='non-numeric scope stays in indexed original')
    (q/'Q00/training/training.jsonl').write_text(json.dumps(update)+'\n')
    report=matrix_publish(root,out)
    assert report['stages']['configuration']['groups']['Q00']['closed_loop']['valid']==2
    with gzip.open(out/'configuration_Q00_updates.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['geometry_sampling.selected_rows']=='2.0'
    assert rows[0]['components.hand_geometry']=='0.01' and 'note' not in rows[0]
