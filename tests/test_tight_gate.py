import copy
import json
from pathlib import Path

import numpy as np
import pytest

from visuotactile.tight_gate import validate_config, paired_admission, validate_formal_manifest
from visuotactile.narrow_data import direct_configuration,angle


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/tight_state.json').read_text())


def test_historical_baseline_override_is_explicit_and_does_not_mutate_checkpoint():
    from tools.tight_finish import baseline_checkpoint
    from visuotactile.position_control import contract
    weights=object();saved=dict(contract=contract(24,True),kind='dp',updates=4500,
        configuration={'seed':17},ema=weights,optimizer=object())
    fixed=baseline_checkpoint(saved,'canonical_joint_state_v1')
    assert fixed['ema'] is weights and fixed['optimizer'] is saved['optimizer']
    assert fixed['configuration']['input_builder']=='canonical_joint_state_v1'
    assert 'input_builder' not in saved['configuration'] and fixed['updates']==4500
    with pytest.raises(ValueError):baseline_checkpoint(saved,'unknown')
    with pytest.raises(ValueError):baseline_checkpoint(dict(saved,contract=contract(32,True)),'canonical_joint_state_v1')
    with pytest.raises(ValueError):baseline_checkpoint(dict(saved,updates=0),'canonical_joint_state_v1')


def test_motion_gate_rejects_detours_even_if_duration_is_long():
    from tools.tight_finish import physical_motion_errors
    rules=dict(maximum_mean_speed_ratio=.9,maximum_insertion_path_ratio=1.25)
    measured={phase:{group:dict(decomposition_valid=True,path_ratio=1.1,mean_speed_ratio=.7)
        for group in ('hand','end_effector','object')} for phase in ('whole','insertion')}
    assert not physical_motion_errors(measured,rules)
    measured['insertion']['object']['path_ratio']=3.2
    assert 'insertion_object_path_excess' in physical_motion_errors(measured,rules)
    measured['insertion']['hand']['mean_speed_ratio']=1.01
    assert 'insertion_hand_actual_speed_not_reduced' in physical_motion_errors(measured,rules)
    del measured['insertion']['end_effector']
    assert 'insertion_end_effector_missing_physical_motion' in physical_motion_errors(measured,rules)
    measured['whole']['hand']['mean_speed_ratio']=float('nan')
    assert 'whole_hand_invalid_physical_motion' in physical_motion_errors(measured,rules)


def test_path_speed_decomposition_separates_longer_route_from_slower_motion():
    from tools.narrow_repair_audit import path_speed_comparison
    groups=('arm','hand','end_effector','object','end_effector_rotation','object_rotation')
    ref=dict(phases={'insertion':dict(seconds=2,**{g:dict(path=1,mean_speed=.5) for g in groups})})
    slow=dict(phases={'insertion':dict(seconds=6,**{g:dict(path=3,mean_speed=.5) for g in groups})})
    x=path_speed_comparison(slow,ref)['insertion']['object']
    assert x['path_contribution_seconds']==4 and x['speed_contribution_seconds']==0
    slow['phases']['insertion'].update({g:dict(path=1,mean_speed=1/6) for g in groups})
    x=path_speed_comparison(slow,ref)['insertion']['object']
    assert x['path_contribution_seconds']==0 and x['speed_contribution_seconds']==4


def row(alpha=1.,duration=3.,speed=1.):
    return dict(family='f0',configuration_sha256='c',reset_sha256='r',expert_sha256='e',alpha=alpha,
        success=True,numeric_valid=True,visual_passed=True,tactile_valid=True,post_success_actions=0,
        duration_s=duration,motion_stages={p:dict(ticks=30,joint_speed_rms_rad_s=speed)
            for p in ('grasp','lift_transport','insertion')})


def test_slow_alpha_or_median_does_not_replace_each_real_duration():
    gate=config()['development'];native=row()
    for duration in (0.,4.05,4.49,9.01,float('nan'),float('inf')):
        assert not paired_admission(row(.45,duration,.6),native,gate)['passed']
    for duration in (4.5,6.,9.):
        assert paired_admission(row(.55,duration,.6),native,gate)['passed']


def test_waiting_missing_phase_or_one_fast_phase_rejects():
    gate=config()['development'];native=row()
    assert not paired_admission(row(.5,6.,1.),native,gate)['passed']
    for phase in native['motion_stages']:
        slow=row(.5,6.,.5);slow['motion_stages'][phase]['joint_speed_rms_rad_s']=1.
        assert not paired_admission(slow,native,gate)['passed']
        slow['motion_stages'][phase]['ticks']=0
        assert not paired_admission(slow,native,gate)['passed']
    assert not paired_admission(dict(row(.5,6.,.5),post_success_actions=1),native,gate)['passed']


def test_same_family_reference_and_all_modalities_are_mandatory():
    gate=config()['development'];native=row();slow=row(.5,6.,.5)
    for key in ('family','configuration_sha256','reset_sha256','expert_sha256'):
        assert not paired_admission(dict(slow,**{key:'different'}),native,gate)['passed']
    for key in ('success','numeric_valid','visual_passed','tactile_valid'):
        assert not paired_admission(dict(slow,**{key:False}),native,gate)['passed']
        assert not paired_admission(slow,dict(native,**{key:False}),gate)['passed']
    assert not paired_admission(slow,dict(native,speed_scheme=dict(arm=1.,hand=1.,hand_target_cap_rad_s=2)),gate)['passed']


def test_direct_sampler_reads_every_configured_range_and_rejects_bad_bounds():
    domain=config()['domain'];domain.update(object_xy_halfwidth=[.001,.002],relative_xy_halfwidth=[.002,.001],
        rotation_radius_degrees=1.,table_z_m=[.38,.38],object_height_above_table_origin_m=[.25,.25],
        native_joint_lerp=.007,joint_velocity_halfwidth=.021)
    for seed in range(30):
        r=direct_configuration(domain,seed);pose=np.asarray(r['fixed_start_pose']);center=domain['center']
        assert np.all(np.abs(pose[:2]-center['object_xyz'][:2])<=domain['object_xy_halfwidth'])
        hole=np.array([r['hole_x_range'][0],r['hole_y_range'][0]])
        assert np.all(np.abs(hole-pose[:2]-center['relative_xyz'][:2])<=domain['relative_xy_halfwidth'])
        assert angle(pose[3:],center['quaternion'])<1.01
        assert r['table_reset_z']==.38 and pose[2]==.63
        assert r['reset_dof_pos_random_interval_arm']==r['reset_dof_pos_random_interval_fingers']==.007
        assert r['reset_dof_vel_random_interval']==.021
    for bad in (dict(domain,object_xy_halfwidth=[-1,1]),dict(domain,table_z_m=[.4,.3]),dict(domain,rotation_radius_degrees=float('nan'))):
        with pytest.raises(ValueError):direct_configuration(bad,1)
    with pytest.raises((ValueError,KeyError)):direct_configuration(dict(center=domain['center']),1)


def test_current_config_rejects_legacy_camera_overlap_and_loose_gate():
    for modify in ('camera','overlap','ratio'):
        c=config()
        if modify=='camera':c['camera_content_validation']=False
        elif modify=='overlap':c['evaluation']['overlap_steps']=4
        else:c['development']['minimum_duration_ratio']=1.35
        with pytest.raises(ValueError):validate_config(c)
    assert validate_config(config())


def test_camera_legacy_path_rejected_before_kit(tmp_path):
    from tools.narrow_collection import environment
    (tmp_path/'session.json').write_text(json.dumps(dict(resource_policy='shared_recent_v1',deadline=1,
        collection_deadline=1,camera_content_validation=False)))
    with pytest.raises(ValueError,match='explicit camera_content_validation'):
        environment(tmp_path)


def test_constant_rgb_flags_true_do_not_pass_content():
    from visuotactile.camera_validity import frame_metrics
    rgb=np.full((3,3,10,10),238,dtype=np.uint8);depth=np.ones((3,1,10,10),dtype=np.float32)
    assert not any(frame_metrics(rgb,depth,np.ones_like(depth,dtype=bool))['content_valid'])


def test_formal_manifest_rejects_historical_missing_or_native_data():
    c=config()
    with pytest.raises(ValueError):validate_formal_manifest(dict(version=3,collection_finished=True,episodes=[]),c)
    m=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,frozen_alpha=.5,
        episodes=[dict(id='x',family='f',split='train',alpha=1.,state_valid=True,multi_view_valid=True,tactile_valid=True,
            admission=dict(passed=True),reference_sha256='r',visual_acceptance_sha256='v')])
    with pytest.raises(ValueError,match='Native or mixed'):validate_formal_manifest(m,c)


def test_old_endpoint_selector_rejects_one_under_ratio_even_when_median_passes():
    from tools.narrow_repair_pipeline import select_speed
    from test_narrow_repair import sample
    rows=[sample(i,1.,duration=3.,speed=1.) for i in range(3)]
    rows += [sample(i,a,duration=4.2 if i==0 else 6.) for i in range(3) for a in (.55,.60,.65)]
    assert not select_speed(rows)['passed']


def test_new_online_proprio_preserves_encoder_bits_legacy_is_explicit():
    import torch
    from visuotactile.state_history import controller_proprio
    from visuotactile.native_data import proprio_radians
    limits=torch.tensor([[-2.967059728,2.967059728]]*29)
    q=torch.linspace(-2.1,2.1,29);qd=torch.linspace(-.1,.1,29)
    normalized=2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1
    packet=dict(proprio=torch.cat((normalized,qd))[None])
    diag=dict(joint_position_rad=q[None],joint_velocity_rad_s=qd[None])
    canonical=controller_proprio(packet,diag,limits,'canonical_joint_state_v1')
    legacy=controller_proprio(packet,diag,limits,'legacy_packet_inverse_v1')
    assert torch.equal(canonical,torch.cat((q,qd)))
    assert torch.equal(legacy,proprio_radians(packet['proprio'][0],limits))
    assert not torch.equal(canonical,legacy), 'Fixture must expose the original roundtrip loss'
    with pytest.raises(ValueError):controller_proprio(packet,diag,limits,'unknown')
    with pytest.raises(ValueError):controller_proprio(packet,dict(diag,joint_position_rad=torch.ones(1,28)),limits,'canonical_joint_state_v1')


def test_lossless_capture_codec_preserves_rgb_depth_and_valid_bits(tmp_path):
    import h5py
    from visuotactile.recording import CompactEpisodeStream
    rng=np.random.default_rng(31)
    fields={'frames/rgb_u8':rng.integers(0,256,(2,3,3,12,20),dtype=np.uint8),
            'frames/depth_m':rng.random((2,3,1,12,20),dtype=np.float32),
            'frames/depth_valid':rng.random((2,3,1,12,20))>.2}
    for codec in ('lzf','gzip4_lossless_v1'):
        writer=CompactEpisodeStream(tmp_path/codec,{},'e',compression=codec)
        for k,v in fields.items():writer._extend(k,v)
        writer.file.close()
        with h5py.File(writer.path) as f:
            for k,v in fields.items():assert np.array_equal(f['episodes/e/'+k][:],v)
    with pytest.raises(ValueError):CompactEpisodeStream(tmp_path/'invalid',{},'e',compression='lossy')


def test_formal_passed_flag_cannot_override_bad_measured_duration_or_motion():
    c=config();c['collection'].update(minimum_accepted_episodes=3,minimum_accepted_families=3)
    entries=[dict(id=k,family=k,split=k,alpha=.525,state_valid=True,multi_view_valid=True,tactile_valid=True,
        reference_sha256='r',visual_acceptance_sha256='v',admission=paired_admission(row(.525,6.,.5),row(),c['development']))
        for k in ('train','validation','test')]
    m=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,frozen_alpha=.525,episodes=entries)
    assert validate_formal_manifest(m,c)
    for field,value in [('duration_ratio',1.35),('duration_ratio',3.1),('actual_phase_speed_ratios',{})]:
        bad=copy.deepcopy(m);bad['episodes'][0]['admission'][field]=value
        with pytest.raises(ValueError):validate_formal_manifest(bad,c)


def test_tactile_requires_correct_past_and_two_substep_clocks(tmp_path):
    import h5py
    from visuotactile.tight_gate import sensor_clock_admission
    with h5py.File(tmp_path/'clock.h5','w') as g:
        stamps=np.arange(4)/60;after=np.arange(4*2*5*3).reshape(4,2,5,3).astype('float32');before=np.concatenate([np.zeros((1,2,5,3),np.float32),after[:-1]])
        fields={'obs/time_s':stamps,'obs/contact_time_s':stamps[:,None]+[-1/120,0.],
            'transition/contact_time_after_s':stamps[:,None]+[1/120,1/60],
            'obs/camera_time_s':np.repeat(stamps[:,None],3,axis=1),'obs/contact_valid':np.array([False,True,True,True]),
            'obs/contact_substeps_w_N':before,'transition/contact_substeps_after_w_N':after,'obs/contact_w_N':before[:,-1]}
        for k,v in fields.items():g.create_dataset(k,data=v)
        assert sensor_clock_admission(g)
        g['obs/contact_time_s'][2,0]+=1/120
        with pytest.raises(ValueError,match='timestamps'):sensor_clock_admission(g)
        g['obs/contact_time_s'][:]=fields['obs/contact_time_s']
        g['obs/contact_substeps_w_N'][2]=after[2]
        with pytest.raises(ValueError,match='prior completed'):sensor_clock_admission(g)


def test_standalone_owner_automatically_advances_only_after_real_gate(tmp_path):
    from unittest.mock import patch,MagicMock
    import fcntl
    from tools.tight_state import run
    from tools.runtime import write
    for passed in (False,True):
        root=tmp_path/str(passed);root.mkdir()
        write(root/'status.json',dict(state='prepared'))
        write(root/'session.json',dict(development_deadline=1e12))
        write(root/'development_gate.json',dict(passed=passed,reason='measured_gate_failure'))
        worker=MagicMock();worker.poll.return_value=0;worker.returncode=0
        def advance(path):
            assert path==root
            with (root/'owner.lock').open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with patch('tools.kit_resource_preflight.inspect',return_value={'resource_admitted':True}), \
             patch('tools.tight_state.subprocess.Popen',return_value=worker), \
             patch('tools.tight_state.proc',return_value=None), \
             patch('tools.tight_state.kill_owned'),patch('tools.tight_state.check_protected'), \
             patch('tools.tight_finish.pipeline',side_effect=advance) as pipeline:
            run(root)
        assert pipeline.call_count==int(passed)
        assert json.loads((root/'status.json').read_text())['owned_workers_exited']


def test_current_entry_has_no_retired_pipeline_imports():
    import ast
    from pathlib import Path
    for name in ('tight_state','tight_finish','tight_final_calibration','tight_check','tight_audit'):
        tree=ast.parse(Path('tools',name+'.py').read_text())
        assert not any(isinstance(n,ast.ImportFrom) and n.module in ('tools.narrow_collection','tools.narrow_repair_pipeline') for n in ast.walk(tree))


def test_owner_sigterm_runs_cleanup_and_restores_handler():
    import os,signal
    from tools.runtime import supervised_owner
    cleaned=[];previous=signal.getsignal(signal.SIGTERM)
    @supervised_owner
    def owner():
        try:os.kill(os.getpid(),signal.SIGTERM)
        finally:cleaned.append(True)
    with pytest.raises(InterruptedError,match='Owner received SIGTERM'):owner()
    assert cleaned==[True] and signal.getsignal(signal.SIGTERM)==previous


def test_passing_fixture_gpu_receipts_cannot_admit_formal_data(tmp_path):
    import hashlib
    from tools.runtime import write
    from visuotactile.tight_gate import validate_training_gates
    data=tmp_path/'state_data';data.mkdir()
    (data/'data.json').write_text('{}');(data/'statistics.pt').write_bytes(b'formal-statistics')
    expected={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in [('source_data_sha256',data/'data.json'),('statistics_sha256',data/'statistics.pt')]}
    for name in ('cpu_audit','gpu_parity','gradient_gate'):write(tmp_path/(name+'.json'),dict(passed=True,**expected))
    for name in ('restore_gate','actuator_gate'):write(tmp_path/(name+'.json'),dict(passed=True))
    assert validate_training_gates(tmp_path,data)
    write(tmp_path/'gpu_parity.json',dict(passed=True,source_data_sha256='old-fixture',statistics_sha256=expected['statistics_sha256']))
    with pytest.raises(ValueError,match='dataset/statistics mismatch'):validate_training_gates(tmp_path,data)


def test_current_cache_rechecks_real_rgbd_despite_manifest_valid_flags(tmp_path):
    from unittest.mock import patch
    import hashlib
    from tools.runtime import write
    from visuotactile.narrow_data import build_cache
    c=config();c['collection'].update(minimum_accepted_episodes=3,minimum_accepted_families=3)
    write(tmp_path/'configuration.json',c)
    source=tmp_path/'source.h5';source.write_bytes(b'bad legacy source')
    entries=[dict(id=str(i),family=str(i),split=split,alpha=.55,path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        state_valid=True,multi_view_valid=True,tactile_valid=True,reference_sha256='r',visual_acceptance_sha256='v',
        admission=dict(passed=True,duration_ratio=2.,actual_phase_speed_ratios={k:.5 for k in ['grasp','lift_transport','insertion']})) for i,split in enumerate(['train','validation','test'])]
    write(tmp_path/'manifest.json',dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,frozen_alpha=.55,episodes=entries))
    with patch('visuotactile.camera_validity.inspect_capture',return_value={'passed':False}) as inspect:
        with pytest.raises(ValueError,match='Actual RGB-D content failed'):build_cache(tmp_path)
        inspect.assert_called_once_with(str(source))
    assert not (tmp_path/'state_data/statistics.pt').exists()


def reliability_fixture():
    from visuotactile.tight_gate import scheme_reliability
    scheme=dict(kind='closed_loop',arm=.5,hand=1.,phase='all')
    plan=dict(speed_scheme=scheme,families=['a','b','c'],slow_attempts=12,repeats_per_family=4,
        reliability=dict(minimum_task_successes=11,minimum_admitted=9,minimum_family_successes=3,minimum_family_admitted=2))
    rows=[dict(identity=f'{f}{i}',family=f,speed_scheme=scheme,success=True,legal=True,software_complete=True,
        admission=dict(passed=True)) for i in range(4) for f in plan['families']]
    return scheme_reliability,plan,rows


def test_reliability_separates_task_success_individual_admission_and_coverage():
    check,plan,rows=reliability_fixture()
    rows[0].update(success=False,admission=dict(passed=False))
    rows[1]['admission']['passed']=False;rows[2]['admission']['passed']=False
    result=check(rows,plan)
    assert result['passed'] and result['successes']==11 and result['admitted']==9
    rows[3]['success']=False
    result=check(rows,plan)
    assert not result['passed'] and not result['task_reliability_passed']
    check,plan,rows=reliability_fixture()
    for i in (0,3,6):rows[i]['admission']['passed']=False
    result=check(rows,plan)
    assert result['task_reliability_passed'] and result['admission_yield_passed'] and not result['coverage_passed']


def test_reliability_short_run_or_mixed_parameters_cannot_pass():
    check,plan,rows=reliability_fixture()
    assert not check(rows[:3],plan)['passed']
    rows[0]['speed_scheme']=dict(plan['speed_scheme'],hand=.5)
    with pytest.raises(ValueError,match='Mixed scheme'):check(rows,plan)
    check,plan,rows=reliability_fixture()
    for row in rows[:2]:row['success']=False
    assert check(rows[:2],plan)['preregistered_threshold_unreachable']


def test_independent_arm_hand_parameters_keep_individual_duration_gate():
    slow=row(.5,6.,.7);slow['speed_scheme']=dict(arm=.5,hand=1.,kind='closed_loop',phase='all')
    assert paired_admission(slow,row(),config()['development'])['passed']
    slow['duration_s']=4.4
    assert not paired_admission(slow,row(),config()['development'])['passed']


def test_event_seed_changes_future_draws_without_changing_reset_or_external_rng():
    import torch
    from visuotactile.environment_rng import IsolatedEnvironment
    class Env:
        device='cpu'
        def reset(self,seed):return torch.rand(5)
        def step(self,command):return torch.rand(5)
    external=torch.get_rng_state().clone()
    first=IsolatedEnvironment(Env(),3);second=IsolatedEnvironment(Env(),3)
    assert torch.equal(first.reset(3),second.reset(3))
    first.begin_events(7);second.begin_events(8)
    assert not torch.equal(first.step(None),second.step(None))
    first.begin_events(9);second.begin_events(9)
    assert torch.equal(first.step(None),second.step(None))
    assert torch.equal(torch.get_rng_state(),external)
    with pytest.raises(ValueError):first.begin_events(-1)


def test_accepted_foundation_rejects_changed_implementation_or_receipt(tmp_path):
    import hashlib
    from tools.runtime import write
    from visuotactile.tight_gate import accepted_foundation
    source=tmp_path/'source.py';source.write_text('accepted math')
    gate=tmp_path/'gate.json';write(gate,dict(passed=True))
    fingerprint=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write(tmp_path/'accepted_foundation.json',dict(baseline_commit='e67bfc6fa1b66359b55e019ae9899240625d5295',
        unchanged_implementations={str(source):fingerprint(source)},gates={k:dict(path=str(gate),sha256=fingerprint(gate))
        for k in ['gpu_parity','gradient_gate','restore_gate','actuator_gate']}))
    assert accepted_foundation(tmp_path)
    source.write_text('changed math')
    with pytest.raises(ValueError,match='implementation changed'):accepted_foundation(tmp_path)
    source.write_text('accepted math');write(gate,dict(passed=False))
    with pytest.raises(ValueError,match='receipt changed'):accepted_foundation(tmp_path)
