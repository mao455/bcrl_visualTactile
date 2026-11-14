import json

import pytest

from tools.tight_batch_throughput_report import report,summarize
from visuotactile.compact import sha256


def write(root,name,value):
    (root/name).write_text(json.dumps(value))


def collection(root,*,batch=False):
    root.mkdir()
    cases=[dict(name=f'case{i}',role='A',spec=dict(source_identity=f'original{i}',family=f'family{i}',
        narrow_reset=dict(x=i),takeover_tick=10**9),dagger=dict(trigger=dict(version='contact',delay_ticks=0)))
        for i in range(2)]
    pair=dict(frontend_sha256='front',backend_sha256='rear')
    plan=dict(kind='takeover',test_used=False,cases=cases,pairs=dict(A=pair),source_commit='source',
        first_plan_forensics=not batch)
    if batch:plan.update(batch=dict(lanes=8,vision_microbatch=8),execution_backend=dict(physics_device='cpu'))
    rows=[]
    for index in range(2):
        valid=8 if index==0 else 4
        value=dict(software_complete=True,legal=True,success=index==0,steps=10+10*index,
            accepted_for_training=index==0,actual_exposure=dict(supervised_ticks=valid),
            dagger=dict(expert_ticks=valid+1,expert_valid_ticks=valid,expert_takeover_success=index==0),
            wall_profile=dict(compact_append_s=2,finalize_verify_s=3),
            async_capture=dict(profile=dict(worker_verify_s=20,submission_wait_s=.1)))
        rows.append(dict(case=f'case{index}',role='A',result=value,timing=dict(total_seconds=10)))
    write(root,'plan.json',plan);write(root,'results.json',rows)
    write(root,'correction_entries.json',dict(case0=dict(episode='case0')))
    write(root,'owner_finished.json',dict(software_complete=True,returncode=0,seconds=20 if batch else 30))
    write(root,'result.json',dict(software_complete=True,worker_error=None,cases=2,elapsed_seconds=18))
    write(root,'session.json',dict(deadline=10**12));write(root,'initialization.json',dict(seconds=2))
    if batch:write(root,'wave_profiles.json',[dict(wave=0,cases=2,control_ticks=30,seconds=10,
        admission_s=3,capture_writer=dict(peak_reserved_bytes=123,peak_queued_or_copying=2))])
    return root


def test_real_json_denominators_exclude_failed_expert_suffix_and_async_overlap(tmp_path):
    serial=collection(tmp_path/'serial');batch=collection(tmp_path/'batch',batch=True)
    value=report(serial,batch)
    counts=value['batch']['counts'];profile=value['batch']['profile']
    assert counts['attempts']==2 and counts['accepted_trajectories']==1
    assert counts['all_valid_expert_ticks']==12 and counts['accepted_valid_expert_ticks']==8
    assert profile['primary_rates']['accepted_valid_expert_ticks_per_hour']==1440
    assert profile['async_profile_sums']['worker_verify_s']==40
    assert profile['owner_seconds']==20 and profile['worker_elapsed_seconds']==18 and profile['wave_seconds']==10
    assert profile['admission_seconds']==3 and profile['outside_wave_after_initialization_seconds']==6
    assert value['comparison']['ratios_batch_over_serial']['accepted_trajectories_per_hour']==1.5
    assert value['comparison']['same_registered_case_conditions'] and value['comparison']['same_pairs']
    assert value['serial']['bindings']['results.json']['sha256']==sha256(serial/'results.json')
    assert value['serial']['profile']['serial_case_seconds']==20


def test_missing_separate_admission_timer_stays_unknown(tmp_path):
    root=collection(tmp_path/'batch',batch=True)
    waves=json.loads((root/'wave_profiles.json').read_text());del waves[0]['admission_s']
    write(root,'wave_profiles.json',waves)
    value=summarize(root,batch=True)
    assert value['profile']['admission_seconds'] is None
    assert value['profile']['outside_wave_after_initialization_seconds']==6


@pytest.mark.parametrize('failure',['owner_incomplete','case_missing','accepted_mismatch','wrong_exposure','wave_clock','nonfinite_time'])
def test_partial_or_mismatched_actual_records_cannot_be_reported_complete(tmp_path,failure):
    root=collection(tmp_path/'batch',batch=True)
    if failure=='owner_incomplete':
        write(root,'owner_finished.json',dict(software_complete=False,returncode=-15,seconds=20))
    elif failure=='case_missing':
        rows=json.loads((root/'results.json').read_text());write(root,'results.json',rows[:1])
    elif failure=='accepted_mismatch':write(root,'correction_entries.json',{})
    elif failure=='wrong_exposure':
        rows=json.loads((root/'results.json').read_text());rows[0]['result']['actual_exposure']['supervised_ticks']=9
        write(root,'results.json',rows)
    elif failure=='wave_clock':
        waves=json.loads((root/'wave_profiles.json').read_text());waves[0]['control_ticks']=29
        write(root,'wave_profiles.json',waves)
    else:write(root,'owner_finished.json',dict(software_complete=True,returncode=0,seconds=float('nan')))
    with pytest.raises(ValueError):summarize(root,batch=True)


def test_mismatched_model_or_initials_are_explicit(tmp_path):
    serial=collection(tmp_path/'serial');batch=collection(tmp_path/'batch',batch=True)
    plan=json.loads((batch/'plan.json').read_text());plan['pairs']['A']['frontend_sha256']='new'
    plan['cases'][0]['spec']['narrow_reset']['x']=7;write(batch,'plan.json',plan)
    comparison=report(serial,batch)['comparison']
    assert not comparison['same_pairs'] and not comparison['same_registered_case_conditions']
