"""CPU-only fixtures; no torch, GPU, simulator or experiment source mutation."""
import copy
import json
from pathlib import Path
import shutil
import pytest
from outputs.diagnostics.tight_paired_dagger_20261122_01 import audit_collection_recovery05 as a
from outputs.diagnostics.tight_paired_dagger_20261122_01.test_collection_noise import Fixture


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    f=Fixture(tmp_path);old=tmp_path/'consumed';new=tmp_path/'continued';old.mkdir();new.mkdir()
    payloads=tmp_path/'payloads';payloads.mkdir()
    initial=dict(passed=False,all_recorded_checks_passed=True,full48_verified=False,
        rows=[r for r in f.initial if (r['family'],r['opportunity'])!=('family3','alignment')],
        unverified_rows=[dict(family='family3',opportunity='alignment',passed=None,reason='cross-attempt; counterpart evidence unavailable')])
    f.plan['pairs']={r:dict(role=r,whole_pair=True) for r in 'AB'}
    for i,(row,case,assigned) in enumerate(zip(f.results,f.cases,f.assignments)):
        assigned.update(attempt='original' if i<15 else 'continued',attempt_ordinal=i if i<15 else i-15)
        result=row['result'];result['batch_execution']['queue_ordinal']=assigned['attempt_ordinal']
        if 7<=i<15:
            result.update(software_complete=False,legal=None,success=False,accepted_for_training=False)
        else:
            switched=case['opportunity']=='support';d=result['dagger'];result.update(complete_student_policy=f.plan['pairs'][case['role']],success=switched,accepted_for_training=False)
            d.update(expert_takeover_success=switched,student_autonomous_success=False,expert_valid_ticks=(result['steps']-13) if switched else 0,
                trigger_receipt=dict(reason='support_instability' if switched else 'no_trigger',never_entered_15mm=True,ever_entered_15mm=False,ever_entered_then_left_15mm=False))
            compact=payloads/(case['name']+'.bin');compact.write_bytes(b'synthetic compact '+case['name'].encode())
            result['dataset_capture']=dict(path=str(compact),sha256=a.sha(compact),verified_source_fields=186,accepted_success=switched)
        f.put(f.root/case['name']/'result.json',result)
        dest=(old if i<15 else new)/case['name'];shutil.move(str(f.root/case['name']),dest)
        row['evidence_directory']=str(dest)
        if result.get('software_complete'):
            f.put(dest/'capture_complete.json',dict(status='source_verified',steps=result['steps'],dataset_capture=result['dataset_capture'],numeric_path=str(dest/'numeric.h5')))

    byname={v['case']:v for v in f.assignments}
    for row in initial['rows']:
        row['attempt']=byname[row['reference']]['attempt']
        row.update(differences={'position':0.0},reference_legal=True,legal=True)
    pairs=a.recovered_assignments(f.plan,f.assignments,initial);checked={(v['family'],v['opportunity']) for v in initial['rows']}
    context=dict(plan=f.plan,assignments=f.assignments,initial=initial,pairs=pairs,checked=checked,rows=f.results,entries={},
        binding=dict(source_directories=[str(f.root),str(old),str(new)]),source_bindings={},summary=dict(opportunities=96,observable_pair_checks=dict(passed=47,unverified=1,failed=0,all_verified=False)))
    monkeypatch.setattr(a,'inputs',lambda _: (context['plan'],context['assignments'],context['initial'],context['pairs'],context['rows'],context['entries'],context['summary']))
    monkeypatch.setattr(a,'numeric_schedule',lambda p:f.numeric[p.name])
    return f,context


def run_noise(f,c,**kwargs):
    return a.noise('fixture',rng_auditor=f.rng,**kwargs)


def test_noise_88_available_8_incomplete_47_observed_pairs(fixture):
    f,c=fixture;r=run_noise(f,c)
    assert r['assignments']==96 and r['paired_assignments']==48
    assert r['observable_pair_checks']['passed']==47 and r['observable_pair_checks']['unverified']==1
    assert r['observable_pair_checks']['all_verified'] is False and r['cpu_receipt_checks_passed']==88
    assert len(r['unavailable'])==8 and len(f.calls)==88
    assert r['exact_cuda_full_endpoint_checks']==0 and not r['cuda_used']
    assert len(r['pairs'])==48
    assert r['all96_execution_endpoints_certified'] is False


def test_continued_attempt_local_ordinal_preserved(fixture):
    f,c=fixture;r=run_noise(f,c)
    row=next(r for r in r['rows'] if r['ordinal']==15)
    assert row['attempt_ordinal']==0 and row['attempt']=='continued'


def test_full_endpoint_includes_shadow_after_takeover(fixture):
    f,c=fixture;r=run_noise(f,c)
    assert r['rows'][0]['actual_native_plans']==3 and r['rows'][0]['student_executed_plan_calls']==2
    assert r['rows'][0]['shadow_after_switch_plan_calls']==1
    assert r['observed_takeover_rng_states']==0


def test_simulated_cuda_reconstructs_complete_cases_only(fixture):
    f,c=fixture
    def rng(directory,plans,*,reconstruct_cuda):
        assert reconstruct_cuda
        return dict(receipt_hashes_verified=True,planning_calls=plans,native_draw_reconstruction_verified=True,start_state=[7091,0],end_state=[7091,plans])
    r=a.noise('fixture',reconstruct_cuda=True,rng_auditor=rng,prefix_reconstructor=lambda draws:[7091,draws])
    assert r['exact_cuda_full_endpoint_checks']==88 and r['student_prefix_cuda_reconstructed']==88
    assert r['observable_pair_checks']['all_verified'] is False


@pytest.mark.parametrize('mutation',['all48','missing_gap','wrong_gap','pair_name','attempt_order','attempt_ordinal','seed'])
def test_assignment_or_observed_pair_change_rejected(fixture,mutation):
    f,c=fixture
    if mutation=='all48':c['initial']['passed']=True
    elif mutation=='missing_gap':c['initial']['unverified_rows']=[]
    elif mutation=='wrong_gap':c['initial']['unverified_rows'][0]['opportunity']='support'
    elif mutation=='pair_name':c['initial']['rows'][0]['case']='other'
    elif mutation=='attempt_order':c['assignments'][15]['attempt']='original'
    elif mutation=='attempt_ordinal':c['assignments'][15]['attempt_ordinal']=15
    else:c['assignments'][0]['sampler_seed']=99
    with pytest.raises(ValueError):a.recovered_assignments(c['plan'],c['assignments'],c['initial'])


def test_evidence_path_recorded_for_both_attempts(fixture):
    f,c=fixture
    assert all(Path(row['evidence_directory']).parent.name==('consumed' if i<15 else 'continued') for i,row in enumerate(c['rows']))


def test_missing_complete_endpoint_fails_instead_of_skipping(fixture):
    f,c=fixture;(Path(c['rows'][0]['evidence_directory'])/'sampler_rng_end.json').unlink()
    with pytest.raises(FileNotFoundError):run_noise(f,c)


def test_actual_local_ordinal_mismatch_rejected(fixture):
    f,c=fixture;c['rows'][15]['result']['batch_execution']['queue_ordinal']=15
    with pytest.raises(ValueError,match='local physical lane/order'):run_noise(f,c)


def test_changed_rng_reseed_rejected(fixture):
    f,c=fixture
    def rng(directory,plans,**kwargs):
        r=f.rng(directory,plans,**kwargs)
        if directory.name.startswith('B'):r['start_state']=[99]
        return r
    with pytest.raises(ValueError,match='reseeding differs'):
        a.noise('fixture',rng_auditor=rng)


def test_integrity_checks_every_complete_capture_and_retains_failures(fixture,monkeypatch):
    f,c=fixture;calls=[]
    def numeric(directory,result,*,compact_path):
        calls.append(directory.name)
        return dict(legal_expert_ticks=result['dagger']['expert_valid_ticks'],student_prefix_ticks=result['dagger']['student_ticks'],queue_or_invalid_expert_ticks=0,compact_complete_numeric_history_checked=True)
    from tools import tight_paired_collection as tc
    monkeypatch.setattr(a,'check_numeric',numeric)
    monkeypatch.setattr(tc,'audit_trigger',lambda p,r:dict(actual_switch_tick=r['dagger']['recovery_coverage']['switch_tick']))
    r=a.integrity('fixture')
    assert len(calls)==len(r['complete_payload_sha256'])==88
    assert len(r['technical_failures'])==8 and r['observable_pair_checks']['all_verified'] is False
    assert sum(v['software_incomplete'] for x in r['by_assigned_opportunity'].values() for v in x.values())==8
    assert sum(v['unknown_initialization'] for x in r['by_assigned_opportunity'].values() for v in x.values())==8
    assert sum(v['opportunities'] for x in r['by_assigned_opportunity'].values() for v in x.values())==96


def test_incomplete_cannot_enter_own_supervision(fixture,monkeypatch):
    f,c=fixture;c['entries'][c['rows'][7]['case']]={}
    from tools import tight_paired_collection as tc
    monkeypatch.setattr(a,'check_numeric',lambda p,r,**k:dict(legal_expert_ticks=0))
    monkeypatch.setattr(tc,'audit_trigger',lambda p,r:dict(actual_switch_tick=r['dagger']['recovery_coverage']['switch_tick']))
    with pytest.raises(ValueError,match='Unsafe incomplete/illegal admission'):
        a.integrity('fixture')


@pytest.mark.parametrize('value',[1.0001e-5,float('nan'),float('inf'),-1e-9])
def test_numeric_observable_tolerance_checked_even_with_pass_flag(fixture,value):
    f,c=fixture;c['initial']['rows'][0]['differences']['position']=value
    with pytest.raises(ValueError,match='numeric tolerance'):
        a.recovered_assignments(c['plan'],c['assignments'],c['initial'])


@pytest.fixture
def capture_receipts(tmp_path):
    compact=tmp_path/'data.h5';compact.write_bytes(b'fixture')
    numeric=tmp_path/'numeric.h5';numeric.write_bytes(b'fixture')
    capture=dict(path=str(compact),sha256='fixed-sha',verified_source_fields=186)
    receipt=dict(numeric_path=str(numeric),dataset_capture=copy.deepcopy(capture),steps=10,status='source_verified')
    result=dict(steps=10,dataset_capture=capture,async_capture=copy.deepcopy(receipt))
    camera=dict(passed=True,frames=10,views=['fixed_high','fixed_side','wrist_rgbd'],invalid_frames_per_view=[0,0,0],
        depth_mask_exact=True,camera_flags_all_true=True,synchronized_clock=True,frames_strictly_increase=True)
    return tmp_path,compact,result,receipt,camera


def run_capture_receipts(fixture):
    directory,compact,result,receipt,camera=fixture
    (directory/'capture_complete.json').write_text(json.dumps(receipt))
    (directory/'camera_acceptance.json').write_text(json.dumps(camera))
    return a.admitted_capture_receipts(directory,result,compact,'fixed-sha')


def test_full_field_and_camera_receipt_binding(capture_receipts):
    value=run_capture_receipts(capture_receipts)
    assert value==dict(source_verified_fields=186,all_frame_camera_acceptance_bound=True)


@pytest.mark.parametrize('mutation',['status','steps','numeric_path','source_hash','source_fields','camera_pass','camera_frames','camera_clock'])
def test_admitted_capture_receipt_changes_rejected(capture_receipts,mutation):
    directory,compact,result,receipt,camera=capture_receipts
    if mutation=='status':receipt['status']='pending'
    elif mutation=='steps':receipt['steps']=9
    elif mutation=='numeric_path':receipt['numeric_path']=str(directory/'foreign.h5')
    elif mutation=='source_hash':receipt['dataset_capture']['sha256']='changed'
    elif mutation=='source_fields':
        for value in (receipt['dataset_capture'],result['dataset_capture'],result['async_capture']['dataset_capture']):value['verified_source_fields']=0
    elif mutation=='camera_pass':camera['passed']=False
    elif mutation=='camera_frames':camera['frames']=9
    else:camera['synchronized_clock']=False
    with pytest.raises(ValueError):run_capture_receipts(capture_receipts)
