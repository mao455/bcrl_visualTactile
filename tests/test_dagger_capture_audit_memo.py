"""Real closed files and checkpoint contracts; no CUDA or simulator imports."""
import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from visuotactile.compact import sha256
from visuotactile.models.native_dp import TargetResidual
from visuotactile.state_benchmark import (
    CAPTURE_AUDIT_MEMO_PROFILE, DaggerCaptureAuditMemo, audit_dagger_capture,
    dagger_policy_identity, validate_capture_audit_profile,
)


def pair_files(directory, *, contract=None):
    directory.mkdir(parents=True, exist_ok=True)
    contract = contract or dict(supervision='q_reached', modality='G00')
    front = directory/'front.pt'; rear = directory/'rear.pt'
    torch.save(dict(contract=contract, tensor=torch.arange(4)), front)
    torch.save(dict(group='I-OfficialDP', residual_version=TargetResidual.version, updates=1,
        contract=contract, configuration=dict(base_checkpoint=str(front), base_sha256=sha256(front))), rear)
    return dagger_policy_identity(torch.load(rear, weights_only=False), rear)


def capture(directory, pair):
    directory.mkdir(); n=6
    nominal=np.full((n,29),.1,dtype=np.float32); residual=np.full_like(nominal,.01)
    command=nominal+residual; source=np.array([0,0,1,1,1,1]);valid=np.array([False]*3+[True]*3)
    np.savez(directory/'trace.npz', nominal=nominal, residual_applied_rad=residual,
        command=command, dagger_control_source=source, dagger_expert_label_valid=valid,
        dagger_segment_id=np.array([0,0,1,2,2,2]), residual_contact_valid=np.ones(n,bool),
        residual_contact_gate=np.ones(n,bool))
    fields=dict(nominal=nominal, student_proposed_target_rad=command, actual_issued=command,
        actual_applied=command,u_expert=command,q_reached=command,expert_label_valid=valid)
    with h5py.File(directory/'numeric.h5','x') as stream:
        g=stream.create_group('episode'); tr=g.create_group('transition')
        for key,value in fields.items():tr.create_dataset('dagger/'+key,data=value)
        tr.create_dataset('target_after_rad',data=command)
        tr.create_dataset('diagnostics_after/joint_position_rad',data=command)
        tr.create_dataset('actuator_substep_target_rad',data=np.repeat(command[:,None],2,axis=1))
    path=directory/'data.h5'
    with h5py.File(path,'x') as stream:
        for key,value in fields.items():stream.create_dataset('episodes/case/transition/dagger/'+key,data=value)
    binding=dict(path=str(path),sha256=sha256(path))
    result=dict(software_complete=True,legal=True,dagger=dict(expert_valid_ticks=3),identity='case',
        complete_student_policy=pair,checkpoint_sha256=pair['backend_sha256'],steps=n,
        residual_forward_ticks=n,residual_executed_ticks=2,dataset_capture=binding)
    (directory/'result.json').write_text(json.dumps(result))
    receipt=dict(status='source_verified',steps=n,numeric_path=str(directory/'numeric.h5'),dataset_capture=binding)
    (directory/'capture_complete.json').write_text(json.dumps(receipt))
    return result


def test_memo_reuses_only_hashes_and_pair_inspection_numeric_audit_still_runs(tmp_path,monkeypatch):
    pair=pair_files(tmp_path/'pair');directory=tmp_path/'capture';capture(directory,pair)
    ordinary=audit_dagger_capture(directory,pair)
    memo=DaggerCaptureAuditMemo();calls=[];load=torch.load
    monkeypatch.setattr(torch,'load',lambda *a,**k:(calls.append(a[0]),load(*a,**k))[1])
    assert audit_dagger_capture(directory,pair,audit_memo=memo)==ordinary
    assert audit_dagger_capture(directory,pair,audit_memo=memo)==ordinary
    assert len(calls)==2
    stats=memo.statistics()
    assert stats['pair_loads']==1 and stats['pair_hits']==1
    assert stats['payload']['full_hashes']==3 and stats['payload']['memo_hits']>=5
    assert stats['retains_tensors'] is False and all(v is True for v in memo._pairs.values())
    with h5py.File(directory/'numeric.h5','r+') as stream:
        stream['episode/transition/dagger/actual_applied'][0,0]=99
    with pytest.raises(ValueError,match='numerical chain mismatch'):
        audit_dagger_capture(directory,pair,audit_memo=memo)


@pytest.mark.parametrize('replacement',[False,True])
def test_changed_expected_source_is_rejected_even_with_same_size_or_path(tmp_path,replacement):
    pair=pair_files(tmp_path/'pair');directory=tmp_path/'capture';result=capture(directory,pair)
    memo=DaggerCaptureAuditMemo();memo.closed_source(directory,result)
    path=Path(result['dataset_capture']['path']);old=path.stat();data=bytearray(path.read_bytes());data[-1]^=1
    if replacement:
        other=tmp_path/'replacement';other.write_bytes(data);other.replace(path)
    else:path.write_bytes(data)
    os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
    with pytest.raises(ValueError,match='expected SHA'):memo.closed_source(directory,result)


def test_same_bytes_replaced_checkpoint_reloads_pair_metadata(tmp_path):
    pair=pair_files(tmp_path/'pair');memo=DaggerCaptureAuditMemo();memo.validate_pair(pair)
    path=Path(pair['frontend']);replacement=tmp_path/'replacement.pt';replacement.write_bytes(path.read_bytes());replacement.replace(path)
    memo.validate_pair(pair)
    assert memo.statistics()['pair_loads']==2
    assert memo.statistics()['pair_hits']==0


def test_checkpoint_corruption_or_expected_identity_change_rejected(tmp_path):
    pair=pair_files(tmp_path/'pair');memo=DaggerCaptureAuditMemo();memo.validate_pair(pair)
    wrong=dict(pair,frontend_sha256='0'*64)
    with pytest.raises(ValueError,match='expected SHA'):memo.validate_pair(wrong)
    path=Path(pair['backend']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='expected SHA'):memo.validate_pair(pair)


def test_pair_contract_and_mmap_load_races_rejected(tmp_path,monkeypatch):
    pair=pair_files(tmp_path/'pair');front=Path(pair['frontend']);rear=Path(pair['backend'])
    changed=torch.load(front,weights_only=False);changed['contract']=dict(supervision='u_expert',modality='G00')
    torch.save(changed,front)
    saved=torch.load(rear,weights_only=False);saved['configuration']['base_sha256']=sha256(front);torch.save(saved,rear)
    pair.update(frontend_sha256=sha256(front),backend_sha256=sha256(rear))
    with pytest.raises(ValueError,match='matched complete policy'):DaggerCaptureAuditMemo().validate_pair(pair)
    pair=pair_files(tmp_path/'second');load=torch.load
    def mutate(*a,**k):
        value=load(*a,**k)
        if str(a[0])==pair['frontend']:
            p=Path(a[0]);p.write_bytes(p.read_bytes()+b'changed')
        return value
    monkeypatch.setattr(torch,'load',mutate)
    with pytest.raises(ValueError):DaggerCaptureAuditMemo().validate_pair(pair)


def test_pair_bound_and_process_boundary(tmp_path,monkeypatch):
    one=pair_files(tmp_path/'one');two=pair_files(tmp_path/'two')
    memo=DaggerCaptureAuditMemo(max_pairs=1,max_payloads=2)
    for pair in (one,two,one):memo.validate_pair(pair)
    stats=memo.statistics();assert stats['pair_loads']==3 and stats['pair_evictions']==2
    assert stats['retained_pairs']==1 and stats['payload']['retained_entries']<=2
    monkeypatch.setattr(os,'getpid',lambda:memo.payload_audit.pid+1)
    with pytest.raises(RuntimeError,match='process boundary'):memo.validate_pair(one)


@pytest.mark.parametrize('field,value',[('status','closed'),('steps',7),('numeric_path','wrong')])
def test_source_receipt_cannot_be_used_as_an_unchecked_external_digest(tmp_path,field,value):
    pair=pair_files(tmp_path/'pair');directory=tmp_path/'capture';result=capture(directory,pair)
    p=directory/'capture_complete.json';receipt=json.loads(p.read_text());receipt[field]=value;p.write_text(json.dumps(receipt))
    memo=DaggerCaptureAuditMemo()
    with pytest.raises(ValueError,match='receipt differs'):memo.closed_source(directory,result)
    assert memo.statistics()['payload']['full_hashes']==0


def test_dagger_record_optional_payload_hash_preserves_default_and_expected_identity(tmp_path,monkeypatch):
    from visuotactile import narrow_data
    path=tmp_path/'source';path.write_bytes(b'closed source')
    entry=dict(path=str(path),source_sha256=sha256(path),trajectory_kind='expert_takeover_v1',split='train',episode='case')
    memo=DaggerCaptureAuditMemo();memo.sha256(path,expected=entry['source_sha256'])
    class ReachedNumeric(Exception):pass
    monkeypatch.setattr(narrow_data,'record',lambda *a,**k:(_ for _ in ()).throw(ReachedNumeric()))
    for options in ({},dict(payload_audit=memo.payload_audit)):
        with pytest.raises(ReachedNumeric):narrow_data.dagger_record(entry,**options)
    assert memo.statistics()['payload']['full_hashes']==1
    path.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='Takeover source changed'):
        narrow_data.dagger_record(entry,payload_audit=memo.payload_audit)


def test_profile_and_request_validation_keep_default_and_reject_silent_mode_changes():
    from tools.tight_fresh_batch_run import validate_capture_audit_request
    assert validate_capture_audit_request(dict(kind='autonomous',batch={})) is None
    assert validate_capture_audit_request(dict(kind='takeover',batch=dict(capture_audit_memo=CAPTURE_AUDIT_MEMO_PROFILE)))==CAPTURE_AUDIT_MEMO_PROFILE
    for value in ({},False,dict(CAPTURE_AUDIT_MEMO_PROFILE,max_pairs=5),dict(CAPTURE_AUDIT_MEMO_PROFILE,retains_tensors=0)):
        with pytest.raises(ValueError):validate_capture_audit_profile(value)
    with pytest.raises(ValueError):validate_capture_audit_request(dict(kind='autonomous',batch=dict(capture_audit_memo=CAPTURE_AUDIT_MEMO_PROFILE)))


def test_formal_prepare_registers_only_explicit_collection_memo(tmp_path,monkeypatch):
    from tools import tight_fresh_batch_run as formal,tight_fresh_stage_takeover as stage
    def prepare(root,output,*args,**kwargs):
        output.mkdir();(output/'plan.json').write_text(json.dumps(dict(kind='takeover')))
        (output/'session.json').write_text(json.dumps(dict(expert_sha256='fixture')))
    monkeypatch.setattr(stage,'prepare',prepare)
    for enabled in (False,True):
        output=tmp_path/str(enabled)
        plan=formal.prepare(tmp_path,output,lanes=16,identities=['train'],delays=[0],
            verification_workers=2,verification_max_pending=16,capture_audit_memo=enabled)
        assert ('capture_audit_memo' in plan['batch']) is enabled
        assert plan['batch']['verification_max_pending']==16
        if enabled:assert plan['batch']['capture_audit_memo']==CAPTURE_AUDIT_MEMO_PROFILE
    for invalid in (1,'true',None):
        with pytest.raises(ValueError):formal.prepare(tmp_path,tmp_path/'bad',lanes=16,identities=['train'],capture_audit_memo=invalid)
    assert not (tmp_path/'bad').exists()


def test_expansion_prepare_registers_memo_before_any_trajectory(tmp_path,monkeypatch):
    # Reuse metadata-only registry setup; no train/raw/test HDF files opened.
    from test_training_expansion import setup_registry
    from tools.tight_fresh_expansion import prepare
    setup=setup_registry.__wrapped__(tmp_path)
    import time
    (setup['source']/'session.json').write_text(json.dumps(dict(deadline=time.time()+1000,domain=setup['domain'],expert_sha256='fixture')))
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair',lambda root,role:dict(role=role))
    plan=prepare(setup['source'],tmp_path/'prepared',session_root=setup['source'],registry_id='memo_fixture',
        seeds=[6000],partition_manifest=setup['partition'],frozen_families=setup['families'],schedules=['fixed40'],
        lanes=16,event_seed=8000,round_index=10,capture_audit_memo=True,verification_workers=2,verification_max_pending=16)
    assert plan['batch']['capture_audit_memo']==CAPTURE_AUDIT_MEMO_PROFILE
    assert plan['batch']['verification_max_pending']==16


def test_owner_rejects_bad_registered_memo_before_spawn(tmp_path,monkeypatch):
    from tools import tight_batch_owner as owner
    (tmp_path/'plan.json').write_text(json.dumps(dict(kind='takeover',batch=dict(capture_audit_memo={}))))
    (tmp_path/'session.json').write_text('{}')
    monkeypatch.setattr(owner.subprocess,'Popen',lambda *a,**k:pytest.fail('Bad profile launched a process'))
    with pytest.raises(ValueError,match='memo profile'):owner.run(tmp_path,tmp_path)
    assert not (tmp_path/'owner_started.json').exists()


def test_formal_admission_uses_memo_and_keeps_every_camera_acceptance(tmp_path,monkeypatch):
    from tools import tight_fresh_batch_run as formal
    pair=pair_files(tmp_path/'pair');memo=DaggerCaptureAuditMemo();entries={};record_calls=[];camera_calls=[]
    monkeypatch.setattr(formal,'classify_capture',lambda *a:('stage_audit.json','fixed_rescue',None))
    def record(entry,*,include_geometry,payload_audit):
        assert payload_audit is memo.payload_audit and include_geometry is False
        assert payload_audit.sha256(entry['path'])==entry['source_sha256']
        record_calls.append(entry);return {},None,None
    def camera(path):
        camera_calls.append(path);return dict(passed=len(camera_calls)==1)
    monkeypatch.setattr('visuotactile.narrow_data.dagger_record',record)
    monkeypatch.setattr('visuotactile.anchor_sampling.supervision_population',lambda *a:dict(supervised_ticks=3))
    monkeypatch.setattr('visuotactile.camera_validity.inspect_capture',camera)
    (tmp_path/'plan.json').write_text('{}');plan=dict(round_index=10)
    for index in range(2):
        name='capture'+str(index);directory=tmp_path/name;result=capture(directory,pair)
        result.update(success=True);result['dagger'].update(expert_takeover_success=True,
            recovery_coverage=dict(switch_tick=2))
        (directory/'result.json').write_text(json.dumps(result))
        case=dict(name=name,role='A',spec=dict(identity='case',family='train',seed=1,takeover_tick=2))
        if index:
            with pytest.raises(ValueError,match='camera content invalid'):
                formal.admit_capture(tmp_path,case,result,pair,plan,entries,audit_memo=memo)
            assert result['accepted_for_training'] is False
        else:
            formal.admit_capture(tmp_path,case,result,pair,plan,entries,audit_memo=memo)
            assert result['accepted_for_training'] is True
    assert len(record_calls)==len(camera_calls)==2
    assert memo.statistics()['pair_loads']==1 and memo.statistics()['pair_hits']==1
