import json
from pathlib import Path
import pytest
from tools import tight_execution_learning as repair


def write(path,value):path.write_text(json.dumps(value))


def test_child_completion_receipt_and_no_implicit_repeat(tmp_path):
    write(tmp_path/'source_freeze.json',{})
    repair.run_child(tmp_path,'cpu',['-c','print("bounded CPU child")'],repair.time.time()+30)
    report=json.loads((tmp_path/'cpu_receipt.json').read_text())
    assert report['exit_code']==0
    with pytest.raises(ValueError,match='already attempted'):
        repair.run_child(tmp_path,'cpu',['-c','raise RuntimeError()'],repair.time.time()+30)


def test_expired_stage_does_not_spawn(tmp_path,monkeypatch):
    write(tmp_path/'source_freeze.json',{})
    monkeypatch.setattr(repair.subprocess,'Popen',lambda *a,**kw:pytest.fail('expired stage spawned'))
    with pytest.raises(TimeoutError):repair.run_child(tmp_path,'expired',[],0)
    assert not (tmp_path/'expired_receipt.json').exists()


def test_failed_child_retains_exit_evidence_and_no_repeat(tmp_path):
    write(tmp_path/'source_freeze.json',{})
    with pytest.raises(RuntimeError,match='failed'):
        repair.run_child(tmp_path,'failed',['-c','raise SystemExit(7)'],repair.time.time()+30)
    assert json.loads((tmp_path/'failed_receipt.json').read_text())['exit_code']==7


def test_source_mutation_blocks_next_stage(tmp_path,monkeypatch):
    file=tmp_path/'source.py';file.write_text('before')
    write(tmp_path/'source_freeze.json',{str(file):repair.sha256(file)})
    file.write_text('after')
    monkeypatch.setattr(repair.subprocess,'Popen',lambda *a,**kw:pytest.fail('source drift spawned'))
    with pytest.raises(ValueError,match='drift'):repair.run_child(tmp_path,'changed',[],repair.time.time()+30)


def test_formal_preparation_cannot_bypass_paired_probe(tmp_path,monkeypatch):
    for name,value in [('manifest.json',{}),('registration.json',{}),('probe_results.json',{}),('probe_gates.json',{'passed':False})]:write(tmp_path/name,value)
    with pytest.raises(ValueError,match='Both candidate'):repair.prepare_formal(tmp_path)
    assert not (tmp_path/'formal').exists()


def test_existing_registration_never_resets_deadline(tmp_path):
    write(tmp_path/'registration.json',{'absolute_deadline':123})
    with pytest.raises(ValueError,match='never reset'):repair.bind(tmp_path)
    assert json.loads((tmp_path/'registration.json').read_text())['absolute_deadline']==123


def test_formal_continuation_binds_latest_complete_joint_pair(tmp_path,monkeypatch):
    import torch
    from tools import tight_finish,tight_dagger
    parent=tmp_path/'parent';parent.mkdir();out=tmp_path/'repair';out.mkdir()
    monkeypatch.setattr(repair,'PARENT',parent)
    write(parent/'evaluation_plan.json',[{'identity':'dev_only'}])
    for round_name in ['round_01','round_02']:
        for group in repair.GROUPS:
            old=parent/round_name/group/'backend';train=old/group/'training';train.mkdir(parents=True)
            nominal=old/group/'nominal';nominal.mkdir();data=old/'state_data';data.mkdir()
            dp=old/'dp.pt';dp.write_bytes(b'frozen frontend '+group.encode())
            torch.save({'configuration':{'base_checkpoint':str(dp),'base_sha256':repair.sha256(dp),
                'residual_statistics_sha256':'common'}},train/'best.pt')
            write(train/'exposure_endpoint.json',{})
            write(old/'evaluation_complete.json',{'complete':True})
            write(nominal/'scales.json',{'control_tick_population':100})
            write(nominal/'index.json',{'episodes':{'train':{}}})
            write(data/'data.json',{'snapshot_id':round_name,'splits':{'train':['train'],'test':['test22']}})
            write(old/'configuration.json',{'residual_pair':{'groups':[group],
                'bases':{group:{}},'continuation_from':{'path':'prior','sha256':'prior'}}})
            write(old/'session.json',{})
    write(out/'manifest.json',{})
    write(out/'registration.json',{'compute_deadline':repair.time.time()+20000,
        'absolute_deadline':repair.time.time()+20600,'started_utc':1})
    write(out/'probe_results.json',{'groups':{g:{'old':{'optimization_wall_s':1,'supervised_ticks':100}} for g in repair.GROUPS}})
    write(out/'probe_gates.json',{'passed':True});write(out/'launch.json',{'source_head':'reviewed'})
    prepared=[];reused=[]
    def prepare(dest,configuration):
        prepared.append(configuration);dest.mkdir(parents=True)
        group=configuration['residual_pair']['groups'][0];(dest/group/'nominal').mkdir(parents=True)
        write(dest/'session.json',{});write(dest/group/'session.json',{})
    monkeypatch.setattr(tight_finish,'residual_prepare',prepare)
    monkeypatch.setattr(tight_dagger,'share_monitor',lambda *args:None)
    monkeypatch.setattr(tight_dagger,'reuse_nominal_cache',lambda *args:reused.append(args))
    repair.prepare_formal(out)
    sources=json.loads((out/'formal_sources.json').read_text())
    assert sources['parent_round']==str(parent/'round_02') and not sources['same_probe_initial_pair']
    assert len(prepared)==len(reused)==2
    for cfg in prepared:
        v=cfg['residual_pair'];assert v['loss_contract']==repair.PROJECTION_FY_LOSS
        assert v['history_contract']==repair.HISTORY and v['maximum_control_tick_epochs']==6
        assert 'round_02' in v['continuation_from']['path']
    assert json.loads((out/'formal_budget.json').read_text())['population']==dict(QG01=100,G01=100)
