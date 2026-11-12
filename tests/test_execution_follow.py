import json
from pathlib import Path
import torch
import numpy as np
from tools import tight_execution_follow as follow

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def test_status_terminal_keeps_explicit_failure_and_root_identity(tmp_path):
    root=tmp_path/'adaptive';root.mkdir()
    follow.state(root,'technical_stop',state='FAILED',reason='evidence')
    assert json.loads((root/'follow_status.json').read_text())['state']=='FAILED'
    assert json.loads((root.parent/'status.json').read_text())['reason']=='evidence'


def test_physical_diagnosis_uses_real_events_and_legal_aligned_takeovers(tmp_path):
    for g in follow.GROUPS:
        d=tmp_path/'formal'/g/'backend'/g/'evaluation/dev';d.mkdir(parents=True)
        force=np.zeros((100,1,3));force[40:60,:,0]=1;lift=np.zeros((100,1),dtype=bool)
        np.savez(d/'trace.npz',command=np.zeros((100,29)),object_contact=force,lifted=lift)
    ticks=follow.physical_diagnosis(tmp_path)
    assert ticks and all((t+2)%12==0 and 22<=t<=298 for t in ticks)
    rows=json.loads((tmp_path/'physical_diagnosis.json').read_text())['rows']
    assert all(r['first_contact']==40 and r['first_contact_loss']==59 for r in rows)


def test_output_change_update_zero_never_deploys_as_trained_candidate(tmp_path):
    torch.save(dict(updates=0),tmp_path/'best.pt');torch.save(dict(updates=10),tmp_path/'last.pt')
    selected=follow.select_safe_candidate(tmp_path)
    assert selected['regression'] and selected['updates']==10
    assert torch.load(tmp_path/'update0_reference.pt',weights_only=False)['updates']==0


def test_formal_adaptation_can_continue_failed_first_gate_with_explicit_spec(tmp_path,monkeypatch):
    from tools import tight_finish,tight_dagger
    from visuotactile.execution_learning import PROJECTION_FY_LOSS
    from visuotactile.residual_output import REACHABLE_TAIL_OUTPUT
    parent=tmp_path/'parent';source=parent/'round_01';source.mkdir(parents=True)
    write(source/'round_finished.json',{});write(parent/'evaluation_plan.json',[{}])
    root=tmp_path/'repair/adaptive';root.mkdir(parents=True)
    for g in follow.GROUPS:
        old=source/g/'backend';train=old/g/'training';train.mkdir(parents=True)
        nominal=old/g/'nominal';nominal.mkdir();dp=old/'dp.pt';dp.write_bytes(b'fixed')
        config=dict(base_checkpoint=str(dp),base_sha256=follow.sha(dp),residual_statistics_sha256='same')
        torch.save(dict(configuration=config),train/'best.pt');write(train/'exposure_endpoint.json',{})
        write(old/'evaluation_complete.json',dict(complete=True))
        write(old/'state_data/data.json',dict(snapshot_id='same',splits=dict(train=['train'],test=['test22'])))
        write(nominal/'index.json',dict(episodes={'train':{}}));write(nominal/'scales.json',dict(control_tick_population=100))
        write(old/'configuration.json',dict(residual_pair=dict(groups=[g],bases={g:{}},continuation_from={})))
    spec=dict(loss_contract=PROJECTION_FY_LOSS,output_contract=REACHABLE_TAIL_OUTPUT,lr=1e-4)
    write(root/'adaptive_selection.json',dict(chosen=spec,paired_probe_passed=False,results={'candidate':{g:dict(optimization_wall_s=1,supervised_ticks=100) for g in follow.GROUPS}}))
    write(root/'registration.json',dict(compute_deadline=follow.time.time()+80000,absolute_deadline=follow.time.time()+80600,started_utc=1))
    write(root/'follow_launch.json',dict(source_head='reviewed'))
    monkeypatch.setattr(follow,'PARENT',parent);created=[]
    def prepare(dest,configuration):
        g=configuration['residual_pair']['groups'][0];(dest/g/'nominal').mkdir(parents=True);created.append(configuration)
        write(dest/'session.json',{});write(dest/g/'session.json',{})
    monkeypatch.setattr(tight_finish,'residual_prepare',prepare)
    monkeypatch.setattr(tight_dagger,'share_monitor',lambda *a:None)
    monkeypatch.setattr(tight_dagger,'reuse_nominal_cache',lambda *a:None)
    follow.prepare(root)
    assert len(created)==2
    assert all(v['residual_pair']['output_contract']==REACHABLE_TAIL_OUTPUT for v in created)
    assert json.loads((root/'formal_budget.json').read_text())['epochs_each']==6


def test_publication_preserves_parent_report_and_rejects_other_local_edits():
    import pytest
    start='<!-- EXECUTION_LEARNING_STATUS_START -->';end='<!-- EXECUTION_LEARNING_STATUS_END -->'
    base=start+'old'+end+'\n\nbody\n'
    ours=start+'new'+end+'\n\nbody\n'
    theirs='parent finished\n'+base
    merged=follow.merge_parent_status(base,ours,theirs)
    assert 'new' in merged and 'parent finished' in merged and 'body' in merged and 'old' not in merged
    with pytest.raises(ValueError):follow.merge_parent_status(base,ours+'unreviewed',theirs)
