"""Throughput separates B2 optimizer receipts from actual trajectory visits."""
import json
import pytest
from tools.tight_fresh_cycle_throughput import Sources,frontend_summary
from visuotactile.trajectory_training import profile
from test_cycle_throughput import fixture,write
from test_full_trajectory_training import receipt_rows


def b2_frontend(tmp_path):
    cycle,_,_,_=fixture(tmp_path);root=cycle/'A/training'
    rows=receipt_rows()
    for row in rows:row['epoch']=1
    write(root/'learning.jsonl',rows,lines=True)
    last=rows[-1]
    validation=json.loads((root/'validation.jsonl').read_text())
    validation.update(full_trajectory_batch=profile(),visits_seen=3,anchors_seen=last['anchors_seen'],target_ticks=last['target_ticks'])
    validation['training_epoch_io']['prefetch']=None
    write(root/'validation.jsonl',[validation],lines=True)
    final=json.loads((root/'finished.json').read_text())
    final.update(full_trajectory_batch=profile(),visits_seen=3,ema_updates=2,anchors_seen=last['anchors_seen'],
        target_ticks=last['target_ticks'],sampling_exposure=dict(x=2,y=1),episode_multiplicities=dict(x=2,y=1))
    write(root/'finished.json',final)
    return cycle/'A'


def test_b2_actual_visit_and_update_denominators_remain_separate(tmp_path):
    directory=b2_frontend(tmp_path);report=frontend_summary(directory,Sources())
    assert report['counters']==dict(optimizer_updates=2,anchors=7,legal_horizon_target_slots=55,
        full_history_frame_visits=51,episode_visits=3)
    assert report['ema_updates']==2 and report['prefetch_totals']=={}
    assert report['rates_per_worker_wall_second']['episode_visits']==3/8
    assert report['rates_per_worker_wall_second']['optimizer_updates']==2/8


@pytest.mark.parametrize('field,value',[('visits_seen',2),('ema_updates',3),
    ('sampling_exposure',dict(x=1,y=2)),('full_trajectory_batch',None)])
def test_b2_incomplete_or_conflicting_receipts_rejected(tmp_path,field,value):
    directory=b2_frontend(tmp_path);path=directory/'training/finished.json'
    final=json.loads(path.read_text());final[field]=value;write(path,final)
    with pytest.raises(ValueError):frontend_summary(directory,Sources())
