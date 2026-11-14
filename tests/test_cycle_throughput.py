import hashlib
import json
from pathlib import Path

import pytest

from tools.tight_fresh_cycle_throughput import (Sources, comparison, development_summary,
    frontend_summary, stage_receipts, summarize, union_seconds)


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = '\n'.join(json.dumps(row) for row in value) if lines else json.dumps(value)
    path.write_text(raw + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    cycle = root / 'round'; stages = root / 'round_stages'; collection = root / 'collect'; old = root / 'source'
    cases = [dict(name='new', role='A', spec=dict(family='newfamily'), dagger={}),
             dict(name='rejected', role='B', spec=dict(family='otherfamily'), dagger={})]
    plan_sha = write(collection / 'plan.json', dict(kind='takeover', test_used=False, cases=cases))
    write(collection / 'owner_started.json', dict(started=10., plan_sha256=plan_sha))
    write(collection / 'owner_finished.json', dict(software_complete=True, returncode=0, seconds=10., ended=20., plan_sha256=plan_sha))
    write(collection / 'result.json', dict(software_complete=True, worker_error=None, cases=2, elapsed_seconds=9.))
    rows = []
    for name, role, accepted in [('new', 'A', True), ('rejected', 'B', False)]:
        rows.append(dict(case=name, role=role, result=dict(software_complete=True, legal=True, success=accepted,
            steps=10, accepted_for_training=accepted, actual_exposure=dict(supervised_ticks=5),
            dagger=dict(expert_ticks=6, expert_valid_ticks=5, expert_takeover_success=accepted))))
    write(collection / 'results.json', rows)
    entries = dict(new=dict(episode='new', family='newfamily', length=10, source_sha256='rawnew'))
    entries_sha = write(collection / 'correction_entries.json', entries)
    data = dict(splits=dict(train=['original'], validation=['val'], test=[]),
        entries=dict(original=dict(family='originalfamily', length=10, source_sha256='rawold'),
                     val=dict(family='valfamily', length=10, source_sha256='rawval')))
    updated = json.loads(json.dumps(data)); updated['entries'].update(entries); updated['splits']['train'].append('new')
    updated['aggregation'] = dict(new_training_ids=['new'])
    shards = {}
    for identity in updated['entries']:
        path = root / 'shared' / (identity + '.pt'); path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'x' * 10)
        shards[identity] = dict(path=str(path), frames=10, sha256=identity, source_sha256=updated['entries'][identity]['source_sha256'])
    for role in 'AB':
        write(old / role / 'state_data/data.json', data)
        data_sha = write(cycle / role / 'state_data/data.json', updated)
        write(cycle / role / 'cache_index.json', dict(complete=True, data_sha256=data_sha, entries=shards))
        learning = [dict(epoch=1, updates=i, anchors_seen=2*i, target_ticks=20*i, visual_frames=10)
                    for i in (1, 2)]
        write(cycle / role / 'training/learning.jsonl', learning, lines=True)
        start = 104. if role == 'A' else 105.
        write(cycle / role / 'training/validation.jsonl', [dict(epoch=1, updates=2, anchors_seen=4, target_ticks=40,
            utc=start+3, train={}, validation={}, training_epoch_io=dict(wall_seconds=2., started_utc=start, ended_utc=start+2,
                process_io=dict(read_bytes=100), prefetch=dict(visits=2, peak_owned_clone_bytes=5, load_s=1.)))], lines=True)
        write(cycle / role / 'training/finished.json', dict(complete=True, epochs=1, updates=2, anchors_seen=4,
            target_ticks=40, wall_seconds=8.))
        write(cycle / role / 'backend/cache_complete.json', dict(complete=True, episodes=3, wall_s=1.))
        write(cycle / role / 'backend' / role / 'nominal/index.json', dict(episodes={key: dict(ticks=10, plans=1) for key in shards}))
        write(cycle / role / 'backend' / role / 'training/finished.json', dict(target_reached=True, run_kind='formal',
            new_optimizer_updates=4, supervised_ticks=30, control_ticks=40, control_tick_population=20,
            supervised_tick_population=15, train_episode_count=2, wall_s=4., training_wall_s=3., control_tick_epochs=2))
    d = cycle / 'development'
    write(d / 'plan.json', dict(kind='autonomous', test_used=False, cases=[
        dict(name='valid', role='A', spec=dict(family='dev')), dict(name='invalid', role='B', spec=dict(family='dev'))]))
    write(d / 'result.json', dict(software_complete=True, worker_error=None, cases=2, elapsed_seconds=4.))
    write(d / 'results.json', [dict(case='valid', role='A', result=dict(software_complete=True, legal=True, steps=6, success=False)),
        dict(case='invalid', role='B', result=dict(software_complete=True, legal=False, not_started=True, success=False))])
    write(d / 'wave_profiles.json', [dict(wave=0, role='A', control_ticks=6, seconds=2.),
                                   dict(wave=1, role='B', control_ticks=0, seconds=1.)])
    write(cycle / 'prepared.json', dict(complete=True)); write(cycle / 'cached.json', dict(complete=True))
    registrations = dict(output=str(cycle), source_root=str(old), collection=str(collection), test_used=False,
        started=100., collection_entries_sha256=entries_sha, parallel_frontends=True, source_commit='frozen')
    write(stages / 'registration.json', registrations)
    schedule = [('aggregate', 100., 1., cycle/'prepared.json'), ('visual_cache', 101., 1., cycle/'cached.json')]
    for role in 'AB':
        schedule.append(('frontend_'+role, 102., 10. if role=='A' else 8., cycle/role/'training/finished.json'))
        for step, start, seconds, path in [('prepare',112.,1.,cycle/'prepared.json'),
            ('cache',113.,1.,cycle/role/'backend/cache_complete.json'),
            ('train',114.,4.,cycle/role/'backend'/role/'training/finished.json'),
            ('offline',118.,1.,cycle/'cached.json')]:
            schedule.append(('backend_'+step+'_'+role, start, seconds, path))
    schedule += [('development_prepare',119.,1.,d/'plan.json'), ('development',120.,5.,d/'result.json')]
    receipts = []
    for name, start, seconds, path in schedule:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        row = dict(stage=name, started=start, seconds=seconds, returncode=0, stop_reason=None,
                   completion=str(path), completion_sha256=sha, utc=start+seconds)
        receipts.append(row); write(stages/'ledger'/(name+'.json'),row)
    write(stages/'finished.json',dict(software_complete=True,stages=receipts,utc=125.1))
    outer=root/'outer.json';write(outer,dict(output=str(cycle),collection=str(collection),data_source=str(old),
        collection_plan_sha256=plan_sha,cycle_command=['future','registered','command']))
    return cycle, collection, stages, outer


def test_union_counts_nested_touching_and_parallel_intervals_once():
    assert union_seconds([(10,20),(12,18),(20,25),(30,32),(10,20)]) == 17
    assert union_seconds([]) == 0


@pytest.mark.parametrize('intervals', [[(2,1)],[(float('nan'),3)],[(1,float('inf'))],[(-1,2)]])
def test_union_rejects_unusable_clocks(intervals):
    with pytest.raises(ValueError): union_seconds(intervals)


def test_complete_cycle_denominators_actual_counts_and_shared_storage(tmp_path):
    cycle, collection, stages, outer=fixture(tmp_path)
    result=summarize(cycle,collection_root=collection,outer_registration=outer)
    assert result['complete']
    timeline=result['timeline']
    assert timeline['cycle_wall_seconds']==25
    assert timeline['stage_interval_union_seconds']==25
    assert timeline['stage_worker_seconds']==40 # 8s frontend + 7s rear overlap
    assert timeline['overlapping_worker_seconds']==15
    assert timeline['collection_to_evaluation_envelope_seconds']==115
    assert timeline['observed_collection_and_cycle_interval_union_seconds']==35
    assert timeline['envelope_unclassified_or_other_work_seconds']==80
    assert result['frontend_aggregate']['counters']['anchors']==8 # cumulative last4 each, not 2+4 each
    assert result['frontend_aggregate']['counters']['legal_horizon_target_slots']==80
    assert result['frontend_aggregate']['rates_per_stage_union_second']['optimizer_updates']==.4
    assert result['frontend_aggregate']['training_epoch_interval_union_seconds']==3
    assert result['frontend_aggregate']['training_epoch_process_read_bytes']==200
    assert result['frontends']['A']['other_frontend_seconds']==6
    assert result['backends']['A']['training']['counters']['supervised_ticks']==30
    assert result['backend_aggregate']['training']['rates_per_stage_union_second']['supervised_ticks']==15
    assert result['population']['before']['train']['trajectories']==1
    assert result['population']['after']['train']['families']==2
    assert result['frozen_visual_storage']['unique_file_bytes']==30
    assert result['frozen_visual_storage']['combined_logical_bytes']==60
    assert result['collection']['counts']['actual_takeovers']==2
    assert result['collection']['counts']['accepted_valid_expert_ticks']==5
    assert 'triggered_attempts' not in result['collection']['counts']
    assert result['development']['per_role']['B']==dict(attempts=1,legal_attempts=0,successes=0,aggregate_control_ticks=0)
    assert result['development']['aggregate_control_ticks_per_worker_second']==1.5
    assert str((stages/'registration.json').resolve()) in result['bindings']


def test_complete_receipt_rejects_changed_content(tmp_path):
    cycle, _, stages, _=fixture(tmp_path)
    write(cycle/'prepared.json',dict(complete=True,changed=True))
    with pytest.raises(ValueError,match='Completion receipt changed'):stage_receipts(stages,Sources())


def test_missing_finish_never_gets_complete_even_with_all_successful_stages(tmp_path):
    cycle, _, stages, _=fixture(tmp_path)
    (stages/'finished.json').unlink()
    result=summarize(cycle)
    assert not result['complete']
    assert 'cycle_wall_seconds' not in result['timeline']
    assert comparison(result,result)['ratios'] is None


def test_future_cycle_can_bind_outer_without_fabricating_training(tmp_path):
    cycle, collection, stages, outer=fixture(tmp_path)
    for p in stages.rglob('*'):
        if p.is_file():p.unlink()
    (collection/'owner_finished.json').unlink()
    value=summarize(cycle,collection_root=collection,outer_registration=outer)
    assert not value['complete'] and not value['collection']['complete']
    assert value['frontends']=={} and value['backends']=={}
    assert value['configuration'] is None and value['registered_future_command']==['future','registered','command']
    assert 'cycle_wall_seconds' not in value['timeline']


def test_outer_cannot_silently_rebind_collection(tmp_path):
    cycle, collection, _, outer=fixture(tmp_path)
    with pytest.raises(ValueError,match='collection binding'):
        summarize(cycle,collection_root=collection.parent/'different',outer_registration=outer)


def test_cumulative_learning_and_finished_counts_must_agree(tmp_path):
    cycle, _, _, _=fixture(tmp_path);path=cycle/'A/training/learning.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['target_ticks']=41;write(path,rows,lines=True)
    with pytest.raises(ValueError,match='final counter differs'):
        frontend_summary(cycle/'A',Sources())


def test_missing_old_epoch_io_stays_unknown(tmp_path):
    cycle, _, _, _=fixture(tmp_path);path=cycle/'A/training/validation.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]['training_epoch_io'];write(path,rows,lines=True)
    result=frontend_summary(cycle/'A',Sources())
    assert result['training_epoch_seconds'] is None and result['other_frontend_seconds'] is None
    assert result['timed_metrics'] is None
    assert result['process_io_epoch_coverage']==dict(expected=1,observed=0,read_bytes_observed=0)


def test_missing_optimizer_visit_is_not_hidden_by_final_counter(tmp_path):
    cycle, _, _, _=fixture(tmp_path);path=cycle/'A/training/learning.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    write(path,rows[1:],lines=True)
    with pytest.raises(ValueError,match='Missing, duplicated or reordered optimizer'):
        frontend_summary(cycle/'A',Sources())


def test_missing_epoch_read_counter_is_unknown_not_zero_aggregate(tmp_path):
    cycle, _, _, _=fixture(tmp_path);path=cycle/'A/training/validation.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]['training_epoch_io']['process_io']['read_bytes'];write(path,rows,lines=True)
    result=summarize(cycle)
    assert result['complete']
    assert result['frontend_aggregate']['training_epoch_process_read_bytes'] is None
    assert result['frontend_aggregate']['observed_training_epoch_process_read_bytes']==100


@pytest.mark.parametrize('mutation',['censored','diagnostic_only','missing_legal_clock','wave_clock'])
def test_development_rejects_diagnostic_or_unaccounted_clocks(tmp_path,mutation):
    cycle, _, _, _=fixture(tmp_path);root=cycle/'development'
    rows=json.loads((root/'results.json').read_text())
    if mutation=='missing_legal_clock':del rows[0]['result']['steps']
    elif mutation=='wave_clock':rows[0]['result']['steps']=7
    else:rows[0]['result'][mutation]=True
    write(root/'results.json',rows)
    with pytest.raises(ValueError):development_summary(root,Sources())


def test_changed_accepted_set_cannot_enter_complete_population(tmp_path):
    cycle, collection, stages, _=fixture(tmp_path)
    path=cycle/'A/state_data/data.json';data=json.loads(path.read_text());data['aggregation']['new_training_ids']=[]
    write(path,data)
    with pytest.raises(ValueError,match='New training IDs'):
        summarize(cycle)


def test_missing_required_stage_rejects_complete_flag(tmp_path):
    _, _, stages, _=fixture(tmp_path);path=stages/'finished.json';value=json.loads(path.read_text())
    value['stages']=[v for v in value['stages'] if v['stage']!='backend_train_B'];write(path,value)
    with pytest.raises(ValueError,match='Incomplete finished stage ledger'):
        stage_receipts(stages,Sources())
