import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import pytest
import torch

from tools import tight_fresh_followup_session as followup
from visuotactile.compact import sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def change(path, fn):
    value = json.loads(path.read_text())
    fn(value)
    write(path, value)


@pytest.fixture
def source(tmp_path, monkeypatch):
    old = tmp_path/'old'
    root = old/'round04'
    root.mkdir(parents=True)
    output = tmp_path/'followup'
    session = dict(version='tight_fresh_every_tick_v1', deadline=1., cache_deadline=.5,
        resource_policy='full_gpu_v1', resources=dict(min_gpu_free_mib=4096, min_ram_gib=10),
        reserve_bytes=1024, visual_stride=1, control_hz=60, planning_hz=5,
        expert_checkpoint='preserved_expert.pt', expert_sha256='expert',
        dino_sha256='dino', dino_checkpoint='dino.pt', domain={'fixed':'narrow'},
        source_commit='original', trainable_initialization='random',
        owned_worker_roots=[str(old)], authorization='old window')
    write(root/'session.json', session)
    write(root/'metadata.json', {'task':'original'})
    write(root/'gate.json', {'passed':True})
    write(old/'shared_resources.json', {'old_telemetry':True})
    (root/'shared_resources.json').symlink_to(old/'shared_resources.json')
    entries = {}
    for identity in ('train', 'validation'):
        path = old/(identity+'.h5')
        with h5py.File(path, 'w') as stream:
            stream.attrs['result_json'] = json.dumps(dict(seed=7, narrow_reset={'fixed':'reset'}))
        entries[identity] = dict(path=str(path), source_sha256=sha256(path),
            family=identity, length=2)
    for role in ('A', 'B'):
        base = root/role
        state = base/'state_data'
        rear = base/'backend'/role
        (state/'perception').mkdir(parents=True)
        (rear/'nominal').mkdir(parents=True)
        (base/'training').mkdir()
        data = dict(visual_stride=1, supervision='q_reached' if role=='A' else 'u_expert',
            entries=entries, splits=dict(train=['train'], validation=['validation'], test=[]),
            normalization_train_ids=['train'])
        write(state/'data.json', data)
        torch.save({'affine':role}, state/'statistics.pt')
        torch.save({'data_sha256':sha256(state/'data.json'), 'records':{k:{} for k in entries}}, state/'numeric.pt')
        if role == 'A':
            torch.save({}, state/'loss_context.pt')
        visual = dict(complete=True, stride=1, trainable_projection_cached=False,
            data_sha256=sha256(state/'data.json'), dino_sha256='dino', entries={})
        for identity in entries:
            path = state/'perception'/(identity+'.pt')
            path.write_bytes(b'frozen visual shard')
            visual['entries'][identity] = dict(path=str(path), sha256=sha256(path),
                source_sha256=entries[identity]['source_sha256'], frames=2)
        write(base/'cache_index.json', visual)
        config = dict(data_sha256=sha256(state/'data.json'), statistics_sha256=sha256(state/'statistics.pt'),
            dino_sha256='dino', cache_index_sha256=sha256(base/'cache_index.json'))
        torch.save(dict(configuration=config, contract={'visual_stride':1}), base/'training/best.pt')
    pairs = {}
    for role in ('A', 'B'):
        base = root/role
        state = base/'state_data'
        rear = base/'backend'/role
        nominal = dict(base_sha256=sha256(base/'training/best.pt'), data_sha256=sha256(state/'data.json'),
            statistics_sha256=sha256(state/'statistics.pt'),
            residual_statistics_sha256=sha256(root/'B/state_data/statistics.pt'), episodes={})
        for identity in entries:
            path = rear/'nominal'/(identity+'.pt')
            path.write_bytes(b'current DP plans')
            nominal['episodes'][identity] = dict(sha256=sha256(path))
        write(rear/'nominal/index.json', nominal)
        write(rear/'nominal/scales.json', {'scales':[1.]})
        config = {k:v for k,v in nominal.items() if k!='episodes'}
        config.update(dino_sha256='dino', cache_index_sha256=sha256(rear/'nominal/index.json'),
            base_checkpoint=str(base/'training/best.pt'))
        (rear/'training').mkdir()
        torch.save(dict(configuration=config, contract={'visual_stride':1}), rear/'training/best.pt')
        write(rear/'training/finished.json', {'target_reached':True})
        write(rear/'offline/result.json', dict(complete=True, test_used=False,
            checkpoint_sha256=sha256(rear/'training/best.pt')))
        pairs[role] = dict(frontend=str(base/'training/best.pt'), frontend_sha256=sha256(base/'training/best.pt'),
            backend=str(rear/'training/best.pt'), backend_sha256=sha256(rear/'training/best.pt'),
            residual_version='target_residual_transformer60_v1')
    def completed(candidate, role):
        pair = copy.deepcopy(pairs[role])
        for field, relative in [('frontend','training/best.pt'), ('backend',f'backend/{role}/training/best.pt')]:
            path = candidate/role/relative
            assert path.resolve() == Path(pair[field])
            if sha256(path) != pair[field+'_sha256']:
                raise ValueError('Changed selected model')
        return pair
    monkeypatch.setattr(followup, 'completed_pair', completed)
    cases = [dict(name=role+'_trial', role=role, spec=dict(identity='trial', seed=7)) for role in ('A', 'B')]
    evaluation = root/'development'
    plan = dict(kind='autonomous', test_used=False, cases=cases, pairs=pairs,
        resource_root=str(root), source_commit='evaluated_source', source_sha256={'worker.py':'frozen'})
    write(evaluation/'plan.json', plan)
    write(evaluation/'run_started.json', dict(pid=999999999, plan_sha256=sha256(evaluation/'plan.json')))
    rows = []
    for case in cases:
        result = dict(case['spec'], legal=True, software_complete=True, success=False,
            steps=601, residual_ticks=601, checkpoint_sha256=pairs[case['role']]['backend_sha256'],
            residual_version=pairs[case['role']]['residual_version'], simulation_step_hz=60, plan_hz=5,
            action_control_mode='absolute_joint_position_target_rad')
        write(evaluation/case['name']/'result.json', result)
        rows.append(dict(case=case['name'], role=case['role'], result=result))
    write(evaluation/'results.json', rows)
    write(evaluation/'result.json', dict(software_complete=True, worker_error=None, cases=len(rows)))
    return SimpleNamespace(root=root, old=old, output=output, evaluation=evaluation,
        session=session, pairs=pairs, completed=completed)


def test_expired_source_creates_new_window_and_collection_compatible_links(source, monkeypatch):
    x = source
    before = {str(p):sha256(p) for p in x.old.rglob('*') if p.is_file()}
    binding = followup.prepare(x.root, x.output, 2)
    assert binding['version'] == 'independent_followup_binding_v1'
    assert 'data_pool_source' not in binding and 'role_policy_sources' not in binding
    session = followup.read(x.output/'session.json')
    assert session['deadline'] == session['started_utc']+7200
    assert session['cache_deadline'] == session['training_deadline'] == session['deadline']
    assert session['domain'] == x.session['domain']
    assert session['followup']['previous_deadline'] == 1
    assert session['owned_worker_roots'] == [str(x.output)]
    assert session['resource_root'] == str(x.output)
    assert 'Continuation' in session['trainable_initialization']
    assert not (x.output/'shared_resources.json').exists()
    assert not (x.output/'shared_resources.json').is_symlink()
    assert (x.output/'A/shared_resources.json').resolve() == x.output/'shared_resources.json'
    assert not (x.output/'A').is_symlink() and not (x.output/'A/training').is_symlink()
    assert (x.output/'A/state_data').resolve() == x.root/'A/state_data'
    assert binding['selected_pairs'] == x.pairs
    assert binding['roles']['A']['large_artifacts']['train']['raw']['bytes_rehashed'] is False
    assert followup.read(x.output/'prepared.json')['complete']
    assert before == {str(p):sha256(p) for p in x.old.rglob('*') if p.is_file()}
    from tools import tight_fresh_dagger
    monkeypatch.setattr(tight_fresh_dagger, 'completed_pair', x.completed)
    tight_fresh_dagger.prepare_collection(x.output, x.output/'takeover_05', [60], 5)
    plan = followup.read(x.output/'takeover_05/plan.json')
    assert plan['resource_root'] == str(x.output) and plan['pairs'] == x.pairs
    assert followup.read(x.output/'takeover_05/session.json')['deadline'] == session['deadline']
    assert before == {str(p):sha256(p) for p in x.old.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault', ['incomplete', 'missing_case', 'case_drift', 'plan_drift', 'takeover',
    'model', 'affine', 'visual_manifest', 'nominal_shard', 'numeric', 'gate', 'live_owner', 'duplicate',
    'dangling_destination', 'inside_old_session'])
def test_rejects_unreviewed_or_live_sources_without_destination(source, fault):
    x = source
    worker = None
    if fault=='incomplete':change(x.evaluation/'result.json', lambda v:v.update(software_complete=False))
    elif fault=='missing_case':change(x.evaluation/'results.json', lambda v:v.pop())
    elif fault=='case_drift':change(x.evaluation/'A_trial/result.json', lambda v:v.update(residual_ticks=0))
    elif fault=='plan_drift':change(x.evaluation/'plan.json', lambda v:v.update(source_commit='changed'))
    elif fault=='takeover':change(x.evaluation/'plan.json', lambda v:v.update(kind='takeover'))
    elif fault=='model':(x.root/'A/training/best.pt').write_bytes(b'changed')
    elif fault=='affine':(x.root/'A/state_data/statistics.pt').write_bytes(b'changed')
    elif fault=='visual_manifest':change(x.root/'A/cache_index.json', lambda v:v.update(stride=12))
    elif fault=='nominal_shard':(x.root/'A/backend/A/nominal/train.pt').write_bytes(b'old plan')
    elif fault=='numeric':torch.save(dict(data_sha256='old', records={}), x.root/'A/state_data/numeric.pt')
    elif fault=='gate':change(x.root/'gate.json', lambda v:v.update(passed=False))
    elif fault=='live_owner':
        worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', '--root', str(x.old)])
        write(x.old/'run_started.json', dict(pid=worker.pid))
    elif fault=='duplicate':x.output.mkdir()
    elif fault=='dangling_destination':x.output.symlink_to(x.old/'missing')
    elif fault=='inside_old_session':x.output = x.old/'new'
    try:
        with pytest.raises((ValueError, FileExistsError)):
            followup.prepare(x.root, x.output, 2)
    finally:
        if worker is not None:
            worker.terminate()
            worker.wait(timeout=5)
    assert not (x.output/'prepared.json').exists()


def test_study_variant_inherits_only_registered_matching_context(source):
    x = source
    variant = x.old/'study/control'
    variant.mkdir(parents=True)
    for name in ('session.json', 'A', 'B', 'development', 'shared_resources.json'):
        (variant/name).symlink_to(x.root/name, target_is_directory=(x.root/name).is_dir())
    # The evaluated root identity must be this variant; all selected artifacts stay unchanged.
    change(x.evaluation/'plan.json', lambda v:v.update(resource_root=str(variant)))
    change(x.evaluation/'run_started.json', lambda v:v.update(plan_sha256=sha256(x.evaluation/'plan.json')))
    protected = {str(x.root/role/'state_data'/name):sha256(x.root/role/'state_data'/name)
        for role in ('A', 'B') for name in ('data.json', 'statistics.pt')}
    write(variant.parent/'registration.json', dict(source_root=str(x.root), protected=protected))
    binding = followup.prepare(variant, x.output, 1)
    assert binding['context_root'] == str(x.root)
    assert binding['upstream_study_registration']['sha256'] == sha256(variant.parent/'registration.json')
    assert binding['context_inputs']['metadata.json']['sha256'] == sha256(x.output/'metadata.json')


def test_rejects_unbound_study_context(source):
    variant = source.old/'study/control'
    variant.mkdir(parents=True)
    for role in ('A', 'B'):
        (variant/role).symlink_to(source.root/role, target_is_directory=True)
    write(variant.parent/'registration.json', dict(source_root=str(source.root), protected={}))
    with pytest.raises(ValueError, match='protect'):
        followup.context_root(variant)


@pytest.mark.parametrize('hours', [0, -1, float('inf'), float('nan'), True])
def test_rejects_invalid_new_window(source, hours):
    with pytest.raises(ValueError, match='finite and positive'):
        followup.prepare(source.root, source.output, hours)


def test_process_scan_detects_bound_outputs_ignores_embedded_shell_text(tmp_path):
    proc = tmp_path/'proc'
    worker = proc/'199999999'
    worker.mkdir(parents=True)
    worker.joinpath('stat').write_text('199999999 (worker) S 1 2 3')
    worker.joinpath('cwd').symlink_to(tmp_path)
    root = tmp_path/'old'
    worker.joinpath('cmdline').write_bytes(b'python\0worker.py\0--root\0unrelated\0--output\0old/child\0')
    with pytest.raises(ValueError, match='Live process'):
        followup.assert_quiescent({root}, [], proc=proc)
    worker.joinpath('cmdline').write_bytes(b'bash\0-c\0python worker.py --root old/child\0')
    write(tmp_path/'started.json', dict(pid=199999999))
    # A reused historical PID now runs an unrelated command and must not block.
    followup.assert_quiescent({root}, [tmp_path/'started.json'], proc=proc)
    worker.joinpath('stat').write_text('199999999 (worker) Z 1 2 3')
    write(tmp_path/'started.json', dict(pid=199999999))
    followup.assert_quiescent({root}, [tmp_path/'started.json'], proc=proc)


@pytest.fixture
def data_pool(source):
    """A newer accepted-data pool whose weights are deliberately absent."""
    x = source
    pool = x.old.parent/'newer/round05'
    for name in ('session.json', 'metadata.json', 'gate.json'):
        destination = pool/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(x.root/name, destination)
    new_path = pool/'new.h5'
    with h5py.File(new_path, 'w') as stream:
        stream.attrs['result_json'] = json.dumps(dict(seed=8, narrow_reset={'fixed':'reset'}))
    entry = dict(path=str(new_path), source_sha256=sha256(new_path), family='train', length=2,
        split='train', trajectory_kind='expert_takeover_v1', episode='new', takeover_tick=1)
    for role in ('A', 'B'):
        old_state, state = x.root/role/'state_data', pool/role/'state_data'
        (state/'perception').mkdir(parents=True)
        data = followup.read(old_state/'data.json')
        data['entries']['new'] = entry
        data['splits']['train'].append('new')
        data.update(source_statistics_sha256=sha256(old_state/'statistics.pt'),
            source_data_sha256=sha256(old_state/'data.json'),
            aggregation=dict(version='expert_takeover_aggregate_v1',
                selection='successful_expert_executed_corrections', student_prefix_supervised=False,
                new_training_ids=['new']))
        write(state/'data.json', data)
        shutil.copyfile(old_state/'statistics.pt', state/'statistics.pt')
        numeric = torch.load(old_state/'numeric.pt', weights_only=False)
        numeric.update(data_sha256=sha256(state/'data.json'))
        numeric['records']['new'] = {}
        torch.save(numeric, state/'numeric.pt')
        if role == 'A':
            shutil.copyfile(old_state/'loss_context.pt', state/'loss_context.pt')
        visual = followup.read(x.root/role/'cache_index.json')
        for identity, cached in visual['entries'].items():
            (state/'perception'/(identity+'.pt')).symlink_to(cached['path'])
        path = state/'perception/new.pt'
        path.write_bytes(b'new actual-frame frozen features')
        visual['entries']['new'] = dict(path=str(path), sha256=sha256(path),
            source_sha256=entry['source_sha256'], frames=2)
        visual['data_sha256'] = sha256(state/'data.json')
        write(pool/role/'cache_index.json', visual)
    return pool


def test_explicit_pool_keeps_evaluated_weights_bound_to_original_data(source, data_pool, monkeypatch):
    x = source
    before = {str(p):sha256(p) for root in (x.root, data_pool) for p in root.rglob('*') if p.is_file()}
    binding = followup.prepare(x.root, x.output, 2, data_source=data_pool)
    assert binding['version'] == 'independent_followup_binding_v2'
    assert binding['split_policy_data'] and binding['selected_pairs'] == x.pairs
    assert binding['data_pool_source']['roles']['A']['added_training_ids'] == ['new']
    for role in ('A', 'B'):
        assert (x.output/role/'state_data').resolve() == data_pool/role/'state_data'
        assert (x.output/role/'cache_index.json').resolve() == data_pool/role/'cache_index.json'
        assert (x.output/role/f'backend/{role}/state_data').resolve() == x.root/role/'state_data'
        assert (x.output/role/f'backend/{role}/nominal').resolve() == x.root/role/f'backend/{role}/nominal'
        for relative in ('training/best.pt', f'backend/{role}/training/best.pt'):
            saved = torch.load(x.output/role/relative, weights_only=False)
            assert saved['configuration']['data_sha256'] == sha256(x.root/role/'state_data/data.json')
            assert saved['configuration']['data_sha256'] != sha256(x.output/role/'state_data/data.json')
    from tools import tight_fresh_dagger
    monkeypatch.setattr(tight_fresh_dagger, 'completed_pair', x.completed)
    tight_fresh_dagger.prepare_collection(x.output, x.output/'takeover_06', [40], 6)
    plan = followup.read(x.output/'takeover_06/plan.json')
    assert plan['pairs'] == x.pairs
    for case in plan['cases']:
        assert len(followup.read(case['source_data'])['splits']['train']) == 2
        assert Path(case['source_data']).parent == data_pool/case['role']/'state_data'
    assert before == {str(p):sha256(p) for root in (x.root, data_pool) for p in root.rglob('*') if p.is_file()}
    # A mixed root cannot silently become a checkpoint-trained-on-new-data source.
    with pytest.raises(ValueError, match='populations differ'):
        followup.audit_role(x.output, 'A', x.pairs['A'], followup.read(x.output/'session.json'))


@pytest.mark.parametrize('fault', ['reorder', 'removed_old', 'old_entry', 'heldout_split', 'heldout_test',
    'heldout_family', 'normalization_ids', 'statistics', 'data_contract', 'session_contract',
    'metadata', 'numeric_old_record', 'visual_identity', 'visual_incomplete', 'geometry',
    'prefix_supervision', 'unknown_family', 'not_expert', 'ab_added_entry', 'gate'])
def test_explicit_pool_rejects_nonappend_or_contract_drift(source, data_pool, fault):
    x = source
    state = data_pool/'A/state_data'
    if fault == 'reorder': change(state/'data.json', lambda v:v['splits']['train'].reverse())
    elif fault == 'removed_old': change(state/'data.json', lambda v:v['splits']['train'].remove('train'))
    elif fault == 'old_entry': change(state/'data.json', lambda v:v['entries']['train'].update(length=3))
    elif fault == 'heldout_split': change(state/'data.json', lambda v:v['splits']['validation'].clear())
    elif fault == 'heldout_test': change(state/'data.json', lambda v:v['splits']['test'].append('new'))
    elif fault == 'heldout_family': change(state/'data.json', lambda v:v['entries']['new'].update(family='validation'))
    elif fault == 'normalization_ids': change(state/'data.json', lambda v:v['normalization_train_ids'].append('new'))
    elif fault == 'statistics': (state/'statistics.pt').write_bytes(b'refit affine')
    elif fault == 'data_contract': change(state/'data.json', lambda v:v.update(visual_stride=12))
    elif fault == 'session_contract': change(data_pool/'session.json', lambda v:v.update(domain={'wide':True}))
    elif fault == 'metadata': change(data_pool/'metadata.json', lambda v:v.update(task='changed'))
    elif fault == 'numeric_old_record':
        numeric = torch.load(state/'numeric.pt', weights_only=False)
        numeric['records']['train']['command'] = torch.ones(2, 29)
        torch.save(numeric, state/'numeric.pt')
    elif fault == 'visual_identity': change(data_pool/'A/cache_index.json',
        lambda v:v['entries']['train'].update(sha256='changed'))
    elif fault == 'visual_incomplete': change(data_pool/'A/cache_index.json', lambda v:v.update(complete=False))
    elif fault == 'geometry':
        torch.save({'train':torch.ones(2)}, x.root/'A/state_data/loss_context.pt')
    elif fault == 'prefix_supervision': change(state/'data.json', lambda v:v['aggregation'].update(student_prefix_supervised=True))
    elif fault == 'unknown_family': change(state/'data.json', lambda v:v['entries']['new'].update(family='unknown'))
    elif fault == 'not_expert': change(state/'data.json', lambda v:v['entries']['new'].update(trajectory_kind='student'))
    elif fault == 'ab_added_entry':
        # Keep A's own cache binding valid so this reaches the cross-role guard.
        change(state/'data.json', lambda v:v['entries']['new'].update(takeover_tick=2))
        change(data_pool/'A/cache_index.json', lambda v:v.update(data_sha256=sha256(state/'data.json')))
        numeric = torch.load(state/'numeric.pt', weights_only=False)
        numeric['data_sha256'] = sha256(state/'data.json')
        torch.save(numeric, state/'numeric.pt')
    elif fault == 'gate': change(data_pool/'gate.json', lambda v:v.update(passed=False))
    with pytest.raises(ValueError):
        followup.prepare(x.root, x.output, 2, data_source=data_pool)
    assert not x.output.exists()


def test_explicit_pool_requires_strict_addition_and_quiescent_data_owner(source, data_pool):
    with pytest.raises(ValueError, match='strictly extend'):
        followup.prepare(source.root, source.output, 2, data_source=source.root)
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', '--root', str(data_pool.parent)])
    (data_pool/'shared_resources.json').symlink_to(data_pool.parent/'shared_resources.json')
    try:
        with pytest.raises(ValueError, match='Live process'):
            followup.prepare(source.root, source.output, 2, data_source=data_pool)
    finally:
        worker.terminate()
        worker.wait(timeout=5)
    assert not source.output.exists()


def test_next_aggregate_keeps_all_pool_records_and_original_affine(source, data_pool, monkeypatch):
    from visuotactile import narrow_data
    # Real numeric-cache merge, with only the newly captured raw suffix decoded by a stub.
    def record(prefix=False):
        return dict(robot=torch.zeros(2, 178), command=torch.ones(2, 29), time=torch.arange(2)/60,
            joint_limits=torch.zeros(29, 2), q_reached=torch.full((2, 29), .5),
            supervision_mask=torch.tensor([not prefix, True]), control_segment=torch.tensor([0, 1]))
    for role in ('A', 'B'):
        for root in (source.root, data_pool):
            path = root/role/'state_data/numeric.pt'
            numeric = torch.load(path, weights_only=False)
            numeric['contract'] = {'version':'fixture'}
            numeric['records'] = {key:record(key=='new') for key in numeric['records']}
            torch.save(numeric, path)
    followup.prepare(source.root, source.output, 2, data_source=data_pool)
    monkeypatch.setattr(narrow_data, 'dagger_record', lambda *a, **k:(record(True), {}, {}))
    new = copy.deepcopy(followup.read(data_pool/'A/state_data/data.json')['entries']['new'])
    new['episode'] = 'next'
    for role in ('A', 'B'):
        out = source.output/f'aggregate/{role}'
        merged = narrow_data.build_dagger_aggregate(source.output/role/'state_data', {'next':new}, out,
            'q_reached' if role=='A' else 'u_expert', include_geometry=False, include_frontend_tactile=False)
        assert merged['splits'] == dict(train=['train', 'new', 'next'], validation=['validation'], test=[])
        assert merged['normalization_train_ids'] == ['train']
        assert sha256(out/'statistics.pt') == sha256(source.root/role/'state_data/statistics.pt')
        rows = torch.load(out/'numeric.pt', weights_only=False)['records']
        assert rows['new']['supervision_mask'].tolist() == [False, True]
        assert rows['next']['supervision_mask'].tolist() == [False, True]


@pytest.fixture
def alternate_policy(source, data_pool, monkeypatch):
    """A second evaluated pair trained on the complete newer pool."""
    x, pool = source, data_pool
    pairs = {}
    for role in ('A', 'B'):
        base, old_base = pool/role, x.root/role
        state, rear = base/'state_data', base/'backend'/role
        (base/'training').mkdir()
        (rear/'training').mkdir(parents=True)
        shutil.copytree(old_base/f'backend/{role}/nominal', rear/'nominal')
        saved = torch.load(old_base/'training/best.pt', weights_only=False)
        saved['configuration'].update(data_sha256=sha256(state/'data.json'),
            cache_index_sha256=sha256(base/'cache_index.json'))
        torch.save(saved, base/'training/best.pt')
        nominal = followup.read(rear/'nominal/index.json')
        (rear/'nominal/new.pt').write_bytes(b'new actual-source DP plans')
        nominal.update(base_sha256=sha256(base/'training/best.pt'), data_sha256=sha256(state/'data.json'))
        nominal['episodes']['new'] = dict(sha256=sha256(rear/'nominal/new.pt'))
        write(rear/'nominal/index.json', nominal)
        saved = torch.load(old_base/f'backend/{role}/training/best.pt', weights_only=False)
        saved['configuration'].update(data_sha256=nominal['data_sha256'], base_sha256=nominal['base_sha256'],
            cache_index_sha256=sha256(rear/'nominal/index.json'), base_checkpoint=str(base/'training/best.pt'))
        torch.save(saved, rear/'training/best.pt')
        write(rear/'training/finished.json', dict(target_reached=True))
        write(rear/'offline/result.json', dict(complete=True, test_used=False, checkpoint_sha256=sha256(rear/'training/best.pt')))
        pairs[role] = dict(frontend=str(base/'training/best.pt'), frontend_sha256=sha256(base/'training/best.pt'),
            backend=str(rear/'training/best.pt'), backend_sha256=sha256(rear/'training/best.pt'),
            residual_version=x.pairs[role]['residual_version'])
    shutil.copytree(x.evaluation, pool/'development')
    evaluation = pool/'development'
    change(evaluation/'plan.json', lambda v:v.update(pairs=pairs, resource_root=str(pool)))
    change(evaluation/'run_started.json', lambda v:v.update(plan_sha256=sha256(evaluation/'plan.json')))
    rows = followup.read(evaluation/'results.json')
    for row in rows:
        row['result']['checkpoint_sha256'] = pairs[row['role']]['backend_sha256']
        write(evaluation/row['case']/'result.json', row['result'])
    write(evaluation/'results.json', rows)
    def completed(root, role):
        front = (root/role/'training/best.pt').resolve()
        pair = pairs[role] if front == Path(pairs[role]['frontend']) else x.pairs[role]
        for key, rel in [('frontend','training/best.pt'), ('backend',f'backend/{role}/training/best.pt')]:
            assert (root/role/rel).resolve() == Path(pair[key])
            assert sha256(root/role/rel) == pair[key+'_sha256']
        return copy.deepcopy(pair)
    monkeypatch.setattr(followup, 'completed_pair', completed)
    return SimpleNamespace(root=pool, pairs=pairs, completed=completed)


def test_role_policy_selection_preserves_two_real_source_dependencies(source, alternate_policy, monkeypatch):
    x, alt = source, alternate_policy
    binding = followup.prepare(x.root, x.output, 2, data_source=alt.root, role_sources={'B':alt.root})
    selected = {'A':x.pairs['A'], 'B':alt.pairs['B']}
    assert binding['version'] == 'independent_followup_binding_v3'
    assert binding['selected_pairs'] == selected
    assert binding['role_policy_sources']['A']['source_root'] == str(x.root)
    assert binding['role_policy_sources']['B']['source_root'] == str(alt.root)
    assert binding['role_policy_sources']['B']['evaluation']['files']['plan.json']['sha256'] == sha256(alt.root/'development/plan.json')
    assert binding['data_pool_source']['roles']['A']['added_training_ids'] == ['new']
    assert binding['data_pool_source']['roles']['B']['added_training_ids'] == []
    assert (x.output/'B/backend/B/state_data').resolve() == alt.root/'B/state_data'
    assert (x.output/'B/backend/B/nominal').resolve() == alt.root/'B/backend/B/nominal'
    assert (x.output/'A/backend/A/state_data').resolve() == x.root/'A/state_data'
    for role in ('A', 'B'):
        assert (x.output/role/'state_data').resolve() == alt.root/role/'state_data'
    from tools import tight_fresh_dagger
    monkeypatch.setattr(tight_fresh_dagger, 'completed_pair', alt.completed)
    tight_fresh_dagger.prepare_collection(x.output, x.output/'takeover_06', [40], 6)
    plan = followup.read(x.output/'takeover_06/plan.json')
    assert plan['pairs'] == selected
    assert all('new' in followup.read(case['source_data'])['splits']['train'] for case in plan['cases'])


@pytest.mark.parametrize('fault', ['incomplete_evaluation', 'wrong_evaluated_pair', 'no_common_pool', 'unknown_role',
    'pool_smaller_than_selected_training', 'changed_policy_contract'])
def test_role_policy_selection_requires_complete_evidence_and_shared_superset(source, alternate_policy, fault):
    x, alt = source, alternate_policy
    kwargs = dict(data_source=alt.root, role_sources={'B':alt.root})
    if fault == 'incomplete_evaluation':
        change(alt.root/'development/result.json', lambda v:v.update(software_complete=False))
    elif fault == 'wrong_evaluated_pair':
        change(alt.root/'development/plan.json', lambda v:v.update(pairs=x.pairs))
        change(alt.root/'development/run_started.json', lambda v:v.update(plan_sha256=sha256(alt.root/'development/plan.json')))
    elif fault == 'no_common_pool': kwargs.pop('data_source')
    elif fault == 'unknown_role': kwargs['role_sources'] = {'C':alt.root}
    elif fault == 'pool_smaller_than_selected_training': kwargs['data_source'] = x.root
    elif fault == 'changed_policy_contract': change(alt.root/'session.json', lambda v:v.update(planning_hz=10))
    with pytest.raises(ValueError):
        followup.prepare(x.root, x.output, 2, **kwargs)
    assert not x.output.exists()


def test_mixed_policies_accept_a_third_untrained_complete_data_pool(source, alternate_policy):
    x, alt = source, alternate_policy
    pool = x.old.parent/'untrained_pool'
    pool.mkdir()
    for name in ('session.json', 'metadata.json', 'gate.json'):
        shutil.copyfile(alt.root/name, pool/name)
    entry = copy.deepcopy(followup.read(alt.root/'A/state_data/data.json')['entries']['new'])
    entry['episode'] = 'batch_gate'
    for role in ('A', 'B'):
        state = pool/role/'state_data'
        old_state = alt.root/role/'state_data'
        (state/'perception').mkdir(parents=True)
        data = followup.read(old_state/'data.json')
        data['entries']['batch_gate'] = entry
        data['splits']['train'].append('batch_gate')
        data['aggregation']['new_training_ids'] = ['batch_gate']
        data['source_data_sha256'] = sha256(old_state/'data.json')
        write(state/'data.json', data)
        shutil.copyfile(old_state/'statistics.pt', state/'statistics.pt')
        if role == 'A': shutil.copyfile(old_state/'loss_context.pt', state/'loss_context.pt')
        numeric = torch.load(old_state/'numeric.pt', weights_only=False)
        numeric['data_sha256'] = sha256(state/'data.json')
        numeric['records']['batch_gate'] = {}
        torch.save(numeric, state/'numeric.pt')
        visual = followup.read(alt.root/role/'cache_index.json')
        visual['data_sha256'] = sha256(state/'data.json')
        visual['entries']['batch_gate'] = copy.deepcopy(visual['entries']['new'])
        for identity, cached in visual['entries'].items():
            (state/'perception'/(identity+'.pt')).symlink_to(cached['path'])
        write(pool/role/'cache_index.json', visual)
    binding = followup.prepare(x.root, x.output, 2, data_source=pool, role_sources={'B':alt.root})
    assert not (pool/'A/training').exists() and not (pool/'B/training').exists()
    assert not (pool/'development').exists()
    assert binding['selected_pairs'] == {'A':x.pairs['A'], 'B':alt.pairs['B']}
    assert binding['data_pool_source']['roles']['A']['added_training_ids'] == ['new', 'batch_gate']
    assert binding['data_pool_source']['roles']['B']['added_training_ids'] == ['batch_gate']
    assert followup.read(x.output/'A/state_data/data.json')['splits']['train'] == ['train', 'new', 'batch_gate']
