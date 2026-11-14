import copy
import json
import os
from pathlib import Path
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
