import json
import signal
import time
from types import SimpleNamespace

import pytest

from tools import tight_fresh_collection_owner as owner
from tools.tight_fresh_stage_takeover import CONTRACT, CONTACT_HEIGHT_CONTRACT, NEVER_FALLBACK
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import VERSION, CONTACT_HEIGHT_VERSION


@pytest.fixture
def registered(tmp_path, monkeypatch):
    output = tmp_path / 'collection'
    tree = tmp_path / 'frozen'
    resource = tmp_path / 'session'
    for path in (output, resource, tree / 'tools', tree / 'visuotactile'):
        path.mkdir(parents=True)
    write = lambda path, value: path.write_text(json.dumps(value))
    session = dict(deadline=time.time() + 1000)
    for path in (output, resource):
        write(path / 'session.json', session)
    (tree / 'tools/worker.py').write_text('# frozen\n')
    case = dict(name='A_one', role='A', spec=dict(takeover_tick=NEVER_FALLBACK),
        dagger=dict(trigger=dict(version=VERSION, delay_ticks=0)))
    plan = dict(kind='takeover', stage_takeover_contract=CONTRACT, never_fallback=NEVER_FALLBACK,
        cases=[case], test_used=False, source_commit='commit', resource_root=str(resource),
        source_session_sha256=sha256(output / 'session.json'),
        source_sha256={'tools/worker.py': sha256(tree / 'tools/worker.py')})
    write(output / 'plan.json', plan)
    monkeypatch.setattr(owner.subprocess, 'check_output', lambda *a, **k: 'commit\n')
    calls = []
    worker = SimpleNamespace(pid=123456, returncode=0)
    worker.poll = lambda: worker.returncode
    def spawn(command, **kwargs):
        calls.append(dict(command=command, **kwargs))
        write(output / 'run_started.json', dict(pid=worker.pid, utc=time.time(),
            plan_sha256=sha256(output / 'plan.json')))
        write(output / 'initialization.json', dict(seconds=1))
        write(output / 'result.json', dict(software_complete=True, worker_error=None, cases=1, utc=time.time()))
        write(output / 'results.json', [dict(case='A_one', role='A', result=dict(software_complete=True))])
        (output / 'A_one').mkdir()
        write(output / 'A_one/stable_stage_audit.json', dict(no_expert_fallback=True,
            source_plan_sha256=sha256(output / 'plan.json')))
        return worker
    monkeypatch.setattr(owner.subprocess, 'Popen', spawn)
    return SimpleNamespace(output=output, tree=tree, resource=resource, write=write, plan=plan,
                           session=session, worker=worker, calls=calls)


def test_owner_launches_exact_stage_cli_and_binds_complete_opportunities(registered):
    x = registered
    owner.run(x.output, x.tree)
    call = x.calls[0]
    assert call['command'] == [str(x.tree / '.venv_isaacsim/bin/python'), '-m',
        'tools.tight_fresh_stage_takeover', 'run', '--output', str(x.output)]
    assert call['cwd'] == x.tree and call['start_new_session'] is True
    assert call['env']['PYTHONFAULTHANDLER'] == '1'
    assert owner.read(x.output / 'owner_worker_started.json')['created_process_group'] == x.worker.pid
    assert owner.read(x.output / 'owner_finished.json')['software_complete'] is True
    with pytest.raises(FileExistsError):
        owner.run(x.output, x.tree)
    assert len(x.calls) == 1


@pytest.mark.parametrize('wrong_kind', [False, True])
def test_contact_collection_requires_its_own_stage_receipt(registered, monkeypatch, wrong_kind):
    x = registered
    x.plan['stage_takeover_contract'] = CONTACT_HEIGHT_CONTRACT
    x.plan['cases'][0]['dagger']['trigger']['version'] = CONTACT_HEIGHT_VERSION
    x.write(x.output / 'plan.json', x.plan)
    def wait(*args, **kwargs):
        if not wrong_kind:
            (x.output / 'A_one/stable_stage_audit.json').rename(
                x.output / 'A_one/contact_height_stage_audit.json')
    monkeypatch.setattr(owner, 'wait_owned', wait)
    if wrong_kind:
        with pytest.raises(FileNotFoundError):
            owner.run(x.output, x.tree)
        assert owner.read(x.output / 'owner_finished.json')['software_complete'] is False
    else:
        owner.run(x.output, x.tree)
        assert owner.read(x.output / 'owner_finished.json')['software_complete'] is True


@pytest.mark.parametrize('name', ['initialization.json', 'result.json', 'results.json', 'owner_started.json'])
def test_prior_markers_cannot_bypass_startup_bound_or_reuse_results(registered, name):
    x = registered
    x.write(x.output / name, {})
    with pytest.raises(FileExistsError):
        owner.run(x.output, x.tree)
    assert not x.calls


@pytest.mark.parametrize('fault', ['count', 'rows', 'pid', 'late_result', 'wrong_audit'])
def test_success_flag_alone_cannot_claim_collection_complete(registered, monkeypatch, fault):
    x = registered
    def wait(*args, **kwargs):
        if fault == 'count':
            result = owner.read(x.output / 'result.json'); result['cases'] = 2
            x.write(x.output / 'result.json', result)
        elif fault == 'rows':
            x.write(x.output / 'results.json', [])
        elif fault == 'pid':
            launch = owner.read(x.output / 'run_started.json'); launch['pid'] += 1
            x.write(x.output / 'run_started.json', launch)
        elif fault == 'late_result':
            result = owner.read(x.output / 'result.json'); result['utc'] = x.session['deadline'] + 1
            x.write(x.output / 'result.json', result)
        else:
            audit = owner.read(x.output / 'A_one/stable_stage_audit.json'); audit['source_plan_sha256'] = 'other'
            x.write(x.output / 'A_one/stable_stage_audit.json', audit)
    monkeypatch.setattr(owner, 'wait_owned', wait)
    with pytest.raises(ValueError):
        owner.run(x.output, x.tree)
    assert owner.read(x.output / 'owner_finished.json')['software_complete'] is False


def test_owner_sigterm_stops_only_its_created_worker_and_restores_handlers(registered, monkeypatch):
    x = registered
    x.worker.returncode = None
    old = signal.getsignal(signal.SIGTERM)
    stopped = []
    def stop(worker):
        stopped.append(worker.pid)
        worker.returncode = -signal.SIGTERM
    def wait(worker, deadline, **kwargs):
        assert deadline == x.session['deadline'] and kwargs['startup_seconds'] == 180
        assert kwargs['startup_marker'] == x.output / 'initialization.json'
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(owner, 'stop_owned', stop)
    monkeypatch.setattr(owner, 'wait_owned', wait)
    with pytest.raises(SystemExit):
        owner.run(x.output, x.tree)
    assert stopped == [x.worker.pid] and signal.getsignal(signal.SIGTERM) == old
    receipt = owner.read(x.output / 'owner_finished.json')
    assert receipt['software_complete'] is False and receipt['stop_reason'] == 'owner_interrupted'


@pytest.mark.parametrize('fault', ['expired', 'session_changed', 'source_extra', 'source_changed', 'commit'])
def test_launch_rejects_changed_deadline_or_frozen_source(registered, monkeypatch, fault):
    x = registered
    if fault == 'expired':
        x.write(x.output / 'session.json', dict(deadline=time.time() - 1))
    elif fault == 'session_changed':
        x.write(x.resource / 'session.json', dict(deadline=x.session['deadline'] + 1))
    elif fault == 'source_extra':
        (x.tree / 'tools/extra.py').write_text('# extra\n')
    elif fault == 'source_changed':
        (x.tree / 'tools/worker.py').write_text('# changed\n')
    else:
        monkeypatch.setattr(owner.subprocess, 'check_output', lambda *a, **k: 'different\n')
    with pytest.raises((TimeoutError, ValueError)):
        owner.run(x.output, x.tree)
    assert not x.calls and not (x.output / 'owner_started.json').exists()
