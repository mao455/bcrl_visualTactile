import copy
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from visuotactile.compact import sha256
from visuotactile.latest_takeover_sampling import episode_deck
from visuotactile.paired_episode_io import PairWorker, configure_pairing, process_alive


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def dataset(root, *, shared=True):
    entries = {}
    for role in ('A', 'B'):
        directory = root / role / 'state_data'
        (directory / 'perception').mkdir(parents=True)
        for identity in ('short', 'long'):
            path = directory / 'perception' / (identity + '.pt')
            if role == 'A' or not shared:
                path.write_bytes(('frozen-' + identity).encode())
            else:
                os.link(root / 'A/state_data/perception' / path.name, path)
            entries[identity] = dict(path=str(path), sha256=sha256(path), frames=8,
                                    source_sha256='0'*64, camera_time_sha256='1'*64)
        data = dict(visual_pair=True, visual_stride=1, splits=dict(train=['short', 'long'], validation=[]),
                    entries={key: {} for key in entries})
        write(directory / 'data.json', data)
        write(root / role / 'cache_index.json', dict(complete=True, stride=1, trainable_projection_cached=False,
            data_sha256=sha256(directory / 'data.json'), entries=copy.deepcopy(entries)))
    write(root / 'session.json', dict(deadline=time.time() + 60))


def train_toy(role, pairing=None):
    """Full histories, independent RNG/parameters/AdamW/EMA, one step per visit."""
    torch.set_num_threads(1)
    torch.manual_seed(101 if role == 'A' else 202)
    cell = torch.nn.GRUCell(4, 5).double()
    head = torch.nn.Linear(5, 1).double()
    params = list(cell.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=.001)
    ema = [value.detach().clone() for value in params]
    schedule = torch.Generator().manual_seed(2026111402)
    noise = torch.Generator().manual_seed(17)
    order_seen = []; updates = 0
    for epoch in (1, 2):
        order = episode_deck(['short', 'long'], {'short': 1, 'long': 1}, schedule)
        if pairing: pairing.begin_epoch(epoch, order)
        for index, identity in enumerate(order):
            if pairing: pairing.before_visit(epoch, index, identity)
            optimizer.zero_grad(set_to_none=True)
            n = 5 if identity == 'short' else 8
            inputs = torch.arange(n*4, dtype=torch.float64).reshape(n, 4) / 30
            hidden = None; predictions = []
            for tick in range(n):
                hidden = cell(inputs[tick:tick+1], hidden)
                predictions.append(head(hidden))
            truth = torch.randn(n, 1, generator=noise, dtype=torch.float64) * .1
            legal = torch.arange(n) >= 2
            loss = (torch.cat(predictions)[legal] - truth[legal]).square().mean()
            loss.backward(); optimizer.step(); updates += 1
            for mean, value in zip(ema, params):
                mean.mul_(.9).add_(value.detach(), alpha=.1)
            order_seen.append(identity)
            if pairing: pairing.after_visit(updates)
    return dict(parameters=[value.detach() for value in params], ema=ema, optimizer=optimizer.state_dict(),
        schedule_rng=schedule.get_state(), noise_rng=noise.get_state(), torch_rng=torch.get_rng_state(),
        optimizer_updates=updates, ema_updates=updates, order=order_seen)


def child(root_string, role, mode='normal'):
    root = Path(root_string)
    try:
        with PairWorker(root, role, 2) as pairing:
            if role == 'B' and mode == 'crash':
                os._exit(9)
            if role == 'B' and mode == 'failure':
                raise ValueError('injected peer failure')
            if role == 'B' and mode == 'wrong_deck':
                pairing.begin_epoch(1, ['wrong', 'deck'])
            result = train_toy(role, pairing)
        torch.save(result, root / (role + '_toy.pt'))
    except BaseException as error:
        write(root / (role + '_error.json'), dict(error=repr(error)))
        raise SystemExit(1)


def launch(root, mode='normal'):
    context = multiprocessing.get_context('spawn')
    processes = [context.Process(target=child, args=(str(root), role, mode)) for role in ('A', 'B')]
    for process in processes: process.start()
    try:
        for process in processes: process.join(15)
        assert all(not process.is_alive() for process in processes), 'A paired worker waited without bound'
        return [process.exitcode for process in processes]
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join(2)


def equal_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b): equal_tree(left, right)
    else:
        assert a == b


def test_two_real_processes_preserve_independent_training_exactly(tmp_path):
    dataset(tmp_path)
    state = torch.get_rng_state().clone()
    configure_pairing(tmp_path, 2, startup_timeout_s=8, barrier_timeout_s=8, poll_s=.02)
    assert torch.equal(state, torch.get_rng_state())
    assert launch(tmp_path) == [0, 0]
    for role in ('A', 'B'):
        actual = torch.load(tmp_path / (role + '_toy.pt'), weights_only=False)
        expected = train_toy(role)
        equal_tree(actual, expected)
        status = json.loads((tmp_path / 'paired_episode_io' / role / 'status.json').read_text())
        assert status['state'] == 'finished' and status['completed_visits'] == 4
        assert status['summary']['visits'] == 4
    a = torch.load(tmp_path / 'A_toy.pt', weights_only=False)
    b = torch.load(tmp_path / 'B_toy.pt', weights_only=False)
    assert not torch.equal(a['parameters'][0], b['parameters'][0])


@pytest.mark.parametrize('mode,reason', [('failure', 'peer failed'), ('crash', 'peer exited'), ('wrong_deck', 'peer failed')])
def test_peer_failure_exit_and_wrong_deck_stop_both(tmp_path, mode, reason):
    dataset(tmp_path)
    configure_pairing(tmp_path, 2, startup_timeout_s=8, barrier_timeout_s=3, poll_s=.02)
    codes = launch(tmp_path, mode)
    assert codes[0] == 1 and codes[1] != 0
    assert reason in json.loads((tmp_path / 'A_error.json').read_text())['error']
    assert json.loads((tmp_path / 'paired_episode_io/A/status.json').read_text())['state'] == 'failed'


def test_missing_peer_startup_has_a_persistent_bounded_failure(tmp_path):
    dataset(tmp_path)
    configure_pairing(tmp_path, 2, startup_timeout_s=.15, barrier_timeout_s=.2, poll_s=.01)
    began = time.monotonic()
    with pytest.raises(TimeoutError, match='startup'):
        with PairWorker(tmp_path, 'A', 2): pass
    assert time.monotonic() - began < 1
    assert json.loads((tmp_path / 'paired_episode_io/A/status.json').read_text())['state'] == 'failed'
    with pytest.raises(FileExistsError): PairWorker(tmp_path, 'A', 2)


def test_live_but_stalled_peer_barrier_and_session_deadline_are_bounded(tmp_path):
    live = tmp_path / 'live'; dataset(live)
    configure_pairing(live, 2, startup_timeout_s=.3, barrier_timeout_s=.15, poll_s=.01)
    a, b = PairWorker(live, 'A', 2), PairWorker(live, 'B', 2)
    order = episode_deck(['short', 'long'], {'short': 1, 'long': 1}, torch.Generator().manual_seed(2026111402))
    with pytest.raises(TimeoutError, match='barrier'):
        with a:
            a.begin_epoch(1, order)
            a.before_visit(1, 0, order[0])  # B's process is alive, but never ready.
    with pytest.raises(RuntimeError, match='peer failed'):
        with b: pass
    expired = tmp_path / 'expired'; dataset(expired)
    write(expired / 'session.json', dict(deadline=time.time()-1))
    configure_pairing(expired, 2)
    with pytest.raises(TimeoutError, match='session deadline'):
        with PairWorker(expired, 'A', 2): pass


def test_reject_copied_shards_and_changed_cache_or_completed_attempt(tmp_path):
    copied = tmp_path / 'copied'; dataset(copied, shared=False)
    with pytest.raises(ValueError, match='same immutable frozen shard inode'):
        configure_pairing(copied, 2)
    shared = tmp_path / 'shared'; dataset(shared)
    configure_pairing(shared, 2)
    with (shared / 'A/state_data/perception/short.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='frozen inputs changed'): PairWorker(shared, 'A', 2)
    assert json.loads((shared / 'paired_episode_io/A/status.json').read_text())['state'] == 'failed'
    existing = tmp_path / 'existing'; dataset(existing)
    (existing / 'A/training').mkdir()
    with pytest.raises(ValueError, match='no automatic training resume'): configure_pairing(existing, 2)
    resumed = tmp_path / 'resumed'; dataset(resumed); configure_pairing(resumed, 2)
    (resumed / 'B/training').mkdir(); (resumed / 'B/training/last.pt').write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='automatic resume is unsupported'): PairWorker(resumed, 'A', 2)


def test_reused_pid_and_zombie_are_not_live(monkeypatch):
    from visuotactile import paired_episode_io
    monkeypatch.setattr(paired_episode_io, 'process_identity', lambda pid: dict(pid=pid, proc_start_ticks=5, state='S'))
    assert not process_alive(dict(pid=1, proc_start_ticks=4))
    monkeypatch.setattr(paired_episode_io, 'process_identity', lambda pid: dict(pid=pid, proc_start_ticks=4, state='Z'))
    assert not process_alive(dict(pid=1, proc_start_ticks=4))


def test_default_train_dispatch_has_no_pairing_and_serial_cycle_rejected(tmp_path, monkeypatch):
    from tools import tight_fresh_train, tight_fresh_cycle
    calls = []
    monkeypatch.setattr(tight_fresh_train, '_train', lambda *args, **kwargs: calls.append((args, kwargs)))
    tight_fresh_train.train(tmp_path, 'A', 2)
    assert calls == [((tmp_path, 'A', 2), {})]
    with pytest.raises(ValueError, match='parallel independent'):
        tight_fresh_cycle.run(tmp_path, tmp_path, tmp_path, 2, 1, 4, 1, paired_episode_io=True)


@pytest.mark.parametrize('admitted', [False, True])
def test_cycle_registers_after_cache_and_requires_joint_admission(tmp_path, monkeypatch, admitted):
    from tools import tight_fresh_cycle as cycle
    from test_fresh_parallel_cycle import admission_clock
    clock=admission_clock(monkeypatch)
    root, collection, output = (tmp_path / name for name in ('source', 'collection', 'next'))
    write(root / 'session.json', dict(deadline=time.time()+60, resource_policy='full_gpu_v1'))
    write(root / 'shared_resources.json', dict(utc_epoch=time.time(),
        gpu_free_mib=32*1024 if admitted else 1024, ram_available_bytes=32*1024**3))
    def fresh_sample(duration):
        clock[0]+=duration
        write(root / 'shared_resources.json', dict(utc_epoch=clock[0],
            gpu_free_mib=32*1024 if admitted else 1024, ram_available_bytes=32*1024**3))
    monkeypatch.setattr(cycle.time,'sleep',fresh_sample)
    write(collection / 'result.json', dict(software_complete=True))
    write(collection / 'correction_entries.json', {'real': {}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cycle.subprocess, 'check_output', lambda *args, **kwargs: 'source commit')
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        if 'aggregate' in command:
            dataset(output)
            write(output / 'prepared.json', dict(complete=True))
        return SimpleNamespace(pid=os.getpid(), returncode=0, poll=lambda: 0)
    monkeypatch.setattr(cycle.subprocess, 'Popen', launch)
    def only_after_registration(*args):
        assert len(calls) == 2  # Aggregate and completed immutable visual-cache stage.
        assert (output / 'paired_episode_io.json').exists()
        raise RuntimeError('registered CPU dispatch boundary')
    monkeypatch.setattr(cycle, 'paired_frontends', only_after_registration)
    match = 'registered CPU dispatch' if admitted else 'no serial fallback'
    with pytest.raises(RuntimeError, match=match):
        cycle.run(root, collection, output, 2, 1, 4, 1, parallel_frontends=True, paired_episode_io=True)
    assert len(calls) == 2  # Neither branch launches a real frontend in this CPU test.
    assert (output / 'paired_episode_io.json').exists() == admitted
    admission=cycle.read(output.with_name(output.name+'_stages')/'parallel_admission.json')
    assert admission['frontend_mode']==('parallel' if admitted else 'paired_rejected_no_serial_fallback')
