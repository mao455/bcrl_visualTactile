"""Opt-in bounded A/B I/O rendezvous; independent learning and RNG stay in each process.

Only immutable shared-inode frozen shards are coordinated. This changes wall-time
scheduling, not either learner's episode order, optimizer/EMA count, or history.
Failed attempts are retained and cannot resume through this synchronization root.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .compact import sha256
from .training_resources import atomic_json

VERSION = 'paired_episode_io_v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, proc_start_ticks=int(fields[19]), state=fields[0])
    except (OSError, ValueError, IndexError):
        return None


def process_alive(identity):
    current = process_identity(identity['pid'])
    return (current is not None and current['proc_start_ticks'] == identity['proc_start_ticks']
            and current['state'] not in ('Z', 'X'))


def shard_identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def shared_inputs(root):
    """Audit small indexes/stat identities, without rereading tens of GB of payload."""
    from .latest_takeover_sampling import sampling_plan
    roles = {}; shards = {}; common = None
    for role in ('A', 'B'):
        directory = root / role
        data_path = directory / 'state_data/data.json'
        continuation_path = directory / 'continuation.json'
        index_path = directory / 'cache_index.json'
        data, index = read(data_path), read(index_path)
        continuation = read(continuation_path) if continuation_path.exists() else {}
        if (data.get('visual_pair') is not True or data.get('visual_stride') != 1
                or index.get('complete') is not True or index.get('stride') != 1
                or index.get('trainable_projection_cached') is not False
                or index.get('data_sha256') != sha256(data_path)):
            raise ValueError('Paired I/O requires complete current dense frozen visual cache')
        _, multiplicities = sampling_plan(data, continuation.get('sampling_contract'))
        population = dict(ids=data['splits']['train'], multiplicities=multiplicities)
        if common is not None and population != common:
            raise ValueError('Paired I/O requires exactly the same ordered training deck population')
        common = population
        roles[role] = dict(data_sha256=sha256(data_path), cache_index_sha256=sha256(index_path),
            continuation_sha256=sha256(continuation_path) if continuation_path.exists() else None)
        for identity in population['ids']:
            entry = index['entries'][identity]
            path = directory / 'state_data/perception' / (identity + '.pt')
            actual = shard_identity(path)
            if actual != shard_identity(entry['path']):
                raise ValueError('Frozen visual shard differs from its registered cache path')
            payload = dict(file=actual, sha256=entry['sha256'], source_sha256=entry['source_sha256'],
                           frames=entry['frames'], camera_time_sha256=entry['camera_time_sha256'])
            if identity in shards and payload != shards[identity]:
                raise ValueError('A/B must share the same immutable frozen shard inode and cache identity')
            shards[identity] = payload
    return dict(roles=roles, population=common, shared_shards=shards)


def configure_pairing(root, epochs, *, startup_timeout_s=300., barrier_timeout_s=600., poll_s=.02):
    """Register once after visual-cache completion and before either trainer starts."""
    import torch
    from .latest_takeover_sampling import episode_deck
    root = Path(root).resolve()
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError('Positive bounded paired epoch count required')
    for value in (startup_timeout_s, barrier_timeout_s, poll_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive paired I/O waits required')
    if poll_s > min(startup_timeout_s, barrier_timeout_s):
        raise ValueError('Polling interval exceeds bounded wait')
    if any((root / role / 'training').exists() for role in ('A', 'B')):
        raise ValueError('Paired I/O only supports new attempts; no automatic training resume')
    if (root / 'paired_episode_io.json').exists() or (root / 'paired_episode_io').exists():
        raise FileExistsError('Preserve previous paired I/O attempt')
    inputs = shared_inputs(root)
    population = inputs['population']
    generator = torch.Generator().manual_seed(2026111402)
    expected = [digest(episode_deck(population['ids'], population['multiplicities'], generator))
                for _ in range(epochs)]
    session = read(root / 'session.json')
    resource_root = Path(session.get('resource_root', root)).resolve()
    stop_paths = list(dict.fromkeys(str(path) for path in (
        root / 'STOP', resource_root / 'STOP', root.parent / (root.name + '_stages') / 'STOP')))
    config = dict(version=VERSION, attempt=uuid.uuid4().hex, epochs=epochs,
        episodes_per_epoch=sum(population['multiplicities'].values()), epoch_deck_sha256=expected,
        inputs=inputs, startup_timeout_s=float(startup_timeout_s), barrier_timeout_s=float(barrier_timeout_s),
        poll_s=float(poll_s), deadline=session['deadline'], stop_paths=stop_paths,
        semantics='Readiness/done rendezvous around unchanged independent episode updates. '
            'Same shared-inode frozen inputs, no learned-state cache, no optimizer/RNG sharing. '
            'Wall scheduling only; speed requires measured acceptance; failed root cannot resume.')
    (root / 'paired_episode_io').mkdir()
    atomic_json(root / 'paired_episode_io.json', config)
    return config


class PairWorker:
    def __init__(self, root, role, epochs):
        self.root = Path(root).resolve()
        if role not in ('A', 'B'):
            raise ValueError('Paired I/O role must be A or B')
        self.role, self.peer = role, 'B' if role == 'A' else 'A'
        self.config = read(self.root / 'paired_episode_io.json')
        self.directory = self.root / 'paired_episode_io'
        self.own = self.directory / role
        self.own.mkdir(exist_ok=False)
        self.identity = dict(role=role, token=uuid.uuid4().hex,
            **{k: v for k, v in process_identity(os.getpid()).items() if k != 'state'})
        self.registration_sha256 = sha256(self.root / 'paired_episode_io.json')
        self.completed = 0; self.current = None; self.epoch = None
        self.waits = []; self.startup_wait_s = 0.; self.closed = False
        self._state('starting')
        try:
            if self.config['version'] != VERSION or self.config['epochs'] != epochs:
                raise ValueError('Paired I/O registration/epoch identity differs')
            if shared_inputs(self.root) != self.config['inputs']:
                raise ValueError('Paired I/O frozen inputs changed')
            if any((self.root / role / 'training' / name).exists()
                   for role in ('A', 'B') for name in ('last.pt', 'progress.json', 'finished.json')):
                raise ValueError('Paired I/O automatic resume is unsupported; preserve the failed attempt')
        except BaseException as error:
            self.fail(error)
            raise
        self._state('registered')

    def _state(self, state, **extras):
        atomic_json(self.own / 'status.json', dict(self.identity, attempt=self.config['attempt'],
            registration_sha256=self.registration_sha256, state=state, completed_visits=self.completed,
            current=self.current, utc=time.time(), **extras))

    def _peer_status(self):
        path = self.directory / self.peer / 'status.json'
        if not path.exists():
            return None
        value = read(path)
        if (value['attempt'] != self.config['attempt'] or value['registration_sha256'] != self.registration_sha256
                or value['role'] != self.peer):
            raise ValueError('Peer belongs to a different registered paired attempt')
        if value['state'] in ('failed', 'aborted'):
            raise RuntimeError('Paired peer failed: ' + value.get('error', value['state']))
        return value

    def _wait(self, predicate, *, startup=False):
        began = time.monotonic()
        budget = self.config['startup_timeout_s' if startup else 'barrier_timeout_s']
        while True:
            if any(Path(path).exists() for path in self.config.get('stop_paths', [])):
                raise InterruptedError('Registered paired I/O STOP during synchronization')
            if time.time() >= self.config['deadline']:
                raise TimeoutError('Original paired session deadline exhausted')
            peer = self._peer_status()
            if peer is not None:
                # A normal finished peer may have left its exact final done receipt.
                # Any active peer exit, including a zombie or reused PID, aborts.
                if peer['state'] != 'finished' and not process_alive(peer):
                    raise RuntimeError('Paired peer exited without successful completion')
                if predicate(peer):
                    return time.monotonic() - began
                if peer['state'] == 'finished':
                    raise RuntimeError('Paired peer finished before the required matching receipt')
            if time.monotonic() - began >= budget:
                raise TimeoutError('Bounded paired I/O ' + ('startup' if startup else 'barrier') + ' wait exhausted')
            time.sleep(min(self.config['poll_s'], max(0., budget - (time.monotonic() - began))))

    def join(self):
        self.startup_wait_s = self._wait(lambda peer: peer['state'] in ('registered', 'active'), startup=True)
        self._state('active', startup_wait_s=self.startup_wait_s)

    def begin_epoch(self, epoch, order):
        if (epoch != self.completed // self.config['episodes_per_epoch'] + 1
                or not 1 <= epoch <= self.config['epochs']
                or digest(order) != self.config['epoch_deck_sha256'][epoch - 1]):
            raise ValueError('Paired I/O actual epoch deck or cursor differs')
        self.epoch = epoch
        self.order = list(order)
        atomic_json(self.own / ('epoch_%03d.json' % epoch), dict(order=order, sha256=digest(order)))

    def _receipt(self, stage, update, identity):
        path = self.directory / ('visit_%06d' % update) / stage
        path.mkdir(parents=True, exist_ok=True)
        own = path / (self.role + '.json')
        if own.exists():
            raise FileExistsError('Never repeat a paired episode receipt')
        receipt = dict(self.identity, attempt=self.config['attempt'], registration_sha256=self.registration_sha256,
            update=update, identity=identity, epoch=self.epoch, stage=stage, utc=time.time())
        atomic_json(own, receipt)
        def matches(peer):
            other = path / (self.peer + '.json')
            if not other.exists():
                return False
            row = read(other)
            for key in ('attempt', 'registration_sha256', 'update', 'identity', 'epoch', 'stage'):
                if row[key] != receipt[key]:
                    raise ValueError('Paired episode receipt mismatch: ' + key)
            if (row['token'] != peer['token'] or row['pid'] != peer['pid']
                    or row['proc_start_ticks'] != peer['proc_start_ticks']):
                raise ValueError('Paired episode receipt belongs to another worker')
            return True
        return self._wait(matches)

    def before_visit(self, epoch, index, identity):
        update = (epoch - 1) * self.config['episodes_per_epoch'] + index + 1
        if self.epoch != epoch or update != self.completed + 1 or self.current is not None:
            raise ValueError('Paired I/O visit cursor differs')
        # The complete order is independently verified, not generated by this worker.
        order = self.order
        if not 0 <= index < len(order) or order[index] != identity:
            raise ValueError('Paired I/O visit identity differs from actual deck')
        if (shard_identity(self.root / self.role / 'state_data/perception' / (identity + '.pt'))
                != self.config['inputs']['shared_shards'][identity]['file']):
            raise ValueError('Paired frozen shard changed before its episode visit')
        self.current = dict(update=update, identity=identity, epoch=epoch, index=index)
        self.ready_wait_s = self._receipt('ready', update, identity)
        return self.ready_wait_s

    def after_visit(self, actual_updates):
        if self.current is None or actual_updates != self.current['update']:
            raise ValueError('Paired I/O optimizer update count differs')
        waiting = self._receipt('done', actual_updates, self.current['identity'])
        row = dict(self.current, ready_wait_s=self.ready_wait_s, done_wait_s=waiting,
                   total_wait_s=self.ready_wait_s + waiting, utc=time.time())
        with (self.own / 'waits.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        self.waits.append(row); self.completed += 1; self.current = None

    def summary(self, *, epoch=None):
        rows = [row for row in self.waits if epoch is None or row['epoch'] == epoch]
        return dict(version=VERSION, registration_sha256=self.registration_sha256, role=self.role,
            visits=len(rows), startup_wait_s=self.startup_wait_s,
            ready_wait_s=sum(row['ready_wait_s'] for row in rows),
            done_wait_s=sum(row['done_wait_s'] for row in rows),
            total_wait_s=sum(row['total_wait_s'] for row in rows),
            optimizer_and_ema_updates_unchanged=True, learned_state_cached=False)

    def fail(self, error):
        if not self.closed:
            self._state('failed', error=repr(error), summary=self.summary())
            self.closed = True

    def finish(self):
        if self.current is not None or self.completed != self.config['epochs'] * self.config['episodes_per_epoch']:
            raise ValueError('Cannot finish an incomplete paired training attempt')
        self._state('finished', summary=self.summary()); self.closed = True

    def __enter__(self):
        try:
            self.join()
        except BaseException as error:
            self.fail(error)
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception is not None:
            self.fail(exception)
        else:
            try:
                self.finish()
            except BaseException as error:
                self.fail(error)
                raise
