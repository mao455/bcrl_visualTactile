import hashlib
import json
import os
from pathlib import Path

import pytest

from visuotactile.frozen_payload_audit import FrozenPayloadAudit


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_real_file_aliases_reuse_only_a_digest_read_in_this_process(tmp_path):
    data = b'real frozen payload' * 100000
    source = tmp_path / 'source';source.write_bytes(data)
    symlink = tmp_path / 'symbolic';symlink.symlink_to(source)
    hardlink = tmp_path / 'hard';hardlink.hardlink_to(source)
    audit = FrozenPayloadAudit()
    assert [audit.sha256(p) for p in [source,symlink,hardlink]] == [digest(data)] * 3
    stats = audit.statistics()
    assert stats['actual_bytes_read'] == len(data)
    assert stats['full_hashes'] == 1 and stats['memo_hits'] == 2
    assert stats['avoided_full_hash_bytes'] == 2 * len(data)
    assert stats['requested_regular_file_bytes'] == 3 * len(data)
    assert stats['file_descriptors_retained'] == 0


def test_distinct_inodes_and_eviction_read_full_bytes(tmp_path):
    one = tmp_path / 'one';two = tmp_path / 'two'
    one.write_bytes(b'same');two.write_bytes(b'same')
    audit = FrozenPayloadAudit(max_entries=1)
    for path in [one,two,one]:assert audit.sha256(path) == digest(b'same')
    stats = audit.statistics()
    assert stats['full_hashes'] == 3 and stats['actual_bytes_read'] == 12
    assert stats['memo_hits'] == 0 and stats['evictions'] == 2
    assert stats['retained_entries'] == 1


def test_same_size_rewrite_with_restored_mtime_invalidates_via_ctime(tmp_path):
    path = tmp_path / 'payload';path.write_bytes(b'original')
    audit = FrozenPayloadAudit();assert audit.sha256(path) == digest(b'original')
    before = path.stat()
    path.write_bytes(b'changed!')
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    assert audit.sha256(path) == digest(b'changed!')
    assert audit.statistics()['memo_hits'] == 0
    assert audit.statistics()['actual_bytes_read'] == 16


def test_symlink_retarget_cannot_return_prior_target_digest(tmp_path):
    one = tmp_path / 'one';two = tmp_path / 'two';link = tmp_path / 'link'
    one.write_bytes(b'one');two.write_bytes(b'two');link.symlink_to(one)
    audit = FrozenPayloadAudit();assert audit.sha256(link) == digest(b'one')
    link.unlink();link.symlink_to(two)
    assert audit.sha256(link) == digest(b'two')
    assert audit.statistics()['full_hashes'] == 2


@pytest.mark.parametrize('replace_path',[False,True])
def test_changes_during_full_hash_are_rejected_and_not_memoized(tmp_path,monkeypatch,replace_path):
    source = tmp_path / 'source';other = tmp_path / 'other';link = tmp_path / 'link'
    source.write_bytes(b'old');other.write_bytes(b'new');link.symlink_to(source)
    audit = FrozenPayloadAudit();read = audit._read_digest
    def changed(stream):
        value = read(stream)
        if replace_path:
            link.unlink();link.symlink_to(other)
        else:source.write_bytes(b'new')
        return value
    monkeypatch.setattr(audit,'_read_digest',changed)
    with pytest.raises(ValueError,match='changed or was replaced'):audit.sha256(link)
    stats = audit.statistics()
    assert stats['actual_bytes_read'] == 3 and stats['full_hash_attempts'] == 1
    assert stats['full_hashes'] == 0 and stats['retained_entries'] == 0
    assert stats['failures'] == 1
    monkeypatch.setattr(audit,'_read_digest',read)
    assert audit.sha256(link) == digest(b'new')


def test_memo_hit_still_checks_descriptor_after_lookup(tmp_path,monkeypatch):
    from visuotactile import frozen_payload_audit as module
    path = tmp_path / 'payload';path.write_bytes(b'old')
    audit = FrozenPayloadAudit();audit.sha256(path)
    fstat = os.fstat;calls = 0
    def mutate_on_second_stat(fd):
        nonlocal calls
        calls += 1
        if calls == 2:path.write_bytes(b'new')
        return fstat(fd)
    monkeypatch.setattr(module.os,'fstat',mutate_on_second_stat)
    with pytest.raises(ValueError,match='changed or was replaced'):audit.sha256(path)
    assert audit.statistics()['memo_hits'] == 0
    assert audit.statistics()['retained_entries'] == 0


@pytest.mark.parametrize('kind',['directory','fifo'])
def test_nonregular_file_rejected_without_read_or_block(tmp_path,kind):
    path = tmp_path / kind
    if kind == 'directory':path.mkdir()
    else:os.mkfifo(path)
    audit = FrozenPayloadAudit()
    with pytest.raises((ValueError,IsADirectoryError)):audit.sha256(path)
    assert audit.statistics()['actual_bytes_read'] == 0
    assert audit.statistics()['retained_entries'] == 0


@pytest.mark.parametrize('bound',[0,-1,True,1.5])
def test_invalid_bound_rejected(bound):
    with pytest.raises(ValueError):FrozenPayloadAudit(max_entries=bound)


def test_inherited_memo_rejected_after_pid_change(tmp_path,monkeypatch):
    from visuotactile import frozen_payload_audit as module
    path = tmp_path / 'payload';path.write_bytes(b'one')
    audit = FrozenPayloadAudit();audit.sha256(path)
    monkeypatch.setattr(module.os,'getpid',lambda:audit.pid+1)
    with pytest.raises(RuntimeError,match='process boundary'):audit.sha256(path)
    assert audit.statistics()['memo_hits'] == 0


def prepare_pair(tmp_path,monkeypatch,name):
    from visuotactile import native_data
    from visuotactile.compact import sha256
    from visuotactile.training_resources import atomic_json
    root = tmp_path / name;root.mkdir()
    checkpoint = tmp_path / 'dino';checkpoint.write_bytes(b'frozen backbone')
    payloads = [tmp_path / 'feature1',tmp_path / 'feature2']
    for index,path in enumerate(payloads):path.write_bytes(bytes([index+1]) * (1024+index))
    entries = {f'episode{i}':dict(path=str(path),sha256=sha256(path),source_sha256=str(i)*64,frames=2)
               for i,path in enumerate(payloads)}
    index = tmp_path / 'reuse.json'
    atomic_json(index,dict(entries=entries,dino_sha256=sha256(checkpoint),dino_revision=native_data.DINO_REVISION,
        stride=12,cameras=native_data.CAMERAS,resolution=[448,252],patches=[384,18,32],depth=[2,18,32],
        trainable_projection_cached=False,random_augmentation=False,complete=True))
    for role in ['A','B']:
        state = root / role / 'state_data';state.mkdir(parents=True)
        atomic_json(state/'data.json',dict(splits=dict(train=list(entries),validation=[]),
            entries={i:dict(path='/NEVER_READ_RGBD',length=20,source_sha256=e['source_sha256']) for i,e in entries.items()}))
        atomic_json(root/role/'session.json',dict(dino_checkpoint=str(checkpoint),dino_sha256=sha256(checkpoint),cache_deadline=1))
        atomic_json(root/role/'continuation.json',dict(reuse_index=str(index)))
    def no_gpu(*args,**kwargs):raise AssertionError('Reused frozen features must not construct DINO')
    monkeypatch.setattr('visuotactile.models.native_vision.NativeDinoSpatial',no_gpu)
    monkeypatch.setattr('visuotactile.state_resources.lease_for',lambda *args,**kwargs:None)
    monkeypatch.setattr('visuotactile.precision.state_fp32',lambda:None)
    return root,payloads,index


def test_pair_integration_reduces_real_reads_without_changing_accepted_indices(tmp_path,monkeypatch):
    from tools.tight_fresh_dagger import cache
    from visuotactile import native_data
    root,payloads,index = prepare_pair(tmp_path,monkeypatch,'ordinary')
    ordinary_sha = native_data.sha256;reads = []
    def counted(path):
        if Path(path).resolve() in payloads:reads.append(Path(path))
        return ordinary_sha(path)
    monkeypatch.setattr(native_data,'sha256',counted)
    cache(root)
    assert len(reads) == 4
    assert not list(root.glob('payload_audit_memo_*.json'))
    candidate,_,_ = prepare_pair(tmp_path,monkeypatch,'memo')
    cache(candidate,payload_audit_memo=True)
    report = json.loads(next(candidate.glob('payload_audit_memo_*.json')).read_text())
    assert report['status'] == 'complete'
    assert report['statistics']['full_hashes'] == 2
    assert report['statistics']['memo_hits'] == 2
    assert report['statistics']['actual_bytes_read'] == sum(p.stat().st_size for p in payloads)
    assert report['statistics']['avoided_full_hash_bytes'] == report['statistics']['actual_bytes_read']
    assert [p['role'] for p in report['phases']] == ['A','B']
    for role in ['A','B']:
        assert json.loads((root/role/'cache_index.json').read_text()) == json.loads((candidate/role/'cache_index.json').read_text())


@pytest.mark.parametrize('fault',['expected_sha','source_sha','frames'])
def test_memo_does_not_bypass_registered_expected_digest_or_contract(tmp_path,monkeypatch,fault):
    from tools.tight_fresh_dagger import cache
    from visuotactile import native_data
    root,_,_ = prepare_pair(tmp_path,monkeypatch,'candidate')
    original = native_data.cache_visual_pair
    def changed(root,**kwargs):
        if root.name == 'B':
            path = kwargs['reuse_index'];index = json.loads(path.read_text())
            key = {'expected_sha':'sha256','source_sha':'source_sha256','frames':'frames'}[fault]
            index['entries']['episode0'][key] = 3 if fault == 'frames' else 'f'*64
            path.write_text(json.dumps(index))
        return original(root,**kwargs)
    monkeypatch.setattr(native_data,'cache_visual_pair',changed)
    with pytest.raises(ValueError):cache(root,payload_audit_memo=True)
    report = json.loads(next(root.glob('payload_audit_memo_*.json')).read_text())
    assert report['status'] == 'failed'
    assert not (root/'B/cache_index.json').exists()
    assert report['statistics']['memo_hits'] == (0 if fault == 'source_sha' else 1)
    assert [p['role'] for p in report['phases']] == ['A']


def test_cli_opt_in_only_for_cache(monkeypatch,tmp_path):
    import sys
    from tools import tight_fresh_dagger as tool
    calls = []
    monkeypatch.setattr(tool,'cache',lambda *args,**kwargs:calls.append(kwargs))
    monkeypatch.setattr(sys,'argv',['dagger','cache','--root',str(tmp_path),'--output',str(tmp_path),'--payload-audit-memo'])
    tool.main();assert calls == [dict(reuse_index=None,payload_audit_memo=True)]
    monkeypatch.setattr(sys,'argv',['dagger','aggregate','--root',str(tmp_path),'--output',str(tmp_path),'--payload-audit-memo'])
    with pytest.raises(SystemExit):tool.main()
