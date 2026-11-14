import hashlib
import json
import mmap
import os
from pathlib import Path
import pickle
import zipfile

import pytest

from tools import tight_checkpoint_cleanup as cleanup


def record(root, relative, content=b'x'*65536):
    path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return dict(path=relative, **cleanup.identity(path))


def fixture_manifest(tmp_path):
    root = tmp_path/'outputs'; root.mkdir()
    candidate = record(root, 'run/retired/training/last.pt')
    candidate.update(aliases=[candidate['path']], non_receipt_references=[],
        finished_receipt='run/retired/training/finished.json', completion_evidence={'complete':True})
    (root/candidate['finished_receipt']).write_text('{"complete":true}')
    protected = record(root, 'run/retired/training/best.pt', b'protected')
    manifest = dict(root=str(root), candidate_files=[candidate], protected_models=[protected])
    return root, manifest


def save_manifest(tmp_path, manifest):
    path = tmp_path/'manifest.json'; raw = json.dumps(manifest).encode(); path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def load_fixture(tmp_path, manifest):
    path, digest = save_manifest(tmp_path, manifest)
    return cleanup.load_manifest(path, approved_sha256=digest,
        approved_root=Path(manifest['root']), expected_counts=(1, 1))[0]


@pytest.mark.parametrize('mutation', ['path_escape', 'protected_name', 'new_reference', 'hardlink', 'duplicate'])
def test_whitelist_refuses_scope_changes(tmp_path, mutation):
    root, manifest = fixture_manifest(tmp_path); row = manifest['candidate_files'][0]
    if mutation == 'path_escape': row['path'] = '../escape/training/last.pt'
    elif mutation == 'protected_name': row['path'] = 'run/retired/training/best.pt'
    elif mutation == 'new_reference': row['non_receipt_references'] = [{'source':'new.json'}]
    elif mutation == 'hardlink': row['nlink'] = 2
    elif mutation == 'duplicate': manifest['candidate_files'].append(dict(row))
    with pytest.raises(cleanup.CleanupRefused): load_fixture(tmp_path, manifest)
    assert (root/'run/retired/training/last.pt').exists()


def test_reviewed_manifest_hash_cannot_be_replaced(tmp_path):
    _, manifest = fixture_manifest(tmp_path); path, digest = save_manifest(tmp_path, manifest)
    path.write_text(path.read_text()+' ')
    with pytest.raises(cleanup.CleanupRefused, match='immutable whitelist'):
        cleanup.load_manifest(path, approved_sha256=digest,
            approved_root=Path(manifest['root']), expected_counts=(1, 1))


@pytest.mark.parametrize('change', ['content', 'hardlink', 'symlink'])
def test_file_identity_drift_refuses_before_deletion(tmp_path, change):
    root, manifest = fixture_manifest(tmp_path); load_fixture(tmp_path, manifest)
    row = manifest['candidate_files'][0]; target = root/row['path']
    if change == 'content': target.write_bytes(b'new')
    elif change == 'hardlink': os.link(target, tmp_path/'alias')
    else:
        target.unlink(); target.symlink_to(root/manifest['protected_models'][0]['path'])
    with pytest.raises(cleanup.CleanupRefused):
        cleanup.operate(manifest, execute=True, ledger_root=tmp_path/'ledger',
            process_check=lambda:{}, source_check=lambda:{})
    failed = list((tmp_path/'ledger').glob('*/failed.json'))
    assert len(failed) == 1 and json.loads(failed[0].read_text())['deleted'] == []
    assert target.exists()


def fake_proc(root, pid, *, command='python harmless.py', fd=None, maps='', parent=1):
    directory = root/str(pid); directory.mkdir(parents=True)
    (directory/'fd').mkdir(); (directory/'cmdline').write_bytes(command.replace(' ', '\0').encode())
    (directory/'comm').write_text('python')
    fields = ['S', str(parent)]+['0']*17+['100']
    (directory/'stat').write_text(f'{pid} (python) '+' '.join(fields))
    (directory/'maps').write_text(maps)
    if fd: (directory/'fd/3').symlink_to(fd)


@pytest.mark.parametrize('kind', ['fd', 'mmap'])
def test_open_and_closed_fd_mmap_consumers_block(tmp_path, kind):
    root, manifest = fixture_manifest(tmp_path); row = manifest['candidate_files'][0]
    proc = tmp_path/'proc'; proc.mkdir()
    maps = (f'1000-2000 r--p 00000000 {os.major(row["dev"]):x}:{os.minor(row["dev"]):x} '
            f'{row["ino"]} {root/row["path"]}\n') if kind == 'mmap' else ''
    fake_proc(proc, 91234, fd=root/row['path'] if kind == 'fd' else None, maps=maps)
    with pytest.raises(cleanup.CleanupRefused, match=f'"kind": "{kind}"'):
        cleanup.inspect_processes({(row['dev'], row['ino'])}, uid=os.getuid(),
            allowed_session=root/'run/current', proc=proc, gpu_pids=[])


def test_real_mmap_remains_visible_after_file_descriptor_closes(tmp_path):
    target = tmp_path/'mapped'; target.write_bytes(b'x'*8192)
    with target.open('rb') as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        row = cleanup.identity(target)
        line = next(line for line in Path('/proc/self/maps').read_text().splitlines()
                    if str(target) in line)
        assert int(line.split()[4]) == row['ino']
        # Limit the real /proc fixture to this process, while preserving actual maps.
        proc = tmp_path/'proc'; proc.mkdir()
        fake_proc(proc, 91234, maps=line+'\n')
        with pytest.raises(cleanup.CleanupRefused, match='mmap'):
            cleanup.inspect_processes({(row['dev'], row['ino'])}, uid=os.getuid(),
                allowed_session=tmp_path/'current', proc=proc, gpu_pids=[])
    finally: mapping.close()


def test_live_unreadable_maps_fails_closed(tmp_path):
    proc = tmp_path/'proc'; proc.mkdir(); fake_proc(proc, 91234)
    (proc/'91234/maps').unlink()  # Live proc exists but the required map census is incomplete.
    with pytest.raises(cleanup.CleanupRefused, match='unreadable'):
        cleanup.inspect_processes(set(), uid=os.getuid(), allowed_session=tmp_path,
            proc=proc, gpu_pids=[])


@pytest.mark.parametrize('mutation', [None, 'reused_pid', 'different_boot'])
def test_reviewed_daemon_exception_binds_identity_and_reports_visibility(tmp_path, monkeypatch, mutation):
    proc=tmp_path/'proc'; proc.mkdir(); fake_proc(proc, 91234, command='known daemon')
    boot=proc/'sys/kernel/random/boot_id'; boot.parent.mkdir(parents=True); boot.write_text('bootA')
    binding=dict(pid=91234, uid=os.getuid(), command='known daemon', comm='python',
        start_ticks=100, boot_id='bootA')
    if mutation=='reused_pid': binding['start_ticks']=99
    if mutation=='different_boot': binding['boot_id']='bootB'
    original=Path.read_text
    def protected_read(path, *args, **kwargs):
        if path == proc/'91234/maps': raise PermissionError(13, 'maps restricted')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', protected_read)
    def check():
        return cleanup.inspect_processes(set(), uid=os.getuid(), allowed_session=tmp_path/'current',
            proc=proc, gpu_pids=[], visibility_exceptions=[binding])
    if mutation:
        with pytest.raises(cleanup.CleanupRefused): check()
    else:
        report=check()
        assert report['visibility_exceptions'][0]['binding']==binding
        assert not report['all_same_uid_processes_fully_visible']


def test_new_gpu_or_research_process_requires_exact_allowed_session(tmp_path):
    proc = tmp_path/'proc'; proc.mkdir(); session=tmp_path/'session05'
    fake_proc(proc, 91234, command=f'python -m tools.tight_fresh_train --root {session}evil')
    with pytest.raises(cleanup.CleanupRefused, match='disallowed_processes'):
        cleanup.inspect_processes(set(), uid=os.getuid(), allowed_session=session,
            proc=proc, gpu_pids=[91234])
    (proc/'91234/cmdline').write_text(f'python -m tools.tight_fresh_train --root {session}/pool')
    assert cleanup.inspect_processes(set(), uid=os.getuid(), allowed_session=session,
        proc=proc, gpu_pids=[91234])['disallowed_processes'] == []


def test_source_checkpoint_reference_inside_pt_metadata_is_retained(tmp_path):
    root = tmp_path/'outputs'; root.mkdir(); pt = root/'source.pt'
    candidate = dict(path='run/retired/training/last.pt', finished_receipt='run/retired/training/finished.json')
    with zipfile.ZipFile(pt, 'w') as archive:
        archive.writestr('archive/data.pkl', pickle.dumps({'source':str(root/candidate['path'])}))
        archive.writestr('archive/data/0', b'tensor bytes must not be read')
    with pytest.raises(cleanup.CleanupRefused, match='retained source now references'):
        cleanup.reference_scan(root, [candidate], worktrees=[], ledger_root=tmp_path/'ledger')


def test_verify_is_read_only_and_execute_releases_real_small_files(tmp_path):
    root, manifest = fixture_manifest(tmp_path); load_fixture(tmp_path, manifest)
    target = root/manifest['candidate_files'][0]['path']; ledger_root=tmp_path/'ledger'
    kwargs = dict(ledger_root=ledger_root, process_check=lambda:{'checked':True}, source_check=lambda:{})
    report = cleanup.operate(manifest, **kwargs)
    assert report['mode']=='verified_read_only' and target.exists() and not ledger_root.exists()
    report = cleanup.operate(manifest, execute=True, **kwargs)
    assert not target.exists() and report['protected_models_unchanged']
    assert report['unlinked_allocated_bytes'] == manifest['candidate_files'][0]['blocks_bytes']
    events = [json.loads(line) for line in (Path(report['ledger'])/'events.jsonl').read_text().splitlines()]
    assert [e['kind'] for e in events] == ['started','delete_intent','deleted','complete']
    assert 'filesystem_before' in events[2] and 'filesystem_after' in events[2]
    # A second attempt cannot silently resume or overwrite the first evidence.
    with pytest.raises(cleanup.CleanupRefused): cleanup.operate(manifest, execute=True, **kwargs)
    assert len(list(ledger_root.glob('*/complete.json'))) == 1
    assert len(list(ledger_root.glob('*/failed.json'))) == 1


def test_changed_completion_receipt_preserves_checkpoint(tmp_path):
    root, manifest=fixture_manifest(tmp_path)
    (root/manifest['candidate_files'][0]['finished_receipt']).write_text('{"complete":false}')
    with pytest.raises(cleanup.CleanupRefused, match='Completed-training evidence changed'):
        cleanup.operate(manifest, process_check=lambda:{}, source_check=lambda:{})
    assert (root/manifest['candidate_files'][0]['path']).exists()


def test_midway_failure_preserves_prior_deletion_ledger_and_remaining_file(tmp_path):
    root, manifest=fixture_manifest(tmp_path); row=record(root, 'run/retired/training/comparison_1.pt')
    row.update(aliases=[row['path']], finished_receipt='run/retired/training/finished.json',
        completion_evidence={'complete':True})
    manifest['candidate_files'].append(row)
    calls=[]
    def process_check():
        calls.append(1)
        if len(calls)==4: raise cleanup.CleanupRefused('new consumer')
        return {}
    with pytest.raises(cleanup.CleanupRefused, match='new consumer'):
        cleanup.operate(manifest, execute=True, ledger_root=tmp_path/'ledger',
            process_check=process_check, source_check=lambda:{})
    report=json.loads(next((tmp_path/'ledger').glob('*/failed.json')).read_text())
    assert report['deleted']==[manifest['candidate_files'][0]['path']]
    assert report['remaining']==[row['path']] and (root/row['path']).exists()
    assert report['protected_check_error'] is None
