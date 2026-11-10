"""Guarded unlink tests use a temporary fake repository, never real artifacts."""
import os
from pathlib import Path
from types import SimpleNamespace as NS

from tools import tight_dagger as owner


def setup(tmp_path,monkeypatch):
    code=tmp_path/'repo';run=code/'outputs/visuotactile/run';run.mkdir(parents=True)
    root=run/'new';root.mkdir();parent=run/'predecessor';parent.mkdir()
    monkeypatch.setattr(owner,'CODE',code)
    monkeypatch.setattr(owner,'predecessor_finished',lambda r:True)
    owner.write(root/'configuration.json',dict(groups={'G11':{}}))
    owner.write(code/'configs/tight_state.json',{})
    owner.write(root/'registration.json',dict(predecessor=str(parent)))
    owner.write(root/'status.json',dict(state='COMPLETE',owned_workers_exited=True))
    owner.write(root/'owned_cleanup.json',dict(verified=True,workers=[]))
    frontend=root/'front.pt';backend=root/'back.pt'
    frontend.write_bytes(b'current frontend');backend.write_bytes(b'current backend')
    owner.write(root/'round_01/G11/initial_pair.json',dict(frontend=str(frontend),frontend_sha256=owner.sha(frontend),
        backend=str(backend),backend_sha256=owner.sha(backend)))
    real_scandir=os.scandir
    fake_proc=[]
    monkeypatch.setattr(os,'scandir',lambda path:iter(fake_proc) if str(path)=='/proc' else real_scandir(path))
    def checkpoint(relative,contents=b'obsolete checkpoint'):
        p=run/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(contents);return p
    return code,root,run,checkpoint,fake_proc


def test_cleanup_exact_allowlist_hardlink_accounting_and_existing_audit(tmp_path,monkeypatch):
    code,root,run,checkpoint,_=setup(tmp_path,monkeypatch)
    best=checkpoint('narrow_state_20261105_02/training/best.pt')
    last=best.with_name('last.pt');os.link(best,last);blocks=best.stat().st_blocks*512
    other=checkpoint('tight_vision_20261109_01/G10/training/last.pt');blocks+=other.stat().st_blocks*512
    protected=checkpoint('tight_vision_20261109_01/G10/training/best.pt',b'selected original frontend')
    raw=checkpoint('tight_palm_20261108_01/episodes/source/data.h5',b'original source')
    audit=code/'research_docs/CLEANUP_AUDIT.md';audit.parent.mkdir();audit.write_text('Existing audit.\n')
    report=owner.cleanup_obsolete(root)
    assert report['executed'] and report['deleted_files']==3
    assert report['released_allocated_bytes']==blocks
    assert all(row.get('sha256') for row in report['rows'] if row['deleted'])
    assert not best.exists() and not last.exists() and not other.exists()
    assert protected.read_bytes()==b'selected original frontend' and raw.read_bytes()==b'original source'
    assert all(row['unchanged'] for row in report['retained_initialization_checks'])
    assert '删除 3 个目录项' in audit.read_text()
    assert owner.cleanup_obsolete(root)==report


def test_cleanup_blocks_live_predecessor_and_missing_fixed_pair(tmp_path,monkeypatch):
    code,root,run,checkpoint,_=setup(tmp_path,monkeypatch)
    path=checkpoint('tight_palm_20261108_01/training/last.pt')
    monkeypatch.setattr(owner,'predecessor_finished',lambda r:False)
    report=owner.cleanup_obsolete(root)
    assert report['blocked'] and report['deleted_files']==0 and path.exists()


def test_cleanup_retains_config_references_open_inode_and_symlinks(tmp_path,monkeypatch):
    code,root,run,checkpoint,processes=setup(tmp_path,monkeypatch)
    referenced=checkpoint('tight_palm_20261108_01/training/best.pt')
    owner.write(root/'configuration.json',dict(groups={'G11':{}},checkpoint=str(referenced)))
    opened=checkpoint('tight_residual_20261110_01/G01/training/best.pt')
    alias=opened.with_name('last.pt');os.link(opened,alias)
    fake=tmp_path/'proc/345/fd';fake.mkdir(parents=True);(fake/'3').symlink_to(alias)
    processes.append(NS(name='345',path=str(fake.parent)))
    linked=run/'dp_command_20261029/b0/best.pt';linked.parent.mkdir(parents=True);linked.symlink_to(referenced)
    report=owner.cleanup_obsolete(root);rows={Path(r['path']):r for r in report['rows']}
    assert report['deleted_files']==0
    assert rows[referenced]['skip_reason']=='referenced by current or queued dependency'
    assert rows[opened]['skip_reason']==rows[alias]['skip_reason']=='open file descriptor'
    assert rows[linked]['skip_reason']=='symlink or redirected parent'
    assert report['audit_notice'].startswith('CLEANUP_AUDIT.md absent')
