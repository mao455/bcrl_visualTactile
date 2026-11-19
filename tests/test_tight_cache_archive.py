import json
from pathlib import Path
import pytest
from tools.tight_cache_archive import archive_file,restore_file,sha


def test_lossless_archive_restore_and_corruption_guard(tmp_path):
    p=tmp_path/'perception'/'cache.pt';p.parent.mkdir();p.write_bytes(bytes(range(256))*10000)
    expected=sha(p);row=archive_file(p,expected);assert not p.exists()
    receipt=p.with_suffix('.pt.archive.json')
    restore_file(receipt);assert sha(p)==expected
    with pytest.raises(AssertionError):restore_file(receipt)
    p.unlink();target=Path(row['archive']);target.write_bytes(target.read_bytes()+b'bad')
    with pytest.raises(AssertionError):restore_file(receipt)
    assert not p.exists()


def test_mismatch_preserves_original(tmp_path):
    p=tmp_path/'perception'/'cache.pt';p.parent.mkdir();p.write_bytes(b'original')
    with pytest.raises(AssertionError):archive_file(p,'incorrect')
    assert p.read_bytes()==b'original'


def test_completed_training_log_archive_is_exact_and_opt_in(tmp_path):
    p=tmp_path/'training'/'learning.jsonl';p.parent.mkdir();p.write_bytes(b'{"update":1}\n'*100)
    before=p.read_bytes()
    with pytest.raises(AssertionError):archive_file(p,sha(p),kind='completed_training_log')
    (p.parent/'finished.json').write_text('{"complete":true}')
    row=archive_file(p,sha(p),kind='completed_training_log')
    restore_file(p.with_suffix('.jsonl.archive.json'))
    assert p.read_bytes()==before
