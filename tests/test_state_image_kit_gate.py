import json







def test_restricted_helper_fds_remain_visible_without_fabricated_zero_counts(monkeypatch):
    from pathlib import Path
    import os
    from tools.runtime import proc
    original=Path.iterdir
    def denied(path):
        if path==Path('/proc',str(os.getpid()),'fd'):
            raise PermissionError(13,'Permission denied')
        return original(path)
    monkeypatch.setattr(Path,'iterdir',denied)
    row=proc(os.getpid())
    assert row['pid']==os.getpid() and row['start_ticks']>0
    assert row['fds'] is None and row['watches'] is None
    assert row['telemetry_missing']==['fd_directory_permission_denied']
