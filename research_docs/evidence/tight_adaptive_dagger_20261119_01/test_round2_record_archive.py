import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec=importlib.util.spec_from_file_location('r2_record_archive_test',Path(__file__).with_name('archive_round2_closed_records.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_only_closed_exact_receipts_are_eligible(tmp_path):
    p=tmp_path/'round02/learning/A/training/learning.jsonl';p.parent.mkdir(parents=True)
    p.write_text('{}\n');(p.parent/'finished.json').write_text('{"complete":true}')
    with patch.object(m.a,'ROOT',tmp_path):
        assert m.safe(p)==p
        (p.parent/'finished.json').write_text('{"complete":false}')
        with pytest.raises(AssertionError):m.safe(p)
        raw=p.with_name('best.pt');raw.write_bytes(b'model')
        with pytest.raises(AssertionError):m.safe(raw)


def test_four_audits_and_freeze_required_before_archiving(tmp_path):
    learning=tmp_path/'round02/learning';learning.mkdir(parents=True)
    accepted=dict(passed=True,components=[dict(eligible=True,receipt_replay_exact=True,minimum_tick_steps=50,minimum_visits=50) for _ in range(4)])
    (learning/'exposure_acceptance.json').write_text(json.dumps(accepted))
    (tmp_path/'round02/complete.json').write_text('{}')
    with patch.object(m.a,'ROOT',tmp_path):
        with pytest.raises(AssertionError):m.gate()
        (tmp_path/'final_freeze.json').write_text('{}');m.gate()
        accepted['components'][0]['minimum_tick_steps']=49
        (learning/'exposure_acceptance.json').write_text(json.dumps(accepted))
        with pytest.raises(AssertionError):m.gate()
