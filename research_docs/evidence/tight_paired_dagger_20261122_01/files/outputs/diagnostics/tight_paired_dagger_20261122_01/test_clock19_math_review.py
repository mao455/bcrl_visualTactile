"""Independent CPU-only interval arithmetic review for inactive helper19.

No process, lifecycle receipt, resource or affinity calls are made. Only the
named source is imported, and every clock used by a fixture is synthetic.
These checks establish the stated interval assumptions, not the historical
absence of realtime clock steps before a sample was taken.
"""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


_spec = importlib.util.spec_from_file_location(
    'clock19_independent_math', Path(__file__).with_name('host_init_capacity19.py'))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


@pytest.mark.parametrize('lower,upper,utc,expected', [
    (100., 100.01, 100., True),
    (100., 100.01, 99.97, True),   # Entire interval fits the original +.05s.
    (100., 100.01, 220., True),   # Oldest feasible birth is exactly 120s old.
    (100., 100.01, 220.0001, False),
    (100., 100.06, 220.05, False), # Intersection alone would accept this.
    (100., 100.06, 100.005, False),
    (100., 100.01, 99.9599, False),
    (100., 100.05, 100., True),
    (100., 100.0501, 100., False),
    (100., 100., 100., True),
])
def test_receipt_requires_entire_interval_in_original_window(lower, upper, utc, expected):
    assert m.receipt_birth(dict(lower=lower, upper=upper), utc) is expected


@pytest.mark.parametrize('lower,upper,utc,expected', [
    (100., 100.01, 100., True),
    (100., 100.01, 100.04, True),  # Original -.05s launch tolerance remains.
    (100., 100.06, 100.06, False), # Lower endpoint violates that tolerance.
    (100., 100.01, 100.0501, False),
    (100., 100.01, 1., True),
    (100., 121., 1., True),       # Latest feasible birth is exactly 120s later.
    (100., 121.0001, 1., False),
    (100., 121.01, 1.005, False), # Intersection alone would accept this.
    (100., 100., 100., True),
])
def test_launch_requires_entire_interval_in_original_window(lower, upper, utc, expected):
    assert m.launch_birth(dict(lower=lower, upper=upper), utc) is expected


@pytest.mark.parametrize('value', [True, False, None, '100', float('nan'), float('inf'), -1., 0.])
def test_invalid_endpoint_or_timestamp_is_rejected(value):
    for predicate in (m.receipt_birth, m.launch_birth):
        assert predicate(dict(lower=value, upper=100.01), 100.) is False
        assert predicate(dict(lower=100., upper=value), 100.) is False
        assert predicate(dict(lower=100., upper=100.01), value) is False


@pytest.mark.parametrize('mapped', [None, True, [], {}, {'lower': 100.}, {'upper': 100.},
                                    {'lower': 100.01, 'upper': 100.}])
def test_missing_or_reversed_interval_is_rejected(mapped):
    assert m.receipt_birth(mapped, 100.) is False
    assert m.launch_birth(mapped, 100.) is False


def _clock(monkeypatch, before, boot, after):
    walls = iter((before, after))
    requested = []
    def gettime(clock_id):
        requested.append(clock_id)
        return boot
    # Replace this module's time reference, never the shared Python time module.
    monkeypatch.setattr(m, 'time', SimpleNamespace(time=lambda: next(walls),
                        clock_gettime=gettime, CLOCK_BOOTTIME=917))
    return requested


@pytest.mark.parametrize('before,boot,after', [
    (1024., 512., 1024.),
    (1024., 512., 1024.03125),
    (2., 1., 2.001),
])
def test_clock_sample_returns_measured_offset_bounds(monkeypatch, before, boot, after):
    requested = _clock(monkeypatch, before, boot, after)
    sample = m.clock_sample()
    assert requested == [917]
    assert sample['offset_lower'] == before-boot
    assert sample['offset_upper'] == after-boot
    assert sample['wall_before'] == before and sample['wall_after'] == after
    assert sample['boottime'] == boot
    assert 0 <= sample['offset_upper']-sample['offset_lower'] <= m.MAX_CLOCK_SAMPLE_SECONDS


@pytest.mark.parametrize('before,boot,after', [
    (1024., 512., 1023.999),
    (1024., 512., 1024.0625),
    (float('nan'), 512., 1024.),
    (1024., float('inf'), 1024.),
    (1024., 512., float('nan')),
    (True, 512., 1024.),
    (1024., True, 1024.),
    (1024., 512., True),
    (0., 512., .001),
    (1024., 0., 1024.),
])
def test_clock_sample_rejects_nonfinite_reversed_or_wide_bracket(monkeypatch, before, boot, after):
    _clock(monkeypatch, before, boot, after)
    with pytest.raises(ValueError):
        m.clock_sample()


def test_both_predicates_reject_any_feasible_endpoint_outside_original_domain():
    # Deterministic arithmetic grid, without randomness, system clocks or PIDs.
    for utc in (1., 100., 1000.):
        for relative_lower in (-120.01, -120., -.051, -.05, 0., .049, .05, 119.99, 120.):
            for width in (0., .001, .01, .06):
                lower, upper = utc+relative_lower, utc+relative_lower+width
                mapped = dict(lower=lower, upper=upper)
                if lower <= 0:
                    assert not m.receipt_birth(mapped, utc)
                    assert not m.launch_birth(mapped, utc)
                    continue
                receipt_expected = lower >= utc-120. and upper <= utc+.05
                launch_expected = lower >= utc-.05 and upper <= utc+120.
                assert m.receipt_birth(mapped, utc) is receipt_expected
                assert m.launch_birth(mapped, utc) is launch_expected
                if receipt_expected:
                    assert all(utc-120. <= birth <= utc+.05 for birth in (lower, (lower+upper)/2., upper))
                if launch_expected:
                    assert all(utc-.05 <= birth <= utc+120. for birth in (lower, (lower+upper)/2., upper))


def test_quantization_union_is_bracket_width_plus_one_tick():
    # For a constant realtime-BOOTTIME offset within the measured bracket,
    # tick truncation gives a conservative interval including every birth.
    for hz in (100, 250, 1000):
        for bracket_width in (0., .001, .03125):
            offset_lower = 1700000000.
            offset_upper = offset_lower+bracket_width
            tick_seconds = 12345/hz
            lower = offset_lower+tick_seconds
            upper = offset_upper+tick_seconds+1/hz
            assert math.isclose(upper-lower, bracket_width+1/hz, abs_tol=5e-7)
            for offset_fraction in (0., .5, 1.):
                for tick_fraction in (0., .5, .999):
                    feasible = offset_lower+bracket_width*offset_fraction+tick_seconds+tick_fraction/hz
                    assert lower <= feasible <= upper
