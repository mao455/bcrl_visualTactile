"""Inactive exact-metadata adapter over immutable final15 sources.

Only the already-consumed development failure's exact observed rc -2 replaces
an incorrect rc 1 expectation. No receipt/binding/model/scientific rule changes.
Explicit root amendment and review are required before any actual entry gate.
"""
from pathlib import Path
import hashlib
import json
import types

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE_NAME = 'final_recovery15_common.py'
BASE_SHA256 = '2d706ea709fde903f58efdf9c801ad83bd9d9553f7b28d096dd2f8073933ace3'
AMENDMENT_NAME21 = 'FINAL_RECOVERY21_AMENDMENT.json'
REVIEW_NAME21 = 'ROOT_EVIDENCE_TOOLS_REVIEW21.json'
TOOLS21 = ('final_recovery21_common.py', 'extract_selected_ledgers_recovery21.py',
           'final_compute_recovery21.py', 'final_report_assembler_recovery21.py',
           'export_final_evidence_recovery21.py', 'test_final_recovery21.py')
OLD_RC = "and _pinned(root/'cycle_logs_recovery10/evaluate_development_exit.json',bindings).get('returncode')==1,"
NEW_RC = "and _pinned(root/'cycle_logs_recovery10/evaluate_development_exit.json',bindings).get('returncode')==-2,"
OLD_EXTRACTOR = "sha(audit/'extract_selected_ledgers_recovery15.py')"
NEW_EXTRACTOR = "sha(audit/'extract_selected_ledgers_recovery21.py')"


def exact_replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Exact private source substitution must occur once: '+old)
    return source.replace(old, new)


def source_module(name, expected, replacements=(), *, actual_file=None):
    path = HERE/name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Immutable inherited source changed: '+name)
    source = raw.decode()
    for old, new in replacements:
        source = exact_replace(source, old, new)
    value = types.ModuleType('_final21_private_'+Path(name).stem)
    value.__file__ = str(actual_file or path)
    exec(compile(source, str(path), 'exec'), value.__dict__)
    return value


_BASE21 = source_module(BASE_NAME, BASE_SHA256,
                       ((OLD_RC, NEW_RC), (OLD_EXTRACTOR, NEW_EXTRACTOR)))
# Preserve original format/constants and scientific validators. Functions retain
# their original private module globals; only entry closure gets the new guard.
for _key21, _value21 in vars(_BASE21).items():
    if not _key21.startswith('__'):
        globals().setdefault(_key21, _value21)


def _file21(path):
    path = Path(path)
    require(path.resolve().is_relative_to(PROJECT) and not path.is_symlink()
            and path.is_file() and path.suffix in ('.json', '.py', '.md'),
            'Only concrete regular project small sources/receipts allowed')
    return sha(path)


def validate_actual_failure21(root, audit):
    """Additional exact checks; original failure/reaping gates still all execute."""
    stage = read_small(root/'cycle_logs_recovery10/evaluate_development_exit.json')
    parent = read_small(root/'evaluation/development/project_failure.json')
    exited = read_small(root/'evaluation/development/project_exit.json')
    driver = read_small(audit/'driver_backend_recovery10/failure.json')
    driver_exit = read_small(audit/'driver_backend_recovery10/driver_exit.json')
    require(type(stage.get('returncode')) is int and stage['returncode'] == -2,
            'Only exact observed original development rc -2 accepted')
    require(parent.get('error') == "KeyboardInterrupt('Owner received signal 15')"
            and exited.get('children_reaped') is True
            and driver.get('error') == "ValueError('evaluate_development failed; no retry or replacement is launched')"
            and driver_exit.get('returned_after_owned_helpers') is True
            and parent['utc'] <= exited['utc'] <= stage['utc'] <= driver['utc'] <= driver_exit['utc'],
            'Exact registered original development interruption/reaping required')
    return dict(returncode=-2, failed=True, retry=False, replacement=False,
                parent_error=parent['error'], children_reaped=True)


def amendment_gate21(root, audit):
    root, audit = Path(root).resolve(), Path(audit).resolve()
    require(audit == HERE and root == ROOT, 'Exact original final15 root/audit only')
    amendment_path, review_path = audit/AMENDMENT_NAME21, audit/REVIEW_NAME21
    amendment, review = read_small(amendment_path), read_small(review_path)
    require(amendment.get('version') == 'final15_exact_olddev_rc2_amendment21_v1'
            and amendment.get('root') == str(root)
            and amendment.get('original_deadline') == 1795684571.2261844
            and amendment.get('change') == dict(original_expected_returncode=1,
                actual_expected_returncode=-2, original_development_consumed=True,
                retry=False, replacement=False, scientific_changes=0),
            'Exact registered metadata-only21 amendment required')
    require(review.get('status') == 'cpu_review_passed'
            and review.get('amendment_sha256') == sha(amendment_path),
            'Concrete root review21 required')
    for field, name in (('binding15_sha256', BINDING_NAME), ('review15_sha256', REVIEW_NAME)):
        require(amendment.get(field) == review.get(field) == sha(audit/name),
                'Original15 binding/review must remain unchanged: '+field)
    actual_sources = {name: _file21(audit/name) for name in TOOLS21}
    require(amendment.get('source_sha256') == review.get('source_sha256') == actual_sources,
            'Exact six21 source pins required')
    # Original15 review still verifies every old bound source/receipt and source15.
    _BASE21.reviewed_binding(root, audit)
    required = [root/'cycle_logs_recovery10/evaluate_development_exit.json',
        root/'evaluation/development/project_failure.json', root/'evaluation/development/project_exit.json',
        audit/'driver_backend_recovery10/status.json', audit/'driver_backend_recovery10/failure.json',
        audit/'driver_backend_recovery10/driver_exit.json', audit/'EVAL_STARTUP12_OWNED_TERMINATION.json',
        audit/'selected_ledgers_recovery15_final01/extraction_status.json']
    receipts = amendment.get('receipt_sha256', {})
    require(set(receipts) == {str(p) for p in required}, 'Exact original failure and failed15 extraction population required')
    for filename, expected in receipts.items():
        require(_file21(filename) == expected, 'Registered failure/failed-extraction receipt changed')
    failed = read_small(required[-1])
    require(failed.get('status') == 'failed' and failed.get('checkpoint_payloads_published') == 0
            and failed.get('raw_data_opened') is False and failed.get('model_forward_calls') == 0
            and failed.get('error') == "ValueError('Original development failed owner/stage must remain truthful')",
            'Original failed15 attempt must remain preserved before extraction')
    failure = validate_actual_failure21(root, audit)
    supplements = amendment.get('supplement_sha256', {})
    require(isinstance(supplements, dict), 'Explicit bounded extra supplements required')
    for filename, expected in supplements.items():
        require(_file21(filename) == expected, 'Supplement21 changed')
    return dict(amendment_sha256=sha(amendment_path), review_sha256=sha(review_path),
                source_sha256=actual_sources, receipt_sha256=receipts,
                supplement_sha256=supplements, original_failure=failure,
                payload_reads=0, original15_unchanged=True)


def reviewed_binding(root, audit):
    amendment_gate21(root, audit)
    return _BASE21.reviewed_binding(root, audit)


def closure_gate(root, audit):
    amendment = amendment_gate21(root, audit)
    value = _BASE21.closure_gate(root, audit)
    value['exact_metadata_amendment21'] = amendment
    return value


def load_entry21(name, expected, actual_file, replacements=()):
    return source_module(name, expected,
        (("HERE/'final_recovery15_common.py'", "HERE/'final_recovery21_common.py'"),)+tuple(replacements),
        actual_file=actual_file)


def export_supplements21(root, audit):
    gate = amendment_gate21(root, audit)
    files = dict(gate['receipt_sha256'])
    files.update(gate['supplement_sha256'])
    files.update({str(HERE/name): digest for name, digest in gate['source_sha256'].items()})
    files[str(HERE/AMENDMENT_NAME21)] = gate['amendment_sha256']
    files[str(HERE/REVIEW_NAME21)] = gate['review_sha256']
    return files
