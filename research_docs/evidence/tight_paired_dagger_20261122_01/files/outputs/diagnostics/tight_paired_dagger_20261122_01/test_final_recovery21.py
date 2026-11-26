"""CPU-only acceptance boundaries; no model/runtime actions or actual root approval."""
from pathlib import Path
from unittest import mock
import importlib.util
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_test21_common', HERE/'final_recovery21_common.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class ExactAdapter21(unittest.TestCase):
    def test_two_private_changes_only(self):
        original = (HERE/c.BASE_NAME).read_text()
        self.assertEqual(original.count(c.OLD_RC), 1)
        self.assertEqual(original.count(c.OLD_EXTRACTOR), 1)
        self.assertNotIn(c.NEW_RC, original)
        derived = c.exact_replace(c.exact_replace(original, c.OLD_RC, c.NEW_RC), c.OLD_EXTRACTOR, c.NEW_EXTRACTOR)
        self.assertEqual(c.exact_replace(c.exact_replace(derived, c.NEW_RC, c.OLD_RC), c.NEW_EXTRACTOR, c.OLD_EXTRACTOR), original)

    def test_missing_or_ambiguous_replacement_refused(self):
        for source in ('absent', 'one one'):
            with self.assertRaises(ValueError): c.exact_replace(source, 'one', 'two')

    def test_missing_actual_root_amendment_refused(self):
        # Test the real guard with its first metadata read forced absent. No actual
        # amendment/root review is created or inferred from this fixture.
        with mock.patch.object(c, 'read_small', side_effect=FileNotFoundError('fixture')):
            with self.assertRaises(FileNotFoundError): c.amendment_gate21(c.ROOT, HERE)

    def failure_rows(self, rc=-2, parent="KeyboardInterrupt('Owner received signal 15')", reaped=True):
        return [dict(returncode=rc, utc=3),dict(error=parent, utc=1),
                dict(children_reaped=reaped, utc=2),
                dict(error="ValueError('evaluate_development failed; no retry or replacement is launched')", utc=4),
                dict(returned_after_owned_helpers=True, utc=5)]

    def test_exact_actual_failure_accepted(self):
        with mock.patch.object(c, 'read_small', side_effect=self.failure_rows()):
            self.assertEqual(c.validate_actual_failure21(c.ROOT,HERE)['returncode'], -2)

    def test_other_nonzero_and_boolean_refused(self):
        for rc in (-9,-1,0,1,2,True):
            with self.subTest(rc=rc), mock.patch.object(c,'read_small',side_effect=self.failure_rows(rc)):
                with self.assertRaises(ValueError): c.validate_actual_failure21(c.ROOT,HERE)

    def test_parent_error_and_unreaped_refused(self):
        for rows in (self.failure_rows(parent='unknown'), self.failure_rows(reaped=False)):
            with mock.patch.object(c,'read_small',side_effect=rows):
                with self.assertRaises(ValueError): c.validate_actual_failure21(c.ROOT,HERE)

    def test_all_entry_sources_import_inactive(self):
        for name in ('extract_selected_ledgers_recovery21.py','final_compute_recovery21.py',
                     'final_report_assembler_recovery21.py','export_final_evidence_recovery21.py'):
            with self.subTest(name=name):
                spec = importlib.util.spec_from_file_location('_inactive21_'+name[:-3], HERE/name)
                module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.assertEqual(Path(module.engine.__file__), HERE/name)

    def test_original_common_still_requires_one(self):
        self.assertIn(c.OLD_RC, (HERE/c.BASE_NAME).read_text())


if __name__ == '__main__': unittest.main()
