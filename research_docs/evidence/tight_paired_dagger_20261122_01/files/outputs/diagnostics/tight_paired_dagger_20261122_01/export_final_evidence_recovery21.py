"""Inactive thin21 entry over pinned15 engine; explicit reviewed amendment required."""
from pathlib import Path
import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_entry21_common', HERE/'final_recovery21_common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
engine = common.load_entry21('export_final_evidence_recovery15.py', 'ffc258ab91ca33facbf94a2e59aa183af73c6788c5e58f8deab4be1d20b614b1', __file__, replacements=(("project/AUDIT_REL/'final_report_assembler_recovery15.py'", "project/AUDIT_REL/'final_report_assembler_recovery21.py'"),))
ORIGINAL_ALLOWLIST21 = engine.allowlist

def allowlist21(project, run, audit, assembly):
    entries = {row['source']: row for row in ORIGINAL_ALLOWLIST21(project, run, audit, assembly)}
    for filename, digest in common.export_supplements21(run, audit).items():
        path = Path(filename).resolve()
        relative = str(path.relative_to(Path(project).resolve()))
        old = entries.get(relative)
        common.require(old is None or old.get('expected_sha256') in (None, digest),
                       'Conflicting21 exact export supplement')
        entries[relative] = dict(source=relative, category='reviewed_exact_metadata_amendment21', expected_sha256=digest)
    return [entries[key] for key in sorted(entries)]

engine.base.allowlist = allowlist21
export = engine.export

if __name__ == '__main__': engine.main()
