"""Inactive thin21 entry over pinned15 engine; explicit reviewed amendment required."""
from pathlib import Path
import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_entry21_common', HERE/'final_recovery21_common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
engine = common.load_entry21('final_report_assembler_recovery15.py', 'fba539cf6cf28503e4baa0ee43899297cf02cac589b434a7503f3b40f2a6fb67', __file__, replacements=())
assemble = engine.assemble

if __name__ == '__main__': engine.main()
