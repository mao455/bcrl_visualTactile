"""Inactive thin21 entry over pinned15 engine; explicit reviewed amendment required."""
from pathlib import Path
import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_entry21_common', HERE/'final_recovery21_common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
engine = common.load_entry21('final_compute_recovery15.py', '0eea5f07351e56a6b0cab295c7f1eb019e4bc5e2379244b1f2bf813697f9d8a4', __file__, replacements=())
inventory = engine.inventory

if __name__ == '__main__': engine.main()
