"""Inactive thin21 entry over pinned15 engine; explicit reviewed amendment required."""
from pathlib import Path
import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_entry21_common', HERE/'final_recovery21_common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
engine = common.load_entry21('extract_selected_ledgers_recovery15.py', '5963851279881f6f30fcc6dd98361aea1f2d66f1e212b8455fb7404659721505', __file__, replacements=())
extract = engine.extract

if __name__ == '__main__': engine.main()
