"""Re-run the independent raw-mask/deck/window reconstruction in this audit root."""
from pathlib import Path
import sys,importlib.util
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location('actual_backend_raw_auditor',SRC/'research_docs/evidence/tight_adaptive_dagger_20261119_01/audit_backend_masks.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for rnd in (1,2):
 for role in ('A','B'):
  module.audit(CAN/f'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01/round{rnd:02d}/learning',role,OUT/f'backend_raw_round{rnd:02d}_{role}.json')
