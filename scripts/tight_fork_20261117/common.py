from pathlib import Path
import json
TREE=Path(__file__).resolve().parents[2]
RUNS=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=RUNS/'tight_balance76_20261117_01_recovery02'
ROOT=RUNS/'tight_fork_diagnosis_20261117_01_recovery01'
OUT=TREE/'research_docs/evidence/tight_long_20261117/fork_diagnosis'
CASES=[('A',28824002,1),('A',28824006,3),('B',28824002,1)]
def read(p):return json.loads(Path(p).read_text())
def path(role,seed,part):return OLD/f'comparison{part:02d}'/'development'/f'{role}_development_{seed}'
