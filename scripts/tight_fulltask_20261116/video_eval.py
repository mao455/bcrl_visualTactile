"""Read-only 60 Hz actual RGB video; no extra camera exposure or policy calls."""
import argparse,time
from pathlib import Path
from tools.tight_fresh_rollouts import read
from visuotactile.training_resources import atomic_json

def main(*,allow_heldout=False):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.name=='updated_evaluation' and (a.output.parent/'paired_request.json').exists():
        from scripts.tight_fulltask_20261116.paired_eval import upgrade_plan,install
        upgrade_plan(a.output);install(a.output)
    from tools.tight_evaluate import run
    return run(a.output,allow_heldout=allow_heldout)
if __name__=='__main__':main()
