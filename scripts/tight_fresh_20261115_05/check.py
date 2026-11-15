"""Single short CPU check for the actual grasp-first implementation changes."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import time

TREE=Path(__file__).resolve().parents[2]


def main():
    began=time.time();out=TREE/'research_docs/evidence/tight_fresh_20261114/grasp_followup05/check.json'
    result=subprocess.run([str(TREE/'.venv_isaacsim/bin/python'),'-m','pytest','-q',
        'tests/test_grasp_curriculum.py','tests/test_fresh_balanced_followup.py'],cwd=TREE,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    from tools.tight_fresh_research_loop import validate_configuration
    from visuotactile.training_expansion import make_registry,registry_binding,read_registry
    config=json.loads((Path(__file__).parent/'loop_config.json').read_text());validate_configuration(config)
    source=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_04/round_14')
    session=json.loads((source/'session.json').read_text());row=config['rounds'][0]
    with tempfile.TemporaryDirectory(prefix='tight-grasp-check-') as temporary:
        registry=make_registry('grasp_cpu_check',session['domain'],row['seeds'],
            {r:source/r/'state_data/data.json' for r in ('A','B')},
            config['inputs']['partition_manifest'],config['inputs']['frozen_families'],
            created_utc=time.time(),revisit_families=row['revisit_families'])
        p=Path(temporary)/'registry.json';p.write_text(json.dumps(registry))
        checked=read_registry(registry_binding(p));assert len(checked['families'])==10
    changed=subprocess.check_output(['git','diff','--name-only','--','*.py'],cwd=TREE,text=True).splitlines()
    for relative in changed:ast.parse((TREE/relative).read_text())
    value=dict(passed=True,seconds=time.time()-began,pytest=result.stdout.strip(),
        actual_training_registry_families=len(checked['families']),new_training_families=0,
        software_changes_only=True,gpu_updates=0,new_simulation_trials=0,
        existing_behavior_evidence='prior.json; no claim of new rescue or student improvement')
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(TREE));main()
