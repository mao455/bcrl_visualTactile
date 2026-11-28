"""Output-only admission waits; unchanged canonical rollout/controller and RNG."""
import sys
from outputs.diagnostics.tight_full_curve_20261127_01.train_recovery import bounded_admission
if len(sys.argv)>1 and sys.argv[1]=='project':
 from tools import tight_project
 tight_project.ENTRIES['curve_eval']=('evaluate','outputs.diagnostics.tight_full_curve_20261127_01.eval_recovery',[])
 sys.argv.pop(1);tight_project.main()
else:
 from tools.tight_machine_review import main
 main()
