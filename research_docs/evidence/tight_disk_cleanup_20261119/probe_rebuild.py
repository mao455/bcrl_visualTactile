"""Bounded verification only: reconstruct one complete old feature payload."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rebuild_visual_cache import rebuild,P,PUBLIC
rows=json.loads((PUBLIC/'cold_cache_dependencies.json').read_text())['archives']
# Preregister smallest complete payload to limit GPU verification; not outcome selection.
idx=min(range(len(rows)),key=lambda i:rows[i]['recipe']['source']['length']);row=rows[idx]
registration=dict(row=idx,archive=row['archive'],frames=row['recipe']['source']['length'],scope='Full deterministic cache rebuild; zero training; one archived episode; no simulation',utc=time.time())
(PUBLIC/'cache_rebuild_registration.json').write_text(json.dumps(registration,indent=2)+'\n')
out=P/'outputs/disk_audit_20261119/rebuild_probe'/Path(row['original']).name
result=rebuild(row,out);result.update(registration)
result['bytewise_equal']=result['sha256']==result['expected_sha256']
(PUBLIC/'cache_rebuild_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='telemetry'},indent=2),flush=True)
assert result['bytewise_equal'],'Cannot claim exact rebuild; keep all candidate archives'
