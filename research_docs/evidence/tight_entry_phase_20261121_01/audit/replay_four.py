"""Re-run original four complete, raw-vision histories without touching old receipts."""
from pathlib import Path
import runpy,json,hashlib,time
OUT=Path(__file__).resolve().parent
ORIGINAL=OUT.parent/'tight_adaptive_audit_20261121_01'
for name in ('replay_exact_vision.py','replay_A_exact_vision.py'):
 path=ORIGINAL/name
 code=path.read_text().replace('OUT=Path(__file__).resolve().parent',f'OUT=Path({str(OUT)!r})')
 scope={'__file__':str(path),'__name__':'entryfix_replay'}
 exec(compile(code,str(path),'exec'),scope)
 scope['run']()
receipt=dict(started_from_scripts={n:hashlib.sha256((ORIGINAL/n).read_bytes()).hexdigest() for n in ('replay_exact_vision.py','replay_A_exact_vision.py')},optimizer_updates=0,physics_steps=0)
for n in ('online_self_replay_exact_vision.json','online_self_replay_A_exact_vision.json'):
 rows=json.loads((OUT/n).read_text());assert all(r['maximum_command_error_rad']<1e-6 for r in rows)
 receipt[n]=[{k:r[k] for k in ('round','identity','frames','maximum_command_error_rad','maximum_nominal_error_rad','maximum_robot_error')} for r in rows]
(OUT/'four_replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
