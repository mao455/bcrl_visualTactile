"""Read-only experiment observer; writes only its own receipt and no model state."""
from pathlib import Path
import json,time,shutil
import psutil
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

while time.time()<read(ROOT/'session.json')['deadline']:
    progress=read(ROOT/'progress.json');round_index=progress.get('round',1);target=ROOT/f'round{round_index:02d}/learning';components=[]
    for role in 'AB':
        for name,folder in [('DP',target/role/'training'),('Transformer',target/role/f'backend/{role}/training')]:
            current=read(folder/'progress.json')
            components.append(dict(model=role+'-'+name,epoch=current.get('epoch'),updates=current.get('updates',current.get('update')),age_s=round(time.time()-current['utc']) if current.get('utc') else None))
    failure={p.name:read(p) for p in ROOT.glob('round1_boundary_failure.json')}
    receipt=dict(round=round_index,stage=progress.get('stage'),components=components,
        free_disk_gib=round(shutil.disk_usage(ROOT).free/2**30,2),free_ram_gib=round(psutil.virtual_memory().available/2**30,2),
        boundary_failure=failure,utc=time.time())
    (ROOT/'observer_progress.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    if failure or (ROOT/'round1_review_ready.json').exists() or (ROOT/'cycle_complete.json').exists():break
    time.sleep(50)
