"""Artifact integrity and no-policy-mutation closure for this forensic task."""
from pathlib import Path
import json,hashlib,subprocess,shutil,re,datetime,py_compile
OUT=Path(__file__).resolve().parent;CAN=OUT.parents[2];SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
for name in ['backend_raw_round01_A.json','backend_raw_round01_B.json','backend_raw_round02_A.json','backend_raw_round02_B.json']:
 d=read(OUT/name);assert d['step_receipts_exact'] and d['raw_masks_recomputed'] and d['effective_exposure']['eligible']
for f in OUT.glob('*.py'):py_compile.compile(str(f),doraise=True)
for f in OUT.glob('*.json'):read(f)
broken=[]
for target in re.findall(r'\]\(([^)]+)\)',(OUT/'REPORT.md').read_text()):
 if not Path(re.sub(r':\d+$','',target)).exists():broken.append(target)
assert not broken
models=[]
for p,d in read(OUT/'source_and_launch.json')['model_identities'].items():
 actual=sha(p);assert actual==d['actual_sha256']==d['expected'];models.append(dict(path=p,sha256=actual))
canonical_status=subprocess.check_output(['git','status','--short'],cwd=CAN,text=True)
source_status=subprocess.check_output(['git','status','--short'],cwd=SRC,text=True)
assert not canonical_status.strip() and source_status.strip()=='?? pretrained_policy'
processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
workers=[line for line in processes if str(OUT) in line and 'python' in line and 'close_audit' not in line and not any(w in line for w in ('bash -l','exec_command'))]
assert not workers, workers
closure=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='End-to-end forensic audit only',optimizer_updates=0,EMA_updates=0,new_physics_steps=0,new_collection_episodes=0,policy_edits=0,protected_model_hashes_rechecked=models,canonical_git_status=canonical_status,experiment_source_git_status=source_status,preexisting_untracked='pretrained_policy symlink',broken_report_links=broken,own_diagnostic_workers_remaining=workers,free_disk_gib=shutil.disk_usage(OUT).free/2**30,diagnostic_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),report='REPORT.md',limitations=['No lossless RGB-D or DINO patches in final-test records','Original ephemeral per-update gradients not retained','Original first post-resume epoch2 numerical checkpoint replaced; receipt only'],findings=['P1 native onset anchor excluded on48/50 new corrections despite absolute-tick exposure>=50','P2 capture DINO microbatch1 differs numerically from cached microbatch2; final-test uses2','P3 generated report says online noise7093 but checkpoints specify7091'],causal_status='No finding alone proves causation of lost final-test successes')
(OUT/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
manifest=[f'{sha(p)}  {p.relative_to(OUT)}' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts]
(OUT/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ('protected_model_hashes_rechecked',)}));print('protected models',len(models),'artifacts',len(manifest))
