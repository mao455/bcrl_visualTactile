"""Read back only the explicitly authorized large public repository after push."""
from pathlib import Path
import json,subprocess,time,csv,urllib.request,concurrent.futures
D=Path('outputs/diagnostics/tight_correction_20261127_01')
repository='mao455/bcrl_visualTactile';branch='feature/tight-state-foundation'
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
assert git('remote','get-url','--push','--all','origin')=='https://github.com/'+repository+'.git'
head=git('rev-parse','HEAD');actual=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert actual==head
subprocess.run(['git','fetch','--no-tags','origin','refs/heads/'+branch+':refs/remotes/origin/'+branch],check=True)
files=set(git('ls-tree','-r','--name-only','refs/remotes/origin/'+branch).splitlines())
retired=list(csv.DictReader((D/'retired_paths.csv').open()));assert len(retired)==30 and all(r['path'] not in files for r in retired)
report='research_docs/TIGHT_CORRECTION_20261127.md';evidence='research_docs/evidence/tight_correction_20261127_01/README.md'
assert report in files and evidence in files
paths=[report,evidence,'research_docs/evidence/tight_correction_20261127_01/audit/per_family_before_after.csv','research_docs/evidence/tight_correction_20261127_01/audit/primary_results.json','research_docs/evidence/tight_correction_20261127_01/audit/retired_paths.csv']
def opened(path):
 url='https://raw.githubusercontent.com/'+repository+'/'+branch+'/'+path
 with urllib.request.urlopen(url,timeout=45) as response:
  body=response.read();assert response.status==200 and body
  if path.endswith('primary_results.json'):assert json.loads(body)['denominator_per_policy']==8
  return dict(path=path,url=url,status=response.status,bytes=len(body))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:opened_rows=list(pool.map(opened,paths))
receipt=dict(utc=time.time(),repository=repository,branch=branch,commit=head,remote_ref_matches=True,opened=opened_rows,retired_paths_absent=[r['path'] for r in retired],second_repository_accessed=False,second_repository_writes=0)
(D/'remote_publication_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(commit=head,opened=len(opened_rows),retired_absent=len(retired),second_repository_writes=0)))
