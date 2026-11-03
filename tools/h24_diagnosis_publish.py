"""Publish only validated small diagnosis outputs after the persistent owner exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools.h24_diagnosis_common import ROOT, read, sha
from visuotactile.training_resources import atomic_json

CODE=Path(__file__).resolve().parents[1]
PUBLIC=Path('research_docs/evidence/h24_diagnosis_20261104')
REPORT=Path('research_docs/H24_CLOSED_LOOP_DIAGNOSIS.md')
DOCS=[REPORT,Path('research_docs/SESSION_HANDOFF.md'),Path('research_docs/CHANGELOG.md'),Path('AGENTS.md')]

def git(*args):
    return subprocess.check_output(['git',*args],cwd=CODE,text=True).strip()

def insert(path,text):
    original=path.read_text()
    first,rest=original.split('\n',1)
    path.write_text(first+'\n\n'+text.strip()+'\n'+rest)

def verify_artifacts(root):
    result=read(root/'owner_exit.json')
    started=read(root/'recorder_recovery_01/started.json')['utc']
    integrity=read(root/'final_integrity.json')
    if result['utc']<started or result['analysis_exit']!=0 or not integrity['passed'] or integrity['utc']<started:
        raise RuntimeError('No current successful analysis/integrity completion')
    summary=read(root/'analysis/summary.json')
    if read(PUBLIC/'summary.json')!=summary:
        raise RuntimeError('Public/local analysis summary mismatch')
    for row in summary['videos'].values():
        for kind in ('clean','diagnostic'):
            video=row[kind];path=Path(video['path'])
            if not path.resolve().is_relative_to((root/'videos').resolve()):raise RuntimeError('Unexpected video destination')
            if path.stat().st_size!=video['bytes'] or sha(path)!=video['sha256']:raise RuntimeError('Video changed')
            if video['decoded_frames']!=(row['result']['steps']+1)//2:raise RuntimeError('Incomplete decoded video')
            if video['stream']['codec_name']!='h264':raise RuntimeError('Unexpected video codec')
    return result,summary,integrity

def publish(root):
    result,summary,integrity=verify_artifacts(root)
    if git('branch','--show-current')!='feature/native-dp-training-v2':raise RuntimeError('Wrong publication branch')
    if git('remote','get-url','origin')!='https://github.com/mao455/bcrl_visualTactile.git':raise RuntimeError('Wrong origin')
    allowed={str(p) for p in DOCS}
    changed=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all'],cwd=CODE).decode().split('\0')
    for item in changed:
        if not item:continue
        path=item[3:]
        if path not in allowed and not path.startswith(str(PUBLIC)+'/'):
            raise RuntimeError('Unrelated uncommitted change; publication withheld: '+path)
    if git('diff','--cached','--name-only'):raise RuntimeError('Existing staged user changes; publication withheld')
    # Generated data is reviewed by schema, identity, decode, bounds and an exact
    # path allowlist; it cannot stage experiment data, MP4s, weights or large logs.
    for path in PUBLIC.rglob('*'):
        if path.is_file() and (path.suffix not in {'.json','.csv','.png','.jpg'} or path.stat().st_size>3*1024**2):
            raise RuntimeError('Unapproved public artifact: '+str(path))
    attempts=read(root/'results.json')
    completed=[r for r in attempts.values() if r['state']=='completed']
    software=[k for k,r in attempts.items() if r['state']!='completed']
    live=sorted(summary['videos'])
    success=summary['reproduced_success_videos']
    recovery=read(root/'recorder_recovery_01/registration.json')
    atomic_json(PUBLIC/'final_integrity.json',integrity)
    atomic_json(PUBLIC/'completion.json',dict(owner=result,software_attempts=software,
        videos=live,reproduced_native_success_videos=success,missing_primary_videos=summary['missing_videos'],
        recovery_source_commit=recovery['source_commit'],runtime_source_overrides=recovery['overrides'],
        all_seven_checkpoints_unchanged=integrity['passed'],new_optimizer_updates=0))
    conclusion=(f'本轮后台已退出，状态`{result["state"]}`；{len(completed)}个attempt软件完成，'
        f'软件缺失/故障记录{len(software)}条（单列，不计任务失败）。\n'
        f'已核验录像组：{", ".join(live) or "无"}；8000真实原生成功录像：{", ".join(success) or "未取得"}；'
        f'指定六段缺失：{summary["missing_videos"] or "无"}。\n'
        f'原始七个受保护checkpoint最终SHA全部匹配；无参数更新。原deadline/启动预算未重置。\n'
        f'本轮首动作前记录器错误及恢复保留，恢复源码`{recovery["source_commit"]}`；'
        '详细结果与视频索引见H24_CLOSED_LOOP_DIAGNOSIS和evidence/h24_diagnosis_20261104/completion.json。')
    insert(REPORT,'## 后台完成与最终完整性\n\n'+conclusion)
    for path in DOCS[1:]:insert(path,'## 2026-11-04 H24诊断后台结束\n\n'+conclusion+'\n不自动重启已完成诊断或追加训练/录像。')
    subprocess.run(['git','diff','--check'],cwd=CODE,check=True)
    subprocess.run(['git','add','--',*[str(p) for p in DOCS],str(PUBLIC)],cwd=CODE,check=True)
    staged=git('diff','--cached','--name-only').splitlines()
    if any(p not in allowed and not p.startswith(str(PUBLIC)+'/') for p in staged):raise RuntimeError('Staging allowlist violation')
    subprocess.run(['git','commit','-m','Publish validated H24 diagnosis results and video index'],cwd=CODE,check=True)
    commit=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:feature/native-dp-training-v2'],cwd=CODE,check=True,timeout=120)
    remote=git('ls-remote','origin','refs/heads/feature/native-dp-training-v2').split()[0]
    if remote!=commit:raise RuntimeError('Remote SHA mismatch')
    atomic_json(root/'publication.json',dict(state='completed',commit=commit,remote_verified=True,
        github_commit='https://github.com/mao455/bcrl_visualTactile/commit/'+commit,
        videos=live,new_optimizer_updates=0,utc=time.time()))

def main(root):
    registration=read(root/'publish_registration.json')
    if sha(Path(__file__))!=registration['script_sha256']:raise RuntimeError('Publisher source drift')
    if (root/'publication.json').exists():raise RuntimeError('Publication already attempted')
    deadline=read(root/'plan.json')['deadline']+1800
    try:
        while not (root/'owner_exit.json').exists():
            if time.time()>deadline:raise TimeoutError('Owner did not finish within fixed deadline plus postprocessing window')
            time.sleep(5)
        publish(root)
    except BaseException as error:
        atomic_json(root/'publication.json',dict(state='blocked',reason=repr(error),utc=time.time()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();os.chdir(CODE);main(args.root.resolve())
