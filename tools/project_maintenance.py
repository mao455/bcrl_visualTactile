"""Dependency inventory and reviewed, identity-checked artifact removal.

No implicit age/size deletion. A request names every candidate subtree, its reason,
references and protected paths. Plan is durable before apply; symlinks are refused.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

CODE = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def source_inventory(entries):
    paths=[CODE/p for p in subprocess.check_output(['git','ls-files','*.py'],cwd=CODE,text=True).splitlines()]
    modules={'.'.join(p.relative_to(CODE).with_suffix('').parts).removesuffix('.__init__'):p for p in paths if p.exists()}
    graph={};errors=[]
    for name,path in modules.items():
        try: tree=ast.parse(path.read_text())
        except (SyntaxError,UnicodeError) as error:
            errors.append(dict(path=str(path.relative_to(CODE)),error=str(error)));continue
        deps=set();package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(tree):
            candidates=[]
            if isinstance(node,ast.Import):candidates=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package.split('.')[:len(package.split('.'))-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []))
                candidates=[base]+[base+'.'+a.name for a in node.names]
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                # Includes configured '-m package.module' entries and importlib strings.
                candidates=[s.strip('"\'(),[]') for s in node.value.split()]
            deps.update(c for c in candidates if c in modules and c!=name)
        graph[name]=sorted(deps)
    live=set();pending=list(entries)
    while pending:
        name=pending.pop()
        if name in live:continue
        if name not in graph:raise ValueError('Unknown entrypoint: '+name)
        live.add(name);pending.extend(graph[name])
    return dict(entries=entries,edges=graph,reachable=sorted(live),parse_errors=errors,
        unreachable=[dict(module=n,path=str(p.relative_to(CODE))) for n,p in modules.items() if n not in live],
        scope='Python AST imports and literal module entrypoints; shell/config/runtime references require separate review')


def identity(path):
    st=path.lstat()
    return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,
        links=st.st_nlink,allocated_bytes=st.st_blocks*512)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def open_inodes():
    found=set();inaccessible=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            for fd in (proc/'fd').iterdir():
                try:st=fd.stat();found.add((st.st_dev,st.st_ino))
                except FileNotFoundError:pass
        except (PermissionError,ProcessLookupError):inaccessible.append(int(proc.name))
        except FileNotFoundError:pass
    return found,inaccessible


def within(path, root):return path==root or root in path.parents


def protection_reason(path, protected):
    return next((protected[p] for p in (path,*path.parents) if p in protected),None)


def restricted_system_process(pid):
    """These nondumpable session daemons cannot be experiment workers; document them."""
    p=Path('/proc')/str(pid)
    try:
        command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        group=(p/'cgroup').read_text().strip();name=(p/'comm').read_text().strip()
        accepted=(group.endswith('/init.scope') and name in ('systemd','(sd-pam)') and
            (command=='/usr/lib/systemd/systemd --user' or command=='(sd-pam)')) or (
            name=='ssh-agent' and group.endswith('/gnome-keyring-daemon.service') and
            command.startswith('/usr/bin/ssh-agent -D -a /run/user/'))
        return dict(pid=pid,command=command,cgroup=group,reason='Protected session daemon, not an experiment worker') if accepted else None
    except FileNotFoundError:return dict(pid=pid,reason='Process exited during inventory')


def plan(request_path, ledger):
    assert not ledger.exists(), 'Never replace a previous deletion ledger'
    request=json.loads(request_path.read_text());allowed=Path(request['allowed_root']).resolve()
    assert within(allowed,(CODE/'outputs').resolve()) or within(allowed,CODE), 'Outside this project'
    protected={Path(p).resolve():why for p,why in request['protected'].items()}
    opened,inaccessible=open_inodes();rows=[];kept=[];seen=set()
    for candidate in request['candidates']:
        original=Path(candidate['path'])
        assert not original.is_symlink(), 'No symlink candidate'
        base=original.resolve();assert within(base,allowed) and base!=allowed
        paths=sorted(base.rglob('*')) if base.is_dir() else [base]
        for path in paths:
            if path.is_symlink():kept.append(dict(path=str(path),reason='symlink: unresolved ownership'));continue
            if not path.is_file() or path in seen:continue
            seen.add(path);st=identity(path)
            why=protection_reason(path,protected)
            if (st['device'],st['inode']) in opened:why='open in a current user process'
            if why:kept.append(dict(path=str(path),reason=why));continue
            rows.append(dict(path=str(path),**st,reason=candidate['reason'],
                references=candidate.get('references',[]),sha256=digest(path) if candidate.get('hash',False) else None))
    write(ledger,dict(state='planned',created=time.time(),request=str(request_path.resolve()),
        request_sha256=digest(request_path),allowed_root=str(allowed),protected=request['protected'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        files=rows,retained=kept,inaccessible_user_processes=inaccessible,
        logical_candidate_bytes=sum(x['bytes'] for x in rows)))
    return dict(files=len(rows),retained=len(kept),logical_bytes=sum(x['bytes'] for x in rows))


def apply(ledger):
    data=json.loads(ledger.read_text());assert data['state'] in ('planned','applying')
    allowed=Path(data['allowed_root']);protected={Path(p).resolve():why for p,why in data['protected'].items()}
    journal=ledger.with_suffix('.deletions.jsonl')
    done={r['path']:r for r in (json.loads(s) for s in journal.read_text().splitlines())} if journal.exists() else {}
    opened,inaccessible=open_inodes()
    exclusions=[restricted_system_process(pid) for pid in inaccessible]
    assert all(exclusions), ('Unreadable unclassified user fd inventories',inaccessible)
    data['restricted_process_exclusions']=exclusions
    # Preflight the entire plan before the first irreversible removal.
    for row in data['files']:
        if row['path'] in done:continue
        path=Path(row['path']);assert within(path.resolve(),allowed) and not path.is_symlink()
        assert not protection_reason(path.resolve(),protected)
        now=identity(path)
        assert all(now[k]==row[k] for k in ('device','inode','bytes','mtime_ns')), 'Candidate changed: '+str(path)
        assert (now['device'],now['inode']) not in opened, 'Candidate opened: '+str(path)
        if row['sha256']:assert digest(path)==row['sha256']
    data.update(state='applying',free_bytes_before=os.statvfs(allowed).f_bavail*os.statvfs(allowed).f_frsize)
    write(ledger,data)
    with journal.open('a') as stream:
        for row in data['files']:
            if row['path'] in done:continue
            path=Path(row['path']);now=identity(path)
            assert all(now[k]==row[k] for k in ('device','inode','bytes','mtime_ns'))
            path.unlink()
            result=dict(path=row['path'],utc=time.time(),released_blocks_bytes=now['allocated_bytes'] if now['links']==1 else 0)
            stream.write(json.dumps(result)+'\n');stream.flush();os.fsync(stream.fileno());done[row['path']]=result
    directories=sorted({p for x in data['files'] for p in Path(x['path']).parents if within(p,allowed) and p!=allowed},key=lambda p:len(p.parts),reverse=True)
    removed=[]
    for path in directories:
        try:path.rmdir();removed.append(str(path))
        except (OSError,FileNotFoundError):pass
    data.update(state='completed',finished=time.time(),deleted_files=len(done),removed_empty_directories=removed,
        deleted_checkpoints=sum(Path(x['path']).suffix in ('.pt','.pth','.ckpt') for x in data['files']),
        released_blocks_bytes=sum(x['released_blocks_bytes'] for x in done.values()),
        free_bytes_after=os.statvfs(allowed).f_bavail*os.statvfs(allowed).f_frsize)
    write(ledger,data)
    return {k:data[k] for k in ('deleted_files','deleted_checkpoints','released_blocks_bytes','free_bytes_before','free_bytes_after')}


def checkpoint_probe(session_root, output, state_checkpoint=None):
    """Real-checkpoint CPU load/AdamW/EMA/data/forward probe, zero optimization.

    CUDA RNG payloads are fingerprinted, not consumed on CPU; the existing live
    contract command separately verifies an actual CUDA restoration after cutover.
    """
    import gc
    import torch
    from visuotactile.state_history import PersistentState, PersistentImage
    from visuotactile.native_data import Episode
    from visuotactile.anchor_sampling import AnchorSampler
    torch.set_num_threads(2);torch.manual_seed(20261102)
    session=json.loads((session_root/'session.json').read_text())
    data=json.loads((Path(session['source_root'])/'data.json').read_text())
    assert [len(data['splits'][k]) for k in ('train','validation','test')]==[320,40,40]
    assert len(set(sum(data['splits'].values(),[])))==400
    cache=torch.load(session['numeric_cache'],map_location='cpu',weights_only=False)
    identities=[data['splits']['train'][0],data['splits']['validation'][0]]
    result={};fingerprints={}
    def tensors(value):
        h=hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v,dict):
                for key in sorted(v,key=str):h.update(str(key).encode());visit(v[key])
            elif isinstance(v,(tuple,list)):
                for item in v:visit(item)
            else:h.update(repr(v).encode())
        visit(value);return h.hexdigest()
    for job,path in [('state',state_checkpoint or session_root/'state_train/last.pt'),('image',Path(session['checkpoints']['image']['path']))]:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=PersistentState('dp',saved['contract']) if job=='state' else PersistentImage(saved['group'],saved['contract'])
        model.load_state_dict(saved['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
        optimizer.load_state_dict(saved['optimizer'])
        assert tensors(optimizer.state_dict())==tensors(saved['optimizer'])
        assert all(int(v['step'])==saved['updates'] for v in optimizer.state.values())
        assert tensors(model.state_dict())==tensors(saved['model'])
        model.load_state_dict(saved['ema']);assert tensors(model.state_dict())==tensors(saved['ema'])
        generator=torch.Generator();generator.set_state(saved['schedule_rng'])
        assert torch.equal(generator.get_state(),saved['schedule_rng'])
        if 'anchor_sampler' in saved:
            sampler=AnchorSampler(session['lengths'],generator,session['effective_batch']);sampler.load_state_dict(saved['anchor_sampler'])
            assert tensors(sampler.state_dict())==tensors(saved['anchor_sampler'])
        fingerprints[job]=dict(checkpoint=str(path),updates=saved['updates'],ema_step=saved['ema_optimization_step'],
            model=tensors(saved['model']),ema=tensors(saved['ema']),optimizer=tensors(saved['optimizer']),
            schedule_rng=tensors(saved['schedule_rng']),noise_rng=tensors(saved['noise_rng']),cuda_rng=tensors(saved['cuda_rng']),
            optimizer_steps=True,sampler_schema=saved.get('anchor_sampler',{}).get('version'),parameters=sum(p.numel() for p in model.parameters()))
        model.eval();del optimizer,saved;gc.collect()
        for identity in identities:
            rec=cache['records'][identity]
            if job=='image':rec=dict(**{k:rec[k] for k in ('robot','command','time','joint_limits')},identity=identity,source_root=session['source_root'])
            with torch.no_grad():
                condition,anchors,_=model.encode(rec)
                plan=model.predict(condition[:2],torch.Generator().manual_seed(7091))
            assert torch.isfinite(condition).all() and torch.isfinite(plan).all()
            result[job+'/'+identity]=dict(condition=condition,anchors=anchors,plan=plan)
            if job=='image':
                episode=Episode(session['source_root'],identity,'I-HistoryDP')
                result[job+'/'+identity]['frame']=episode.frame(0,'cpu')
        del model;gc.collect()
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,output)
    write(output.with_suffix('.json'),dict(passed=True,optimizer_updates=0,checkpoints=fingerprints,
        identities=identities,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        implementation_sha256=digest(Path(__file__)),output_sha256=digest(output),cuda_rng_restore='deferred to actual CUDA contract check'))
    return dict(passed=True,optimizer_updates=0,outputs=len(result))


def markdown_links():
    missing=[]
    for path in [CODE/'README.md',CODE/'AGENTS.md',*(CODE/'research_docs').glob('*.md')]:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('https:','http:','/','#','mailto:')):continue
            relative=target.split('#')[0]
            if relative and not (path.parent/relative).exists():missing.append(dict(file=str(path),target=target))
    return missing


def migrate_freezes(parent, reviewed_changes, destination):
    """Explicit, recorded source-only migration. Frozen data may not change."""
    destination.mkdir(parents=True,exist_ok=False);amendments={}
    for name in ('offline_freeze.json','runtime_freeze.json'):
        original=json.loads((parent/name).read_text());write(destination/('before_'+name),original)
        updated={};changes=[]
        for key,before in original.items():
            path=Path(key) if Path(key).is_absolute() else CODE/key
            relative=str(path.relative_to(CODE)) if path.is_relative_to(CODE) else None
            if not path.exists():
                assert relative in reviewed_changes and reviewed_changes[relative]=='D', 'Unreviewed missing frozen input: '+key
                changes.append(dict(path=key,before=before,after=None,reason='reviewed retired source'));continue
            after=digest(path)
            if before!=after:
                assert relative in reviewed_changes, 'Unreviewed frozen input drift: '+key
                changes.append(dict(path=key,before=before,after=after,reason='reviewed organization; real checkpoint parity and regression required'))
            updated[key]=after
        for path in [*(CODE/'visuotactile').rglob('*.py'),*(CODE/'tools').glob('*.py')]:
            key=str(path) if name=='offline_freeze.json' else str(path.relative_to(CODE))
            updated[key]=digest(path)
        write(parent/name,updated);amendments[name]=changes
    write(destination/'amendments.json',amendments)
    return amendments


def final_summary(root, request, stage):
    """Publish only measured outcomes and actual completed cleanup/checks."""
    maintenance=root/'maintenance';offline=json.loads((root/'offline/comparison.json').read_text())['checkpoints']
    loop=json.loads((CODE/'research_docs/evidence/state_dp/closed_loop.json').read_text())
    by_update={x['update']:x for x in offline};old,new=by_update[16584],by_update[20918]
    def score(item):
        w=item['cohorts']['validation']['windows']['full']
        return .5*(w['arm_rmse_rad']**2+w['hand_rmse_rad']**2)
    gain=(score(old)-score(new))/score(old)
    train_loss_ratio=new['cohorts']['train']['loss']['loss']/old['cohorts']['train']['loss']['loss']
    val_loss_ratio=new['cohorts']['validation']['loss']['loss']/old['cohorts']['validation']['loss']['loss']
    validation=loop['cohorts']['validation']['new'];train=loop['cohorts']['train']['new']
    progressed=bool(validation['success'] or validation['lift'] or validation['first_subgoal'])
    decision=dict(validation_selection_mse_gain_100_to150=gain,train_loss_ratio=train_loss_ratio,
        validation_loss_ratio=val_loss_ratio,validation=validation,train_fitting=train,
        recommendation='review_bounded_200_epoch_continuation' if progressed and gain>=.005 else 'stop_blind_budget_extension',
        training_state='stopped_at_complete_150_epoch_checkpoint',image_training='not_started',
        caveat='4 validation and4 reconstructed train observable initial states; no significance or exact historical PhysX-state claim')
    write(maintenance/'decision.json',decision)
    ledgers=[json.loads(p.read_text()) for p in maintenance.glob('artifact_ledger*.json') if json.loads(p.read_text()).get('state')=='completed']
    rows=[x for ledger in ledgers for x in ledger['files']]
    model_names=lambda p:bool(re.fullmatch(r'(?:(?:[bc]|b[01])_)?(?:best|last|diagnostic)\.pt|milestone_\d+\.pt|continued_\d+\.pt|checkpoint\.pt',Path(p).name))
    deleted_model_files=sum(model_names(x['path']) for x in rows)
    changes=request['reviewed_changes']
    kit=stage['regression'].get('kit',{})
    live_passed=isinstance(kit,dict) and kit.get('valid',0)>0
    kit_text=(f"通过；首plan最大差{kit.get('first_plan_max_abs_rad')} rad" if live_passed else
        '尚未通过。额外2次启动均未取得有效episode：首次为整理引入的ElementTree漏导入（首次模型动作前失败，已修复并通过真实资产初始化测试）；修复后的唯一补充启动在场景创建前FileThread SIGABRT（exit−6）。全部进程/GPU资源已回收，停止自动重试。两次原始日志与状态保留。')
    if not live_passed and not (isinstance(kit,dict) and kit.get('actual_starts')==2 and kit.get('first_failure')):
        kit_text='未验证通过：'+str(kit)
    summary=dict(state='completed' if live_passed else 'completed_cleanup_live_acceptance_blocked',deleted_files=sum(x['deleted_files'] for x in ledgers),
        deleted_model_checkpoint_paths=deleted_model_files,
        other_deleted_torch_artifacts=sum(Path(x['path']).suffix in ('.pt','.pth','.ckpt') and not model_names(x['path']) for x in rows),
        released_allocated_bytes=sum(x['released_blocks_bytes'] for x in ledgers),
        removed_empty_directories=sum(len(x['removed_empty_directories']) for x in ledgers),
        deleted_python_source_files=sum(v=='D' and p.endswith('.py') and p.startswith(('tools/','visuotactile/')) for p,v in changes.items()),
        retired_test_files=sum(v=='D' and p.startswith('tests/') for p,v in changes.items()),
        retired_documents=sum(v=='D' and p.endswith('.md') for p,v in changes.items()),
        review_commit=request['review_commit'],cutover_commit=stage['cutover_commit'],
        regression=stage['regression'],decision=decision,
        live_acceptance_passed=live_passed,
        ledger_paths=[str(p) for p in maintenance.glob('artifact_ledger*.json')],
        retained_reasons=['original400 and split/index/statistics/current caches','latest/best/milestone complete resumable State and current Image10000',
            'paired final/interim/expert traces and necessary videos','single isolated Kit cache source',
            'unknown data/checkpoints and small upstream reference/provenance files','representative unique historical crash logs'],
        storage_note='Allocated blocks released on final hardlink removal; directory and checkpoint counts are paths, not independent models; df includes unrelated activity')
    previous_request=json.loads((maintenance/'artifact_request_remaining.json').read_text())
    summary['removed_retired_experiment_roots']=sum(not Path(x['path']).exists() for x in previous_request['candidates'])
    summary['retained_checkpoint_sha256']=stage['regression']['checkpoint_sha256']
    write(CODE/'research_docs/evidence/cleanup_summary.json',summary)
    lines=['# 清理与回归验收','',f"实际释放 **{summary['released_allocated_bytes']/2**30:.3f} GiB**；删除{summary['deleted_files']}文件、{summary['deleted_model_checkpoint_paths']}模型checkpoint路径（含别名）、{summary['other_deleted_torch_artifacts']}其他Torch缓存/初始状态文件。完整移除{summary['removed_retired_experiment_roots']}个已淘汰实验/数据根目录，累计移除{summary['removed_empty_directories']}个空目录。硬链接只在最后一条链接移除时计算物理空间，不用逻辑大小冒充释放量。",'',
        f"当前分支删除{summary['deleted_python_source_files']}个研究源码文件、{summary['retired_test_files']}个旧方案专属测试文件、{summary['retired_documents']}份重复/过时Markdown。Git历史保留旧代码；不是抹去失败结论。",'',
        '公共checkpoint/EMA/AdamW/RNG逻辑集中至`visuotactile.checkpointing`；进程与durable JSON至`tools.runtime`；视频至`visuotactile.video`；FK/轨迹统计至`tools.trajectory_metrics`。训练由anchor_continue按job配置，独立Kit worker统一为anchor_continue_worker，报告由anchor_interim_analysis维护。','',
        f"实际验证：{stage['regression']['tests_passed']}项保留测试全部通过；真实State16584/Image10000 CPU checkpoint、EMA/AdamW/采样状态一致，20个真实条件/预测/图像张量字段逐值一致；整理后当前State和Image的实际CUDA恢复与历史合同检查通过（0优化更新）；原控制/环境源码保护，视频真实编码/完整解码，Markdown本地链接检查通过。",
        f"整理后额外闭环验收：{kit_text} 这些是代码整理验收，单独记录，不加入150 epoch的train4/validation4分母；原8次均有效、软件缺失0。",'',
        '保留400成功数据、当前320/40/40索引、现用numeric/DINO缓存及train统计；当前/最佳/150里程碑完整恢复状态、Image10000与本轮三阶段比较的精简数据。旧目录中的仍用缓存不能整目录清理。来源不明的data/checkpoints、小型upstream refs、未知符号链接和代表性崩溃证据保留。',
        '大视频、HDF5和权重留本地，不上传。必要视频与实测结果见[统一结果](STATE_DP_RESULTS.md)。原Kit关闭异常、Image live strict pairing未解决和历史资源违规均保持真实记录。','',
        f"源码审查commit `{request['review_commit']}`；切入commit `{stage['cutover_commit']}`。完整路径/大小/引用/删除理由、逐文件日志、before/after冻结SHA、CPU/CUDA结果在 `{maintenance}`。",'',
        '[精简机器摘要](evidence/cleanup_summary.json) · [当前入口](README.md) · [历史结论](RESEARCH_HISTORY.md)','']
    (CODE/'research_docs/CLEANUP_AUDIT.md').write_text('\n'.join(lines))
    report=CODE/'research_docs/STATE_DP_RESULTS.md'
    text=report.read_text()+f"\n## 150 epoch后的实测决策\n\n100→150端点validation selection MSE相对改善{gain*100:.3f}%；Train总loss比值{train_loss_ratio:.4f}，Validation总loss比值{val_loss_ratio:.4f}。训练拟合初态成功{train['success']}/{train['valid']}、验证成功{validation['success']}/{validation['valid']}；两组分开，不声称统计显著。\n\n"
    for cohort,label in [('train','训练320'),('validation','验证40')]:
        previous=old['cohorts'][cohort]['windows']['first12'];current=new['cohorts'][cohort]['windows']['first12']
        text+=f"{label}首12tick：Arm位置RMSE {previous['arm_rmse_rad']:.6g}→{current['arm_rmse_rad']:.6g} rad，增量RMSE {previous['arm_delta_rmse_rad']:.6g}→{current['arm_delta_rmse_rad']:.6g} rad/tick；静止保持基线{current['arm_hold_rmse_rad']:.6g} rad，专家增量RMS {current['arm_motion']['expert_increment_rms']:.6g} rad/tick。末端误差{previous['ee_rmse_mm']:.4f}→{current['ee_rmse_mm']:.4f} mm，累积误差{previous['ee_integrated_error_mm_s']:.5g}→{current['ee_integrated_error_mm_s']:.5g} mm·s。这些量用于区分静态位置拟合与小幅连续运动，不能只看全局RMSE。\n\n"
    text+=f"150端点Validation−Train gap：Arm {new['train_validation_gap']['arm_rmse_rad']:.6g} rad，Hand {new['train_validation_gap']['hand_rmse_rad']:.6g} rad，FK {new['train_validation_gap']['ee_rmse_mm']:.5g} mm。"
    if train_loss_ratio<.98 and val_loss_ratio>=.98:
        text+='训练loss继续下降而验证总loss未同步改善，支持当前泛化/监督到闭环之间存在瓶颈；不等于证明优化已完全收敛。'
    elif abs(train_loss_ratio-1)<.02 and abs(val_loss_ratio-1)<.02:
        text+='两个端点的总loss均近似平台；有限端点不能证明数学意义的收敛。'
    else:text+='训练/验证变化需连同首12tick和闭环阶段判读；有限端点不能证明已经完全收敛。'
    text+='\n\n'
    if decision['recommendation']=='stop_blind_budget_extension':
        text+='当前证据支持停止单纯增加到200–300的训练量，保留150及最佳完整checkpoint。小样本闭环和当前验证曲线尚未支持继续消耗预算；Image保持未续训，也未绕过严格live输入配对问题。网络/损失不自动修改。下一步最有价值的是针对首12tick小幅动作、累计末端偏差与稳定接触做固定初态的有界因果比较，优先验证观测分布与动作精度，而不是再从头训练两个模型。\n'
    else:
        text+='验证与任务阶段出现实际进展，可讨论在原scheduler和完整状态上作有界200epoch对照；当前仍停在完整150 checkpoint，不自动启动Image或更改网络/损失。四初态不足以宣布显著改进。\n'
    text+='\n训练初态仅恢复记录中的可观察初态与已导出状态，不能声称恢复原四lane的全部PhysX隐藏状态。清理和实际回归见[CLEANUP_AUDIT](CLEANUP_AUDIT.md)。\n'
    window=new['cohorts']['validation']['windows']['first12']
    motion=window['arm_motion']
    text+=f"\n## 失败机制的证据边界\n\n验证首12tick的Arm位置误差比静止保持基线还高{100*(window['arm_rmse_rad']/window['arm_hold_rmse_rad']-1):.1f}%；动作增量误差为专家增量RMS的{window['arm_delta_rmse_rad']/motion['expert_increment_rms']:.2f}倍，方向cos={motion['cosine']:.3f}、幅值比={motion['amplitude_ratio']:.3f}。因此不能把较低的全程绝对位置RMSE解释为已经学会精确小幅连续运动，也不能简单称模型只输出静止动作。训练集首12tick方向和幅值确有改善，验证差距仍大。\n\n"
    validation_rows=loop['cohorts']['validation']['rows']
    text+=f"150验证初态首次持续5mm偏离tick为{[x['new']['first_ee_divergence_tick'] for x in validation_rows]}，均早于专家接触及首次tick12重规划。单独消除重规划边界跳变不能解释最早偏离。\n\n"
    text+='|验证4公共专家前缀，episode宏平均，rad|原10000|150 epoch|配对专家|\n|---|---:|---:|---:|\n'
    for key,label in [('arm_jump_boundary_rms_rad','Arm边界跳变RMS'),('hand_jump_boundary_rms_rad','Hand边界跳变RMS'),('hand_tracking_rms_rad','Hand目标/实际跟踪RMS')]:
        values=[sum(x['common_prefix_metrics'][part][key] for x in validation_rows)/len(validation_rows) for part in ('old','new','expert')]
        text+='|'+label+'|'+'|'.join(f'{v:.6f}' for v in values)+'|\n'
    text+='\n成功专家的手指跟踪RMS本身更大，不能把目标/实际差直接称执行接口bug；边界跳变仍高于专家，但这里只是相关性证据。现有证据较强地支持训练/验证差距与接触前动作精度问题；闭环分布偏移、手指接触失稳的因果权重仍未分离。下一轮先保持checkpoint、网络、loss与控制不变，针对同一首12tick的实际在线观测和专家观测作输入差异归因，确认误差来源后再安排单因素有界对照。当前没有证据要求立即改网络或重新训练。\n'
    text+=f"\n## 整理后的执行验收\n\n{kit_text} 这是模型性能评估完成之后发生的独立验收，不解释或改变原8次闭环失败。\n"
    report.write_text(text)
    acceptance='整理后闭环回归通过' if live_passed else '整理后完整Kit回归仍未通过，失败证据保留；不自动重试'
    handoff=CODE/'research_docs/SESSION_HANDOFF.md';handoff.write_text(handoff.read_text()+f"\n最终验收：150训练、8初态与统一端点分析完成；清理切入、CPU测试和CUDA恢复完成。{acceptance}。建议`{decision['recommendation']}`，训练已安全停在20918，Image未新训。实际结果/最终commit看统一报告及评估root/maintenance/finalization_recovery.json；原finalization.json保留初次blocked证据。\n")
    return summary


def finalize(request_path, ledger):
    """Durable post-assessment cutover of an already-reviewed source commit."""
    request=json.loads(request_path.read_text());root=Path(request['assessment_root']);maintenance=root/'maintenance'
    owner_lock=(maintenance/'finalization.lock').open('a')
    fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not ledger.exists(), 'Inspect the previous finalization before an explicit recovery; never overwrite its evidence'
    session=json.loads((root/'session.json').read_text());parent=Path(session['parent_root'])
    stage=dict(state='running',phase='waiting_existing_assessment',pid=os.getpid(),started=time.time());monitor=None
    def status(phase,**extra):stage.update(phase=phase,heartbeat=time.time(),**extra);write(ledger,stage)
    def run(command,name,timeout=1800):
        timeout=min(timeout,request['deadline']-time.time())
        if timeout<=0:raise TimeoutError('Finalization deadline reached before '+name)
        status(name);path=maintenance/(name+'.log')
        with path.open('a') as log:
            worker=subprocess.Popen(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=worker.wait(timeout=timeout)
            except BaseException:
                if worker.poll() is None:
                    os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
                raise
        assert code==0, f'{name} exited {code}; see {path}'
    runner=['bash','visuotactile/run_cpu.sh']
    try:
        status('waiting_existing_assessment')
        while True:
            if time.time()>request['deadline']:raise TimeoutError('Original bounded assessment/finalization window exhausted')
            if (root/'owner_exit.json').exists():
                owner=json.loads((root/'owner_exit.json').read_text())
                assert owner['state']=='completed', 'Existing evaluation did not complete; no source cutover'
            service=subprocess.check_output(['systemctl','--user','show',request['analysis_service'],
                '--property=ActiveState,ExecMainStatus'],text=True)
            state=dict(line.split('=',1) for line in service.splitlines() if '=' in line)
            assert state.get('ActiveState')!='failed' and state.get('ExecMainStatus','0')=='0', 'Existing analysis service failed'
            if (root/'offline/progress.json').exists() and json.loads((root/'offline/progress.json').read_text())['state']=='completed' and state.get('ActiveState')=='inactive':break
            status('waiting_existing_assessment');time.sleep(20)
        status('verify_completed_inputs')
        summary=json.loads((root/'summary.json').read_text())
        assert summary['planned_episodes']==8 and summary['pending_initials']==0 and summary['acceptance_failed']==0
        assert all(summary['by_cohort'][c]['valid']==4 for c in ('train','validation'))
        cp=json.loads((root/'checkpoint.json').read_text());assert cp['updates']==20918 and digest(Path(cp['path']))==cp['sha256']
        assert digest(maintenance/'refactor_numerical_parity.json')==request['cpu_parity_sha256']
        assert json.loads((maintenance/'refactor_numerical_parity.json').read_text())['passed']
        state_paths=[parent/'state_train'/n for n in ('last.pt','best.pt','continued_10918.pt')]
        state_paths.append(Path(session['checkpoints']['image']['path']))
        before_weights={str(p):digest(p) for p in state_paths}
        pause=json.loads((root/'parent_pause.json').read_text());pid=pause['pid']
        procpath=Path('/proc')/str(pid)
        if procpath.exists():
            fields=(procpath/'stat').read_text().rsplit(')',1)[1].split()
            assert int(fields[19])==pause['start_ticks'] and fields[0]=='T'
            assert b'tools.anchor_continue_queue' in (procpath/'cmdline').read_bytes()
        # The independently completed checkpoint exists; no active trainer is signalled.
        worker=session['parent_worker'];workerpath=Path('/proc')/str(worker['pid'])
        exit_code=None
        if workerpath.exists():
            fields=(workerpath/'stat').read_text().rsplit(')',1)[1].split()
            assert int(fields[19])==worker['start_ticks'] and fields[0]=='Z'
            exit_code=os.waitstatus_to_exitcode(int(fields[49]));assert exit_code==0
        main_pid=int(subprocess.check_output(['systemctl','--user','show',request['parent_service'],'--property=MainPID','--value'],text=True).strip())
        assert main_pid in (0,pid), 'Parent service identity changed'
        run(['systemctl','--user','stop',request['parent_service']],'stop_completed_parent',180)
        parent_status=json.loads((parent/'status.json').read_text())
        parent_status.update(state='completed_state150_for_evidence_review',stage='state_train_epoch_150',pid=None,worker_pid=None,finished=time.time())
        parent_status['model_progress']['state']=json.loads((parent/'state_train/progress.json').read_text())
        write(parent/'status.json',parent_status);write(parent/'owner_exit.json',parent_status)
        train_stage=parent/'stages/state_train_epoch_150';result_path=sorted(train_stage.glob('attempt_*/output/result.json'))[-1]
        result=json.loads(result_path.read_text());assert result['passed'] and result['update']==20918
        record=dict(name='state_train_epoch_150',valid=True,result=result,resources_released=True,exit_code=exit_code,
            recovered_after_verified_owner_hold=True,output=str(result_path.parent),kit=False,finished=time.time())
        write(train_stage/'committed.json',record)
        # Preserve all generated results before applying the reviewed source change.
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).splitlines()
        allowed=request['allowed_generated_paths']
        for line in dirty:
            path=line[3:];assert any(path==p or path.startswith(p.rstrip('/')+'/') for p in allowed), 'Unexpected worktree modification: '+path
        if dirty:
            run(['git','add',*[line[3:] for line in dirty]],'commit_results_add')
            run(['git','diff','--cached','--check'],'commit_results_check')
            run(['git','commit','-m','Record State150 results and verified artifact cleanup before source consolidation'],'commit_results')
        run(['git','cherry-pick',request['review_commit']],'apply_reviewed_source')
        stage['cutover_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
        for path,change in request['reviewed_changes'].items():
            if change=='D':assert not (CODE/path).exists(),path
            else:assert (CODE/path).read_bytes()==subprocess.check_output(['git','show',request['review_commit']+':'+path],cwd=CODE),path
        migrate_freezes(parent,request['reviewed_changes'],maintenance/'freeze_migration')
        run(runner+['-m','pytest','-q','tests'],'postcleanup_tests',600)
        test_count=int(re.findall(r'(\d+) passed',(maintenance/'postcleanup_tests.log').read_text())[-1])
        (parent/'STOP_MONITOR').unlink(missing_ok=True)
        with (maintenance/'regression_monitor.log').open('a') as log:
            monitor=subprocess.Popen(runner+['-m','visuotactile.state_resources','--root',str(parent)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for job in ('state','image'):
            output=maintenance/('cuda_contract_'+job)
            run(runner+['-m','visuotactile.anchor_continue','--root',str(parent),'--output',str(output),'contract','--job',job],'cuda_contract_'+job)
            result=json.loads((output/'result.json').read_text());assert result['passed'] and result['optimizer_updates']==0
        assert before_weights=={str(p):digest(p) for p in state_paths}, 'Regression changed a training checkpoint'
        stage['regression']=dict(tests_passed=test_count,cpu='20 real checkpoint fields exact before/after',
            cuda_state='passed',cuda_image='passed',optimizer_updates=0,checkpoint_hashes_unchanged=True,checkpoint_sha256=before_weights)
        # One separate post-cleanup acceptance, only if the original recovery slot is unused.
        if summary['actual_starts']<session['kit_maximum_starts']:
            regression=root/'code_regression'
            run(runner+['-m','tools.anchor_interim_eval','--root',str(regression),'--parent',str(parent),'--epoch','150',
                '--ordinals','4','--boundary-seconds','0','--checkpoint',cp['path'],'prepare'],'prepare_code_regression')
            config=json.loads((regression/'session.json').read_text());config.update(kit_maximum_starts=1,deadline=min(request['deadline'],time.time()+1800),purpose='postcleanup acceptance; excluded from train4/validation4')
            write(regression/'session.json',config);(regression/'private_runtime').symlink_to(root/'private_runtime')
            run(runner+['-m','tools.anchor_interim_eval','--root',str(regression),'run'],'postcleanup_closed_loop',1900)
            check=json.loads((regression/'summary.json').read_text());assert check['new']['valid']==1 and not check['acceptance_failed']
            import numpy as np
            new_trial=check['rows'][0];old_trial=next(x for x in summary['rows'] if x['result']['trials'][0]['identity']==new_trial['result']['trials'][0]['identity'])
            identity=new_trial['result']['trials'][0]['identity']
            with np.load(Path(new_trial['output'])/identity/'plans.npz') as a,np.load(Path(old_trial['output'])/identity/'plans.npz') as b:
                difference=float(np.max(np.abs(a['plans'][0]-b['plans'][0])))
            assert difference<2e-5, ('Postcleanup first-plan regression',difference)
            stage['regression']['kit']=dict(valid=1,first_plan_max_abs_rad=difference,video=check['rows'][0]['video'],excluded_from_performance=True)
        else:stage['regression']['kit']='No remaining start budget; original8 validated episodes plus unchanged worker function bodies and actual video tests retained'
        (parent/'STOP_MONITOR').touch();monitor.wait(timeout=20);monitor=None
        # The train50 smoke/50-epoch intermediates are now superseded. Keep the
        #100/150 comparison and latest/best payloads; only explicit files enter this plan.
        intermediate=[p for p in (parent/'state_train').glob('continued_*.pt') if p.name not in ('continued_6584.pt','continued_10918.pt')]
        prune_request=maintenance/'artifact_request_intermediate.json';prune_ledger=maintenance/'artifact_ledger_intermediate.json'
        write(prune_request,dict(allowed_root=str((parent/'state_train').resolve()),
            protected={str(p.resolve()):'Latest/best/150 complete recovery checkpoint' for p in state_paths},
            candidates=[dict(path=str(p),reason='Completed supervision/50-epoch checkpoint superseded; numerical history retained',references=['completed150 plus full CUDA restoration'],hash=True) for p in intermediate]))
        plan(prune_request,prune_ledger);apply(prune_ledger)
        run(runner+['-m','tools.anchor_interim_analysis','--root',str(root)],'regenerate_consolidated_report')
        assert not markdown_links(),markdown_links()
        stage['regression']['markdown_links']='passed'
        summary=final_summary(root,request,stage)
        assert not markdown_links(),markdown_links()
        run(['git','diff','--check'],'final_diff_check')
        run(['git','add','research_docs/STATE_DP_RESULTS.md','research_docs/CLEANUP_AUDIT.md','research_docs/SESSION_HANDOFF.md','research_docs/evidence/state_dp','research_docs/evidence/cleanup_summary.json'],'final_report_add')
        run(['git','commit','-m','Report State150 decision and measured cleanup regression'],'final_report_commit')
        run(['git','push','origin','HEAD:feature/native-dp-training-v2'],'publish_reviewed_results',180)
        stage['final_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
        run(['git','worktree','remove',request['review_worktree']],'remove_review_worktree',180)
        status('completed',state='completed',finished=time.time(),summary=summary)
    except BaseException:
        stage.update(state='blocked',error=traceback.format_exc(),finished=time.time());write(ledger,stage);raise
    finally:
        if monitor:
            (parent/'STOP_MONITOR').touch()
            try:monitor.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(monitor.pid,signal.SIGTERM);monitor.wait(timeout=20)
        owner_lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['sources','plan','apply','checkpoint-probe','finalize'])
    p.add_argument('--entries',nargs='+');p.add_argument('--request',type=Path);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--session-root',type=Path)
    p.add_argument('--state-checkpoint',type=Path)
    a=p.parse_args()
    if a.mode=='sources':r=source_inventory(a.entries);write(a.ledger,r);print(json.dumps(dict(reachable=len(r['reachable']),unreachable=len(r['unreachable']),errors=r['parse_errors'])))
    elif a.mode=='checkpoint-probe':print(json.dumps(checkpoint_probe(a.session_root,a.ledger,a.state_checkpoint)))
    elif a.mode=='finalize':finalize(a.request,a.ledger)
    else:print(json.dumps(plan(a.request,a.ledger) if a.mode=='plan' else apply(a.ledger)))
