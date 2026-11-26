"""One finite viewing queue, using the existing Tight environment and recorder."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, supervised_owner, kill_owned, proc


def beat(root, **fields):
    previous = read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json', dict(previous, **fields, updated_utc=time.time()))


def sources():
    return {str(p.relative_to(CODE)): sha(p) for folder in ('tools', 'visuotactile')
            for p in sorted((CODE/folder).rglob('*.py'))}


def plateau_evidence(rows, progress, rules):
    """The existing training plateau criterion; no new relaxed convergence rule."""
    scores = [float(row['metrics']['selection_score']) for row in rows]
    finite = bool(scores) and all(math.isfinite(v) and v >= 0 for v in scores)
    count = rules['plateau_validations']
    recent = scores[-count:]
    gain = 1-min(recent)/max(recent[0], 1e-12) if finite and len(recent) == count else None
    severe = finite and len(scores) >= 5 and all(v > 2*min(scores[:-3]) for v in scores[-3:])
    epochs = float(progress.get('anchor_equivalent_epochs', 0))
    passed = (finite and math.isfinite(epochs) and epochs >= rules['minimum_anchor_epochs']
              and gain is not None and gain < rules['plateau_relative_improvement'] and not severe)
    return dict(passed=passed, anchor_equivalent_epochs=epochs, recent_scores=recent,
                recent_relative_improvement=gain, severe_degradation=severe,
                rule=rules, scope='Finite validation plateau, not proof of global convergence or task success')


def evaluation_counts(root):
    cases = read(root/'plan.json'); rows = []
    started = 0
    for case in cases:
        directory = root/case['identity']
        started += (directory/'started.json').exists()
        if (directory/'result.json').exists(): rows.append(read(directory/'result.json'))
    legal = [r for r in rows if r.get('software_complete') and r['legal']]
    return dict(planned=len(cases), physical_attempts=started, completed=len(rows), valid=len(legal),
        successes=sum(bool(r['success']) for r in legal), initial_rejections=sum(not r['legal'] for r in rows),
        incomplete_attempts=started-len(rows), unstarted=len(cases)-started,
        success_rate=sum(bool(r['success']) for r in legal)/len(legal) if legal else None,
        native_lift=sum(bool(r['native_lift']) for r in legal),
        first_subgoal=sum(bool(r['native_subgoals']) for r in legal),
        videos_verified=sum((root/c['identity']/'video_verified.json').exists() for c in cases))


def prepare_after_training(root, config, parent):
    from visuotactile.narrow_data import direct_configuration
    parent = parent.resolve(); base = read(config); session = read(parent/'session.json')
    rules = base['conditional_evaluation']
    training = session['training']
    for key in ('minimum_anchor_epochs', 'plateau_validations', 'plateau_relative_improvement'):
        if rules[key] != training[key]: raise ValueError('Keep the frozen training plateau criterion: '+key)
    if not 1 <= rules['maximum_attempts'] <= 10 or not 1 <= rules['record_first_attempts'] <= 3:
        raise ValueError('Bounded evaluation/video queue required')
    if rules['maximum_seconds'] > 3600: raise ValueError('Evaluation may not exceed one hour')
    if session['source_head'] != rules['training_source_head'] or not session.get('palm_training'):
        raise ValueError('Unexpected training source')
    if sha(session['expert_checkpoint']) != session['expert_sha256']:
        raise ValueError('Frozen expert identity changed')
    if CODE.resolve() == Path(session['cwd']).resolve():
        raise ValueError('Use an isolated worktree; running training sources must remain frozen')
    # The post-training entry changes; model, environment and control implementations do not.
    for filename, digest in read(parent/'execution_source_freeze.json').items():
        relative = Path(filename).relative_to(session['cwd'])
        if relative.parts[0] != 'tools' and sha(CODE/relative) != digest:
            raise ValueError('Training/deployment implementation differs: '+str(relative))
    plan = []
    for index in range(rules['maximum_attempts']):
        seed = rules['seed_start']+index; identity = f'palm_fresh_{index:03d}'
        spec = dict(identity=identity, family=identity, split='new', cohort='fresh', seed=seed,
            collision_gate='live_collision_mesh_bounds_v1', narrow_reset=direct_configuration(session['domain'], seed))
        plan.append(dict(identity=identity, spec=spec, record_video=index < rules['record_first_attempts']))
    root.mkdir(parents=True, exist_ok=False)
    cfg = dict(source_root=str(parent), checkpoint='training/best.pt', checkpoint_sha256=None,
        maximum_seconds=rules['maximum_seconds'], inference=base['inference'], resources=base['resources'],
        after_training=rules, wait_deadline=session['deadline']+120,
        absolute_deadline=session['deadline']+120+rules['maximum_seconds'],
        purpose='Conditional fresh-start success estimate with prospectively selected videos; never used for model selection')
    write(root/'configuration.json', cfg); write(root/'plan.json', plan); write(root/'source_session.json', session)
    write(root/'inputs.json', {str(p): sha(p) for p in
        (Path(session['expert_checkpoint']), parent/'configuration.json', parent/'manifest.json',
         parent/'state_data/data.json', parent/'state_data/statistics.pt')})
    beat(root, state='PREPARED', stage='conditional_evaluation_prepared', completed=0, total=len(plan),
         exit_code=None, reason='Wait for predecessor exit; require the original >=200 epoch / last4 <1% plateau')
    report(root)


def await_training(root, cfg):
    parent = Path(cfg['source_root'])
    reported = False
    while True:  # Ordinary CPU owner only; no Kit, model loading, or GPU monitor during the wait.
        if (root/'STOP').exists(): raise InterruptedError('Owned evaluation STOP')
        if time.time() >= cfg['wait_deadline']: raise TimeoutError('Predecessor completion deadline')
        status = read(parent/'status.json')
        progress = read(parent/'training/progress.json')
        if status['state'] == 'FAILED' or (parent/'finish_error.json').exists():
            raise RuntimeError('Training/offline predecessor failed; evaluation dependency not satisfied')
        owners = read(parent/'active_owner.json')
        active = []
        for name, record in owners.items():
            current = proc(record['pid']) if record else None
            if current and current['start_ticks'] == record['start_ticks']: active.append(name)
        done = status['state'] in ('COMPLETE', 'BUDGET_STOP') and status['stage'] == 'trained_and_offline_evaluated'
        beat(root, state='WAITING', stage='waiting_for_training_and_offline_exit', pid=os.getpid(),
             source_updates=progress['update'], source_anchor_epochs=progress['anchor_equivalent_epochs'],
             active_predecessor_processes=active, reason='Ordinary CPU dependency wait; no evaluation GPU work started')
        if not reported and (root/'configuration.json').exists():
            report(root); reported = True
        if done and status.get('owned_workers_exited') and not active: break
        if not active: raise RuntimeError('Predecessor exited without successful training/offline completion')
        time.sleep(30)
    rows = [json.loads(line) for line in (parent/'training/validation.jsonl').read_text().splitlines()]
    evidence = plateau_evidence(rows, progress, cfg['after_training'])
    write(root/'eligibility.json', evidence)
    if not evidence['passed']:
        beat(root, state='SKIPPED', stage='validation_plateau_not_reached', exit_code=0,
             reason='Original plateau condition was not met within the training budget; no physical starts')
        return False
    finished = read(parent/'training/finished.json'); offline = read(parent/'offline/result.json')
    if not offline['complete'] or offline['episodes'] != 38 or finished['updates'] != progress['update']:
        raise ValueError('Incomplete predecessor receipts')
    checkpoint = parent/cfg['checkpoint']; digest = sha(checkpoint)
    if digest != offline['checkpoint_sha256']: raise ValueError('Offline-selected checkpoint changed')
    write(root/'selected_checkpoint.json', dict(path=str(checkpoint), sha256=digest,
        updates=offline['selected_update'], selection='Frozen validation4; closed-loop results never select weights'))
    protected = read(root/'inputs.json')
    protected.update({str(p): sha(p) for p in
        (checkpoint, parent/'training/configuration.json', parent/'training/finished.json', parent/'offline/result.json')})
    write(root/'inputs.json', protected)
    return True


def evaluation_report(root):
    status = read(root/'status.json'); cfg = read(root/'configuration.json'); counts = evaluation_counts(root)
    write(root/'evaluation_summary.json', dict(counts, state=status['state'], exit_code=status.get('exit_code'),
        reason=status.get('reason'), original_results_unchanged=True, selection_used_closed_loop=False))
    selected = read(root/'selected_checkpoint.json') if (root/'selected_checkpoint.json').exists() else None
    text = ['# StateDP 掌部几何模型：有条件的小规模成功率与视频', '',
        f"状态 **{status['state']}**：{status.get('reason', '')}",
        '等待原训练和离线诊断结束；仅达到原≥200等效epoch、最后4次validation分数改善<1%时执行。',
        '冻结10个新初态，不按结果挑seed；最多1小时，首次3个身份录制三视角60fps，非法初态不补录。',
        '原生完整成功判据不变；不同进度的自主/专家轨迹不按同tick比较动作标签。', '',
        f"已尝试{counts['physical_attempts']}，完成{counts['completed']}，合法{counts['valid']}，"
        f"成功{counts['successes']}，初态拒绝{counts['initial_rejections']}，中断{counts['incomplete_attempts']}，未启动{counts['unstarted']}。",
        f"完整成功率：{counts['successes']}/{counts['valid']}" if counts['valid'] else '完整成功率：未测。',
        f"原生抬升{counts['native_lift']}，首子目标{counts['first_subgoal']}；视频验收{counts['videos_verified']}。",
        f'选定模型：{selected}' if selected else '模型等待最终validation选择；未加载训练中的可变checkpoint。', '',
        '|身份 / seed|真实结果|已完整解码视频|', '|---|---|---|']
    for case in read(root/'plan.json'):
        path = root/case['identity']; outcome = '未运行'; link = '未生成'
        if (path/'result.json').exists():
            row = read(path/'result.json')
            outcome = '初态拒绝' if not row['legal'] else f"{'成功' if row['success'] else '失败'}；{row['steps']}步；{row['reasons']}"
        elif (path/'started.json').exists(): outcome = '已开始，尚无完整结果'
        if (path/'video_verified.json').exists(): link = f"[MP4]({case['identity']}/rollout.mp4)"
        text.append(f"|{case['identity']} / {case['spec']['seed']}|{outcome}|{link}|")
    text += ['', '本队列不会扩成100例，不补seed、不重试，不训练或更新模型。',
             '实际整卡资源及越界见resource_summary.json；显存限制不是GPU利用率保证。', '']
    temporary = root/'README.md.tmp'; temporary.write_text('\n'.join(text)); temporary.replace(root/'README.md')
    begin, end = '<!-- tight-palm-evaluation:start -->', '<!-- tight-palm-evaluation:end -->'
    fragment = '\n'.join([begin, '## 2026-11-09 有条件成功率测试与视频', '',
        f"状态 **{status['state']}**；成功{counts['successes']}/合法{counts['valid']}，拒绝{counts['initial_rejections']}，"
        f"中断{counts['incomplete_attempts']}，未启动{counts['unstarted']}。无合法完成时成功率未测。",
        f"{status.get('reason', '')}；索引：`{root}/README.md`。",
        '仅原200epoch/last4改善<1%平台达成后使用最终validation选定模型，最多10次新初态/1h、前3身份视频；不扩跑。', end])
    targets = [CODE/'research_docs/STATE_DP_RESULTS.md']
    # Do not race the running predecessor's own report writer.
    if status['state'] in ('COMPLETE', 'FAILED', 'BUDGET_STOP', 'SKIPPED'):
        parent = Path(cfg['source_root']); state = read(parent/'status.json')
        if state.get('owned_workers_exited'):
            target = Path(read(root/'source_session.json')['cwd'])/'research_docs/STATE_DP_RESULTS.md'
            if target not in targets: targets.append(target)
    for target in targets:
        previous = target.read_text()
        if begin in previous: previous = previous[:previous.index(begin)]+fragment+previous[previous.index(end)+len(end):]
        else: previous += '\n\n'+fragment+'\n'
        tmp = target.with_suffix('.md.tmp'); tmp.write_text(previous); tmp.replace(target)


def prepare(root, config, remaining_from=None):
    if root.exists():
        raise FileExistsError('Existing viewing root: inspect its status; never restart automatically')
    cfg = read(config); cfg.pop('conditional_evaluation', None)
    parent = (CODE/cfg['source_root']).resolve()
    checkpoint = parent/cfg['checkpoint']
    if sha(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('Selected Core checkpoint identity changed')
    session = read(parent/'session.json')
    if sha(session['expert_checkpoint']) != session['expert_sha256']:
        raise ValueError('Expert identity changed')
    plan = {s['identity']: s for s in read(parent/'frozen_evaluation_plan.json')}
    selected = []
    for case in cfg['cases']:
        spec = plan[case['identity']]
        old = read(parent/'evaluation'/case['identity']/'result.json')
        if not old['software_complete'] or not old['legal'] or old['success']:
            raise ValueError('Viewing selection must refer to a completed legal historical failure')
        if old['inference'] != cfg['inference'] or spec['cohort'] != 'fresh':
            raise ValueError('Viewing configuration differs from historical evaluation')
        selected.append(dict(case, spec=spec, historical_result=old))
    if len(selected) != 3 or len({r['identity'] for r in selected}) != 3:
        raise ValueError('Exactly three distinct viewing cases required')
    if remaining_from is not None:
        previous = remaining_from.resolve()
        if read(previous/'status.json')['state'] != 'FAILED' or 'GPU mean exceeded 50%' not in read(previous/'worker_error.json')['reason']:
            raise ValueError('Explicit remaining queue only supports the recorded resource interruption')
        if read(previous/'configuration.json') != cfg:
            raise ValueError('Original viewing configuration changed')
        selected = [case for case in selected if not (previous/case['identity']/'started.json').exists()]
        if len(selected) != 2:
            raise ValueError('Only the two never-started identities may continue; no retry/replacement')
        cfg = dict(cfg, recovery_of=str(previous), absolute_deadline=read(previous/'session.json')['deadline'])
        if time.time() >= cfg['absolute_deadline']: raise TimeoutError('Original viewing deadline exhausted')
        cfg['resources'] = dict(cfg['resources'], exceeded_mean_action='wait_for_original_admission')
    root.mkdir(parents=True)
    write(root/'configuration.json', cfg)
    write(root/'plan.json', selected)
    write(root/'source_session.json', session)
    files = [checkpoint, Path(session['expert_checkpoint']), parent/'manifest.json',
             parent/'state_data/data.json', parent/'state_data/statistics.pt', parent/'frozen_evaluation_plan.json']
    write(root/'inputs.json', {str(p): sha(p) for p in files})
    beat(root, state='PREPARED', stage='awaiting_source_freeze', completed=0, total=len(selected), exit_code=None,
         reason='队列已冻结，等待提交源码后启动')
    report(root)


def report(root):
    status = read(root/'status.json')
    cfg = read(root/'configuration.json')
    if cfg.get('after_training'):
        return evaluation_report(root)
    text = ['# Core StateDP 失败案例观看复跑', '',
        f"状态：{status['state']}；完成 {status['completed']}/{status['total']}。{status.get('reason', '')}", '',
        '按历史失败行为挑选；实际复跑结果见下表，不加入原正式评估分母。',
        '同 seed 不保证物理逐帧一致；完整三视角、60fps、1倍仿真时间，末次动作前画面结束，无重置画面。', '',
        '|原案例与历史行为|本次实际结果|已完整解码的视频|', '|---|---|---|']
    for case in read(root/'plan.json'):
        name = case['identity']; path = root/name
        actual = read(path/'result.json') if (path/'result.json').exists() else None
        outcome = '未完成'
        link = '未生成/未验收'
        if actual:
            outcome = ('初态拒绝' if not actual['legal'] else
                f"{'成功' if actual['success'] else '失败'}；{actual['steps']}步；"+
                ','.join(k for k, v in actual['reasons'].items() if v)+
                f"；lift={actual['native_lift']}；子目标={actual['native_subgoals']}")
        if (path/'video_verified.json').exists():
            link = f'[{name}.mp4]({name}/rollout.mp4)'
        text.append(f"|{name}：{case['historical_behavior']}|{outcome}|{link}|")
    text += ['', '模型8500更新EMA，身份见inputs.json；旧trial checkpoint字段误指专家，本次分别记录学生与专家SHA。',
             '实际资源见resource_summary.json；整卡上限不能由显存限额保证。', '']
    temporary = root/'README.md.tmp'; temporary.write_text('\n'.join(text)); temporary.replace(root/'README.md')
    if cfg.get('recovery_of'):
        previous = Path(cfg['recovery_of'])
        relative = os.path.relpath(root, previous)
        combined = ['# Core StateDP 观看视频', '',
            '[第一段三视角视频，592帧/9.87秒；资源中断，非自然终止](evaluation_formal_000/interrupted_592frames.mp4)', '',
            '首次录制在整卡60秒均值51.29%时停止；原FAILED记录保留。',
            '仅接续原队列中从未启动的两个案例，不重跑第一例、不增加seed，沿用原绝对截止时间。', '',
            f"剩余队列：{status['state']}，{status['completed']}/{status['total']}完成。", '']
        combined += [line.replace('](', ']('+relative+'/') for line in text[6:]]
        temporary = previous/'README.md.tmp'; temporary.write_text('\n'.join(combined)); temporary.replace(previous/'README.md')
    target = CODE/'research_docs/STATE_DP_RESULTS.md'
    begin, end = '<!-- tight-failure-videos:start -->', '<!-- tight-failure-videos:end -->'
    fragment = '\n'.join([begin, '## 2026-11-08 用户请求的三个失败案例视频', '',
        f"状态 **{status['state']}**，完成 {status['completed']}/{status['total']}；{status.get('reason', '')}。",
        f'观看索引：`{root}/README.md`；阶段/计数见同目录`status.json`。',
        '同Core8500 EMA与原控制/物理，三个按历史结果挑选的独立观看复跑；不改原27例分母，不训练、不补seed。', end])
    if cfg.get('recovery_of'):
        fragment = fragment.replace(end, '首次队列因整卡60s均值51.29%停止，保留592帧已解码中断视频；本次仅接续未启动两例，原截止时间不变。\n'+end)
    previous = target.read_text()
    if begin in previous:
        previous = previous[:previous.index(begin)]+fragment+previous[previous.index(end)+len(end):]
    else:
        previous += '\n\n'+fragment+'\n'
    temporary = target.with_suffix('.md.tmp'); temporary.write_text(previous); temporary.replace(target)


def worker(root):
    from visuotactile.state_resources import Lease
    cfg = read(root/'configuration.json'); session = read(root/'session.json')
    conditional = bool(cfg.get('after_training'))
    if sha(root/'configuration.json') != session['configuration_sha256'] or sha(root/'plan.json') != session['plan_sha256']:
        raise ValueError('Frozen configuration/plan changed')
    if sources() != read(root/'source_freeze.json'):
        raise ValueError('Frozen source changed before worker startup')
    active = None; last_beat = 0.
    class VideoLease(Lease):
        def check(self, deadline=None):
            nonlocal last_beat
            super().check(deadline)
            if shutil.disk_usage(root).free < cfg['resources']['min_disk_gib']*1024**3:
                raise RuntimeError('Disk reserve reached')
            monitor = read(root/'monitor_owner.json')['pid']
            if not Path('/proc', str(monitor)).exists():
                raise RuntimeError('Resource monitor exited')
            path = root/'resources/resources.json'
            if path.exists():
                resources = read(path); row = resources['latest']
                if time.time()-row['utc_epoch'] > 8:
                    raise RuntimeError('Resource telemetry stale')
                if any(v is not None and v > 50 for v in
                       (resources['gpu_time_weighted_mean_percent'], row.get('rolling_60s_percent'))):
                    if cfg['resources'].get('exceeded_mean_action') != 'wait_for_original_admission':
                        raise RuntimeError('Measured whole-card GPU mean exceeded 50%; stop only this viewing worker')
                    if not (root/'resource_cap_observed.json').exists():
                        write(root/'resource_cap_observed.json', dict(utc=time.time(), resources=resources,
                            action='Lease waits for batch, complete60s and recent means <=45; original deadline unchanged'))
            if time.monotonic()-last_beat >= 5:
                progress = root/active/'progress.json' if active else None
                fields = {}
                if progress is not None and progress.exists():
                    fields['observed_control_steps'] = read(progress)['tick']+1
                beat(root, worker_pid=os.getpid(), **fields); last_beat = time.monotonic()
    lease = VideoLease(root, deadline=session['deadline'])
    beat(root, stage='resource_admission_then_kit_initialization', worker_pid=os.getpid())
    from tools.tight_state import environment
    app, env, player, lease = environment(root, True, lease=lease)
    import torch
    from visuotactile.position_control import contract
    from visuotactile.state_benchmark import trial
    from visuotactile.video import verify_video
    checkpoint = (CODE/cfg['source_root']/cfg['checkpoint']).resolve()
    with lease.block(cuda=False):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        expected = read(root/'selected_checkpoint.json')['updates'] if conditional else 8500
        if saved['updates'] != expected or saved['ema_optimization_step'] != expected or saved['contract'] != contract(24, True):
            raise ValueError('Core EMA/target contract mismatch')
        if conditional:
            parent = Path(cfg['source_root']); selected = read(root/'selected_checkpoint.json')
            if sha(checkpoint) != selected['sha256'] or saved['loss_id'] != 'palm_command_sample_commit12':
                raise ValueError('Selected palm model identity mismatch')
            if saved['config_sha256'] != sha(parent/'training/configuration.json'):
                raise ValueError('Selected model configuration mismatch')
            if saved['configuration']['statistics_sha256'] != sha(parent/'state_data/statistics.pt'):
                raise ValueError('Selected train-only normalizer identity mismatch')
    for index, case in enumerate(read(root/'plan.json')):
        active = case['identity']; output = root/active
        beat(root, stage='recording', identity=active, observed_control_steps=0)
        record_video = case.get('record_video', True)
        result = trial(env, lease, output, case['spec'], {}, player, str(checkpoint),
            saved=saved, position=True, video=record_video, inference=cfg['inference'],
            expert_checkpoint=session['expert_checkpoint'],
            video_title=f"StateDP {'Palm' if conditional else 'Core'}{expected} EMA | seed {case['spec']['seed']} | {'fresh evaluation' if conditional else 'viewing rerun'} | 1x simulation time")
        if result['legal']:
            if not conditional and result['environment_sha256'] != case['historical_result']['environment_sha256']:
                raise ValueError('Environment configuration differs from reference')
            if conditional:
                # Direct-reset parameters differ by seed; the native task and controller sources stay frozen.
                errors = [result['actuator_target_max_error'], result['submitted_target_max_error']]
                if any(not math.isfinite(v) or v >= 1e-6 for v in errors):
                    raise ValueError('Native actuator/command interface mismatch')
            if record_video:
                verified = verify_video(output/'rollout.mp4', expected_frames=result['steps'])
                if verified['frame_rate'] != 60 or verified['shape'] != [316, 1344, 3]:
                    raise ValueError('Video dimensions/rate mismatch')
                write(output/'video_verified.json', dict(verified, sha256=sha(output/'rollout.mp4')))
        beat(root, completed=index+1, observed_control_steps=result.get('steps', 0))
        report(root)
    write(root/'result.json', dict(software_complete=True, completed=len(read(root/'plan.json')),
        formal_trials_added=0, new_evaluation=evaluation_counts(root) if conditional else None,
        reason='Fixed ten-attempt queue ends even if successful; zero success never expands the queue' if conditional else 'Viewing only'))
    from visuotactile.native_evaluation import exit_completed_kit_worker
    exit_completed_kit_worker(root)


@supervised_owner
def run(root):
    lock = (root/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monitor = child = None; code = 1; logs = []
    try:
        if read(root/'status.json')['state'] != 'PREPARED':
            raise ValueError('Run already started; no automatic restart')
        cfg = read(root/'configuration.json')
        for path, digest in read(root/'inputs.json').items():
            if sha(path) != digest: raise ValueError('Frozen input drift: '+path)
        # Results MD is updated by this queue itself; all runtime/config sources must be committed.
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'tools', 'visuotactile', 'configs'], cwd=CODE, text=True)
        if dirty: raise ValueError('Uncommitted runtime/configuration: '+dirty)
        frozen = sources()
        configuration_sha256, plan_sha256 = sha(root/'configuration.json'), sha(root/'plan.json')
        write(root/'owner_source_freeze.json', frozen)
        write(root/'owner_launch.json', dict(pid=os.getpid(), utc=time.time(), cwd=str(CODE), python=sys.executable,
            source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
            configuration_sha256=configuration_sha256,plan_sha256=plan_sha256))
        if cfg.get('after_training'):
            if not await_training(root, cfg):
                code = 0
                return code
            if sources() != frozen: raise ValueError('Evaluation source changed during dependency wait')
            if sha(root/'configuration.json') != configuration_sha256 or sha(root/'plan.json') != plan_sha256:
                raise ValueError('Evaluation configuration/queue changed during dependency wait')
            for path, digest in read(root/'inputs.json').items():
                if sha(path) != digest: raise ValueError('Frozen input drift after waiting: '+path)
        now = time.time(); deadline = min(now+cfg['maximum_seconds'], cfg.get('absolute_deadline', float('inf')))
        if now >= deadline: raise TimeoutError('Original viewing deadline exhausted')
        session = dict(read(root/'source_session.json'), started_utc=now, deadline=deadline,
            training_deadline=deadline, source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
            configuration_sha256=configuration_sha256, plan_sha256=plan_sha256,
            resource_target_percent=35, cwd=str(CODE), python=sys.executable, viewing_only=not bool(cfg.get('after_training')))
        write(root/'session.json', session); write(root/'source_freeze.json', sources())
        write(root/'gpu_schedule.json', cfg['resources'])
        beat(root, state='RUNNING', stage='starting_monitor', pid=os.getpid(),
             reason='Original plateau gate passed; fixed new-start evaluation, at most one hour' if cfg.get('after_training') else 'Frozen viewing queue; original one-hour deadline retained')
        report(root)
        for mode in ('monitor', 'worker'):
            log = (root/(mode+'.log')).open('a'); logs.append(log)
            args = ['-m', 'visuotactile.state_resources'] if mode == 'monitor' else ['-m', 'tools.tight_videos', 'worker']
            process = subprocess.Popen(RUN+args+['--root', str(root)], cwd=CODE,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if mode == 'monitor':
                monitor = process; write(root/'monitor_owner.json', dict(pid=monitor.pid))
            else: child = process
        beat(root, worker_pid=child.pid, monitor_pid=monitor.pid)
        code = child.wait(timeout=max(1, deadline-time.time()))
        if code != 0 or not (root/'result.json').exists():
            raise RuntimeError(f'Viewing worker exited {code}; see worker.log/worker_error.json')
        if sources() != read(root/'source_freeze.json'):
            raise ValueError('Runtime source drift')
        beat(root, state='COMPLETE', stage='videos_verified', exit_code=0,
             reason='Finite evaluation/video queue finished; no additional starts' if cfg.get('after_training') else 'Finite viewing queue finished; no additional starts')
    except BaseException as exc:
        code = 2 if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, InterruptedError)) else 1
        write(root/'error.json', dict(reason=str(exc), traceback=traceback.format_exc()))
        beat(root, state='BUDGET_STOP' if code == 2 else 'FAILED', exit_code=code, reason=str(exc))
    finally:
        if child is not None: kill_owned(child)
        if monitor is not None:
            (root/'STOP_MONITOR').touch()
            try: monitor.wait(timeout=15)
            except subprocess.TimeoutExpired: kill_owned(monitor)
        for log in logs: log.close()
        path = root/'resources/resources.json'
        if path.exists():
            write(root/'resource_summary.json', dict(read(path), configured_limits=cfg['resources'], hard_gpu_quota_guaranteed=False))
        beat(root, owned_worker_exited=child is None or child.poll() is not None,
             owned_monitor_exited=monitor is None or monitor.poll() is not None)
        report(root); lock.close()
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'prepare-after-training', 'run', 'worker', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CODE/'configs/tight_failure_videos.json')
    parser.add_argument('--remaining-from', type=Path)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode == 'prepare-after-training':
        if args.parent is None: parser.error('--parent is required')
        prepare_after_training(root, args.config, args.parent)
    elif args.mode == 'prepare': prepare(root, args.config, args.remaining_from)
    elif args.mode == 'status': print(json.dumps(read(root/'status.json'), indent=2))
    elif args.mode == 'run': raise SystemExit(run(root))
    else:
        try: worker(root)
        except BaseException as exc:
            write(root/'worker_error.json', dict(reason=str(exc), traceback=traceback.format_exc()))
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)


if __name__ == '__main__': main()
