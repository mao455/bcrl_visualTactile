"""One finite viewing queue, using the existing Tight environment and recorder."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, supervised_owner, kill_owned


def beat(root, **fields):
    previous = read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json', dict(previous, **fields, updated_utc=time.time()))


def sources():
    return {str(p.relative_to(CODE)): sha(p) for folder in ('tools', 'visuotactile')
            for p in sorted((CODE/folder).rglob('*.py'))}


def prepare(root, config, remaining_from=None):
    if root.exists():
        raise FileExistsError('Existing viewing root: inspect its status; never restart automatically')
    cfg = read(config); parent = (CODE/cfg['source_root']).resolve()
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
        if saved['updates'] != 8500 or saved['ema_optimization_step'] != 8500 or saved['contract'] != contract(24, True):
            raise ValueError('Core EMA/target contract mismatch')
    for index, case in enumerate(read(root/'plan.json')):
        active = case['identity']; output = root/active
        beat(root, stage='recording', identity=active, observed_control_steps=0)
        result = trial(env, lease, output, case['spec'], {}, player, str(checkpoint),
            saved=saved, position=True, video=True, inference=cfg['inference'],
            expert_checkpoint=session['expert_checkpoint'],
            video_title=f"StateDP Core8500 EMA | seed {case['spec']['seed']} | viewing rerun | 1x simulation time")
        if result['legal']:
            if result['environment_sha256'] != case['historical_result']['environment_sha256']:
                raise ValueError('Environment configuration differs from reference')
            verified = verify_video(output/'rollout.mp4', expected_frames=result['steps'])
            if verified['frame_rate'] != 60 or verified['shape'] != [316, 1344, 3]:
                raise ValueError('Video dimensions/rate mismatch')
            write(output/'video_verified.json', dict(verified, sha256=sha(output/'rollout.mp4')))
        beat(root, completed=index+1, observed_control_steps=result.get('steps', 0))
        report(root)
    write(root/'result.json', dict(software_complete=True, completed=len(read(root/'plan.json')), formal_trials_added=0))
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
        now = time.time(); deadline = min(now+cfg['maximum_seconds'], cfg.get('absolute_deadline', float('inf')))
        if now >= deadline: raise TimeoutError('Original viewing deadline exhausted')
        session = dict(read(root/'source_session.json'), started_utc=now, deadline=deadline,
            training_deadline=deadline, source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
            configuration_sha256=sha(root/'configuration.json'), plan_sha256=sha(root/'plan.json'),
            resource_target_percent=35, cwd=str(CODE), python=sys.executable, viewing_only=True)
        write(root/'session.json', session); write(root/'source_freeze.json', sources())
        write(root/'gpu_schedule.json', cfg['resources'])
        beat(root, state='RUNNING', stage='starting_monitor', pid=os.getpid(), reason='Frozen viewing queue; original one-hour deadline retained')
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
        beat(root, state='COMPLETE', stage='videos_verified', exit_code=0, reason='Finite viewing queue finished; no additional starts')
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
    parser.add_argument('mode', choices=['prepare', 'run', 'worker', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CODE/'configs/tight_failure_videos.json')
    parser.add_argument('--remaining-from', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode == 'prepare': prepare(root, args.config, args.remaining_from)
    elif args.mode == 'status': print(json.dumps(read(root/'status.json'), indent=2))
    elif args.mode == 'run': raise SystemExit(run(root))
    else:
        try: worker(root)
        except BaseException as exc:
            write(root/'worker_error.json', dict(reason=str(exc), traceback=traceback.format_exc()))
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)


if __name__ == '__main__': main()
