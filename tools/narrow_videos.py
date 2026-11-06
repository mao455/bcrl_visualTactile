"""Four fixed State200 viewing reruns; no training, data collection or retries.

The owner has a new fixed one-hour budget, runs independently, and validates all
video frames after its Kit worker exits. Original formal results stay read-only.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, proc, kill_owned
from tools.narrow_collection import freeze, check_freeze, environment

CASES = ('train_family_000', 'train_family_004', 'unseen_00', 'unseen_04')
INFERENCE = dict(inference_steps=16, clip_sample=False, sampler_seed=7091)


def status(root, **fields):
    previous = read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json', dict(previous, **fields, utc=time.time()))


def prepare(root, parent, demos):
    parent, demos = parent.resolve(), demos.resolve()
    previous = read(parent/'session.json')
    complete = read(parent/'evaluation/result.json')
    assert read(parent/'status.json')['state'] == 'complete' and complete['complete']
    assert read(parent/'status.json')['owned_workers_exited']
    checkpoint = parent/'training/best.pt'
    assert sha(checkpoint) == complete['checkpoint_sha256']
    indexed = {s['identity']: s for s in read(parent/'evaluation_plan.json')}
    plan = [indexed[k] for k in CASES]
    for spec in plan:
        result = read(parent/'evaluation'/spec['identity']/'result.json')
        assert result['legal'] and not result['success'] and result['software_complete']
    if demos.exists():
        raise FileExistsError(demos)
    root.mkdir(parents=True, exist_ok=False)
    demos.mkdir(parents=True, exist_ok=False)
    now = time.time()
    budget = dict(started=now, deadline=now+3600, maximum_worker_starts=1,
        maximum_episode_starts=4, maximum_output_bytes=4*1024**3,
        reserve_bytes=60*1024**3, maximum_no_progress_s=900,
        automatic_retries=False, training_updates=0, data_captures=0)
    write(root/'budget.json', budget)
    write(root/'session.json', dict(budget, version='narrow_failure_viewing_v1',
        parent=str(parent), demos=str(demos), checkpoint=str(checkpoint),
        checkpoint_sha256=complete['checkpoint_sha256'], updates=complete['updates'],
        expert_checkpoint=previous['expert_checkpoint'], expert_sha256=previous['expert_sha256'],
        resource_policy='shared_recent_v1', resource_target_percent=40.,
        purpose='fixed viewing reruns, excluded from formal counts and model selection',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()))
    write(root/'plan.json', plan)
    protected = [parent/'budget.json', parent/'manifest.json', checkpoint,
        parent/'training/last.pt', parent/'evaluation/result.json',
        CODE/'outputs/visuotactile/data/close_v4/manifest.json']
    protected += sorted((parent/'evaluation').glob('*/result.json'))
    stopped = parent.parent/'narrow_state_followup_20261106_01'
    protected += [stopped/'budget.json', stopped/'status.json']
    write(root/'protected.json', {str(p): sha(p) for p in protected})
    write(root/'input_hashes.json', {str(root/p): sha(root/p)
        for p in ('plan.json', 'session.json', 'budget.json', 'protected.json')})
    freeze(root)
    status(root, state='prepared', completed=0, owned_workers_exited=True)


def inputs_unchanged(root):
    check_freeze(root)
    for manifest in ('input_hashes.json', 'protected.json'):
        for path, digest in read(root/manifest).items():
            if sha(path) != digest:
                raise RuntimeError('Protected evidence/input changed: '+path)


def worker(root):
    import torch
    from visuotactile.state_benchmark import trial
    s = read(root/'session.json')
    inputs_unchanged(root)
    write(root/'heartbeat.json', dict(stage='startup', utc=time.time()))
    app, env, player, lease = environment(root, evaluation=True)
    with lease.block():
        saved = torch.load(s['checkpoint'], map_location='cpu', weights_only=False)
        assert saved['updates'] == s['updates']
    results = []
    for spec in read(root/'plan.json'):
        identity = spec['identity']
        write(root/'heartbeat.json', dict(stage='viewing', identity=identity, utc=time.time()))
        result = trial(env, lease, root/'episodes'/identity, spec, {}, player,
            s['expert_checkpoint'], saved=saved, position=True, video=True,
            inference=INFERENCE,
            video_title=f"StateDP | update {s['updates']} | {identity} | seed {spec['seed']} | viewing rerun | 1x")
        results.append(result)
        write(root/'results.json', results)
        write(root/'heartbeat.json', dict(stage='viewing', completed=len(results), utc=time.time()))
    write(root/'worker_complete.json', dict(completed=len(results), utc=time.time(),
        teardown='SDK fast exit after durable native outcomes; not normal SDK teardown'))


def finish(root):
    import numpy as np
    from visuotactile.video import verify_video
    from visuotactile.narrow_decision import failure_stage
    s = read(root/'session.json'); parent = Path(s['parent']); demos = Path(s['demos'])
    rows = []
    for result in read(root/'results.json'):
        identity = result['identity']; directory = root/'episodes'/identity
        original = parent/'evaluation'/identity
        old = read(original/'result.json')
        row = dict(identity=identity, seed=result['seed'], split=result['split'],
            legal=result['legal'], success=result['success'], original_success=old['success'],
            original_steps=old['steps'], original_subgoals=old['native_subgoals'])
        if result['legal']:
            movie = directory/'rollout.mp4'
            row.update(video=verify_video(movie, result['steps']), video_sha256=sha(movie),
                steps=result['steps'], native_subgoals=result['native_subgoals'],
                native_lift=result['native_lift'], reasons=result['reasons'],
                contact=result['contact'], failure_stage=failure_stage(result))
            initial, old_initial = read(directory/'initial_observable.json'), read(original/'initial_observable.json')
            keys = ('joint_position_rad', 'joint_velocity_rad_s', 'object_pose_w', 'hole_pose_w',
                'table_pose_w', 'palm_body_pose_w', 'finger_position_w', 'finger_quat_wxyz')
            row['initial_observable_max_abs'] = {k: float(np.abs(np.asarray(initial[k])-old_initial[k]).max()) for k in keys}
            with np.load(directory/'plans.npz') as a, np.load(original/'plans.npz') as b:
                row['first_plan_max_abs'] = float(np.abs(a['plans'][0]-b['plans'][0]).max())
            with np.load(directory/'trace.npz') as a, np.load(original/'trace.npz') as b:
                n = min(len(a['command']), len(b['command']))
                row['common_prefix_trace_max_abs'] = {k: float(np.abs(a[k][:n]-b[k][:n]).max())
                    for k in ('command', 'joint', 'object')}
            destination = demos/(identity+'.mp4')
            if destination.exists():
                raise FileExistsError(destination)
            os.link(movie, destination)  # Same filesystem; no duplicate large payload.
        rows.append(row)
    inputs_unchanged(root)
    report = dict(complete=True, source_commit=s['source_commit'], checkpoint_sha256=s['checkpoint_sha256'],
        updates=s['updates'], inference=INFERENCE, budget=read(root/'budget.json'), rows=rows,
        formal_counts_unchanged=True, selection_allowed=False, exact_hidden_simulator_replay_claimed=False,
        training_updates=0, data_captures=0, resources=read(root/'monitor_exit.json')['summary'],
        worker_exit=read(root/'worker_exit.json'), protected_hashes_verified=True)
    write(root/'report.json', report)
    lines = ['# State200 failure viewing reruns', '',
        f"Checkpoint update {s['updates']}, SHA256 `{s['checkpoint_sha256']}`.", '',
        'Four fixed cases selected from the completed evaluation, not additional formal tests. '
        'Real fresh simulation, no expert takeover or state forcing. Outcomes may differ from the original runs. '
        'Three camera views, native 60 fps / 1x simulation time; every pre-action frame through natural termination. '
        'No terminal/reset image is substituted. All delivered videos fully decoded and frame counts verified.', '',
        '| Case | Original steps / subgoals | Viewing steps / subgoals | Viewing result |',
        '|---|---:|---:|---|']
    for r in rows:
        label = f"[{r['identity']}]({r['identity']}.mp4)" if r['legal'] else r['identity']
        lines.append(f"| {label} | {r['original_steps']} / {r['original_subgoals']} | "
            f"{r.get('steps', 0)} / {r.get('native_subgoals', 0)} | "
            f"{'success' if r['success'] else r.get('failure_stage', 'pre-action rejection')} |")
    lines += ['', f"Numeric traces, original comparison and resource evidence: `{root}`.",
        'Original train 0/4, unseen 0/10 legal plus two pre-action rejections remain unchanged.']
    (demos/'README.md').write_text('\n'.join(lines)+'\n')


def owner(root):
    s = read(root/'session.json'); worker_process = monitor = None
    with (root/'owner.lock').open('a') as lock, (Path(s['parent'])/'owner.lock').open('r') as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exclusive creation makes re-launch fail without resetting any budget.
        with (root/'owner_started.json').open('x') as marker:
            marker.write(str(time.time())+'\n')
        try:
            inputs_unchanged(root)
            if time.time() >= s['deadline'] or (root/'STOP').exists():
                raise TimeoutError('Viewing deadline/STOP')
            from tools.kit_resource_preflight import inspect
            preflight = inspect(parent=root); write(root/'inotify.json', preflight)
            if not preflight['resource_admitted']:
                raise RuntimeError('Inotify admission failed; no Kit started')
            with (root/'monitor.log').open('w') as log:
                monitor = subprocess.Popen(RUN+['-m', 'visuotactile.state_resources', '--root', str(root)],
                    cwd=CODE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (root/'worker.log').open('w') as log:
                worker_process = subprocess.Popen(RUN+['-m', 'tools.narrow_videos', 'worker', '--root', str(root)],
                    cwd=CODE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(root, state='running', owned_workers_exited=False,
                owner=proc(os.getpid()), worker=proc(worker_process.pid), monitor=proc(monitor.pid))
            disk_check = 0
            while worker_process.poll() is None:
                if time.time() >= s['deadline'] or (root/'STOP').exists():
                    raise TimeoutError('Viewing deadline/STOP; no automatic retry')
                if monitor.poll() is not None:
                    raise RuntimeError('Resource monitor exited')
                progress = [root/'owner_started.json', root/'heartbeat.json', *root.glob('episodes/*/progress.json')]
                if time.time()-max(p.stat().st_mtime for p in progress if p.exists()) > s['maximum_no_progress_s']:
                    raise TimeoutError('No durable progress for 900s')
                if time.time()-disk_check > 30:
                    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    if size > s['maximum_output_bytes'] or shutil.disk_usage(root).free < s['reserve_bytes']:
                        raise RuntimeError('Viewing disk limit/reserve')
                    disk_check = time.time()
                time.sleep(2)
            write(root/'worker_exit.json', dict(returncode=worker_process.returncode, utc=time.time(),
                worker_absent=proc(worker_process.pid) is None))
            if worker_process.returncode or not (root/'worker_complete.json').exists():
                raise RuntimeError('Viewing worker failed; no automatic retry')
            (root/'STOP_MONITOR').touch(); monitor.wait(timeout=15)
            status(root, state='validating', owned_workers_exited=True, worker=None, monitor=None)
            finish(root)
            status(root, state='complete', completed=len(read(root/'results.json')))
        except BaseException as error:
            write(root/'owner_error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time()))
            status(root, state='blocked', error=repr(error))
            raise
        finally:
            if worker_process is not None:
                kill_owned(worker_process)
            if monitor is not None:
                (root/'STOP_MONITOR').touch()
                try:
                    monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_owned(monitor)
            status(root, owned_workers_exited=True, worker=None, monitor=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker', 'status'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--demos', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode == 'prepare':
        if args.parent is None or args.demos is None:
            parser.error('prepare requires --parent and --demos')
        prepare(root, args.parent, args.demos)
    elif args.mode == 'run':
        owner(root)
    elif args.mode == 'status':
        print(read(root/'status.json'))
    else:
        code = 0
        try:
            worker(root)
        except BaseException:
            write(root/'worker_error.json', dict(traceback=traceback.format_exc(), utc=time.time()))
            code = 1
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)


if __name__ == '__main__':
    main()
