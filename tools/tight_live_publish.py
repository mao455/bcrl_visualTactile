"""CPU-only, bounded, two-file Git progress publisher. Never controls experiments."""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PATHS = ('research_docs/TIGHT_LIVE_STATUS.md',
         'research_docs/evidence/tight_execution_learning_20261113/live_status.json')
BRANCH = 'feature/tight-state-foundation'


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(encoded(value))
    temp.replace(path)


def checked_summary(path):
    """Only release counts backed by result hashes and matching plan/receipt specs."""
    path = Path(path)
    summary = read(path)
    plan_path = path.parent.parent / 'evaluation_plan.json'
    plan = {r['identity']: r['spec'] for r in read(plan_path)}
    rows = summary['results']
    seen = set()
    for result in rows:
        identity = result['identity']
        if identity in seen or identity not in plan or Path(identity).name != identity:
            raise ValueError('duplicate/unknown evaluation identity')
        seen.add(identity)
        case = path.parent / 'evaluation' / identity
        receipt = read(case / 'evaluation_receipt.json')
        actual = read(case / 'result.json')
        binding = receipt['binding']
        if (sha(case / 'result.json') != receipt['result_sha256'] or actual != result
                or binding['spec'] != plan[identity]
                or binding['plan_sha256'] != sha(plan_path)
                or binding['checkpoint_sha256'] != result['checkpoint_sha256']):
            raise ValueError('evaluation result/plan binding mismatch')
        for k, value in plan[identity].items():
            if result.get(k) != value:
                raise ValueError('actual opportunity differs from plan')
    complete_rows = [r for r in rows if r.get('software_complete')]
    valid = [r for r in complete_rows if r.get('legal')]
    counts = dict(completed=len(complete_rows), valid=len(valid),
                  successes=sum(bool(r['success']) for r in valid),
                  planned=len(plan))
    for key, value in counts.items():
        if summary[key] != value:
            raise ValueError('summary count mismatch: ' + key)
    if summary.get('complete') and len(rows) != len(plan):
        raise ValueError('incomplete cohort marked complete')
    if summary.get('formal_test22_used') is not False or summary.get('expert_takeover') is not False:
        raise ValueError('not an autonomous development cohort')
    return dict(counts, complete=summary['complete'], software_failures=summary['software_failures'],
                native_lifts=summary['native_lifts'], summary_sha256=sha(path),
                evidence='result hashes and plan bindings checked; physical hidden-state equality unproven')


def snapshot(config):
    outer = Path(config['outer']).resolve()
    watch = read(outer / 'agent_supervisor/watch_plan.json')
    watched = Path(watch['root']).resolve()
    if not watched.is_relative_to(outer):
        raise ValueError('watch root outside authorized session')
    status = read(watched / watch.get('status_file', 'status.json'), {})
    # Explicit allowlist excludes local paths, PID, logs, secrets and arbitrary text.
    state = {k: status[k] for k in ('state', 'stage', 'batch', 'label', 'round', 'group',
             'updates', 'epoch') if k in status and isinstance(status[k], (str, int, float, bool))}
    roots = {outer / 'adaptive_contact_recovery_01', outer / 'success_replication_01', watched}
    evaluations = []
    training = []
    errors = []
    files = sorted({p for r in roots for p in r.rglob('evaluation_summary.json')})
    for path in files:
        identity = str(path.relative_to(outer))
        scope = ('B current/previous only; not A/B' if identity.startswith('success_replication_01/')
                 else 'A/B matched increment' if identity.startswith('adaptive_contact_recovery_01/')
                 else 'new stage; A/B comparability requires its own audit')
        try:
            evaluations.append(dict(path=identity, comparison=scope, **checked_summary(path)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(path=identity, error=type(exc).__name__))
    for path in sorted({p for r in roots for p in r.rglob('training/finished.json')}):
        d = read(path)
        training.append(dict(path=str(path.relative_to(outer)), **{k:d[k] for k in
            ('updates', 'supervised_target_ticks', 'supervised_ticks', 'target_reached') if k in d}))
    audit = read(Path(config['audit']))
    fairness = {k:audit[k] for k in ('matched_increment_passed', 'failed', 'actual',
               'strict_all_history_equality', 'limitations')}
    return dict(version=1, session=outer.name, absolute_deadline_utc=config['deadline'],
        watch=watched.name, state=state, completion_record_present=(watched/'finished.json').exists(),
        failure_record_present=(watched/'failure.json').exists(),
        fairness=dict(round='adaptive_contact_recovery_01/round_01', **fairness),
        training=training, evaluations=evaluations, evidence_errors=errors,
        limitations=['B-only replication is separate from A/B; never pool denominators.',
          'Matched incremental budgets do not erase historical initialization/exposure differences.',
          'New stages require a fresh A/B audit; this publisher does not certify future comparisons.'])


def render(data):
    stamp = dt.datetime.fromtimestamp(data['published_utc'], dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# Tight-Insertion 自动进度', '', f'证据读取时间：{stamp}。', '',
        '本页由独立 CPU 脚本更新；只发布有凭据的新进展，不启动训练或解释科研结论。',
        '每分钟检查，有变化时通常五分钟内推送；阶段完成/失败优先推送。原截止北京时间 11月14日 00:05:15 不变。', '',
        f"当前阶段：`{data['watch']}`；状态：`{json.dumps(data['state'], ensure_ascii=False, sort_keys=True)}`。",
        f"完成凭据：{data['completion_record_present']}；失败凭据：{data['failure_record_present']}。", '',
        '公平规则见 [AB_COMPARISON_POLICY](AB_COMPARISON_POLICY.md)。最近 contact 周期增量审计：'
        + ('通过' if data['fairness']['matched_increment_passed'] else '未通过') + '；历史初始化/曝光仍不等同。', '',
        '| 评估队列 | 范围 | 已完成/计划 | 合法 | 已观察成功 | 整队列完成 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for row in data['evaluations']:
        lines.append(f"| `{row['path'].replace('/evaluation_summary.json', '')}` | {row['comparison']} | {row['completed']}/{row['planned']} | {row['valid']} | {row['successes']} | {row['complete']} |")
    lines += ['', '未完成队列只报告进度，不计算最终成功率。B 新旧模型复验不能用于与 A 排名，也不加入原 A0/10、B1/10 的分母。', '',
              '| 已完成训练 | 更新 | 有效监督量 |', '| --- | ---: | ---: |']
    for row in data['training']:
        lines.append(f"| `{row['path'].replace('/training/finished.json','')}` | {row.get('updates','?')} | {row.get('supervised_target_ticks',row.get('supervised_ticks','?'))} |")
    lines += ['', f"未通过凭据检查的条目：{len(data['evidence_errors'])}；此类条目不发布成功统计。", '',
        '[机器可读证据](evidence/tight_execution_learning_20261113/live_status.json)。脚本不上传原数据、模型或视频；网络失败保留待发布状态，恢复后重试。', '']
    return '\n'.join(lines)


def git(repo, *args, deadline=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    remaining = 60 if deadline is None else min(60, deadline-time.time())
    if remaining <= 0:
        raise TimeoutError('original publication deadline reached')
    result = subprocess.run(['git', '-C', str(repo), *args], env=env, text=True,
                            capture_output=True, timeout=remaining)
    if result.returncode:
        raise RuntimeError(f'git {args[0]} failed: {result.stderr[-1000:]}')
    return result.stdout.rstrip('\n')


@contextlib.contextmanager
def lock(path, nonblocking=False):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def publish(repo, data, deadline=None):
    """Rebase only our two-path commits; reject unrelated dirt/history; never force."""
    repo = Path(repo)
    def g(*args):
        return git(repo, *args, deadline=deadline)
    common = Path(g('rev-parse', '--path-format=absolute', '--git-common-dir'))
    with lock(common / 'tight-publication.lock', nonblocking=True):
        dirty = g('status', '--porcelain', '--untracked-files=all').splitlines()
        if any(line[3:] not in PATHS for line in dirty):
            raise RuntimeError('publisher worktree contains unrelated changes')
        # A crash may leave only bot-owned dirty files; finish that transaction first.
        for attempt in range(3):
            if not dirty:
                g('fetch', 'origin', BRANCH)
                upstream = 'origin/' + BRANCH
                if g('rev-list', upstream + '..HEAD'):
                    changed = g('diff', '--name-only', upstream + '...HEAD').splitlines()
                    if any(p not in PATHS for p in changed):
                        raise RuntimeError('unreviewed local commits outside publisher paths')
                    g('rebase', upstream)
                else:
                    g('merge', '--ff-only', upstream)
            (repo / PATHS[0]).parent.mkdir(parents=True, exist_ok=True)
            (repo / PATHS[0]).write_text(render(data))
            write(repo / PATHS[1], data)
            g('add', '--', *PATHS)
            if g('diff', '--cached', '--name-only'):
                g('commit', '-m', 'Update verified Tight-Insertion live progress', '--', *PATHS)
            dirty = []
            try:
                g('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
                return g('rev-parse', 'HEAD')
            except RuntimeError:
                if attempt == 2:
                    raise
        raise AssertionError('unreachable')


def run(config, once=False):
    out = Path(config['runtime'])
    out.mkdir(parents=True, exist_ok=True)
    with lock(out/'owner.lock', nonblocking=True):
        prior = read(out/'receipt.json', {})
        while time.time() < config['deadline']:
            try:
                data = snapshot(config)
                fingerprint = hashlib.sha256(encoded(data).encode()).hexdigest()
                urgent = (data['completion_record_present'] or data['failure_record_present']
                          or data['state'] != prior.get('state'))
                if fingerprint != prior.get('fingerprint') and (urgent or once or
                        time.time()-prior.get('utc', 0) >= config.get('interval', 300)):
                    data = dict(data, published_utc=time.time())
                    commit = publish(config['worktree'], data, deadline=config['deadline'])
                    prior = dict(fingerprint=fingerprint, commit=commit, utc=time.time(), state=data['state'])
                    write(out/'receipt.json', prior)
                write(out/'status.json', dict(state='watching', utc=time.time(), last_publication=prior,
                    evidence_errors=data['evidence_errors'], deadline=config['deadline']))
            except Exception as exc:
                write(out/'status.json', dict(state='publication_retry_pending', utc=time.time(),
                    error=str(exc)[-2000:], last_publication=prior, deadline=config['deadline']))
                if once:
                    raise
            if once:
                return
            time.sleep(min(60, max(0, config['deadline']-time.time())))
        write(out/'stopped.json', dict(reason='original_absolute_deadline', utc=time.time(), last_publication=prior))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    if args.preview:
        print(encoded(snapshot(config)))
    else:
        run(config, once=args.once)


if __name__ == '__main__':
    main()
