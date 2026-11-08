"""Finite diagnostics for the existing StateDP. No training/Kit launch path.

Ordinary systemd execution owns its resource monitor and completes its report.
The no-training decision is frozen from existing validation evidence, not test.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, supervised_owner, kill_owned

BEGIN = '<!-- tight-dp3-diagnostics:start -->'
END = '<!-- tight-dp3-diagnostics:end -->'


def table(path, rows):
    if not rows:
        raise ValueError('Empty metric table: '+str(path))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def source_hashes():
    return {str(p.relative_to(CODE)): sha(p) for folder in ('tools', 'visuotactile')
        for p in sorted((CODE/folder).rglob('*.py'))}


def beat(root, **fields):
    old = read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json', dict(old, **fields, updated_utc=time.time(), pid=os.getpid()))


def prepare(root, config):
    if root.exists():
        raise FileExistsError('Run already exists; inspect status instead of restarting')
    cfg = read(config)
    if cfg['decision'] != 'diagnostics_then_stop_no_training' or cfg['condition_mode'] != 'existing_robot_geometry_short_windows_v1':
        raise ValueError('Only reviewed existing-equivalent diagnostics are supported')
    parent = (CODE/cfg['source_root']).resolve()
    data = read(parent/'state_data/data.json')
    if [len(data['splits'][s]) for s in ('train', 'validation', 'test')] != [156, 30, 22]:
        raise ValueError('Frozen Core split mismatch')
    for split, ids in cfg['evaluation']['cohorts'].items():
        if not ids or not set(ids) <= set(data['splits'][split]):
            raise ValueError('Diagnostic IDs outside declared Core split')
    if sha(parent/cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    rows = [json.loads(x) for x in (parent/'training/validation.jsonl').read_text().splitlines()]
    gain = 1-rows[-1]['metrics']['selection_mse']/rows[-4]['metrics']['selection_mse']
    ratio = rows[-1]['metrics']['selection_mse']/rows[-1]['train8']['selection_mse']
    if gain >= .01 or read(parent/'training/sufficiency.json')['still_improving']:
        raise ValueError('Historical no-continuation evidence differs; do not silently train')
    root.mkdir(parents=True)
    write(root/'configuration.json', cfg)
    write(root/'decision.json', dict(training_updates=0, architecture_changed=False,
        reason='Equivalent direct path already exists; existing complete-sampling curve has <1% last-four gain and large train/validation gap',
        last_four_relative_gain=gain, validation_train_mse_ratio=ratio,
        source_validation_sha256=sha(parent/'training/validation.jsonl'), final_test_used=False))
    paths = [parent/p for p in ('manifest.json', 'composition_manifest.json', 'state_data/data.json',
        'state_data/numeric.pt', 'state_data/statistics.pt', 'training/best.pt', 'training/validation.jsonl',
        'training/training.jsonl', 'training/sufficiency.json', 'evaluation_summary.json')]
    write(root/'inputs.json', {str(p): sha(p) for p in paths})
    beat(root, state='PREPARED', stage='acceptance', formal_updates=0, episodes_completed=0,
        episodes_total=sum(map(len, cfg['evaluation']['cohorts'].values())), exit_code=None,
        reason='Awaiting focused acceptance and source freeze', run=str(root))


def load_cpu(root):
    import torch
    import h5py
    from visuotactile.state_history import PersistentState
    from visuotactile.position_control import contract
    from visuotactile.state_image_loss import GeometryObjective
    cfg = read(root/'configuration.json'); parent = CODE/cfg['source_root']
    data = read(parent/'state_data/data.json')
    saved = torch.load(parent/cfg['checkpoint'], map_location='cpu', weights_only=False)
    cache = torch.load(parent/'state_data/numeric.pt', map_location='cpu', weights_only=False)
    if saved['contract'] != contract(24, True) or cache['contract'] != saved['contract']:
        raise ValueError('H24 absolute-target contract mismatch')
    if cache['data_sha256'] != sha(parent/'state_data/data.json'):
        raise ValueError('Numeric data fingerprint mismatch')
    if saved['configuration']['input_builder'] != 'canonical_joint_state_v1':
        raise ValueError('Reference checkpoint input builder mismatch')
    for key, path in [('data_sha256', 'data.json'), ('statistics_sha256', 'statistics.pt')]:
        if saved['configuration'][key] != sha(parent/'state_data'/path):
            raise ValueError('Checkpoint/data normalization identity mismatch')
    model = PersistentState('dp', saved['contract']).eval()
    model.load_state_dict(saved['ema'], strict=True)
    stats = torch.load(parent/'state_data/statistics.pt', map_location='cpu', weights_only=False)
    for key, value in stats.items():
        if not torch.equal(getattr(model, key), value):
            raise ValueError('EMA normalizer mismatch: '+key)
    if saved['updates'] != 8500 or saved['ema_optimization_step'] != 8500:
        raise ValueError('Reference EMA update mismatch')
    ids = [i for values in cfg['evaluation']['cohorts'].values() for i in values]
    records = {i: cache['records'][i] for i in ids}
    entry = data['entries'][ids[0]]
    with h5py.File(entry['path']) as f:
        meta = json.loads(f.attrs['metadata_json'])
    objective = GeometryObjective(meta); objective.weights = saved['configuration']['loss_weights']
    return cfg, data, records, model, objective, saved


def acceptance(root):
    """One CPU optimizer step is disposable; never modifies a reference model."""
    import copy
    import torch
    from visuotactile.state_history import HistoryBank, short_windows
    from visuotactile.anchor_sampling import anchor_batch
    from visuotactile.models.vendor.ema_model import EMAModel
    from visuotactile.checkpointing import training_state, restore_training, save
    cfg, data, records, model, objective, old = load_cpu(root)
    import hashlib
    for name in ('state_history.py', 'models/native_dp.py', 'native_data.py', 'narrow_data.py', 'position_control.py'):
        path = 'visuotactile/'+name
        original = subprocess.check_output(['git', 'show', cfg['reference_commit']+':'+path], cwd=CODE)
        if hashlib.sha256(original).hexdigest() != sha(CODE/path):
            raise ValueError('Existing-equivalent policy/data code changed: '+path)
    del old
    rec = records[cfg['evaluation']['cohorts']['train'][0]]
    with torch.no_grad():
        full, anchors, hidden = model.encode(rec)
        bank = HistoryBank(); online = []
        for tick in range(len(rec['time'])):
            c = bank.observe(0, 1, tick, model, rec['actor'][tick], rec['robot'][tick], rec['geometry'][tick], rec['time'][tick])
            if c is not None: online.append(c)
        parity = float((full-torch.cat(online)).abs().max())
        if parity >= 2e-5: raise ValueError('Full-prefix train/deploy condition mismatch')
    # Hold Actor and recurrent output constant: only the direct current-state
    # input can produce this derivative/intervention response.
    robots, geos = short_windows(rec, torch.tensor([12]), model)
    robots = robots.detach().requires_grad_(); geos = geos.detach().requires_grad_()
    fixed = torch.zeros(1, 384)
    def condition(r, g):
        return model.condition_projection(torch.cat((model.robot_encoder(r), model.state_encoder(g), fixed), -1))
    c = condition(robots, geos)
    gr, gg = torch.autograd.grad(c.square().mean(), (robots, geos))
    r2, g2 = robots.detach().clone(), geos.detach().clone()
    r2[0, -180] += .1; g2[0, -87] += .1
    delta_r = float((condition(r2, geos)-c).abs().max().detach())
    delta_g = float((condition(robots, g2)-c).abs().max().detach())
    current_grad = [float(gr[0, -180:-2].abs().sum()), float(gg[0, -87:-2].abs().sum())]
    if min(delta_r, delta_g, *current_grad) <= 0: raise ValueError('Direct current state path is inactive')
    model.train(); model.zero_grad(set_to_none=True)
    averaged = copy.deepcopy(model).eval().requires_grad_(False); ema = EMAModel(averaged, power=.75)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6, betas=(.95, .999), weight_decay=1e-6)
    generator = torch.Generator().manual_seed(101); noise = torch.Generator().manual_seed(102)
    before = next(model.condition_projection.parameters()).detach().clone()
    identity = cfg['evaluation']['cohorts']['train'][0]
    cond, y, mask = anchor_batch(model, records, [(identity, 12)])
    loss, _, _ = objective.forward(model, cond, y, mask, noise)
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError('Acceptance nonfinite loss/gradient')
    opt.step(); ema.step(model)
    change = float((next(model.condition_projection.parameters()).detach()-before).abs().max())
    if change <= 0: raise ValueError('Disposable update did not change parameters')
    temporary = root/'disposable_acceptance.pt'
    save(temporary, training_state(model, averaged, opt, ema, generator, noise, updates=1))
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    restore_training(saved, model, averaged, opt, ema, generator, noise)
    if ema.optimization_step != 1 or not all(int(v['step']) == 1 for v in opt.state.values()):
        raise ValueError('Disposable optimizer/EMA restore mismatch')
    with torch.no_grad():
        c, _, _ = averaged.encode(rec, stop=13)
        predicted = averaged.predict(c[-1:], torch.Generator().manual_seed(7091))
    if predicted.shape != (1, 24, 29) or not torch.isfinite(predicted).all():
        raise ValueError('Complete prediction after checkpoint restore failed')
    report = dict(passed=True, source_sha256=source_hashes(), full_prefix_ticks=len(rec['time']),
        condition_shape=list(full.shape), condition_projection_input=896, condition_projection_output=512,
        robot_window_shape=list(robots.shape), geometry_window_shape=list(geos.shape),
        native_history_hidden_shape=list(hidden.shape), train_deployment_max_abs=parity,
        current_robot_condition_delta=delta_r, current_geometry_condition_delta=delta_g,
        direct_current_input_gradient_l1=current_grad, recurrent_and_actor_held_constant=True,
        disposable_updates=1, formal_updates=0, disposable_loss=float(loss), gradient_norm=float(norm),
        parameter_change=change, save_load_optimizer_ema=True, normalization_matches=True,
        complete_prediction_shape=list(predicted.shape), original_model_source_unchanged=True)
    temporary.unlink()  # Only this explicitly disposable smoke checkpoint.
    write(root/'acceptance.json', report)
    beat(root, state='CHECKED', stage='ready_to_freeze', reason='CPU acceptance passed')


def historical_curves(root, parent):
    rows = []
    for line in (parent/'training/validation.jsonl').read_text().splitlines():
        r = json.loads(line)
        for name, key, loss_key in [('train8', 'train8', 'train8_loss'), ('validation30', 'metrics', 'validation_loss')]:
            rows.append(dict(update=r['update'], split=name, loss=r.get(loss_key, {}).get('loss'),
                arm_rmse_rad=r[key]['arm_rmse'], hand_rmse_rad=r[key]['hand_rmse'],
                selection_mse_rad2=r[key]['selection_mse'], protocol='historical_episode_batch_seed7091'))
    table(root/'historical_curves.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # All plots read the emitted raw CSV, including historical curves.
    with (root/'historical_curves.csv').open() as f: plotted = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for split in ('train8', 'validation30'):
        selected = [r for r in plotted if r['split'] == split]
        for ax, key in zip(axes, ('loss', 'arm_rmse_rad', 'hand_rmse_rad')):
            ax.plot([int(r['update']) for r in selected], [float(r[key]) if r[key] else float('nan') for r in selected], label=split)
            ax.set(xlabel='Optimizer updates', ylabel=key); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(root/'historical_curves.png', dpi=140); plt.close(fig)


def target_plot(root, identity, sample, phases, contact, lift):
    import matplotlib.pyplot as plt
    rows = []
    for tick in range(len(phases)):
        row = dict(tick=tick, seconds=tick/60, phase=str(phases[tick]), contact=bool(contact[tick]), lift=bool(lift[tick]), replan=tick%12 == 0)
        for kind in ('truth', 'prediction', 'hold'):
            # Increments within a planned chunk start from the real anchor target.
            before = sample['hold'][tick] if tick%12 == 0 else sample[kind][tick-1]
            for j in range(29):
                row[f'{kind}_{j}_rad'] = float(sample[kind][tick, j])
                row[f'{kind}_delta_{j}_rad'] = float(sample[kind][tick, j]-before[j])
        rows.append(row)
    path = root/(identity+'_targets.csv'); table(path, rows)
    with path.open() as f: rows = list(csv.DictReader(f))
    # Fixed representative joints, chosen before viewing predictions.
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for col, joint in enumerate((3, 7)):
        for row_index in range(2):
            ax = axes[row_index, col]
            for kind in ('truth', 'prediction', 'hold'):
                key = f'{kind}_{joint}_rad' if row_index == 0 else f'{kind}_delta_{joint}_rad'
                ax.plot([float(r['seconds']) for r in rows], [float(r[key]) for r in rows], label=kind, linewidth=.8)
            for r in rows:
                if r['replan'] == 'True': ax.axvline(float(r['seconds']), color='gray', alpha=.15, linewidth=.5)
            for field, color in (('contact', 'green'), ('lift', 'purple')):
                first = next((r for r in rows if r[field] == 'True'), None)
                if first: ax.axvline(float(first['seconds']), color=color, label='first '+field)
            ax.set(xlabel='Expert-history time (s)', ylabel=('Target' if row_index == 0 else 'Within-plan change')+' (rad)', title=f'Canonical joint {joint}')
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/(identity+'_targets.png'), dpi=140); plt.close(fig)


def reuse_closed_loop(root, parent):
    import numpy as np
    from visuotactile.narrow_decision import failure_stage
    rows = []; selected = []
    for p in sorted((parent/'evaluation').glob('*/result.json')):
        r = read(p)
        rows.append(dict(identity=r['identity'], split=r['split'], legal=r['legal'], success=r['success'],
            approach=r.get('approach_proxy'), object_contact=r.get('contact', {}).get('object_contact'),
            stable_grasp=r.get('contact', {}).get('stable_grasp_diagnostic'), lift=r.get('native_lift'),
            alignment=r.get('native_alignment'), subgoals=r.get('native_subgoals'),
            actuator_error_rad=r.get('actuator_target_max_error'), submitted_target_error_rad=r.get('submitted_target_max_error'),
            tracking_rmse_rad=r.get('tracking_rmse_rad'), failure_stage=failure_stage(r),
            source=str(p), source_sha256=sha(p)))
        if r['split'] in ('train', 'validation') and r['split'] not in [x[0] for x in selected]: selected.append((r['split'], p.parent))
    table(root/'closed_loop_reused.csv', rows)
    cohorts = {}
    for split in sorted({r['split'] for r in rows}):
        group = [r for r in rows if r['split'] == split]
        cohorts[split] = dict(attempts=len(group), valid=sum(r['legal'] for r in group),
            rejected=sum(not r['legal'] for r in group), successes=sum(r['success'] for r in group),
            reached={k: sum(bool(r[k]) for r in group) for k in ('approach', 'object_contact', 'stable_grasp', 'lift', 'alignment', 'subgoals')})
    events = []
    for split, directory in selected:
        with np.load(directory/'trace.npz') as z:
            initial_z = read(directory/'initial_observable.json')['object_pose_w'][0][2]
            physical = []
            for t in range(len(z['joint'])):
                err = z['joint'][t]-z['target'][t]
                physical.append(dict(tick=t, seconds=(t+1)/60, object_rise_m=float(z['object'][t, 2]-initial_z),
                    fingertip_min_distance_m=float(z['fingertip_distance'][t]),
                    contact_force_max_N=float(np.linalg.norm(z['object_contact'][t], axis=-1).max()),
                    arm_tracking_rmse_rad=float(np.sqrt(np.mean(err[:7]**2))),
                    hand_tracking_rmse_rad=float(np.sqrt(np.mean(err[7:]**2))),
                    native_lift=bool(np.any(z['lifted'][t])), native_alignment=bool(np.any(z['alignment'][t])),
                    subgoals=int(np.max(z['subgoals'][t]))))
            events.append(dict(identity=directory.name, split=split,
                first_contact_tick=next((r['tick'] for r in physical if r['contact_force_max_N']>.1), None),
                first_native_lift_tick=next((r['tick'] for r in physical if r['native_lift']), None),
                first_native_alignment_tick=next((r['tick'] for r in physical if r['native_alignment']), None),
                maximum_object_rise_m=max(r['object_rise_m'] for r in physical),
                earliest_expert_physical_divergence='not measured: autonomous/expert histories are not same-tick pairs'))
        path = root/(directory.name+'_physical.csv'); table(path, physical)
        with path.open() as f: plotted = list(csv.DictReader(f))
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, keys, unit in [(axes[0], ['object_rise_m', 'fingertip_min_distance_m'], 'm'),
            (axes[1], ['contact_force_max_N'], 'N'), (axes[2], ['arm_tracking_rmse_rad', 'hand_tracking_rmse_rad'], 'rad')]:
            for key in keys: ax.plot([float(r['seconds']) for r in plotted], [float(r[key]) for r in plotted], label=key)
            ax.set(xlabel='Student time (s)', ylabel=unit); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(root/(directory.name+'_physical.png'), dpi=140); plt.close(fig)
    write(root/'closed_loop_summary.json', dict(cohorts=cohorts, representative_events=events,
        new_starts=0, scope='Reused old autonomous trials; no teacher target error across different-progress trajectories'))


def report(root):
    s = read(root/'status.json'); decision = read(root/'decision.json')
    content = f'''{BEGIN}
## 2026-11-08 DP3风格StateDP条件与诊断

状态 **{s['state']}**；阶段 `{s['stage']}`；已完成离线轨迹 {s.get('episodes_completed', 0)}/{s.get('episodes_total', 38)}。
原因：{s.get('reason', '')}。正式新增更新0；新Kit/闭环启动0。
等价当前robot/geometry直达condition已存在，保留原模型/GRU/U-Net与旧checkpoint。
旧最近四次验证改善 {decision['last_four_relative_gain']:.4%}，验证/train8采样MSE比 {decision['validation_train_mse_ratio']:.2f}；依据train/validation冻结为诊断后停止，不自动续训。

本轮root：`{root}`；状态`status.json`，日志`service.log`，验收`acceptance.json`，
冻结身份`session.json`/`inputs.json`/`source_freeze.json`。
图均由本run CSV生成：`historical_curves.csv/png`为原17点完整DDIM曲线（旧episode-batch噪声）；
`sampled_ticks.csv`/`stage_metrics.json`为当前同部署batch1顺序、固定seed7091、真实完整专家前缀、前12步目标误差；
`*_targets.csv/png`含29关节目标/变化量原始数值及固定joint3/7示例、接触/抬升/重规划标记。
`closed_loop_reused.csv`/`closed_loop_summary.json`与`*_physical.csv/png`复用旧27次自主轨迹。
未生成文件代表该阶段未完成，不代表零误差或零成功。离线目标FK误差、命令接口误差和物理tracking各自独立；
自主与专家不同进度下第一次物理偏离未测，不按同tick作动作误差。阶段事件只作诊断，且含transition接触/对齐信息，绝不输入网络。
资源上限：整卡批次/完整60s均值50%，45%准入、35%目标、6GiB显存/14GiB RAM/60GiB磁盘储备；最多1h含等待。
实测资源见`resources/resources.json`与`resource_summary.json`，启动不等于完成或全程资源合规。
{END}'''
    if (root/'stage_metrics.json').exists():
        metrics = read(root/'stage_metrics.json')
        lines = ['\n|专家历史离线集/阶段|Arm RMSE / hold (rad)|Hand RMSE / hold (rad)|有效tick|', '|---|---:|---:|---:|']
        for arm in (r for r in metrics if r['group'] == 'arm' and r['motion'] == 'all'):
            hand = next(r for r in metrics if r['split'] == arm['split'] and r['phase'] == arm['phase'] and r['group'] == 'hand' and r['motion'] == 'all')
            fmt = lambda r: f"{r['rmse_rad']:.6f} / {r['hold_rmse_rad']:.6f}" if r['ticks'] else '未测'
            lines.append(f"|{arm['split']} / {arm['phase']}|{fmt(arm)}|{fmt(hand)}|{arm['ticks']}|")
        content = content.replace(END, '\n'.join(lines)+'\n\n'+END)
    if (root/'closed_loop_summary.json').exists():
        cohorts = read(root/'closed_loop_summary.json')['cohorts']
        text = '\n旧闭环复用：'+ '；'.join(f"{k} {v['successes']}/{v['valid']}成功，拒绝{v['rejected']}" for k,v in cohorts.items())+'。不加入新实验分母。\n\n'
        content = content.replace(END, text+END)
    cfg = read(root/'configuration.json'); path = CODE/cfg['output_report']; old = path.read_text()
    if BEGIN in old:
        start, end = old.index(BEGIN), old.index(END)+len(END)
        updated = old[:start]+content+old[end:]
    else:
        first, rest = old.split('\n', 1); updated = first+'\n\n'+content+'\n\n'+rest
    temporary = path.with_suffix('.md.tmp'); temporary.write_text(updated); temporary.replace(path)


@supervised_owner
def run(root):
    import torch
    import h5py
    import numpy as np
    from visuotactile.state_resources import Lease
    from visuotactile.state_image_offline import deployment_sample, diagnostic_rows, summarize_diagnostics, no_kit
    from tools.narrow_repair_audit import stages
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'session.json').exists(): raise RuntimeError('No automatic rerun/resume of completed or failed diagnostics')
    monitor = None; code = 1
    try:
        cfg = read(root/'configuration.json'); check = read(root/'acceptance.json')
        # Acceptance covers the unmodified policy/data implementation and the
        # real checkpoint. Final CLI/plot tests cover the orchestration separately.
        current_sources = source_hashes()
        if not check['passed'] or any(digest != current_sources.get(path) for path, digest in check['source_sha256'].items()
                if path.startswith('visuotactile/')):
            raise ValueError('Acceptance model/evaluator source changed')
        final_checks = read(root/'runtime_checks.json')
        if not final_checks['passed'] or final_checks['source_sha256'] != current_sources:
            raise ValueError('Final runtime checks do not cover current sources')
        if final_checks['configuration_sha256'] != sha(root/'configuration.json') or sha(root/'configuration.json') != sha(CODE/'configs/tight_dp3_diagnostics.json'):
            raise ValueError('Reviewed run configuration differs from committed configuration')
        for path, digest in read(root/'inputs.json').items():
            if sha(path) != digest: raise ValueError('Frozen input changed: '+path)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True).strip()
        if dirty: raise ValueError('Commit the reviewed runtime before launch: '+dirty)
        now = time.time(); deadline = now+3600*cfg['evaluation']['maximum_hours']
        write(root/'session.json', dict(started_utc=now, training_deadline=deadline, deadline=deadline,
            source_head=head, cwd=str(CODE), python=str(Path('/proc/self/exe').resolve()),
            configuration_sha256=sha(root/'configuration.json'),
            resource_target_percent=35, formal_updates=0, seed=7091))
        write(root/'source_freeze.json', source_hashes())
        write(root/'gpu_schedule.json', cfg['resources'])
        beat(root, state='RUNNING', stage='historical_curves', reason='Frozen existing-model diagnostic queue', exit_code=None)
        parent = CODE/cfg['source_root']; historical_curves(root, parent); reuse_closed_loop(root, parent)
        log = (root/'monitor.log').open('a')
        monitor = subprocess.Popen(RUN+['-m', 'visuotactile.state_resources', '--root', str(root)],
            cwd=CODE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        class BoundedLease(Lease):
            def check(self, deadline=None):
                super().check(deadline)
                if shutil.disk_usage(root).free < cfg['resources']['min_disk_gib']*1024**3:
                    raise RuntimeError('60GiB disk reserve reached')
                if monitor.poll() is not None: raise RuntimeError('Resource monitor exited')
        lease = BoundedLease(root, deadline=deadline)
        beat(root, stage='resource_admission', monitor_pid=monitor.pid,
            reason='Require a complete measured 60-second window before reference load and CUDA')
        with lease.block(cuda=False):
            cfg, data, records, model, objective, saved = load_cpu(root); del saved
        with lease.block(): model.to(cfg['evaluation']['device'])
        rows = []; completed = 0; anchors_total = 0
        def progress(**kw):
            beat(root, **kw)
            if not (root/'first_progress.json').exists():
                write(root/'first_progress.json', read(root/'status.json'))
        for split, ids in cfg['evaluation']['cohorts'].items():
            for index, identity in enumerate(ids):
                beat(root, stage='offline_sampling', identity=identity, split=split, native_ticks=0,
                    sampled_anchors=0, reason='Deployment-order DDIM on a complete expert prefix')
                with lease.block(): rec = {k:v.to(cfg['evaluation']['device']) for k,v in records[identity].items()}
                sample = deployment_sample(model, rec, objective, lease, seed=cfg['evaluation']['seed'],
                    progress=progress)
                e = data['entries'][identity]
                with h5py.File(e['path']) as f:
                    g = f['episodes'][e['episode']]; phases = stages(g)
                    contact = np.linalg.norm(g['transition/object_contact_force_w'][:], axis=-1).reshape(len(phases), -1).max(-1)>.1
                    lift = g['transition/diagnostics_before/lifted_object'][:].reshape(len(phases), -1).any(-1)
                episode_rows = diagnostic_rows(sample, phases, identity=identity, split=split)
                path = root/'sampled_ticks.csv'
                with path.open('a', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
                    if completed == 0: writer.writeheader()
                    writer.writerows(episode_rows)
                rows.extend(episode_rows)
                if index == 0: target_plot(root, identity, sample, phases, contact, lift)
                completed += 1; anchors_total += (len(phases)+11)//12
                write(root/'stage_metrics.json', summarize_diagnostics(rows))
                beat(root, episodes_completed=completed, anchors_total=anchors_total)
                del rec, sample
        no_kit()
        if source_hashes() != read(root/'source_freeze.json'): raise ValueError('Runtime source drift')
        code = 0
        beat(root, state='COMPLETE', stage='diagnostics_complete_no_training', exit_code=0,
            reason='Finite diagnosis complete; equivalent path and existing plateau do not justify retraining')
    except BaseException as exc:
        code = 2 if isinstance(exc, (TimeoutError, InterruptedError)) else 1
        write(root/'error.json', dict(type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc()))
        beat(root, state='BUDGET_STOP' if code == 2 else 'FAILED', exit_code=code, reason=str(exc))
    finally:
        if monitor is not None:
            (root/'STOP_MONITOR').touch()
            try: monitor.wait(timeout=15)
            except subprocess.TimeoutExpired: kill_owned(monitor)
            log.close()
            path = root/'resources/resources.json'
            if path.exists():
                summary = read(path)
                samples = [json.loads(line) for line in (root/'resources/resources.jsonl').read_text().splitlines()]
                summary['configured_limits'] = cfg['resources']
                summary['below_vram_reserve_samples'] = sum(r['gpu_free_mib'] is not None and r['gpu_free_mib']<6144 for r in samples)
                summary['complete_60s_above_cap_samples'] = sum(r['rolling_60s_percent'] is not None and r['rolling_60s_percent']>50 for r in samples)
                summary['actual_min_gpu_free_mib'] = min((r['gpu_free_mib'] for r in samples if r['gpu_free_mib'] is not None), default=None)
                summary['hard_gpu_quota_guaranteed'] = False
                write(root/'resource_summary.json', summary)
        beat(root, owned_monitor_exited=monitor is None or monitor.poll() is not None)
        report(root)
        lock.close()
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'check', 'run', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CODE/'configs/tight_dp3_diagnostics.json')
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode == 'prepare': prepare(root, args.config)
    elif args.mode == 'status': print(json.dumps(read(root/'status.json'), indent=2))
    else:
        import torch
        from visuotactile.precision import state_fp32
        torch.set_num_threads(2); state_fp32()
        if args.mode == 'check': acceptance(root)
        else: raise SystemExit(run(root))


if __name__ == '__main__': main()
