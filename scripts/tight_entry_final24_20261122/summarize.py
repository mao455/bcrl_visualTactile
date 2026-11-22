"""Validate the frozen test and publish the primary results before diagnostics."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import binomtest

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
ROOT = PROJECT / 'outputs/visuotactile/run/tight_entry_final24_20261122_01'
OUT = PROJECT / 'research_docs/evidence/tight_entry_final24_20261122_01'
REPORT = PROJECT / 'research_docs/TIGHT_ENTRY_FINAL24_20261122.md'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    freeze = read(ROOT/'registration.json')
    episodes, waves, file_hashes, initial_states, protocol_checks = [], [], {}, [], []
    assert (ROOT/'project_complete.json').exists(), 'Finish all registered waves before primary delivery'
    assert read(ROOT/'project_exit.json')['children_reaped']
    for wave_path, digest in freeze['wave_plan_sha256'].items():
        wave = Path(wave_path)
        assert sha(wave/'plan.json') == digest
        plan = read(wave/'plan.json')
        assert plan['pairs'] == freeze['pairs'] and plan['test_used']
        assert plan['kind'] == 'autonomous'
        initial = read(wave/'paired_initial_check.json')
        assert initial['passed'] and len(initial['rows']) == 3
        rows = read(wave/'results.json')
        profiles = read(wave/'wave_profiles.json')
        assert len(profiles)==1
        assert len(rows) == 4
        waves.append(dict(wave=wave.name, initial_passed=True, cases=4))
        destination = OUT/wave.name
        destination.mkdir(exist_ok=True)
        for name in ('plan.json', 'session.json', 'results.json', 'result.json',
                     'paired_initial_check.json', 'paired_reset_installed.json',
                     'initialization.json', 'run_started.json', 'videos.json'):
            if (wave/name).exists():
                shutil.copy2(wave/name, destination/name)
                file_hashes[str(wave/name)] = sha(wave/name)
        cases = {c['name']: c for c in plan['cases']}
        for row in rows:
            result = row['result']
            case = cases[row['case']]
            role = row['role']
            assert result['complete_student_policy'] == freeze['pairs'][role]
            assert result['seed'] == case['spec']['seed']
            assert result['legal'] and result['software_complete']
            assert result['residual_ticks'] == result['steps']
            case_directory = wave/row['case']
            timing = read(case_directory/'native_inference_timing.json')['batch_observed_counts']
            assert all(timing[k]==result['steps'] for k in ('history_updates','robot_observations','backend_updates'))
            assert timing['plans']==(result['steps']+11)//12
            vision = profiles[0]['vision_execution']['roles'][role]
            assert vision['vision_microbatch']==2
            assert vision['controller_calls']==vision['real_lanes']==result['steps']
            assert vision['consumed_observations_per_lane'][plan['cases'].index(case)]==result['steps']
            protocol_checks.append(dict(seed=result['seed'],role=role,steps=result['steps'],
                plans=timing['plans'],history_updates=timing['history_updates'],
                backend_updates=timing['backend_updates'],visual_observations=vision['real_lanes'],
                vision_microbatch=2))
            case_destination = destination/row['case']
            case_destination.mkdir(exist_ok=True)
            for name in ('initial_observable.json','initial_collision.json','metadata.json','result.json'):
                shutil.copy2(case_directory/name,case_destination/name)
                file_hashes[str(case_directory/name)] = sha(case_directory/name)
            if plan['cases'].index(case)==0:
                observed = read(case_directory/'initial_observable.json')
                obj,base = np.array(observed['object_pose_w'][0]),np.array(observed['base_pose_w'][0])
                initial_states.append(dict(seed=result['seed'],
                    vector=np.r_[observed['joint_position_rad'][0],obj[:3]-base[:3],obj[3:]].tolist()))
            episodes.append(dict(cohort=case['diagnostic_cohort'], seed=result['seed'],
                role=role, lane=plan['cases'].index(case), success=bool(result['success']),
                native_subgoals=int(result['native_subgoals']), steps=result['steps'],
                legal=result['legal'], software_complete=result['software_complete'],
                previously_known=False, result_path=str(wave/row['case']/'result.json')))
    assert len(episodes) == 96 and len({(e['seed'],e['role']) for e in episodes}) == 96
    assert len({e['seed'] for e in episodes}) == 24
    separations = [float(np.max(np.abs(np.array(a['vector'])-b['vector'])))
                   for i,a in enumerate(initial_states) for b in initial_states[i+1:]]
    assert len(initial_states)==24 and min(separations)>1e-5
    write(OUT/'distinct_initial_states.json',dict(states=initial_states,
        vector='q29 plus object position minus base position and object quaternion; initial observed values only',
        minimum_pairwise_max_component_separation=min(separations),tolerance=1e-5))
    write(OUT/'protocol_checks.json',protocol_checks)
    for role, pair in freeze['pairs'].items():
        for name in ('frontend', 'backend'):
            assert sha(pair[name]) == pair[name+'_sha256']
            file_hashes[pair[name]] = pair[name+'_sha256']
        assert sorted(Counter(e['lane'] for e in episodes if e['role']==role).values()) == [6]*4
    for path, digest in freeze['source_sha256'].items():
        assert sha(PROJECT/path) == digest
    session = read(ROOT/'session.json')
    for name in ('dino','expert'):
        path,digest = session[name+'_checkpoint'],session[name+'_sha256']
        assert sha(path)==digest
        file_hashes[path] = digest
    # Reconstruct only random draws, without loading a policy or advancing physics.
    # Actual start/end CUDA states close the persistent per-lane noise accounting.
    import torch
    generator = torch.Generator(device='cuda:0').manual_seed(7091)
    states = {0:generator.get_state().cpu()}
    for count in range(1,max((e['steps']+11)//12 for e in episodes)+1):
        torch.randn((1,24,29),device='cuda:0',generator=generator)
        states[count] = generator.get_state().cpu()
    noise_checks = []
    for e in episodes:
        directory = Path(e['result_path']).parent
        count = (e['steps']+11)//12
        for stage,expected in [('start',states[0]),('end',states[count])]:
            receipt = read(directory/f'sampler_rng_{stage}.json')
            state_path = directory/receipt['state_path']
            assert sha(state_path)==receipt['state_sha256']
            saved = torch.load(state_path,map_location='cpu',weights_only=False)
            assert torch.equal(saved['generator_state'],expected)
            assert receipt['initial_seed']==receipt['checkpoint_sampler_seed']==7091
            assert receipt['extra_random_draws']==0
            destination = OUT/directory.parent.name/directory.name
            destination.mkdir(parents=True,exist_ok=True)
            for source in (directory/f'sampler_rng_{stage}.json',state_path):
                shutil.copy2(source,destination/source.name)
                file_hashes[str(source)] = sha(source)
        noise_checks.append(dict(seed=e['seed'],role=e['role'],planning_calls=count,
            start_and_end_rng_exact=True))
    write(OUT/'noise_audit.json',noise_checks)
    pairs = []
    for group in 'AB':
        for seed in freeze['seeds']:
            old = next(e for e in episodes if e['seed']==seed and e['role']==group+'_130_50')
            new = next(e for e in episodes if e['seed']==seed and e['role']==group+'_entry')
            pairs.append(dict(group=group, cohort=old['cohort'], seed=seed,
                original_success=old['success'], entry_success=new['success'],
                both_success=old['success'] and new['success'],
                only_original=old['success'] and not new['success'],
                only_entry=new['success'] and not old['success'],
                both_failure=not old['success'] and not new['success'],
                original_subgoals=old['native_subgoals'], entry_subgoals=new['native_subgoals'],
                original_steps=old['steps'], entry_steps=new['steps'],
                original_lane=old['lane'], entry_lane=new['lane'],
                original_software_complete=old['software_complete'],
                entry_software_complete=new['software_complete'],
                original_result=old['result_path'], entry_result=new['result_path']))
    contrasts = []
    for group in 'AB':
        for cohort in ('reserved8', 'fresh16', 'all24'):
            values = [p for p in pairs if p['group']==group and (cohort=='all24' or p['cohort']==cohort)]
            delta = np.array([int(p['entry_success'])-int(p['original_success']) for p in values])
            rng = np.random.default_rng(20261122)
            bootstrap = delta[rng.integers(0,len(values),(200000,len(values)))].mean(1)
            ci = np.quantile(bootstrap, [.025,.975]).tolist()
            lost = sum(p['only_original'] for p in values)
            gained = sum(p['only_entry'] for p in values)
            contrasts.append(dict(group=group, cohort=cohort, n=len(values),
                original_success=sum(p['original_success'] for p in values),
                entry_success=sum(p['entry_success'] for p in values),
                both_success=sum(p['both_success'] for p in values), only_original=lost,
                only_entry=gained, both_failure=sum(p['both_failure'] for p in values),
                paired_difference=float(delta.mean()), paired_bootstrap95=ci,
                mcnemar_exact_two_sided=binomtest(gained,lost+gained,.5).pvalue if lost+gained else 1.,
                software_incomplete_original=0, software_incomplete_entry=0))
    funnels = []
    for role in freeze['pairs']:
        for cohort in ('reserved8', 'fresh16', 'all24'):
            values = [e for e in episodes if e['role']==role and (cohort=='all24' or e['cohort']==cohort)]
            reached = [e for e in values if e['native_subgoals']>=1]
            funnels.append(dict(role=role,cohort=cohort,n=len(values),first_subgoal=len(reached),
                success_after_subgoal=sum(e['success'] for e in reached),
                success_after_subgoal_denominator=len(reached)))
    summary = dict(complete=True, observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        episodes=96, different_initial_states=24, waves=waves, paired_initial_checks_passed=24,
        all_legal=True, software_incomplete=0, optimizer_updates=0,new_dagger_trajectories=0,
        all_model_and_source_hashes_unchanged=True,previously_known_results=[],
        inference_rng_start_end_verified=96,
        contrasts=contrasts, funnels=funnels, uncertainty=freeze['uncertainty'],
        stage='primary results delivered; requested diagnostics follow without changing test models/configuration')
    write(OUT/'primary_summary.json',summary)
    csv_write(OUT/'per_initial_state.csv', pairs)
    csv_write(OUT/'rollouts.csv', episodes)
    csv_write(OUT/'paired_counts.csv', contrasts)
    csv_write(OUT/'subgoal_funnels.csv', funnels)
    write(OUT/'raw_file_hashes.json',file_hashes)
    for name in ('project_job_started.json','project_complete.json','project_exit.json','session.json'):
        shutil.copy2(ROOT/name,OUT/name)
    def counts(c):
        return f"{c['original_success']}/{c['n']} → {c['entry_success']}/{c['n']}"
    lines = ['# 入口修复版对 130-50：冻结终测 24 个初态', '',
        '| 组别 | 预留 8 个 | 新增 16 个 | 全部 24 个 | 是否已证明超过 130-50 |',
        '| --- | --- | --- | --- | --- |']
    for group in 'AB':
        rows = [c for c in contrasts if c['group']==group]
        c=rows[-1]
        verdict='支持本冻结协议下净增益' if c['paired_bootstrap95'][0]>0 and c['mcnemar_exact_two_sided']<.05 else '尚未证明可靠净增益'
        lines.append(f'| {group} | {counts(rows[0])} | {counts(rows[1])} | {counts(c)} | {verdict} |')
    for group in 'AB':
        r8,r16,total = [c for c in contrasts if c['group']==group]
        gains = [r['entry_success']-r['original_success'] for r in (r8,r16,total)]
        direction = ('两批方向相反' if gains[0]*gains[1]<0 else
                     '两批同向' if gains[0]*gains[1]>0 else '至少一批净差为零')
        lines += ['', f'{group}：预留批净差 {gains[0]:+d} 例，新增批 {gains[1]:+d} 例，合计 {gains[2]:+d} 例；{direction}。'
            + ('全部24个仅差一两例，现有样本不足以建立可靠优势。' if 0<abs(gains[2])<=2 else '')]
    lines += ['', '表中为原 130-50 → 入口修复的原生完整插入成功数。四套均为完整 DP + Transformer。全部 96 次合法且软件完成；24/24 可观测初态检查通过。训练更新 0，新增 DAgger 0。', '',
        '| 组别 / 批次 | 双成功 | 仅原版 | 仅修复 | 双失败 | 净成功率差 | 配对 bootstrap 95% 区间 | McNemar 双侧 p |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for c in contrasts:
        lo,hi=c['paired_bootstrap95']
        lines.append(f"| {c['group']} / {c['cohort']} | {c['both_success']} | {c['only_original']} | {c['only_entry']} | {c['both_failure']} | {c['paired_difference']*100:+.1f} pp | [{lo*100:.1f}, {hi*100:.1f}] pp | {c['mcnemar_exact_two_sided']:.4g} |")
    lines += ['', '区间按初态整对重采样 200000 次，固定随机种子 20261122；检验只使用不一致配对。小样本区间离散，无不一致配对时的[0,0]仅是重采样局限，不能解释为总体差异精确为零。表中p为未经多重比较校正的单组描述；单组、单批次和两组合计均不据成功总数直接宣称普遍恢复或因果改善。软件未完成：每组每版每批次均为 0。', '',
        '| 策略 / 批次 | 首子目标达成 / 全部 | 达成后完整插入 / 已达成 |',
        '| --- | ---: | ---: |']
    for f in funnels:
        lines.append(f"| {f['role']} / {f['cohort']} | {f['first_subgoal']}/{f['n']} | {f['success_after_subgoal']}/{f['success_after_subgoal_denominator']} |")
    lines += ['', '条件完成率的分母是各策略自己实际达到首子目标的轨迹，不能据此声称对同一组中间状态的后段因果能力提高。', '',
        '两批在任何本轮结果打开前一次冻结：28851000–007 与 28852000–015。预留 8 个未发现合规旧执行，故 96 次均为本轮新执行，没有此前已知的本轮结果。新增 16 个以同一 domain、相同 reset_configuration 和 10 cm 布局生成，未按表现筛选。每波同一初态四套策略，循环平衡并行位置；每版在每位置恰好 6 次。', '',
        '物理/控制/规划为原生 120/60/5 Hz；逐控制帧 DINO，microbatch=2，保留尾 lane 复制、动作可达投影与严格原生成功判据。四套 sampler_seed=7091，各 lane 独立同 seed 生成器，host scheduler；96次实际起止RNG状态均与按原生规划次数重建的24×29噪声消费完全相符。可观测初态按 1e-5 容差核验，不声称隐藏 PhysX 或未来环境随机流完全相同；本轮只估计这一固定推理噪声方案，DINO沿用cuDNN非逐位确定设置。', '',
        f'本轮输出：`{ROOT}`。所有前后端 SHA 与冻结源码哈希在评估前后均核验。历史源码基点为 `{freeze["source_commit"]}`；新增终测包装仅移除了旧实验全局 root 绑定，实际逐文件源码身份见登记。', '',
        '[逐初态 CSV](evidence/tight_entry_final24_20261122_01/per_initial_state.csv) · [96 次 rollout CSV](evidence/tight_entry_final24_20261122_01/rollouts.csv) · [冻结登记](evidence/tight_entry_final24_20261122_01/registration.json) · [主结果 JSON](evidence/tight_entry_final24_20261122_01/primary_summary.json)', '',
        '入口同历史误差和开发正反案例诊断将在主结果之后补充；终测权重与配置继续冻结。', '']
    REPORT.write_text('\n'.join(lines))
    print(json.dumps(contrasts,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
