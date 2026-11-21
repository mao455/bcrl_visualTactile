"""Publish actual evidence, including the completed independent final test."""
import json
from datetime import datetime
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent
PROJECT = PUBLIC.parents[2]
ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
        '|' + '|'.join(['---'] * len(headers)) + '|'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def export():
    from visuotactile.run_facts import sampler_facts
    sampler_rows=[sampler_facts(ROOT/f'round{n:02d}/learning',r) for n in (1,2) for r in 'AB']
    actual_sampler_seeds=sorted({r['checkpoint_sampler_seed'] for r in sampler_rows},key=str)
    final = PUBLIC / 'final_test/counts.json'
    closure_path = PUBLIC / 'final_closure.json'
    closure = read(closure_path) if closure_path.exists() else None
    outcome = []
    if closure:
        decision = read(PUBLIC / 'final_decision.json')
        assert closure['complete'] and not decision['promote']
        finished = datetime.fromtimestamp(decision['cycle_finished_utc']).astimezone().isoformat()
        seconds = round(decision['elapsed_seconds'])
        duration = f'{seconds // 3600}小时{seconds % 3600 // 60}分{seconds % 60}秒'
        counts = {r['model']: r for r in read(final)}
        outcome = [f'**本轮已完成，不晋升候选。** 实际结束于{finished}，全周期{duration}。最终32例全部合法完整、0软件未完成：完整插入A {counts["A_parent"]["success"]}/8→{counts["A_new"]["success"]}/8，B {counts["B_parent"]["success"]}/8→{counts["B_new"]["success"]}/8（最初130-50父策略→Round 2完整候选）。本轮没有显示自主插入改善；全部旧成功均丢失，B另新增1例。小样本不作统计显著结论。训练、仿真及本轮协调/监控均已退出，测试打开后未再训练或选模。']
    blocks = ['# 两轮自适应DAgger：结果与限制',
        *outcome,
        f'更新时刻：{datetime.now().astimezone().isoformat()}。实验分支`feature/tight-adaptive-dagger-20261119`；原48小时截止未延长。两轮实际采集、八次组件训练、独立曝光验收和各144例自主开发均已完成。',
        'A始终使用原q_reached目标，B使用原u_expert绝对命令目标；各自完整DP+Transformer先后训练，按各自选中DP刷新nominal。原生物理、控制、合法观测、历史及成功判据保持。所有旧合法轨迹进入共同池，原归一化冻结。',
        '## 采集与真实专家恢复',
        '两轮分别使用130-50与Round 1选中的完整学生，保存全部真实学生前缀和专家后缀。学生前缀与交接过渡不作为专家监督。表中完整机会不包含另外保存的资源中断前缀；专家救回不计学生自主成功。']
    collection = []
    training = []
    evaluation = []
    metrics = []
    for index in (1, 2):
        folder = PUBLIC / f'round{index:02d}'
        for r in read(folder / 'collection_summary.json')['table']:
            collection.append([index, r['role'], r['attempts'], r['expert_takeovers'],
                r['accepted'], r['expert_success'], r['legal_ticks']])
        for r in read(folder / 'training_completion.json')['components']:
            assert r['eligible'] and min(r['minimum_visits'], r['minimum_tick_steps']) >= 50
            training.append([index, r['model'], f"{r['actual_epochs']}/{r['actual_updates']}",
                f"{r['selected_epoch']}/{r['selected_updates']}", r['minimum_visits'], r['minimum_tick_steps']])
        counts = read(folder / 'evaluation/counts.json')
        for cohort in ('training', 'development', 'new_development'):
            selected = {r['model']: r for r in counts if r['cohort'] == cohort}
            assert all(r['n'] == 8 for r in selected.values())
            evaluation.append([index, cohort] + [f"{selected[k]['success']}/8" for k in
                ('A_72', 'A_parent', 'A_new', 'B_72', 'B_parent', 'B_new')])
        for model in ('A_parent', 'A_new', 'B_parent', 'B_new'):
            rows = [r for r in counts if r['model'] == model]
            metrics.append([index, model, sum(r['n'] for r in rows)] + [sum(r[k] for r in rows)
                for k in ('success', 'first_subgoal', 'later_subgoal', 'hold30', 'hold60', 'hold120')])
    blocks += [table(['轮', '来源学生', '完整机会', '接管', '准入', '专家救回', '新增合法tick'], collection),
        'Round 1新增29条/4855合法tick，130→159；Round 2新增21条/3500tick，159→180。R2按登记的容量规则结束采集（至少20条且A/B各至少8条），未为凑24条突破资源保护线。',
        'Round 1：接触停滞7/7、搬运8/8、对准6/7、支撑不稳8/9实际接管被专家救回。Round 2：接触停滞2/3、搬运5/5、支撑不稳11/11、第一原生子目标后3/3；另3个未触发机会保留。R2子目标后3条仅127个合法tick，不能把3次救回当作充分后段覆盖。阶段和初态未随机化，不能因救回率不同宣称某阶段因果最优。',
        'R2实际接管91–409tick，包含多个12tick规划相位；首个合法专家标签经原时序合同核对，所有完整记录的执行/关节/时钟/actuator反馈连续。原失败后缀和中断前缀保留并排除训练。',
        '## 实际选中checkpoint的本轮新增曝光',
        table(['轮', '组件', '实际epoch/update', '选中epoch/update', '最低轨迹访问', '最低tick不同提交步'], training),
        'R1全池159条/32363合法tick、R2全池180条/35863合法tick。逐tick正权mask按已提交optimizer step去重，祖先曝光、重叠slot、非法交接和padding均不计数；后端另从原始mask及真实seeded deck独立复算通过。各轮training_completion、selected_tick_exposure.csv.gz与selected_phase_exposure提供逐组件、逐轨、逐tick和阶段证明，A/B原始mask独立验收不依赖训练日志自报。选模仅在本轮达标checkpoint中依各自固定验证损失进行，未按闭环或终测挑权重。',
        '## 完整自主闭环与旧能力',
        '下表每格分母8。R1父为原130-50，R2父为R1完整pair；每个角色父/新在同一两lane物理波实际配对，历史72另波，不能称六策略同波或隐藏物理状态精确重放。两轮均144条合法完整记录、0软件未完成；各72份实际可观测初态检查通过。原开发重复使用，新开发在本轮打开后均不再是未见测试。',
        table(['轮', '初态人口', 'A72', 'A父', 'A新', 'B72', 'B父', 'B新'], evaluation),
        table(['轮', '模型', 'n', '完整插入', '首子目标', '后续子目标', 'hold30', 'hold60', 'hold120'], metrics),
        'R1在各24个配对初态：A保留1/新增2/丢3，B保留0/新增2/丢4。R2：A保留0/新增3/丢3，B保留1/新增5/丢3。R2的A丢失训练28817016/028、新开发28838500；B丢失训练28817016、开发28826003/005。A训练成功2→0/8；B训练1→4/8同时原开发3→1/8，因此A/B收益不一致，旧能力没有稳定保持。hold60仅1秒持物诊断，不是原生成功必要条件。',
        'R1的11个、R2的14个成功变化配对均tick0已出现命令差，物体首次>1mm差异分别60–68/60–69tick；不能把这些时刻解释为首次有害错误或专家反事实真值。全部288条执行审计通过：记录链与可观测反馈连续性最大差0、实际可达命令越界<1.1e-6rad；仍存在大量提议裁剪、可达投影和实际跟踪误差，不能由连续性检查排除闭环反馈问题。',
        'R2六段代表自主视频全解码2269帧并查看三视角关键帧：A28817016父成功305帧，新策略搬到孔口并竖直后失去支撑、461帧跌落；B28826003父成功313帧，新策略搬近孔后转正不稳、约240–360帧失去支撑并超时601帧。A28826000与B28817000新增成功分别287/302帧，完成抓取、搬运、转正与插入。此为结果选取的代表例，不能外推全部失败机制。见round02/autonomous_media_{decode,review}.json与本地media/autonomous_round02。',
        '## 拟合、阶段覆盖与瓶颈判断',
        '同轨、同合法mask、同噪声的共享旧池离线比较：R1原30条完整最终命令arm/hand RMSE相对父模型A下降40.24%/27.39%，B下降42.74%/30.10%；R2进一步下降A37.33%/20.89%、B21.96%/24.39%。但固定validation6最终命令R1反增A3.83%/1.33%、B2.49%/2.44%，R2反增A1.81%/1.39%、B0.40%/0.66%。训练拟合改善没有转成同样的泛化收益。DP自身目标误差及各阶段误差完整列在offline/shared_parent_comparison.csv；两DP目标不同，不跨标签空间比较绝对值。',
        f'新纠偏自身的最终动作arm/hand误差相对当时在线父预测R1约下降A66.56%/46.01%、B79.26%/51.86%，R2约下降A72.45%/55.90%、B83.79%/61.73%。从实际加载checkpoint读取的在线sampler_seed集合为{actual_sampler_seeds}；队列跨episode持续推进RNG，历史未保存每episode状态。这些新数据对比仅是描述性拟合证据；每组只比较来源学生自己的新轨迹，不能冒称完全同噪声对照。',
        '各组件都达到曝光50，但不能据此宣称收敛。R1从首个达标epoch50到60的固定验证损失变化A-DP+1.09%、A后端+1.45%、B-DP+1.11%、B后端+0.57%；R2分别+0.30%、+1.23%、+0.80%、−0.56%。固定train2诊断误差还在下降（不是全训练池损失），全池拟合也改善，而验证未持续改善；当前没有“只是没训练够、继续加epoch就会解决”的证据，也不能证明已达到最优或不存在局部欠拟合。',
        '接管改动取得了真实后段可救状态，其中搬运和支撑不稳提供较多有效纠偏，子目标后覆盖仍很少；无法从非随机采集确定哪种接管最有效。B的部分训练插入和多数持物诊断改善，A/原开发及旧成功并不稳定，不能声称失败整体后移。专家也有无法救回实例，晚期坠落前缀不能一概视为可恢复。',
        '这些结果更支持同时调查数据分布覆盖、组合策略保持与闭环动作反馈，而不是只增加同类数据或训练时长。本轮没有新encoder因果对照，不能将失败归为感知表征唯一瓶颈；下一轮若研究encoder，应先以隔离family的几何/接触读出与有界匹配闭环对照立门槛。本轮不自动扩展第三轮、不做RL或网络改造。',
        '## 最终独立测试']
    if final.exists():
        rows = read(final)
        cases = read(PUBLIC / 'final_test/per_case.json')
        assert len(cases) == 32 and all(r['legal'] and r['software_complete'] for r in cases)
        changes = read(PUBLIC / 'final_test/paired_changes.json')
        execution = read(PUBLIC / 'final_test/execution_audit/receipt.json')
        assert execution['passed'] and execution['cases'] == 32
        blocks += ['预登记28839000–007八初态，冻结全部四套完整pair后一次打开。终测父为最初130-50，候选为R2；每角色父/新两lane同波，A/B分波。此后无训练、选模或重复物理机会。',
            table(['模型', 'n', '完整插入', '首子目标', '后续子目标', 'hold30', 'hold60', 'hold120'], [
                [r['model'], r['n'], r['success'], r['first_subgoal'], r['later_subgoal'], r['hold30'], r['hold60'], r['hold120']] for r in rows]),
            table(['角色', '旧成功保留', '丢失', '新增', '丢失seed', '新增seed'], [
                [role, *[sum(r[key] for r in changes if r['role'] == role) for key in ('retained', 'lost', 'gained')],
                 ', '.join(str(r['seed']) for r in changes if r['role'] == role and r['lost']) or '无',
                 ', '.join(str(r['seed']) for r in changes if r['role'] == role and r['gained']) or '无'] for role in 'AB']),
            table(['模型', '原生lift', '未lift且未首子目标', 'lift后未首子目标', '首子目标后未完成'], [
                [r['model'], r['native_lift'], *[r['failure_stages'].get(key, 0) for key in ('before_lift', 'after_lift', 'after_subgoal')]] for r in rows]),
            '上表失败阶段是原生诊断标签，不代替视频中的几何抬升判断。B候选hold30/60/120均8/8，但完整插入只有1/8，不能用持物诊断代替成功。A首子目标5→3、hold120为6→4，B首子目标4→3，最终新初态也没有一致的阶段改善或旧能力保持。',
            table(['模型', 'arm跟踪RMSE(rad)', 'hand跟踪RMSE(rad)', '提议投影维度比例', '反馈连续性最大差'], [
                [r['role'], f"{r['tracking_arm_rmse_rad']:.6f}", f"{r['tracking_hand_rmse_rad']:.6f}",
                 f"{100*r['reachable_projection_component_fraction']:.2f}%", r['continuity_max']] for r in execution['summary']]),
            '32例执行链全部通过，16份实际可观测初态配对检查通过；实际命令相对原可达区间超差小于1.1e-6rad。各模型均持续存在提议投影与实际关节跟踪误差，A候选arm跟踪误差更大，但轨迹分布不同，不能将此表视为单因素控制因果证据。',
            '终测六段代表视频全解码2537帧并查看三视角关键帧：A28839004父成功254帧，新策略搬近孔并转正后失去支撑，421帧跌落；B28839005父成功286帧，新策略持续竖直持物靠近孔口直至601帧超时，没有完成原生插入，不能将此例归为跌落。B28839002父在孔口反复调整后失败超时601帧，新策略完成插入及释放374帧。选例来自结果，未重跑物理，未人工逐帧审阅全部视频；见final_test/autonomous_media_{decode,review}.json。',
            '完整逐例结果、实际初态配对和执行审计见final_test。最终A旧成功保留0/1、B保留0/2；B新增1例不能抵消旧能力损失。当前证据不支持晋升Round 2候选。样本量每模型8，不据此宣称统计显著退化或彻底否定所有自适应DAgger方法。']
    else:
        done = sum(len(read(q)) for q in (ROOT / 'final_test').glob('paired*/results.json'))
        blocks += [f'终测实际已开始，当前完整批次提交{done}/32例，尚未完成。预登记八初态28839000–007；冻结后原130-50与R2的完整A/B各8例。不能据部分结果报告最终成功率。冻结后禁止新增训练与选模。']
    blocks += ['## 复现、资源及证据',
        '原始模型、全部合法数据、必要best/last与恢复断点继续保留。保护SHA、模型SHA、选中checkpoint路径、训练曲线、逐阶段曝光和逐例结果均在本目录evidence/tight_adaptive_dagger_20261119_01。大模型/原轨/视频只保存在本地，不入Git。统一执行入口为python -m tools.tight_project，必须传显式job及有效有限session；旧已结束session不可直接重启。',
        '本轮多次RAM/磁盘守卫中断与恢复均保留原failure/前缀/断点；不能称全程无中断。用户后续明确授权本会话清理本项目无用派生产物、无损压缩关闭记录，并另授权TinyObjGrasp在2026-11-06 00:00(+08)前的脚本生成数据（cleandataset除外）。清理ledger记录逐文件身份及恢复限制：旧生成数据已删除需重建，记录归档可按清单恢复。未清理其他项目、未终止其他会话进程。运行底线为12GiB磁盘/10GiB RAM/4GiB显存；精确采样限制和所有准入等待如实保留。',
        '轻量结果生成命令：PYTHONPATH=. python research_docs/evidence/tight_adaptive_dagger_20261119_01/quantitative_report.py --round 1（或2）；本报告由同目录publish_results.py只读汇总实际证据。完整本地run为outputs/visuotactile/run/tight_adaptive_dagger_20261119_01，状态及历史恢复记录见原实验报告TIGHT_ADAPTIVE_DAGGER_20261119.md。']
    if closure:
        totals = {kind: sum(r['bytes'] for r in closure['evidence_files'] if r['kind'] == kind)
                  for kind in ('protected_or_selected', 'raw', 'frozen_visual')}
        blocks += [f'最终CPU收尾核验finalize_evidence.py已通过：原保护与选中模型共{closure["protected_and_selected_sha_files"]}份/{totals["protected_or_selected"]:,}字节、当前原始轨迹{closure["raw_sha_files"]}份/{totals["raw"]:,}字节、冻结视觉缓存{closure["frozen_visual_sha_files"]}份/{totals["frozen_visual"]:,}字节逐文件全内容SHA一致；两轮A/B共同人口、冻结归一化、八个选中曝光门与终测冻结模型身份通过。原400条另核验manifest SHA、文件存在与大小，**没有新做400条全部内容SHA**。收尾时余盘{closure["remaining_disk_gib"]:.2f}GiB，活动依赖清单持续保护、无本轮自有worker。正式决定见final_decision.json，逐文件SHA与核验范围见final_closure.json。此次收尾0策略更新、0物理步，不自动开始第三轮或encoder实验。']
    target = PROJECT / 'research_docs/TIGHT_ADAPTIVE_DAGGER_20261119_RESULTS.md'
    target.write_text('\n\n'.join(blocks) + '\n')
    return target


if __name__ == '__main__':
    print(export())
