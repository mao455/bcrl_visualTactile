# 第八轮首次学习数据与阶段拟合入口

本摘要按不可变 `source_sha256` 对齐源模型实际 `train_ids`、当前训练池和合法动作掩码，不依赖轨迹名字相同。只读审计没有读取视觉或验证集张量，没有运行模型。源模型权重 SHA 沿用已绑定的 continuation/finished 记录，本次没有重新扫描大权重文件。

当前训练池为 118 条轨迹、42 个训练初态家族，包含 32,920 个合法专家动作时刻。新增轨迹不等于新增初态。

| 当前选定来源 | 本轮首次接触轨迹 | 涉及家族 | 合法专家步 | 相对来源新增初态家族 |
|---|---:|---:|---:|---:|
| A05 selected | 34 | 19 | 9,005 | 12 |
| B06 selected | 25 | 17 | 7,155 | 12 |

第六、七、八批分别提供 9 / 6 / 19 条，覆盖 1,850 / 1,360 / 5,795 个合法专家步。A05 尚未学习这三批；B06 已学习第六批。B06 selected 是第 3 epoch，第六批每条实际访问 6 次，共 54 次完整历史、21,984 个训练目标槽；last 是第 10 epoch，每条 20 次、共 180 次访问、73,280 个目标槽。last 的剂量没有算入 selected。这里均为该来源训练阶段的剂量，没有把更早继承的阶段累计进去。

只检查最新 19 条会漏掉 A 的 15 条首次学习纠偏、3,210 个合法动作，以及 B 的 6 条首次学习纠偏、1,360 个合法动作。因此新增显式 v3 合集入口，保留 v2 默认行为：`--collection` 仍绑定最新一批，重复的 `--include-collection` 才启用合集。

v3 同时报告全部 34 条、最新 19 条、每批、来源已见与未见。A 的已见/未见为 0/34，B 为 9/25。空分层保留零计数与未知误差。各层先累加合法动作槽的平方误差再计算 RMSE，不平均轨迹 RMSE。不同分层存在重叠，不能再次相加。

每个模型各自从每条真实学生前缀重建完整历史；只复用固定 DINO 特征，不缓存可学习历史隐藏态。源模型和候选模型使用相同 5 Hz / 接管起点规划锚点、相同合法 commit12 槽。保持锚点关节位置的基线只读取该锚点的观测；B 另有保持锚点实际执行器目标的基线。没有使用目标时刻之后的观测。这些是专家轨迹上的前端拟合检查，不能证明完整学生能自主拿起、转正或插入。

41 项相关 CPU 测试通过，其中 15 项为新增合集测试；真实第六至八批 manifest 的只读绑定也通过，A/B 均得到 34 条，批次计数为 9/6/19。尚未对本次候选运行真实阶段拟合。候选注册预算为每角色 10 epoch、每 epoch 137 次访问；最新 19 条各访问两次，其余各一次。这些仍是计划剂量，须在训练完成后由 selected/last 实际记录确认。

待正式 cycle 结束、候选 `finished.json` 存在且评估器代码冻结后，可使用以下命令。`RESEARCH_ROOT` 指原研究项目，`STAGE_FIT_TREE` 指包含本次 v3 的新冻结评估器目录；两个环境变量由操作者明确设置，不复用活动训练源码目录。下列输出必须尚不存在。

```bash
set -euo pipefail
cd "$STAGE_FIT_TREE"
PY="$RESEARCH_ROOT/.venv_isaacsim/bin/python"
RUNS="$RESEARCH_ROOT/outputs/visuotactile/run"
S05="$RUNS/tight_fresh_20261114_05"
S06="$RUNS/tight_fresh_20261114_06"
TREES="$(dirname "$RESEARCH_ROOT")"
SELECTED_TREE="$TREES/bc+rl_RSS_fresh_cycle08_20261114"
REPORT="$S06/cpu_audits/round08_stage_fit_union_01"

"$PY" -m tools.tight_fresh_stage_fit prepare \
  --root "$S06/round_08" --role A \
  --collection "$S06/new_family_08" \
  --include-collection "$S05/takeover_06_tasking4_01" \
  --include-collection "$S05/takeover_07_batch8_m8_01" \
  --selected-frozen "$SELECTED_TREE" \
  --source-frozen "$TREES/bc+rl_RSS_fresh_round05_train_20261114" \
  --output "$REPORT/A"

"$PY" -m tools.tight_fresh_stage_fit prepare \
  --root "$S06/round_08" --role B \
  --collection "$S06/new_family_08" \
  --include-collection "$S05/takeover_06_tasking4_01" \
  --include-collection "$S05/takeover_07_batch8_m8_01" \
  --selected-frozen "$SELECTED_TREE" \
  --source-frozen "$TREES/bc+rl_RSS_fresh_repeat06_20261114" \
  --output "$REPORT/B"

for role in A B; do
  "$PY" -m tools.tight_fresh_stage_fit run-arm --registration "$REPORT/$role/registration.json" --arm source
  "$PY" -m tools.tight_fresh_stage_fit run-arm --registration "$REPORT/$role/registration.json" --arm selected
  "$PY" -m tools.tight_fresh_stage_fit compare --registration "$REPORT/$role/registration.json"
done
```

A05/B06/候选训练源码分别实核为 `32b3988` / `3e648db` / `82d5052`；v3 评估器另行冻结。上述 GPU 命令应沿用正式资源监督和剩余 deadline，不在本次 CPU 审计中执行。

旧 B06 的 9 条、3,038 帧检查实际推理 16.975 秒、总耗时 20.615 秒。本次 34 条有 12,963 帧，按帧数粗估每臂推理约 72 秒、四臂约 5 分钟；不是测得的本次耗时。合集固定特征约 16.1 GiB，还需阶段掩码整理、模型与特征 SHA 校验及文件读取。建议预留 10～15 分钟并记录实际耗时；共享磁盘拥塞时可能更久。

机器路径已从 [summary.json](summary.json) 去除；其私有凭据 SHA 可用于核对原审计。
