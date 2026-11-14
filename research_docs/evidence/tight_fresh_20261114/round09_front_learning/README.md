# 第九轮前端学习日志摘要

第九轮使用冻结源码 `87577e84f5f9183ba739e483b34729029629bb99`：单轨完整历史、原固定规划起点、最新52条每 epoch 重复两次。这里没有运行 B2、随机训练起点或 eager 特征读取。A 选中 epoch10/2220 updates，B 选中 epoch9/1998 updates；两者最后均完成 epoch10/2220。新数据的曝光与验证误差不代表自主任务成功。

`summarize.py` 是实际统计调用的脚本字节，SHA256：`5b5c40444a915d8587304261efd681e3fa490ead9cc68f896c9e589398a2dc54`。统计后只将报告里的来源显示从角色目录名 A/B 展开为 session 相对的来源视图；没有改数字或输入绑定，也没有重新运行统计。这个显示修订保存在 `summarize_display_revision.py`，SHA256：`c9c8876d94c5f8bbdaaf72f36745d1879b6026ecb21e2743c7f85e68def3d374`，报告的 `post_run_presentation` 明确区分两份脚本。

实际调用时工作目录是 `/home/CNS2025331827/docs/bc+rl_RSS_continuous_vision_20261114`：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv_isaacsim/bin/python \
  /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/cpu_audits/round09_front_learning01/summarize.py \
  --session /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06 \
  --public research_docs/evidence/tight_fresh_20261114/round09_front_learning \
  > /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/cpu_audits/round09_front_learning01/stdout.json
```

现有输出使用独占创建，不应原地重跑。需要复现时，将脚本复制到新的私有审计目录，并传入新的公开输出目录。

`report.json` 的 `input_bindings` 列出相对 session 的每个 JSON/JSONL 输入、字节数与 SHA256。输入包括最终128机会采集摘要、52条准入索引、A/B训练清单和配置、完整 learning/validation 日志、finished 与后端 prepared 凭据。来源轨迹按既有 source SHA 对齐，没有重新散列轨迹文件。报告复用 `tight_fresh_learning_report.exposure_rows` 与 `tight_fresh_stage_exposure.update_counts`，从实际累计计数差分统计 visits、anchors 与合法 H24 标签槽，并核对 selected/last 的逐轨曝光和最终计数。

选中 epoch 按冻结训练器的“验证 selection 首次严格最小值”规则读取日志；finished 的选中权重 SHA 与后端 prepared 相符。本次没有重新打开检查点元数据或权重，也没有读取 numeric、RGB 或验证载荷。52条来自34个家族；held13来自12个家族。held13统计的是这些轨迹的全部合法专家后段，不是逐步持续持物掩码。重复访问和重叠 H24 标签槽会重复计数，不能当作独立数据量、几何样本数或梯度权重。B epoch9未执行新的纠偏 cohort 指标，因此报告保留空值，没有借用 epoch10指标。

私有完整逐轨统计在 `cpu_audits/round09_front_learning01/report.json`，其最终 SHA 记录于公开报告的 `private_receipt`。本归档没有新增仿真、参数更新或测试。

## 已有逐轨纠偏误差

`fit_report.json` 单独汇总 initial_metrics 与 epoch10 已有的52条逐轨误差，没有修改前述曝光报告。实际脚本是 `fit_summary.py`，SHA256 `7c85bbbc27686c48ec00aa08c0b6315356a710f5195fcce9bec36e30824fc40e`。实际调用为：

```bash
.venv_isaacsim/bin/python research_docs/evidence/tight_fresh_20261114/round09_front_learning/fit_summary.py \
  --session /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06 \
  --evidence research_docs/evidence/tight_fresh_20261114/round09_front_learning \
  --private /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/cpu_audits/round09_front_learning01 \
  > /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/cpu_audits/round09_front_learning01/fit_stdout.json
```

按每轨等权平均 RMSE，新52条的 A 手臂/手指误差分别降低59.09%/64.12%，held13降低67.50%/71.65%；A epoch10正好是选中模型。B **末 epoch10，非选中 epoch9**，新52降低58.60%/62.75%，held13降低65.23%/70.22%。两角色各52条的手臂和手指误差均下降。B选中模型的纠偏误差仍缺失，没有估算或补评。

A拟合的是下一控制步实际 `q[t+1]`，仅相隔1/60秒，不是稳定后的最终位置；B拟合实际专家绝对目标命令 `u`，不能直接跨标签比较误差。held13汇总覆盖这些轨迹的全部合法专家后段，不能据此断言持物阶段已经学会。报告同时保留原日志52条合并后的 RMSE；held13缺少逐轨合法 commit12误差数，故只提供明确等轨权重的平均值，不制造合并加权指标。所有输入是既有小JSON，SHA与前一报告绑定一致，无新模型推理、仿真或测试。
