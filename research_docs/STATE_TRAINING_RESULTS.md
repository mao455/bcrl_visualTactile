# State observation/history repair — actual results

## 2026-11-03：H32直接位置研究，训练尚未开始

新root `position_20261103_01`，基准f1953c7，最高优先级StateDP，停止新StateBC。
400/81,894直接位置标签核验通过；新鲜专家2/2成功，位置重放2/2成功，基础Gate已通过。
两条分别168/153步，14项控制/物理字段逐字节一致；提交→target及真实执行器target误差均0。
实际q相对target的RMSE分别0.085416/0.106316rad，和专家完全相同；未强制写入实际q或物体轨迹。
首次4个启动probe有1次恢复成功和1次写出结果后的FileThread abort；正常关闭对照也在closing阶段SIGSEGV。
改为明确记录的独立worker进程退出后，4/4预启动已通过，父进程确认PID和GPU进程记录消失。
服务position-state-20261103-01随后执行train4/validation8在线专家对照，再自动进入H32原生命令短对照和位置训练。
源代码审查修正了一处优化器循环变量覆盖模型group身份的问题；修订前无新optimizer更新，已完成轨迹不重复。
89项针对性及相关控制回归测试通过。禁止将旧BC10000/DP19000、CPU测试或专家重放成功当作新学生任务成功。
小型实测证据见[foundation_review.json](evidence/position_20261103/foundation_review.json)。
原始train/validation各一条共310tick经过在线特征路径，robot178最大差3.58e-7，
Actor/geometry/位置目标差0；这是教师强制接口核验，不是学生成功，见[input_parity.json](evidence/position_20261103/input_parity.json)。
位置合同见POSITION_ACTION_CONTRACT；启动证据见KIT_RELIABILITY。

预注册：H32 native-command控制最多1000 updates，H32 position最多19000，
分1000/3000/6000/10000/19000里程碑；相同19000 cosine horizon，train320/val40，test40只在选择冻结后。
每次执行12步，保留原任务。State条件仍为Actor141+robot178+geometry85+持续GRU256。
合法validation按原40条排序事先几何筛选前8条，不能按成功结果挑选。
Image自动启动要求至少8条合法validation完整评价、至少2个不同起点完整原生成功，且基础Gate仍有效。
尚无本轮State成功，Image未启动。小预算数值拟合只是继续实验的条件，不是研究成功。


## 2026-11-03 00:19 UTC+8：训练完成，评估不完整，尚无学生完整成功

旧state_full服务inactive，owner/全部自有worker/monitor于11月2日23:21:53退出。
机器状态为`training_complete_evaluation_incomplete`，不是整个实验complete。
StateBC完成10000updates，StateDP完成19000updates；两者均train320/validation40，
覆盖全部320训练身份，按验证集选择10000/19000的EMA，未使用test选择。

|模型/评估|完整成功/已完成|其他实测结果|缺失项|
|---|---:|---|---|
|BC所选10000|0/12|fresh0/4、train0/4、val0/4；全601timeout，无lift/子目标|无|
|DP19000里程碑|0/4|train1条lift后254步fall；另3条601timeout；无子目标|无|
|DP所选19000|0/5|fresh4条601timeout；train1条lift后273步fall；无子目标|第6条启动失败，后6条未启动|

DP里程碑与所选模型诊断分开计数，不合并为9个独立起点。
第6条`dp_single_05`在建立场景前发生FileThread/std::system_error SIGABRT(-6)，
只有runtime_controls.json，无动作或轨迹；该条未知任务结果，不能记为任务失败或成功。
另有DP10000里程碑第4条启动SIGSEGV缺失。三次自动Kit恢复已消耗，未追加重试。
因为两项deferred未解决，旧State12条对照、新test40离线和各组正式20条均未开始。
因此当前不能给出完整正式成功率或新旧State在同一新cohort上的最终改善幅度。

最终validation前12步arm/hand RMSE：BC0.617104/0.491214，DP0.715060/0.500374。
DP从10000步0.798099/0.559021继续下降，仍未形成完整成功。
训练指标仅固定前16条：BC0.540785/0.468841，DP0.612155/0.436233；
优化器实际使用全部train320，不能将16条指标当成全部训练分布指标。

所选DP已完成5条的reset Actor与对应专家均逐字节相同，第0步arm/hand命令已偏离；
reset动作RMSE跨轨迹平均0.619776/0.275060。chunk边界/内部跳变量RMS平均
0.455739/0.188447，提示重规划不连续；尚无因果干预证明其为唯一原因。
在线专家12/12成功与fresh replay2/2 exact16仍支持可运行场景下的控制基础设施。
软件启动不稳定解释缺失评估，不能解释已经实际完成的学生闭环失败。

只读收尾核验169源码哈希、88已提交stage结果、两份best权重SHA；
复用14条视频完整解码结果并重新核对文件哈希，全部通过，未新增训练或仿真。
资源报告含13279.81秒监测间隔（包括停机），整卡采样均值36.599%、最大完整60秒69.208%；
当前协议无固定利用率上限，且无另一项目吞吐指标，不能据此声称零干扰或全时段覆盖。
证据：[训练完成核验](evidence/state_shared_20261102/training_complete_review.json)、
[学习曲线](evidence/state_shared_20261102/results/learning_curves.png)、
[逐轨迹诊断](evidence/state_shared_20261102/results/summary.json)。
旧运行的逐次恢复快照和自动表已从当前报告去重；完整历史仍可由基准提交f1953c7中的本文件及上述小证据恢复。


## 当前运行与成功边界

同一root的deadline不重置：Gate为11月3日03:55:10，训练为21:55:10，绝对终点11月4日00:55:10（UTC+8）。
Stage前后核验源码/数据哈希，位置normalizer只用train320，第一阶段1000后的arm/hand误差阈值均0.25rad。
学习初态专家控制要求完整12条合法结果、至少10条成功且validation至少6/8；原生成功定义不变。
State较长训练在验证连续两次无1%改善时停止，低误差不能替代完整任务成功。
两组Image各按300/1000/3000有界预算顺序执行，只有State成功Gate通过才会开始。
Image只读三路RGB-D、机器人本体和严格过去位置历史，不读Actor/特权几何。

原400、旧best及所有milestone保留。已删除4份与best内部逐成员SHA一致的last副本及旧可重建numeric cache，
释放2,512,265,377字节；清单见[evidence/position_20261103/cleanup.json](evidence/position_20261103/cleanup.json)。
本报告删去重复的运行中快照，不删历史Git或重要旧失败结果。普通owner结束后自动补充曲线/表/视频校验；
未经复核的后续自动产物不自动push。


## 自动收尾实测表（待人工复核与Git提交）

此表只读取普通owner结束后的产物，不新增训练或rollout。旧/前置失效样本不混入新分母。

|组|cohort|已尝试|合法|完整成功|native lift|至少1子目标|
|---|---|---:|---:|---:|---:|---:|
|expert|fresh|2|2|2|0|2|
|expert_learning|train|4|4|4|2|4|
|position_replay|fresh|2|2|2|0|2|

|组|updates|train arm/hand RMSE|validation arm/hand RMSE|
|---|---:|---|---|

最终owner状态：`{"state": "blocked", "reason": "ValueError('Registered source drift: visuotactile/state_full_training.py')", "utc": 1793640372.3562098}`。

逐条reset误差、接触前后首差及阶段误差见公开`summary.json/trace_diagnostics`。
这些观测不唯一证明网络失败机制；同输入/同命令验收与完整成功分开报告。
