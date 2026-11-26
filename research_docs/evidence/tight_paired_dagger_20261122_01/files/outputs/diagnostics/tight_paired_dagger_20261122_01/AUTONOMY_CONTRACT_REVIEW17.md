# 自主评估权限：冻结源码与配置复核

2026-11-24，CPU静态复核。**未发现专家权限阻断项。** 开发8波及终测64波的全部72份计划均与原登记SHA一致，均为 `kind=autonomous`、`expert_permission=false`；每波四套完整学生，case均无 `dagger` 或 `source_data` 权限。两个job绑定通过，两份登记中的295个生产源码SHA全部匹配当前文件。continuation15保留原17–63波的worker参数，最后仍调用原冻结自主worker。

| 位置 | 实际权限约束 |
| --- | --- |
| `tight_paired_evaluate.validate_wave` | 校验原登记、计划SHA、四角色及自主标记；拒绝纠偏配置或源数据权限。 |
| `no_expert_scope` → `worker` | worker全程包在禁专家scope内；专家factory返回仅有reset协议的惰性stub，`get_action`一旦调用即报错；不创建或加载专家网络。队列的expert actor输入替换为零。 |
| `tight_fresh_batch_run` → `run_queue` | 冻结计划选择queue路径；`capture=(kind==takeover)`为false，专家bridge、模型hook、接管trigger、合法专家label及训练准入分支均不执行。 |
| 队列控制循环 | `active_expert_action`仅在capture分支调用；环境实际执行 `controller.act(...)` 产生的完整学生命令。expert shadow数组为零占位，无监督标签含义。 |
| 结果协议复核 | 汇总要求专家禁用回执，并拒绝 `expert_shadow_computed` 不为false或存在dagger结果的轨迹。 |

通用原生环境初始化里仍出现 `create_expert` 调用：它在scope生效后才从专家模块导入factory，因此实际调用的是stub factory。通用setup也仍核验历史专家checkpoint的SHA，作为原任务来源依赖；核验文件身份、调用reset协议与专家动作权限是不同的操作。本复核没有读取该checkpoint载荷。

结论范围是**源码与配置绑定**。未读取本轮64家族结果、轨迹或运行生命周期，也未读取权重/缓存载荷；没有改变生产源码、队列、权限或进程。最终实际运行是否完整仍应由终态protocol/禁专家回执审计确认。stub回执里的 `expert_action_calls=0` 是声明字段，单独看它不是独立计数器；禁止动作的执行约束来自 `get_action` 直接报错及capture分支关闭。

精确输入SHA及逐位置证据见 [JSON](AUTONOMY_CONTRACT_REVIEW17.json)：376个输入均为源码、冻结配置或登记文件。线程上限2；最终绑定复算约0.043秒，另有只读源码检查；新增仿真、GPU调用、策略前向、优化更新均为0。
