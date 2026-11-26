# Host-init19 首次实际 intent 独立复核

wave46 的首次 placement 已恢复核验通过；19 helper 仍在运行，**本记录不是最终收尾**。

精确 service 为 `tight-paired-host-init-capacity19.service`，InvocationID `34b4aabd29454f7ab57f414fbb67a41a`，helper PID3542303/birth1795506788.47，当前 active/running；其 argv/cwd 与19登记一致。运行中 Result=success/ExecMainStatus0 不能当成已完成成功。

wave46/PID3538948/birth1795506550.85 的 intent、expansion-intent、exit 齐全。只借 CPU5，原134个线程均为 mask8–23/nice10，token不重复；实际扩展134次、恢复134次、总268次 affinity 写。记录实际0.4397364秒后因 `external_cpu_activity` 提前撤回，`restored=true`、`placement_end_verified=true`，无 error、restore_error 或 infrastructure_abort。

独立对仍存活的同birth worker再次逐TID观察：134个线程的 Tgid、前后 starttick 一致，所有 mask 均为8–23、nice10。此为有身份校验的顺序观察，不是原子全线程快照。四份 source pin 全部保持，exit中launch/dispatch摘要与实际小回执一致。观察时仍处于 constructor_started，initialization提交不存在；没有读取main64科学结局或轨迹载荷，也没有停止进程、发信号或改变任何 affinity/nice。

外部活动撤回只有已记录的标记，wave回执没有保存 external snapshot 的逐进程计数，因此本审查不能独立归因外部活动或证明零竞争。CPU4在采样中100%而未借用；实际借核满足原预登记候选和容量门限。该现象不构成提速或科学结果一致性的证据。

最终证据绑定需要包含19 registration/review/activation、实际 preflight、started、本次三份wave46回执、本独立复核，以及后续实际出现的各波intent/expansion/exit/abort、helper exit和最终独立service/PID及逐intent收尾核验。不能预造未发生文件的SHA，也不能用closure15替代19收尾。

完整16项核验、逐TID数值、source/receipt SHA及绑定清单见同名JSON，SHA256 `391395c586ef08b91a1b0ec3eff40f745f30c4fee190aa03786510f465cce80e`。
