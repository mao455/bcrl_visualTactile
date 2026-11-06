# Command StateDP当前数据字段

此页是[当前合同](CONTRACTS.md)的字段附录。历史nominal/Correction、epsilon、重复末动作padding的旧schema保留在Git642f5a2，不适用于本轮。

|字段/构造|当前语义|
|---|---|
|metadata|真实joint_order、joint_limits_rad、task配置、专家SHA、camera标定、physics/control dt、family和split|
|obs/time_s、step_id|动作前60Hz观察，连续0…T−1；transition/time_after_s=t+1/60|
|transition/target_after_rad|29维canonical7Arm+22Hand绝对目标弧度；唯一command监督|
|transition/target_before_rad|等于obs/actuator_target_rad；t>0等于上一target_after_rad|
|transition/actuator_substep_target_rad|T×2×29；两个120Hz物理子步实际目标均等于target_after_rad|
|diagnostics_before joint q/qd|canonical rad/rad/s；实际q_after等于下一tick q_before；q不是command监督|
|Actor141|真实原生expert_actor_input_diagnostic；冻结专家RMS与embedding保持|
|robot178|q29、qd29、当前目标29、目标−q29、严格过去请求29/应用29、两age、valid、elapsed|
|geometry85|原base-frame构造；米、rad/s、wxyz，原q/-q等价规则保持|
|frames/rgb_u8|T×3×3×252×448 uint8，真实固定高位/侧位/腕部三视角|
|frames/depth_m/depth_valid|T×3×1×252×448，精确float32米制depth与显式有效位|
|frames/post_index|下一原生曝光索引，terminal为−1；不接reset图像|
|obs/camera_time_s/frame/valid|真实观察时钟、frame counter及逐视角有效位；另验真实内容，拒绝空白固定相机|
|obs/contact_substeps_w_N|T×2×5×3，两次已完成120Hz触觉；时间t−1/120和t|
|transition/contact_substeps_after_w_N|动作后的两次真实样本；时间t+1/120和t+1/60；下一obs逐值相同|
|obs/contact_valid|首tick false，其后true；obs/contact_w_N等于已观察最后子步|

`narrow_data.record`生成完整trajectory缓存time/command/robot/actor/geometry/joint_limits。H24标签末端补零并mask；future与invalid padding不计损失。机器人统计用train所有native tick，geometry统计用train每12tick，action仿射统计也只用train。家族不能跨split；原速/开发数据不进入正式慢速manifest。

新在线构造器用显式canonical_joint_state_v1直接读取同源q/qd，避免归一化再反算误差。旧checkpoint保持legacy_packet_inverse_v1。历史在真实episode ID更换时清空，每60Hz更新；replan和子目标不清空。训练past expert targets明确属于teacher forcing，部署用实际学生提交/应用反馈。

新capture可选gzip4_lossless_v1，旧默认LZF保留；RGB/depth/valid与原始field digest无损核对。压缩不是改帧率、时间戳或复制传感器帧。原400/275保持原文件和身份。
