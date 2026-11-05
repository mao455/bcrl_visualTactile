# 当前交接：窄分布StateDP数据采集

2026-11-05本轮准备完成，实际live状态必须读取新root JSON，不能从本文推断已经启动。
工作树`/home/CNS2025331827/docs/bc+rl_RSS_narrow_state`，分支`feature/narrow-state-data`，
新root`outputs/visuotactile/run/narrow_state_20261105_01`。主文档[SLOW_EXPERT_RESULTS](SLOW_EXPERT_RESULTS.md)
包含范围、split、速度、预算及状态/停止/恢复/未来训练命令；当前合同见CONTRACTS。

原400只读；旧State143有效/175尝试评估保持paused，旧专家减速数据捕获blocked，均不恢复。
新数据预计15旧+9补充家族，旧split保持，总16/4/4，约64生产尝试；不为数量扩大条件域。
真实完整传感器采集、多个episode reset和State单更新验收尚待实际结果。
验收后独立systemd继续本阶段，只写盘/审计/manifest/统计，不自动State/Image/触觉训练。
启动只称“通过短监督”，COMPLETE.json存在且验收通过才称数据就绪。

历史合同/结果从expert-speed84cbb98及native-v2 50c06a1查看；不使用旧运行PIDs。
原IDE工作树的CHANGELOG/SESSION_HANDOFF既有改动未改写、未混入本提交。
