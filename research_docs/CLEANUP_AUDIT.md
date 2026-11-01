# Native-command清理审计

当前仅盘点；run02已完成S/Official拟合与诊断，正在恢复History前端Gate，正式训练/评估尚未完成。
本次恢复删除文件0，释放空间0 bytes；400原数据、旧DINO cache、Gate权重及全部失败证据保留。
初始化生成 `cleanup_inventory.json`，按文件记录实际路径、大小，默认保留。

训练评估完成后才核对用途、源码/文档引用、manifest和hash，逐文件生成删除清单并执行。
优先无引用临时checkpoint、重复导出与已由保留证据覆盖的缓存。
每项必须记录删除前SHA256、大小、理由、引用者检查、实际删除结果和释放字节。
不能证明无用的文件保留；不使用未经审计的批量删除。

必保留：原400、split、关键best、220步成功样本、历史T0必要证据、旧正式0成功摘要与必要恢复状态。
未在本轮删除旧tracked入口或历史MD；它们仍可能用于复现旧模型身份，不能因为新入口存在就判为无价值。
失败的Oracle/Kit/History attempts、日志、资源记录及归档blocked保留；不覆盖旧数据。
本次只恢复原run02，新增native尾帧overlay保留明确原cache引用及hash，不借清理重置预算。
