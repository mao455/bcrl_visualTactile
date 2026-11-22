# 无损数值审计输入

训练180、开发96、终测96共372轨迹，仅机械复制冻结analyze实际消费的17个数值dataset（缺失可选字段为null），不含视觉、原始HDF5、大权重或缓存。每行保留实际group、source SHA引用、metadata所需config/joint_order/FK身份、完整实际result JSON，以及数组dtype/shape/C-order原始字节base64和逐数组SHA。没有新增分析、阈值或选种。

标准库验证：

```bash
python decode_audit_inputs.py . --verify-only
python decode_audit_inputs.py . --out /tmp/tight_decoded
```

后者生成每轨迹的transition/<字段>.bin、manifest.json、metadata_selected.json和result.json；无需NumPy或h5py即可解码并校验每个原始数组字节。原始大HDF5的SHA引用前次已核验的source manifests，本导出不重新扫描原始视觉载荷。

若需重建最小HDF5，使用NumPy/h5py：读取manifest中的dtype与shape，`a=np.frombuffer(raw,dtype=np.dtype(spec['dtype'])).reshape(spec['shape'])`，在manifest记录的group下创建`transition/<字段>`。在HDF5根写metadata_json属性为metadata_selected.json文本，result_json属性为result.json文本。group、字段、dtype、shape和每个元素的位模式均可还原；缺失字段保持不创建。完整原始HDF5其他未消费字段、视觉帧、压缩布局和未选metadata不在此包，不能据此宣称重建了整份原始HDF5。冻结分析及URDF/任务几何定义仍使用仓库已登记源码和资产。

`array_sha256_manifest.json`列出372条记录每数组的原始SHA；`shards.json`给每个gzip大小/SHA/身份；`verification.json`是标准库完整解码核验。

可选的最小HDF5重建辅助脚本（需要NumPy/h5py；只写新文件、不运行分析）：

```bash
python rehydrate_hdf5.py . --out /tmp/tight_numeric_rebuild --cohort train --identity layout10_train_28817000
```

省略cohort/identity可重建全部372个最小数值HDF5。采用exclusive-create，已有目标文件不会覆盖。三cohort各取一条机械重建并逐字段回比原字节的检查记录在`reconstruction_verification.json`；终测重建仅作字节完整性检查，不运行analyze或改变阈值。

标准库decoder还强制核对shards.json中的每片文件SHA/大小/轨迹数/身份顺序、注册cohort总数，以及每份metadata_selected_json和result_json的原UTF8 SHA；缺失或多出任何jsonl.gz分片会拒绝。
