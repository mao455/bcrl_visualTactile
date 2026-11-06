# 感知与数据来源

## 2026-11-06 数据模态有效性修订

原400条与新200条必须区分。200条直接窄域采集的固定RGB首帧各200/200为空白，保留低维State用途，Multi-view-valid计0，不可用于完整三视角Image训练。本轮已用真实同初态A/B定位direct入口遗漏固定相机configure；修复新入口和camera_valid内容判据。首条199帧RGB-D三视角全过程验收及开头/接触中/末目视通过；后续新数据仍逐条门控。单lane持久Kit避免未验证的并行相机状态串扰，不声称多lane已验收。

[修复后三视角起/中/末预览](evidence/narrow_repair_camera.jpg)。

以下原400说明不覆盖新200；实际后续可用集合以manifest的state_valid/multi_view_valid及视觉验收SHA为准。State/Image/触觉比较须披露不同集合或限定同一有效子集。


当前训练使用原400自然成功记录的三视角RGB-D：fixed_high、fixed_side、wrist_rgbd，448×252，原60Hz控制/120Hz物理。记录布局、内外参、joint_order与URDF SHA由每个HDF5 metadata给出；不能把旧双相机224×126配置冒充当前数据。

ImageDP每12ticks读取固定缓存的三视角DINOv2-S/14空间特征与显式depth validity，native tick持续更新历史；离线Episode reader与在线NativeDinoSpatial使用冻结revision和批量算术规则。DINO pretrained权重不训练，Image不读state privileged geometry/tactile。

StateDP使用已声明的Actor141/robot178/geometry85，不声称这些几何状态来自真实视觉估计器。触觉与物体状态只按合同作数据/诊断，不暗中增加模型输入。

当前合同见[CONTRACTS](CONTRACTS.md)。原有图像精度、时间和live配对证据边界见[抓取诊断](GRASP_FAILURE_DIAGNOSIS.md)；历史相机布局完整验收见[48a3516 CAMERA_VALIDATION](https://github.com/mao455/bcrl_visualTactile/blob/48a3516/research_docs/CAMERA_VALIDATION.md)。
