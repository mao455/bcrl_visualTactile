# 感知与数据来源

当前训练使用原400自然成功记录的三视角RGB-D：fixed_high、fixed_side、wrist_rgbd，448×252，原60Hz控制/120Hz物理。记录布局、内外参、joint_order与URDF SHA由每个HDF5 metadata给出；不能把旧双相机224×126配置冒充当前数据。

ImageDP每12ticks读取固定缓存的三视角DINOv2-S/14空间特征与显式depth validity，native tick持续更新历史；离线Episode reader与在线NativeDinoSpatial使用冻结revision和批量算术规则。DINO pretrained权重不训练，Image不读state privileged geometry/tactile。

StateDP使用已声明的Actor141/robot178/geometry85，不声称这些几何状态来自真实视觉估计器。触觉与物体状态只按合同作数据/诊断，不暗中增加模型输入。

当前合同见[CONTRACTS](CONTRACTS.md)。原有图像精度、时间和live配对证据边界见[抓取诊断](GRASP_FAILURE_DIAGNOSIS.md)；历史相机布局完整验收见[48a3516 CAMERA_VALIDATION](https://github.com/mao455/bcrl_visualTactile/blob/48a3516/research_docs/CAMERA_VALIDATION.md)。
