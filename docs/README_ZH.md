# MultiAttCalib 中文使用说明

本压缩包为 MultiAttCalib 作者团队的实验代码投稿版，包含论文所用的网络、训练流程、迭代位姿校正、评价指标及消融配置。v0.1.1 仅更新文档、代码说明、数据可用性声明和版本信息，核心算法与实验配置保持不变。主要实验参数见 configs/paper.json，具体实现细节见 IMPLEMENTATION_NOTES.md。

## 文件导航

- multiattcalib/geometry.py：投影、最近深度选择、SE(3) 指数映射、旋转误差。
- multiattcalib/matching.py：有效性引导、互惠权重、chance-centered kappa、置信度门控以及所有消融。
- multiattcalib/model.py：双分支 ResNet18、CCAM 风格 RCB、位置编码、自注意力、融合与迭代位姿修正。
- multiattcalib/losses.py：旋转、平移、固定有效点投影监督。
- multiattcalib/data.py：KITTI Raw 标定读取、按 drive 划分、固定测试扰动以及通用私有数据读取。
- multiattcalib/engine.py：训练、验证、独立消融、检查点和推理。
- multiattcalib/metrics.py：旋转和平移 MAE、边际及联合成功率。
- multiattcalib/visualize.py：实验箱线图及实际投影可视化。
- scripts/：合成数据冒烟测试、五种消融批处理、外部方法输入导出及结果汇总。
- tests/：数学性质、极端输入、数据泄漏与端到端前后向测试。

## 私有数据说明

KITTI Raw 可从官方网站获取。课题组自采的 VLP-16–FLIR 数据集及相关标定信息不公开分发，可向通讯作者提出合理申请获取。

通讯作者：Tao Cao（caotao@sqxy.edu.cn）。

本工程不附带原始传感器数据、实验清单、训练权重或预测结果。真实平台的测试输入格式见 DATA.md。精确复现实验应采用原始训练、验证、测试清单与固定扰动；可联系通讯作者咨询原始实验协议。自动生成的划分可用于独立实验，但不可直接视为论文原始划分。

## 常量门控对照

constant_gate 只去掉 kappa 对门控的依赖，保留 n>=2 指示项及可训练参数 b；独立训练后，使用相同测试帧与初始外参。评估程序根据实际预测自动生成 summary.json。

## 软件测试与复现实验

README 中包含安装、训练、评估和消融实验的英文使用步骤，VALIDATION.md 记录软件功能测试范围。合成数据测试用于检验程序运行，不用于证明论文真实数据集上的定量结果。MIT 软件许可不包含第三方数据集或课题组私有数据的分发授权。
