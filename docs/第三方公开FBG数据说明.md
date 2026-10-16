# 第三方公开 FBG 光谱数据说明

## 1. 数据来源

本项目新增 Şerif Ali Sadık 发布的 *Optical Spectrum Dataset for Dual-Peak Temperature Measurements*：

- 数据仓库：Mendeley Data，version 2
- DOI：<https://doi.org/10.17632/fymrd8js55.2>
- 许可：CC BY 4.0
- 仪器：Anritsu MS9710B 光谱分析仪
- 数据规模：17 条双峰 FBG 实测光谱，每条 1001 个波长采样点
- 波长范围：1548.68–1550.68 nm
- 工况：感温 FBG 为 2–50 °C，参考 FBG 固定在 20 °C

原始压缩包、OSA TXT 文件、仪器截图和元数据均保留在 `data/external_mendeley_dual_peak/raw`，未修改原文件。正式使用或发表时需引用原作者、数据集名称和 DOI。

## 2. 在本项目中的定位

这组数据来自其他研究团队和另一套测试设备，可作为独立于本项目仿真程序及自有设备的外部实测证据。由于只有 17 个温度工况且没有应变标签，它适合验证光谱解析、温度灵敏度和双峰重叠处理，不适合直接训练深度网络，也不能单独验证温度—应变解耦精度。

## 3. 已完成的处理

`scripts/process_external_dual_peak_dataset.py` 完成以下工作：

1. 自动跳过 OSA 文件头，解析波长和光功率数值区；
2. 检查 17 个文件是否具有一致的 1001 点波长网格；
3. 使用 Savitzky–Golay 平滑并进行逐光谱归一化；
4. 保存标准化 NPZ、元数据 CSV 和长表 CSV.GZ；
5. 根据原始元数据中的双峰位置计算灵敏度、峰间距和重叠区；
6. 使用留一交叉验证建立简单线性温度标定基线；
7. 输出两张 300 dpi PNG 及 PDF/SVG 矢量科研图。

## 4. 分析结果

| 指标 | 结果 |
|---|---:|
| 感温峰灵敏度 | 9.285 pm/°C |
| 峰位—温度线性拟合 R² | 0.9763 |
| 留一法温度 MAE | 2.034 °C |
| 留一法温度 RMSE | 2.494 °C |
| 留一法温度 R² | 0.9712 |
| 参考峰标准差 | 12.917 pm |
| 最小双峰间距 | 67 pm |

外部感温 FBG 的 9.29 pm/°C 与本项目四通道实测的 9.48–10.21 pm/°C 处于同一量级，说明自有温度标定结果具有合理的物理尺度。约 23–35 °C 时双峰间距接近或低于仪器 96 pm 的有效分辨率，是峰值提取最困难的区间，也是留一法误差增大的主要原因之一。

## 5. 生成文件

- 标准化光谱：`data/external_mendeley_dual_peak/processed/external_dual_peak_spectra.npz`
- 元数据和预测：`data/external_mendeley_dual_peak/processed/external_dual_peak_metadata.csv`
- 光谱长表：`data/external_mendeley_dual_peak/processed/external_dual_peak_spectra_long.csv.gz`
- 数值摘要：`outputs/external_mendeley_dual_peak/analysis_summary.json`
- 外部光谱图：`figures/publication/fig06_external_dual_peak_dataset.*`
- 外部验证图：`figures/publication/fig07_external_temperature_benchmark.*`

在论文或汇报中建议把这组数据表述为“第三方公开实测数据上的独立温度验证”，不能表述为“外部应变—温度解耦测试”。
