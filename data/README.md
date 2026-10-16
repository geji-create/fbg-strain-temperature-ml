# 数据目录说明

- `source_synthetic_exports/`：最初导出的两组 70 工况仿真数据、六特征表和对比表；
- `source_archives/`：原始实验数据压缩包；
- `real_raw/`：解压后的四通道实测光谱；
- `synthetic_expanded_v1/`、`v2_nonuniform/`、`v3_same_device/`：逐步改进的仿真扩增数据；
- `synthetic_ablation_uniform_same_device/`：消融实验使用的均匀应变对照数据；
- `external_mendeley_dual_peak/`：第三方公开实测 FBG 温度光谱及处理结果。

原始文件只做归档和读取，不应覆盖；新增预处理结果应写入新的子目录或 `outputs/`。
