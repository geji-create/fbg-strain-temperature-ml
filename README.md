# FBG 温度–应变解耦：论文复现与实验验证

这是我在大创项目中整理的一套 FBG 温度–应变机器学习实验。最初的目标是复现论文 *Machine Learning Methods for Discriminating Strain and Temperature Effects on FBG-Based Sensors*，随后又加入了真实四通道温度标定、仿真数据扩增、一维 CNN、位置/谱形消融实验，以及第三方公开数据验证。

这个仓库主要想回答两个问题：论文中的六特征机器学习方法能否复现，以及模型换到未见工况、不同仿真器或真实光谱后还能保留多少性能。因此，除了常规随机划分，我也保留了完整工况留出和跨数据来源测试，没有只展示最好看的分数。

![位置与谱形消融实验](figures/publication/fig04_ablation_position_shape.png?v=20261016-2)

> 位置/谱形消融实验，柱高为三次重复实验的平均 MAE，误差棒表示标准差。完整光谱与绝对位置结合后，两项任务都取得了当前最好的结果。

## 目前得到的结果

| 实验 | 数据 | 主要结果 |
|---|---|---|
| 真实四通道温度标定 | 40 个温度条件，CH1–CH4 | 灵敏度 9.48–10.21 pm/°C；LOOCV 温度 MAE 0.95–1.29 °C |
| 完整光谱一维 CNN | 同设备物理仿真，10,000 条样本 | 测试集温度 MAE 1.27 °C，应变 MAE 17.82 µε |
| 位置/谱形消融 | 5 个变体 × 3 次重复 | 完整光谱加绝对位置表现最佳，温度 MAE 1.32 ± 0.12 °C |
| 第三方实测温度验证 | 17 条双峰 FBG 光谱 | 峰位灵敏度 9.29 pm/°C；LOOCV 温度 MAE 2.03 °C |

这里需要说明：CNN 和消融实验使用的是仿真或半实物数据，它们能帮助比较方法，但还不能代表真实传感器在温度—应变联合加载下的最终精度。真正的联合加载实验仍然是下一阶段最重要的工作。

## 复现内容

论文中的主要计算流程包括：

1. 用均匀 FBG 耦合模理论生成反射光谱；
2. 建立 10 个应变等级 × 7 个温度点的 70 个工况；
3. 从每条光谱提取主峰、左旁瓣、右旁瓣的波长和功率，共 6 个论文特征；
4. 训练 Random Forest、Bagging、AdaBoost 和 Gradient Boosting 多目标回归模型；
5. 同时比较论文式随机 70/20/10 划分与“完整工况留出”划分。

## 工程目录

- `data/`：原始实测数据、原始仿真导出、扩增数据和第三方公开数据；
- `src/`、`scripts/`：核心算法与数据处理脚本；
- `outputs/`：各次训练产生的模型权重、指标、预测结果和日志；
- `figures/`：科研绘图与论文级图像；
- `presentations/`：汇报材料说明；含个人信息的终稿仅保存在本地；
- `reports/`：项目报告说明；含个人信息的 Word 与讲稿仅保存在本地；
- `references/papers/`：本地参考论文目录，论文 PDF 不上传公开仓库；
- `docs/`：方法、数据和阶段结果说明；
- `archive/`：旧版汇报、构建缓存和质检渲染，仅保存在本地。

公开仓库的文件范围见 `docs/GITHUB_PUBLICATION.md`。

## 论文中已经明确的设置

- 温度范围：46-105 °F（约 7.8-40.6 °C）；
- 7 个温度条件、10 个应变条件；
- 输入：主峰及左右第一旁瓣的波长和功率；
- 输出：应变和温度；
- 模型：Random Forest、Bagging、AdaBoost、Gradient Boosting；
- 数据划分：70% 训练、20% 验证、10% 测试；
- Boosting 学习率：0.01。

论文没有公开全部 10 个应变数值、光栅长度、折射率调制和仪器采样参数。复现时补充的参数统一放在 `config.default.json` 中。其中 `0-900 µε`、10 mm、中心波长 1548.097 nm 等都是可替换的建模假设，并非论文给出的原始参数。

## 运行环境

建议在克隆后的项目目录中创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需运行 PyTorch GPU 模型，再安装 `requirements-pytorch-cu128.txt`。Windows 下的 `run_python.cmd` 会依次查找项目内 `.venv` 和上一级 `.venv`。

## 快速检查

可以直接双击 `运行快速检查.cmd`，也可以在 PowerShell 中运行：

```powershell
cd fbg_ml_reproduction
.\run_python.cmd run_pipeline.py --config config.default.json --output-dir outputs\quick_check --quick
```

快速检查使用每个工况 10 条光谱和 30 棵基学习器。

## 完整基线实验

可以直接双击 `生成完整仿真并训练.cmd`，也可以运行：

```powershell
cd fbg_ml_reproduction
.\run_python.cmd run_pipeline.py --config config.default.json --output-dir outputs\baseline
```

默认生成 70 × 50 = 3500 条光谱，并输出：

- `spectra.csv`：完整反射光谱；
- `six_features.csv`：论文六特征；
- `model_metrics.csv`：训练、验证和测试指标；
- `test_predictions.csv`：测试集预测；
- `models/`：训练后的模型；
- `representative_spectra.png`：代表性反射谱；
- `model_scores.png`：两种划分方式的 R² 对比；
- `predictions.png`：Gradient Boosting 预测散点图；
- `summary.json`：运行摘要。

## 直接使用已有光谱数据

直接双击 `使用现有CMT数据训练.cmd`，即可处理 `data\source_synthetic_exports` 中归档的既有 CMT 数据。对应命令为：

```powershell
.\run_python.cmd run_pipeline.py `
  --config config.default.json `
  --input-csv "data\source_synthetic_exports\fbg_uniform_cmt_spectra.csv" `
  --output-dir outputs\existing_cmt
```

## 结果解释

`paper_random_70_20_10` 会把同一温度-应变工况的重复光谱分散到训练和测试集中，最接近论文描述，但可能高估泛化能力。

`unseen_condition_70_20_10` 按 `condition_id` 留出完整工况，可检查模型能否预测没有见过的温度-应变组合。若随机划分很高而完整工况留出很低，说明模型主要记住了离散工况。

## PyTorch

论文的四种基线模型全部由 scikit-learn 实现，不依赖 PyTorch。为了后续增加一维 CNN 或端到端光谱模型，当前独立环境已经安装：

- PyTorch 2.11.0；
- CUDA 12.8 运行时；
- 已验证 NVIDIA GeForce RTX 4060 Laptop GPU 可用。

重新安装 GPU 版本时可以运行：

```powershell
.\run_python.cmd -m pip install -r requirements-pytorch-cu128.txt
```

检查 PyTorch 和 GPU：

```powershell
.\run_python.cmd check_gpu.py
```

## 一维 CNN 端到端实验

可以直接双击 `训练一维CNN.cmd`。该实验把每条完整反射谱直接输入一维卷积神经网络，同时预测应变和温度，并继续比较论文式随机划分与完整工况留出划分。

对应命令为：

```powershell
.\run_python.cmd run_cnn.py `
  --input-csv "data\source_synthetic_exports\fbg_uniform_cmt_spectra.csv" `
  --output-dir outputs\cnn_baseline
```

主要输出包括：

- `spectrum_cnn_*.pt`：两种数据划分对应的模型及归一化参数；
- `cnn_metrics.csv`：训练、验证和测试指标；
- `cnn_test_predictions.csv`：测试集真实值与预测值；
- `cnn_training_history.png`：训练与验证损失；
- `cnn_predictions.png`：应变和温度预测散点图；
- `cnn_summary.json`：GPU、PyTorch版本和最终测试结果。

生成传统模型与 CNN 的统一比较表和图：

```powershell
.\run_python.cmd compare_results.py `
  --classical outputs\existing_cmt_baseline\model_metrics.csv `
  --cnn outputs\cnn_baseline\cnn_metrics.csv `
  --output-dir outputs\model_comparison
```

## 真实实验数据

`data\real_raw` 中保存了解压后的 `.list` 文件。当前按数值文件名为温度（°C）进行标定；每个文件包含 `Wave` 和 `CH1`～`CH4` 四路真实光谱。

可以双击 `分析真实数据.cmd`，或运行：

```powershell
.\run_python.cmd analyze_real_data.py `
  --input-dir data\real_raw `
  --output-dir outputs\real_data
```

该流程会输出整理后的长表、亚采样精度主峰位置、四通道温度灵敏度、标定误差以及光谱和标定曲线。当前压缩包中没有应变标签，因此这批数据只能用于温度标定，不能单独完成温度—应变解耦训练。

## 半实物数字孪生

`运行半实物数字孪生.cmd` 会把真实温度谱形和噪声与物理仿真的应变效应结合，生成 0–900 µε、40 个温度点、四通道同步采集的半实物数据，并比较：

- 单通道论文六特征；
- 四通道特征融合；
- 四通道峰位物理解算；
- 物理引导两阶段模型。

当前正式场景采用 `hybrid_config.gradient8.json`，其中 8% 非均匀应变是用于可辨识性分析的假设，不是实验测量值。保守的 3.5% 配置保存在 `hybrid_config.default.json`。

正式命令为：

```powershell
.\run_python.cmd run_hybrid.py `
  --input-dir data\real_raw `
  --config hybrid_config.gradient8.json `
  --output-dir outputs\hybrid_digital_twin `
  --skip-cnn
```

完成半实物数据生成后，可以双击 `运行四通道CNN.cmd`。四通道 CNN 是仿真扩展模型；目前在真实温度谱上存在明显域偏移，不应将其仿真分数作为真实实验精度。

完整的方法、假设和结果解释见 `docs\半实物数字孪生说明.md`，最终对比表和图位于 `outputs\hybrid_digital_twin\comparison`。

## 第三方公开实测数据

为了看看标定流程换到另一套实测光谱后是否仍然成立，我又加入了 Şerif Ali Sadık 发布的双峰 FBG 温度数据集（Mendeley Data v2，DOI：`10.17632/fymrd8js55.2`，CC BY 4.0）。其中共有 17 条 Anritsu MS9710B 实测光谱，每条包含 1001 个波长点；感温 FBG 的温度范围为 2–50 °C，参考 FBG 固定在 20 °C。

双击 `处理第三方公开数据.cmd`，或运行：

```powershell
.\run_python.cmd scripts\process_external_dual_peak_dataset.py
```

该流程会生成标准化 NPZ、带温度标签的 CSV/CSV.GZ、留一法温度标定结果和两张论文图。由于样本数只有 17，这组数据定位为独立外部验证集，不作为深度学习训练集。来源与署名要求见 `data\external_mendeley_dual_peak\SOURCE.md`。


