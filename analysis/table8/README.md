# Supplementary Table 8 Reproduction

这个包从保存的预测中位数和对应观测值重新计算附录表 8。输入数据、分析程序和核验用的参考结果都在包内。运行不依赖 Claude 会话、原项目目录或模型 API。范围从保存的预测开始，不包含重新拟合传播模型或重新生成 LLM 预测。

## 运行

解压后，在本目录用 Python 3.13 执行。验证环境为 Python 3.13.9。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_table8.py --verify --output table8.csv
```

依赖安装后可离线运行。省略 `--output` 时将 CSV 写到标准输出。指定输出路径时必须使用新文件名。使用 `--verify` 会逐行比较参考结果，检查模型、组别、起点数、误差比和置信区间等字段。

## 数据与方法

- `data/table8_inputs.csv.gz` 包含 22,272 条记录。它取自原项目 `bench/runs/main/forecasts.parquet` 和 `bench/runs/frontier/forecasts.parquet`。预测只保留四舍五入到四位小数后分位数等于 0.5 的记录，覆盖 13 个模型和三种设计。观测来自 `bench/config.yaml` 指定的 `Data/HK_weekly_6pathogens.csv`，按目标周和病原匹配。每行保留实验组、起点、目标周、预测时距、病原、模型、设计、预测中位数和观测。
- 设计 `direct_blind`、`direct_named`、`r0_semi` 分别为盲法直接预测、具名直接预测和机制锚定预测。分析按模型、预测起点、目标周、病原和时距配对，计算 `abs(log1p(observed) - log1p(max(forecast_median, 0)))`。
- 先对每个病原的配对误差求和，再取锚定与直接预测误差比的几何平均。合并分析对 10 个完整模型等权；另报 11 个主实验模型和 2 个前沿模型的逐模型结果。区间基于 4,000 次 bootstrap，随机种子为 `20260927`。逐模型按预测起点重抽样，合并结果在各模型共享起点抽样后再重抽模型。
- `reference/table8_reference.csv` 是核对过的参考结果，共 36 行，包括逐模型、合并及 1–4 周两个对照的结果。附录表格报告 28 个误差比及区间，表注另报告 4 个盲法对照时距结果。两个合并误差比为盲法 **0.73（95% CI 0.54–0.92）**、具名 **0.90（0.77–1.02）**。

`origins_blind`、`origins_named` 和 `origins_grounded` 是各设计的可评分起点数。`n_origins` 是配对后起点数，合并行取纳入模型中的最小值。`n_cells` 统计配对后的起点和病原组合，合并行按模型求和。`p_better` 是 bootstrap 中锚定误差比小于 1 的比例。

表 8 保留原分析程序实际使用的种子 `20260927`。附录 Note 5 中的 `20260925` 出现在 WIS 统计说明中。包内文件可用 `shasum -a 256 -c SHA256SUMS` 核对。
