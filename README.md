# 农业供需匹配DSS：合成数据库与可复现实验

[English documentation](README_EN.md) | [Dataset card](DATASET_CARD.md)

## 重要声明

本项目中的库存、客户、订单、解析器输出、决策时间、用户评分和人工覆盖记录均为确定性随机种子生成的**合成仿真数据**。它们不是来自真实农场，也没有调用真实LLM端点。因此，本实验适合：

1. 验证论文第5章提出的数据结构、LP分配逻辑、统计分析和审计轨迹；
2. 生成可直接用于Python、SQL和Excel分析的机器可读数据；
3. 作为真实历史数据和真实LLM结果接入前的可复现基准。

它不能被表述为真实农场的实证证据，也不能把合成解析结果写成某一真实LLM的性能。

## 与论文第4、5章的对应关系

- E1 信息抽取：客户、产品、数量、等级、交付日期、替代偏好的Precision、Recall、F1和记录级完全匹配率。
- E2 分配回测：同一日供给和真实需求下，对比B0人工启发式、B1仅优化器和P1端到端DSS。
- E3 效率：对比每日决策周期，并分解解析、验证、求解、解释和人工复核时间。
- E4 人机评价：生成解释清晰度、事实正确性、用途、复核信心、整体可用性以及覆盖原因记录。
- 敏感性/消融：改变客户优先、等级替代和临期库存权重，并分别移除三类权重。
- 稳健性：检查供给上限、需求上限、非负性、过期批次、无效JSON和缺失单位拦截。

优化器利用该问题的运输网络结构，以最小费用流实现论文连续LP的等价求解。目标函数包含销售价值、客户优先、等级替代惩罚和临期剩余惩罚。人工基线被明确形式化为“按消息先后、优先精确等级、偏向新鲜库存、仅不一致地使用允许替代”的启发式；它是仿真基线，不是真实员工表现。

## 数据规模

- 日期：2025-01-01至2025-03-31，共90个日周期；
- 开发集：前60天；冻结测试集：后30天；
- 产品：8种，质量等级A/B；
- 匿名客户：12个；
- 库存批次：5,445条；
- 原始订单消息：1,012条；
- 真实订单行：2,515条；
- 测试集订单行：843条；
- 固定随机种子：`20260820`。

## 文件说明

### dataset

- `synthetic_agri_dss.sqlite`：包含输入、解析、分配、统计检验、敏感性、用户评价和审计结果的完整SQLite数据库。
- `products.csv`：产品、别名、货架期、单位换算和合成价值。
- `customers.csv`：匿名客户与优先权重。
- `grade_compatibility.csv`：等级兼容矩阵与替代惩罚。
- `supply.csv`：按日期、产品、等级和年龄批次展开的可用库存。
- `order_messages.csv`：机器可读的合成非结构化消息、难度和开发/测试标识。
- `order_lines_ground_truth.csv`：人工真值结构，统一包含千克数量。
- `parser_predictions.csv`：合成错误模型产生的解析器输出；`prediction_source`明确说明不是真实LLM。
- `data_dictionary.csv`：主要字段字典。
- `experiment_metadata.json`：版本、日期、样本量、随机种子和合成声明。

### results

- `condition_comparison.csv`：三种条件的测试集均值。
- `daily_metrics.csv`：30天×3种条件的配对日指标。
- `allocations.csv`：批次到订单行的分配明细。
- `extraction_metrics.csv` / `extraction_summary.csv`：E1结果。
- `hypothesis_tests.csv`：H1-H3、效应量、单侧p值和配对Bootstrap置信区间。
- `sensitivity_results.csv`：27个目标权重组合。
- `ablation_results.csv`：基础模型和三项消融。
- `human_evaluation.csv` / `human_evaluation_summary.csv`：E4合成评分。
- `override_logs.csv`：合成人工覆盖记录。
- `robustness_results.csv`：约束与异常处理检查。
- `experiment_summary.json`：紧凑的机器可读实验摘要。

### code

- `generate_dataset.py`：从随机种子重建CSV和SQLite输入表。
- `run_experiments.py`：执行E1-E4、三条件回测、统计检验、敏感性、消融和稳健性检查。
- `schema.sql`：核心关系表的DDL说明。
- `requirements.txt`：仅依赖NumPy和pandas；无需商业求解器。
- `run_all.ps1`：Windows一键重建数据与实验。

## 运行方法

在项目根目录中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python .\run_pipeline.py
python .\validate_repository.py
python -m unittest discover -s .\tests -v
```

`run_pipeline.py`是Windows、macOS和Linux均可使用的推荐入口。若只想复用仓库中已提交的数据集并重新运行实验，可执行：

```powershell
python .\run_pipeline.py --skip-dataset
```

原有分步运行方式如下：

```powershell
python -m pip install -r .\code\requirements.txt
python .\code\generate_dataset.py --output-dir .\dataset
python .\code\run_experiments.py --dataset-dir .\dataset --results-dir .\results --figures-dir .\figures
```

或：

```powershell
.\code\run_all.ps1 -Python python
```

## GitHub仓库结构

```text
.
|-- code/               # 数据生成、实验、SQL结构和Windows脚本
|-- dataset/            # CSV数据、字段字典、元数据和SQLite数据库
|-- results/            # 实验结果与统计检验
|-- figures/            # 实验图形
|-- tests/              # 自动验证测试
|-- .github/workflows/  # GitHub Actions复现检查
|-- run_pipeline.py     # 跨平台一键入口
|-- validate_repository.py
|-- DATASET_CARD.md
|-- CITATION.cff
`-- requirements.txt
```

仓库已排除Python缓存、虚拟环境和本地检查文件。CSV和SQLite文件均小于GitHub单文件限制，可以直接纳入版本控制。

## 公开发布前

本包没有替你选择软件或数据许可证。将仓库设为公开之前，请根据学校、导师和后续发表要求选择适当许可证，并在GitHub仓库中补充`LICENSE`文件。

## SQLite使用示例

```sql
SELECT condition,
       AVG(waste_proxy_rate) AS avg_waste_proxy,
       AVG(quantity_fulfillment_rate) AS avg_fulfillment,
       AVG(decision_cycle_minutes) AS avg_minutes
FROM daily_metrics
GROUP BY condition;

SELECT *
FROM hypothesis_tests
ORDER BY hypothesis;
```

## 本次仿真的主要测试集结果

| 指标 | B0人工基线 | B1仅优化器 | P1端到端DSS |
|---|---:|---:|---:|
| 临期未分配代理率 | 3.84% | 0.65% | 0.77% |
| 数量履约率 | 70.72% | 78.81% | 76.64% |
| 完整订单行履约率 | 57.77% | 65.97% | 64.05% |
| 平均决策周期 | 29.97分钟 | 2.90分钟 | 3.91分钟 |

E1测试集宏平均字段F1为95.74%，记录级完全匹配率为78.77%，无效JSON率为1.82%，进入人工确认的消息比例为31.82%。三个方向性假设在本次合成仿真中均得到支持；具体p值、效应量和95%区间见`results/hypothesis_tests.csv`。

## 结果口径

- `waste_proxy_rate`是“若当日未分配则次日不可用的批次剩余量/当日可用供给量”，属于临期未分配代理，不是实测报废率。
- P1先把检测到的问题送入人工确认；人工确认后的行恢复为真值，未被检测的语义错误会进入求解器，从而形成B1与P1之间的差距。
- H1和H3的正向配对差定义为“人工基线减P1”；H2定义为“P1减人工基线”。
- 近似对称的配对差使用单侧配对t检验；偏斜差使用单侧Wilcoxon符号秩正态近似。所有假设同时给出效应量和配对Bootstrap 95%置信区间。
- E4评分和决策时间是合成值，只能展示分析方法，不能作为真实用户研究结论。

## 替换为真实数据

保留CSV列名与主键关系，替换`supply.csv`、`order_messages.csv`、`order_lines_ground_truth.csv`和`parser_predictions.csv`即可复用实验代码。接入真实LLM时，应冻结提示词和模型版本、保存原始响应、验证状态、人工修正以及端点日期，并重新生成全部结果，不能沿用本合成仿真的统计结论。
