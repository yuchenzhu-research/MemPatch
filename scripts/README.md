# Scripts

所有的实验与评估脚本都保存在 `scripts/` 目录下。

## 评估与审计工具 (无需 GPU)

| 脚本 | 作用 |
|------|------|
| `scripts/evaluate_mempatch_predictions.py` | 官方评测程序，对任何 `predictions.jsonl` 进行指标评分 |
| `scripts/data/audit_decision_boundary.py` | 对数据集做决策边界泄漏与多样性审计 |
| `scripts/data/generate_mempatch.py` | 从 blueprints 渲染生成场景 JSONL |
| `scripts/data/package_mempatch_release.py` | 打包数据集（校验 hash 并生成 manifest.json） |
| `scripts/build_experiment_artifacts.py` | 读取评估输出，自动化构建论文中的 LaTeX 表格和 Matplotlib 绘图 |

## Linux 评测流水线 (`scripts/linux/`)

包含基于 Hugging Face 进行 4-bit 量化推理及评估的 Shell 脚本和 Python 辅助工具。主要脚本如下：

* `scripts/linux/run_smoke_no_lora.sh`: 在本地/云端对 30 个样例的 smoke test 数据集进行完整推理和评估
* `scripts/linux/06_eval_test.sh`: 运行 Path B (五字段 JSON) 在 test500 上的评估
* `scripts/linux/07_eval_path_a.sh`: 运行 Path A (类型化动作序列 + DPA 投影) 在 test500 上的评估
