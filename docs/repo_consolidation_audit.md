# Repository Consolidation & Cleanup Audit

本项目包含两个相关但各自独立的 Paper 的评估/代码体系：
- **Paper 1 (ReTrace-Learn)**: 方法论文，包含 Graph Extractor, Typed Revision Proposer 以及 ReTrace-Engine。
- **Paper 2 (ReTrace-Bench)**: 评测基准论文，包含独立的测试评估架构与基准数据集。

---

## 1. Tracked Files by Major Directory
- `src/retracemem/`: 核心 TM/DPA 执行引擎逻辑。
- `src/retrace_learn/`: ReTrace-Learn 的 learned module 实现。
- `benchmark/retrace_bench/`: 独立的 ReTrace-Bench 评测包。
- `docs/`: 包含 api_providers, architecture, repo_file_map 等说明文档。
- `scripts/`: 提供评测、生成、校验等 python 入口。
- `tests/`: 包含核心单元测试及 `tests/retrace_bench`。

## 2. Untracked Files
- `data/retrace_bench/v1_smoke/`: 刚刚生成的 100 场景 smoke 数据集（除已提交的 `manifest.json` 外）。

## 3. Ignored Files Visible Locally
- `.DS_Store`
- `.venv/` (Python 虚拟环境目录)
- `.pytest_cache/`
- `.pycache_compile/`
- `models/` (本地 MLX community 模型模型缓存)
- `outputs/` (包含基线评测生成的 outputs 和 csv/json 报表)
- `reference/` (外部拉取的仓库指针副本)

## 4. Large Files over 10 MB
- `models/mlx-community/Qwen3-4B-4bit/model.safetensors` (MLX 权重，已 ignore)
- `models/mlx-community/Qwen2.5-1.5B-Instruct-4bit/model.safetensors` (MLX 权重，已 ignore)
- `.venv/lib/.../libmlx.dylib` (动态链接库，已 ignore)
- `reference/` 内各 external repo 的 `.git` 对象及大文件 (已 ignore)

## 5. Generated Data Files
- `outputs/fc_method_results.jsonl`, `fc_metrics_summary.json` 等（评测产生）。

## 6. Cache Directories
- `__pycache__` (遍布项目各目录)
- `.pytest_cache`
- `.pycache_compile`

## 7. Intermediate Iteration Artifacts
- 无明显的临时迭代冲突文件，全管道逻辑均通过 code 生成。

## 8. Experiment Logs
- `outputs/runs/` 及 `experiments/` 目录下的运行 log 文件。

## 9. Old Stage A/B/C Artifacts
- 包含在 `experiments/archive/` 中的老版 Stage A/B 脚本。

## 10. Active ReTrace-Learn Code
- `src/retrace_learn/`

## 11. Active ReTrace-Bench Code
- `benchmark/retrace_bench/`

## 12. Redundant / Obsolete Candidates (将被清理)
- 所有的 `__pycache__/` 目录。
- 所有的 `.pytest_cache/` 目录。
- 所有的 `.DS_Store` 文件。
- 任何临时的 `*.tmp`, `*.bak` 文件。

## 13. Preserved Files (保留并存档)
- `experiments/archive/` 内的历史研究代码（标记为 historical，不删除）。
- `data/retrace_bench/v1_smoke/manifest.json` (小元数据文件，已提交)。
- 所有的核心源码、测试用例和 docs 文档。
