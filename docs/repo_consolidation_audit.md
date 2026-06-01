# Repository Consolidation & Cleanup Audit

> Canonical project structure: see [`docs/project_governance.md`](project_governance.md).
> ReTrace 是 umbrella 项目，按 **两条 active research tracks** 治理（不是三条或四条）：
> ReTrace-Bench（benchmark track）与 ReTrace-Learn（method track）。
> **ReTrace-Engine 是 ReTrace-Learn 内部确定性 Authorization Court 的实现名**，
> 不是独立论文，也不是独立的顶层模块。

本项目包含两条 active research track 的评估/代码体系：
- **ReTrace-Learn (method track)**: 方法论文，pipeline 为 Graph Extractor -> Typed Revision Proposer -> Authorization Court（由 ReTrace-Engine / `authorize(...)` 实现）。
- **ReTrace-Bench (benchmark track)**: 评测基准论文，evaluation-only，包含独立的测试评估架构与基准数据集。

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
- 所有的核心源码、测试用例 and docs 文档。

---

## 14. Pass 2 Consolidation & Hardening Summary (pass2 整理与加固总结)

在 Pass 2 中，我们对仓库进行了进一步的安全加固与接口规范化审计：
- **文献指针化管理**：将 `references/agent_memory/` 和 `references/top_benchmarks/` 下的各类学术与工程相关文献转换为了标准 YAML 结构指针形式，移除了所有 PDF 以及大文件依赖。
- **防止数据污染校验加固**：在 `retrace_learn` 训练链路中新增了递归的数据污染检测器（`check_contamination`），不仅能够检测输入路径，还能够深入递归字典、列表、元组以及集合的所有子项，一旦发现任何包含评测专用的 `data/retrace_bench` 路径，将立即触发拒绝。
- **数据集精简**：完成了 `data/retrace_bench/sample_20/` (20 场景 / 80 问题) 的小规模 canonical 评测数据集生成，并仅将其提交跟踪，而将大规模的 100/1000/2500 级场景保留在 ignored 本地，以防 bloat。
- **接口标准化与向下兼容**：
  - 加固了 `GraphExtractor` 与 `TypedRevisionProposer` 接口规约，支持 `memory_snapshot` 和 `metadata` 等高级特征扩展，同时通过参数类型检查对原有的双参数 positional 调用保持了完全的向下兼容性。
  - 标准化了 `RuntimeResult` 的错误返回格式，直接暴露出与论文架构完全一致的 parser/gate/dpa 结构化 error 及 warning。

