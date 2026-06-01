# ReTrace 项目文件地图 (Repository File Map)

本文件提供了 `yuchenzhu-research/ReTrace` 仓库中所有主要目录和文件的详细映射与说明，方便后续开发、对齐与审计。

> 项目按两条 active research track 治理：**ReTrace-Bench**（benchmark track，见第 3 节）
> 与 **ReTrace-Learn**（method track，见第 1–2 节）。规范定义见
> [`docs/project_governance.md`](project_governance.md)。

---

## 1. Authorization Court / ReTrace-Engine (`src/retracemem/`)
本目录是 **ReTrace-Learn method track 内部**的确定性 Authorization Court，其实现名为 **ReTrace-Engine**。
它实现确定性的信念修正授权机制（Deterministic Defeat-Path Authorization, DPA）和 RevisionGate 准入控制，
通过唯一公共入口 `authorize(...)` 提交。**它不是独立论文，也不是独立的顶层研究 track。**

### 根目录与核心模式
* `src/retracemem/__init__.py`: 暴露公共 API `authorize` 授权内核。
* `src/retracemem/authorization.py`: `authorize(...)` 的核心实现，整合 RevisionGate 和 DPA 算法。
* `src/retracemem/schemas.py`: 核心数据类定义（如 `BeliefNode`, `EvidenceNode`, `EvidenceEdge` 等）。

### 准入控制与 DPA 算法 (`tms/` & `memory/`)
* `src/retracemem/tms/__init__.py`: 导出 TMS 模块。
* `src/retracemem/tms/authorization.py`: 基于 Defeat-Path Authorization (DPA) 的信念状态计算核心逻辑。
* `src/retracemem/tms/gate.py`: `RevisionGate` 的实现，验证新增 edges 的语义和作用域。
* `src/retracemem/tms/rollback.py`: 对历史冲突或证据恢复进行撤销/回滚的诊断逻辑。
* `src/retracemem/memory/belief_store.py`: 维护当前处于活跃/已授权状态的信念库（Belief Store）。
* `src/retracemem/memory/episode_ledger.py`: 记录多 Agent 交互历史及事件的账本。
* `src/retracemem/memory/temporal_validity.py`: 证据及连接边的时间跨度与有效性判定逻辑。

### 接口与检索 (`methods/` & `retrieval/` & `verifier/`)
* `src/retracemem/methods/controlled_retrace.py`: 多阶段受控 DPA 执行器，协调提案生成与提交。
* `src/retracemem/methods/batched_controlled_retrace.py`: 批量受控 DPA 执行器，优化 LLM 调用开销。
* `src/retracemem/methods/contracts.py`: 定义方法层的输入输出协议契约。
* `src/retracemem/methods/directjudge.py`: 基线对照方法 `DirectJudge` 的直接预测推理逻辑。
* `src/retracemem/retrieval/candidate_retriever.py`: 用于拉取受影响信念和关联条件的候选信念检索器。
* `src/retracemem/retrieval/typed_retrievers.py`: 类型化检索器，用于精准匹配图节点与关联边。
* `src/retracemem/verifier/evidence_edge_verifier.py`: 验证新证据与已有条件/信念之间的指向边关系。
* `src/retracemem/verifier/prompt_evidence_edge_verifier.py`: 使用提示词方式让 LLM 验证新边。
* `src/retracemem/verifier/prompt_requirement_inducer.py`: 基于 Prompt 的前置条件归纳验证器。
* `src/retracemem/verifier/prompt_typed_belief_extractor.py`: 基于 Prompt 的细粒度信念节点提取。
* `src/retracemem/verifier/typed_edge_response_parser.py`: 对 LLM 预测的边响应进行确定性格式化解析。

### 大模型服务商与传输 (`providers/` & `cache/`)
* `src/retracemem/providers/base.py`: 定义底层 LLM Transport 规范 (`BaseLLMProvider`)，为方法运行与训练提供统一的带有结构化 `ModelCallTrace` 缓存和记录的**规范抽象**。
* `src/retracemem/providers/provider_factory.py`: 大模型客户端工厂，支持多平台集成。
* `src/retracemem/providers/cached_client.py`: 提供 LLM 请求的本地 JSONL 缓存装饰，防重复调用。
* `src/retracemem/cache/jsonl_cache.py`: 轻量级本地 JSONL 文件缓存实现。
* *注：`benchmark/retrace_bench/llm_providers/` 内包含的精简版 Provider 仅供 Benchmark 基线推理使用，以解耦复杂的 Trace 日志开销。*

---

## 2. ReTrace-Learn 训练模块 (`src/retrace_learn/`)
本模块是 Paper 1 中支持策略训练的重点部分，涵盖了图提取器（Graph Extractor）、修正提案器（Revision Proposer）以及 DPA-in-the-Loop 强化学习奖励系统。

### 数据管道与生成器 (`data/`)
* `src/retrace_learn/data/build_synthetic_raw_dialogue.py`: 生成合成多 Agent 对话语料并计算 DPA 黄金真值的生成引擎。
* `src/retrace_learn/data/export_graph_sft.py`: 导出图提取任务 of SFT 数据。
* `src/retrace_learn/data/export_revision_sft.py`: 导出修正提案预测任务 of SFT 数据。
* `src/retrace_learn/data/export_rl_rollouts.py`: 生成带 DPA 奖励评分 of RL 轨迹数据。
* `src/retrace_learn/data/jsonl_io.py`: 本地读写 JSONL 数据集的辅助工具。

### 运行时与奖励引擎 (`runtime/`)
* `src/retrace_learn/runtime/dpa_runtime.py`: 承载解析 SFT/RL 输出，并打通 RevisionGate -> DPA 的推理生命周期。
* `src/retrace_learn/runtime/engine_errors.py`: 确定性后端各阶段（Parser、Gate、DPA）的结构化错误规约。
* `src/retrace_learn/runtime/graph_extractor.py`: 提取器推理外壳，将原始文本转化为规范的图数据。
* `src/retrace_learn/runtime/learned_proposer.py`: 包装生成策略并向 Engine 提交修改边申请。
* `src/retrace_learn/runtime/path_ranker.py`: 对合法 defeat 路径进行安全评分并审计排序。
* `src/retrace_learn/runtime/reward.py`: DPA-in-the-Loop 强化学习奖励设计。

### 模型训练脚本 (`training/`)
* `src/retrace_learn/training/train_lora_sft.py`: 基于 HuggingFace 对 Extractor / Proposer 进行 LoRA 微调。
* `src/retrace_learn/training/train_dpo.py`: 通过 DPA-in-the-loop 偏好轨迹进行 DPO 强化学习训练。
* `src/retrace_learn/training/train_grpo.py`: 基于 GRPO 算法和在线 DPA-in-the-loop 奖励进行推理优化。

---

## 3. ReTrace-Bench 评准套件 (`benchmark/retrace_bench/`)
独立评测基准套件的代码及各子模块，服务于 Paper 2。
* `benchmark/retrace_bench/taxonomy.py`: v1 常量与 Enum。
* `benchmark/retrace_bench/schemas.py`: v1 评测数据类结构。
* `benchmark/retrace_bench/taxonomy_v2.py`: v2 扩展常量与 Enum。
* `benchmark/retrace_bench/schemas_v2.py`: v2 新评测数据类结构。
* `benchmark/retrace_bench/validation_v2.py`: v2 数据与校验逻辑。
* `benchmark/retrace_bench/generation/`: 数据生成模块。
* `benchmark/retrace_bench/protocols/`: 评测协议数据定义。
* `benchmark/retrace_bench/evaluation/`: 跑分器、评测循环与聚合报告。
* `benchmark/retrace_bench/baselines/`: 基线算法。


---

## 4. 数据与运行输出配置 (`data/` & `outputs/`)
- **方法训练数据 (`data/retrace_learn/`)**: 存放 Module 训练及 SFT 内部合成数据集。
- **独立评测数据 (`data/retrace_bench/`)**: 评估专用的基准测试集，有防污染拦截。
- **运行预测与结果输出 (`outputs/`)**: 运行中产生的 local logs、报告等，通常已被 ignore。

---

## 5. 评估与实验 (`experiments/`)
* `experiments/multiagent/run_stageab_api_eval.py`: 运行 Stage A 与 Stage B 联合 API 评估。
* `experiments/multiagent/select_prompt_smoke_examples.py`: 选取并审计 Stage C API-ICL 的人工作业样本包。
* `experiments/multiagent/apply_smoke_review_decisions.py`: 冻结人工决策包并向 manifest 写入校验哈希。
* `experiments/multiagent/local_training/prepare_mlx_stagec_data.py`: 本地 MLX 训练数据准备。
* `experiments/archive/`: 历史归档遗留代码，不影响 canonical 逻辑。

---

## 6. 文档 (`docs/`)
* `docs/architecture.md`: 系统整体分层架构。
* `docs/repo_consolidation_audit.md`: 仓库整理审计与清理计划。
* `docs/retrace_learn_pipeline.md`: ReTrace-Learn 核心管道逻辑文档。
* `docs/retrace_bench/`: ReTrace-Bench 评测设计及协议说明。
