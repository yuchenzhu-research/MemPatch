# ReTrace 项目文件地图 (Repository File Map)

本文件提供了 `yuchenzhu-research/ReTrace` 仓库中所有主要目录和文件的详细映射与说明，方便后续开发、对齐与审计。

---

## 1. 核心 ReTrace-Engine 后端 (`src/retracemem/`)
核心后端的职责是实现确定性的信念修正授权机制（Deterministic Defeat-Path Authorization, DPA）和 RevisionGate 准入控制。

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
* `src/retracemem/providers/base.py`: 定义底层 LLM Transport 接口规范。
* `src/retracemem/providers/provider_factory.py`: 大模型客户端工厂，支持多平台集成。
* `src/retracemem/providers/cached_client.py`: 提供 LLM 请求的本地 JSONL 缓存装饰，防重复调用。
* `src/retracemem/cache/jsonl_cache.py`: 轻量级本地 JSONL 文件缓存实现。

---

## 2. ReTrace-Learn 训练模块 (`src/retrace_learn/`)
本模块是 Paper 1 中支持策略训练的重点部分，涵盖了图提取器（Graph Extractor）、修正提案器（Revision Proposer）以及 DPA-in-the-Loop 强化学习奖励系统。

### 数据管道与生成器 (`data/`)
* `src/retrace_learn/data/build_synthetic_raw_dialogue.py`: 生成合成多 Agent 对话语料并计算 DPA 黄金真值的生成引擎。
* `src/retrace_learn/data/export_graph_sft.py`: 导出图提取任务的 SFT 数据（Module 1）。
* `src/retrace_learn/data/export_revision_sft.py`: 导出修正提案预测任务的 SFT 数据（Module 2）。
* `src/retrace_learn/data/export_rl_rollouts.py`: 生成带 DPA 奖励评分的 RL 轨迹数据（Module 4）。
* `src/retrace_learn/data/jsonl_io.py`: 本地读写 JSONL 数据集的辅助工具。

### 运行时与奖励引擎 (`runtime/`)
* `src/retrace_learn/runtime/dpa_runtime.py`: 承载解析 SFT/RL 输出，并打通 RevisionGate -> DPA 的推理生命周期。
* `src/retrace_learn/runtime/engine_errors.py`: [新引入] 确定性后端各阶段（Parser、Gate、DPA）的结构化错误规约。
* `src/retrace_learn/runtime/graph_extractor.py`: 提取器推理外壳，将原始文本转化为规范的图数据。
* `src/retrace_learn/runtime/learned_proposer.py`: 包装生成策略并向 Engine 提交修改边申请。
* `src/retrace_learn/runtime/path_ranker.py`: 对合法 defeat 路径进行安全评分并审计排序。
* `src/retrace_learn/runtime/reward.py`: 著名的 DPA-in-the-Loop 强化学习奖励设计，计算 JSON 约束、可达边 grounding、stale propagation penalty、gate rejection penalty 以及 NO_REVISION 超量处罚。

### 模型训练脚本 (`training/`)
* `src/retrace_learn/training/train_lora_sft.py`: 基于 HuggingFace 对 Extractor / Proposer 进行 LoRA 微调。
* `src/retrace_learn/training/train_dpo.py`: 通过 DPA-in-the-loop 偏好轨迹进行 DPO 强化学习训练。
* `src/retrace_learn/training/train_grpo.py`: 基于 GRPO 算法和在线 DPA-in-the-loop 奖励进行推理优化。

---

## 3. 评估与实验 (`experiments/`)
* `experiments/multiagent/run_stageab_api_eval.py`: 运行 Stage A (ReTrace-Prompt) 与 Stage B (DirectJudge) 的联合 API 评估兼容入口。
* `experiments/multiagent/select_prompt_smoke_examples.py`: 选取并审计 Stage C API-ICL 的人工作业样本包。
* `experiments/multiagent/apply_smoke_review_decisions.py`: 冻结人工决策包并向 manifest 写入校验哈希。
* `experiments/multiagent/local_training/prepare_mlx_stagec_data.py`: 将 Stage C 数据导出为适合 Mac 本地 MLX 训练框架的格式。

---

## 4. 文档 (`docs/`)
* `docs/architecture.md`: 系统整体分层架构及确定性后端的结构化错误契约设计。
* `docs/benchmark_risk_and_external_validation.md`: [新引入] 阐明内部合成数据集与外部 STALE/CUPMem 评测库的隔离线，防泄露设计。
* `docs/retrace_learn_full_plan.md`: ReTrace-Learn 的阶段三微调及强化学习计划。

---

## 5. 测试 (`tests/`)
* `tests/retrace_learn/test_engine_errors.py`: [新引入] 单元与端到端测试，验证 parser error, schema error, gate rejection penalty, 和 NO_REVISION 过度惩罚的完整链路。
* `tests/retrace_learn/test_reward.py`: 测试 DPA 在环奖励计算的准确性。
* `tests/gate_unit/`: 独立验证 DPA 状态变迁和 RevisionGate 过滤正确性。

---

## 6. 脚本工具 (`scripts/`)
* `scripts/evaluate.py`: 用于主实验协议的一键评测 CLI 入口。
* `scripts/build_failure_analysis.py`: 生成 Stage A 对比 Stage B 的详细错题归因报告。

---

## 7. 冗余与重叠审计 (Redundancies & Architecture Alignments)

### Scripts 与 ReTrace-Learn Exporters
* **现象**: `scripts/export_*.py` 很多时候是直接对 `src/retrace_learn/data/export_*.py` 的包装。
* **设计理由**: 这符合“src 下存放复用库，scripts 下存放单次调用命令行脚本”的规范。`scripts` 文件中通过显式 `sys.path.insert(0, ...)` 注入 src 保证可用性。无需清理，但后续开发应保持 exporter 逻辑纯净于 `src` 内。

### Local MLX 准备与云端 GPU 训练
* **重叠**: `experiments/multiagent/local_training/` 是为 Mac 设备本地开发 MLX-LM 量身定做的；而 `src/retrace_learn/training/` 是云端 GPU 训练标准架构（LoRA / GRPO）。
* **设计理由**: 双轨并行。本地开发可通过 MLX 快速跑通 Pipeline 验证，大规模训练则使用 `retrace_learn/training/` 代码库。

### 外部遗留与归档隔离 (E4 Boundary)
* **注意**: 所有与外部 stale-memory / STALE / CUPMem 评测挂钩的适配器与数据格式转换器，必须严格保留在 `experiments/archive/` (如 `stale_adapter.py` / `cupmem_bridge.py`)，决不能将它们的特有字段污染到 `src/retrace_learn` 的纯净训练模式中。
