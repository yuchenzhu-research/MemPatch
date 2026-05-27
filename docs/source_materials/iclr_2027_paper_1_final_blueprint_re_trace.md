# 1. 最终决定：第一篇是方法论文，不做完整新 benchmark，也不做 full latent

## 1.1 最终论文身份

**目标会议：ICLR 2027 Main Conference**  
**内部标题：ReTrace: Evidence-Preserving Reversible Belief Revision for Dynamic Agent Memory**  
**更正式的投稿标题候选：Evidence-Preserving Reversible Belief Revision for Evolving Personalized Agents**

这篇 Paper 1 研究的不是“agent 应该如何从大量 episodes 学出 latent semantic memory”，也不是“再造一个大规模 benchmark”。它研究一个更窄、更能按时做完、又能与 CUPMem 正面区分的问题：

> 当长期用户记忆随时间变化，旧 belief 可能被后来证据削弱或推翻时，agent 能否在**保留原始证据**的前提下，形成一份**可回溯、可撤销、只沿有理由路径更新**的当前可用 belief view，从而减少 obsolete-memory misuse？

### 一句话版本

> **Paper 1 = 可审计的、证据保留的、TMS-inspired reversible belief revision method，在 STALE 与 Memora 等现有 benchmark 上证明价值。**

## 1.2 这篇明确不做什么

| 不做的内容 | 为什么第一篇不做 |
|---|---|
| 完整的新 benchmark 论文 | 太占人工标注与验证时间；你现在应把全部精力放在方法与实验闭环上 |
| Full latent memory / memory token learning | 难解释、训练重、与 Auto-Dreamer / MEM1 等近期工作正面碰撞 |
| Learned episodic-to-semantic consolidation policy | NEMORI、Auto-Dreamer、Useful Memories Become Faulty、AgeMem 已经覆盖大块叙事空间 |
| 开放世界 causal topology discovery | 没有现成 gold benchmark，很容易变成 LLM 猜依赖关系 |
| 大规模 RL / GRPO | 第一篇用 SFT/LoRA 训练一个轻量 relation verifier 即可；RL 留给后续 |

## 1.3 这篇和 CUPMem 的真正区别

CUPMem 的核心是：**fixed typed schema + write-time LLM adjudication + propagation-aware candidate search + ACTIVE/STALE/UNKNOWN_CURRENT readout**。

ReTrace 的核心是：

1. **不采用固定生活领域 slot ontology**：不预定义 `health_and_mobility`、`routine_and_transport` 这类领域结构作为传播边界；
2. **不删除或覆盖原始 episodic evidence**：过去的证据永远可以回溯；
3. **不允许仅靠主题相关性扩散 revision**：旧 belief 只有在存在可验证 justification path 时，才会在当前 view 中被阻止或条件化；
4. **revision 是可撤销的授权变化，而不是永久性文本重写**：后续证据可以解除旧 blocker；
5. **学习组件只学习局部 relation / justification edge**，符号式 gate 决定是否允许 revision，不让一个黑箱 judge 直接决定所有 memory fate。

你不要写：“我们首次研究 stale memory。”STALE 已经做了。你要写：

> Recent work exposes implicit state invalidation, but current solutions rely on fixed state ontologies or unconstrained LLM adjudication. We study evidence-preserving, reversible belief revision in which a memory can stop governing current behavior only through a traceable defeat path supported by later evidence.

## 1.4 对后续两条线的决定

- **完整 benchmark paper**：不并入 Paper 1。Paper 1 只做一个很小的人工诊断集作为 error analysis，暂名 `BoundaryAudit`，不把它包装成新 benchmark contribution。等 Paper 1 结果确认 over-revision 确实严重，再决定是否扩成单独论文。
- **真正 latent learning paper**：保留为 Paper 2。它需要建立在 Paper 1 的 evidence path、revision failure taxonomy 与实验管线上。
- **不要现在同时承诺两篇 ICML 2027 主会稿**：在 ICLR 提交后，根据结果选择“扩展 benchmark”或“latent method”中的一条作为下一篇主线；我更推荐优先做 latent method，因为它更接近你真正想解决的深层问题。

---

# 2. Related Work 边界与代码核验：你需要读谁、跑谁、避开谁

下面所有代码状态均按 **2026-05-27** 我已经核验到的公开页面整理。对于没有找到论文专属官方仓库的工作，只作为 novelty threat / related work，不作为你必须复现的 baseline。

## 2.1 你点名的三篇与 AgeMem

| 工作 | 做了什么 | 公开代码情况 | 对 Paper 1 的影响 | 你如何处理 |
|---|---|---|---|---|
| **NEMORI / What Deserves Memory: Adaptive Memory Distillation for LLM Agents** | 将对话组织为 coherent episodes，再基于 prediction error 蒸馏 semantic knowledge；将“值得记忆”近似为“已有知识难以预测的新内容” | **有，论文正文直接链接官方 repo**：[nemori-ai/nemori](https://github.com/nemori-ai/nemori) | 它压缩了“episodic → semantic distillation”创新空间，但不研究 justification-grounded revision | **可运行 baseline**；重点看 episode integration、distillation interface、LoCoMo/LongMemEval 评测脚本 |
| **Useful Memories Become Faulty When Continuously Updated by LLMs** | 诊断持续 textual consolidation 会先提升后退化，甚至低于 no-memory；强调 raw episodes 应保留，consolidation 应被 gate | **未找到论文页面给出的专属官方代码仓库** | 你不能再声称首次发现 false consolidation 有害；但它支持你的 evidence-preserving 动机 | **必读、不作为可运行 baseline**；Introduction 与 failure motivation 引用 |
| **Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents** | fast acquisition + slow offline consolidation；GRPO 用端到端任务表现学习跨 session abstraction；读取 provenance-linked trajectories | **未找到论文专属官方代码仓库**；论文说明实验建立在 [OpenTinker](https://github.com/open-tinker/OpenTinker) 与 [verl](https://github.com/volcengine/verl) 上 | 它直接挡住了“第一篇做 future-value learned consolidation”的路线 | **强 novelty threat，不复现**；用于说明你不做 offline consolidation，而做 belief revision authorization |
| **AgeMem / Agentic Memory: Learning Unified LTM and STM Management** | 将 `store / retrieve / update / summarize / discard` 等 memory operation 变为 agent tool actions，用三阶段 RL 与 step-wise GRPO 学习何时操作记忆 | **有公开实现仓库**：[y1y5/AgeMem](https://github.com/y1y5/AgeMem) | 你不能声称首次学习 memory action；但它没有显式处理 justification path 与 revision boundary | **related work + 可选重型 baseline**；第一版不要求完整复现 RL |

## 2.2 Paper 1 最直接相关工作地图

| 工作 | 核心能力 | 与你的重合 | 你的明确差异 | 代码链接与优先级 |
|---|---|---|---|---|
| **STALE / CUPMem** | implicit conflict benchmark；typed current-state adjudication；propagation-aware search | 旧 belief 被新证据隐式失效 | 你不用 fixed slots；保留 evidence；TMS-style proof gate；可撤销 authorization | [icedreamc/STALE](https://github.com/icedreamc/STALE)；**第一优先级，必须跑** |
| **Memora** | weeks-to-months evolving memory；FAMA 惩罚 obsolete memory use | 多次 mutation 后旧记忆污染回答 | 它是评测，不是方法；你的方法要在 FAMA 上证明价值 | [geniesinc/Memora](https://github.com/geniesinc/Memora)；**第一优先级，必须跑** |
| **MemoryAgentBench / FactConsolidation** | incremental multi-turn memory evaluation；包含 conflict resolution / selective forgetting | 可作为已有 conflict benchmark 补充 | 不作为主 battlefield，仅做第三 benchmark 或 non-regression | [HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)；第二优先级 |
| **LongMemEval** | knowledge update、temporal reasoning、long-context memory | 社区认可度高的 bridge benchmark | mutation 压力低于 Memora；只做 non-regression | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)；第三优先级 |
| **Graphiti / Zep** | temporal knowledge graph；时间有效性与历史事实存储 | 你的 temporal provenance storage 可借鉴 | temporal KG 本身不提供 TMS defeat path 或 over-revision control | [getzep/graphiti](https://github.com/getzep/graphiti)；**看代码结构，不必直接作为主实现** |
| **TriMem** | raw dialogue + atomic facts + entity profiles 三层记忆；source dialogue IDs；TextGrad 风格 prompt evolution | 与 episodic evidence / profile belief 有接近之处 | 不处理 implicit invalidation 与 reversible revision | [tmlr-group/TriMem](https://github.com/tmlr-group/TriMem)；阅读数据结构与 retrieval |
| **StructMemEval** | 测 memory structure，而不只测 retrieval | 支持你论证“结构化 memory 值得研究” | 它不是动态 belief invalidation benchmark | [yandex-research/StructMemEval](https://github.com/yandex-research/StructMemEval)；related work / 可选实验 |
| **MEM1** | RL 学 compact shared state，联合 memory consolidation 与 reasoning | 后续 latent/memory policy 相关 | 不围绕 evolving personalized belief revision 与 provenance defeat path | [MIT-MI/MEM1](https://github.com/MIT-MI/MEM1)；**Paper 2 重点读** |
| **A-MAC** | admission control：future utility、confidence、novelty、recency、content type prior | 都强调可审计 memory control | 它只判断是否写入，不处理旧 belief 如何被后来证据合法阻止 | [GuilinDev/Adaptive_Memory_Admission_Control_LLM_Agents](https://github.com/GuilinDev/Adaptive_Memory_Admission_Control_LLM_Agents)；可选 baseline |
| **A-MEM** | linked note / evolving memory organization | 是常见 memory framework baseline | 不围绕 justification-based revision | [WujiangXu/A-mem-sys](https://github.com/WujiangXu/A-mem-sys)；在 Memora 可作为现成对照 |
| **Mem0** | extract-update + retrieval 的工程型 memory layer | 常见工业 baseline | 缺少 proof-carrying revision | [mem0ai/mem0](https://github.com/mem0ai/mem0)；可跑一个基础对照 |
| **Memory-R1** | RL 学 `ADD / UPDATE / DELETE / NOOP` 与 answer retrieval | 都研究 learned memory update | 它是操作学习，不是 evidence-preserving revision；我在 arXiv 页面未定位作者官方代码链接 | 只作为 related work，除非后续找到作者 repo |
| **ContextWeaver** | 为工具轨迹构造 dependency graph，并使用 validation signal 选择上下文 | dependency-path 思想与你相近 | 它在 SWE-Bench 工具轨迹上工作，验证信号来自测试执行，不是个性化 memory state | 论文必读；我未在论文 HTML 中核实到专属官方代码链接 |

## 2.3 这张 related-work 图应该怎样写进论文

论文 Related Work 不要堆五十篇，正文只围绕三条线：

1. **Dynamic personalized memory evaluation**：STALE、Memora、MemoryAgentBench、LongMemEval；说明现有评测已经揭示 stale/obsolete misuse，因此你不再主张新问题定义。
2. **Memory construction and consolidation**：NEMORI、TriMem、A-MEM、Auto-Dreamer、Useful Memories Become Faulty；说明它们研究写入、蒸馏、合并或其风险，而你研究后续证据如何合法改变当前可用 belief。
3. **Structured and learned memory control**：CUPMem、Graphiti/Zep、A-MAC、AgeMem、MEM1、ContextWeaver；说明 fixed schema、temporal graph、RL actions 或 trajectory dependency 都不足以给出 personalized belief revision 的 evidence-preserving authorization mechanism。

核心对比句可以固定成：

> Unlike distillation and consolidation methods that transform episodes into reusable summaries, and unlike typed-state systems that retire memories inside a fixed schema, ReTrace preserves episodic evidence and revises only the authorized current-belief view through traceable defeat paths induced by later observations.

---

# 3. Abstract、Introduction 与贡献设计：论文到底讲什么故事

## 3.1 论文主张必须收得很准

不要讲：

- “我们首次研究旧记忆失效”；
- “我们首次提出 episodic / semantic memory”；
- “我们首次用图做 agent memory”；
- “我们首次让模型学 memory update”；
- “我们首次发现 consolidation 可能有害”。

这些都已经被最近工作占掉了或部分覆盖。

你要讲：

> Long-term memory agents need not only retain and update information, but also justify when earlier beliefs should cease to govern current responses. Existing typed-state revision systems rely on fixed ontologies and LLM adjudication, while consolidation-based systems risk overwriting useful evidence. We propose an evidence-preserving reversible belief memory that separates immutable episodes from an authorized belief view, and permits a revision only through a traceable later-evidence defeat path.

## 3.2 建议的三条 contribution

### Contribution 1：问题重新落点，不是新 benchmark

**Evidence-preserving reversible belief revision formulation.**  
你将动态 memory 使用形式化为：原始 episode 不被覆盖，系统维护的是一个可撤销的、由证据与理由路径授权的 current belief view。这个 formulation 区别于 retrieval-only、destructive consolidation 以及 fixed-slot stale marking。

### Contribution 2：方法 ReTrace

**ReTrace architecture.**  
一个 slot-free 的 temporal proposition graph：

- immutable episodic ledger；
- open-text belief nodes；
- temporal provenance；
- learned local relation verifier；
- TMS-inspired revision gate；
- query-time authorized current basis；
- natural rollback when later evidence defeats or expires a blocker。

### Contribution 3：现有 benchmark 上的严格实验 + 小型诊断分析

**Evaluation without claiming a new benchmark.**  
主实验使用 STALE 与 Memora，补充使用 MemoryAgentBench FactConsolidation 或 LongMemEval。你另外构造 40–80 个 `BoundaryAudit` contrastive cases，只用于分析系统是否误伤未受影响 beliefs，不把它声称为完整 benchmark。

## 3.3 英文 Abstract 草稿骨架

> Long-term memory agents must adapt to evolving user circumstances without allowing obsolete beliefs to govern current behavior. Recent benchmarks expose implicit invalidation and repeated memory mutation, while existing solutions either revise memories inside fixed state schemas or consolidate experiences into abstractions that may overwrite useful evidence. We introduce **ReTrace**, an evidence-preserving reversible belief memory architecture. ReTrace stores episodic evidence immutably, maintains an open proposition graph with temporal provenance, and authorizes belief revision only through a traceable defeat path predicted by a lightweight relation verifier and constrained by a truth-maintenance-inspired gate. Unlike destructive consolidation, ReTrace can suppress an outdated belief for current use while preserving its historical validity and allowing later rollback. We evaluate ReTrace on STALE and Memora, with additional conflict-resolution and preservation diagnostics under controlled token and call budgets. Our results aim to show that provenance-grounded reversible revision reduces obsolete-memory misuse without relying on predefined state slots or unconstrained LLM adjudication.

在你没有实验结果之前，Abstract 最后一句只能写成 **aim to show** 的规划版本；出结果以后再替换为具体数字。

## 3.4 Introduction 的四段逻辑

### 第 1 段：真实需求

长期个性化 agent 会不断积累关于用户的事实、偏好、计划与约束；现实信息会改变，因此 memory 不能只是 retrieval cache。

### 第 2 段：已有进展与仍然存在的问题

- STALE 已揭示 implicit conflict 与 stale premise；
- Memora 已揭示 repeated mutation 下 obsolete memory 被复用；
- CUPMem 说明显式 state adjudication 有效；
- 但是 CUPMem 依赖 fixed typed slots 与 LLM-based propagation proposal；
- consolidation 系统又可能把 episode 抽象错并覆盖关键证据。

### 第 3 段：你的 insight

> Revision should change which beliefs are currently authorized, not erase the evidence from which those beliefs were derived. A belief should stop governing behavior only if later evidence supplies a traceable defeat path.

这一段用腿骨折例子：

- 历史：用户骑车通勤；
- 新证据：腿骨折；
- 正确：近期骑车方案不能支配通勤建议；
- 但历史 episode 仍应保存，用于回答“过去如何通勤”；
- 若后来医生允许恢复运动，旧习惯可以重新成为候选，而不是被一次永久删除。

### 第 4 段：方法与实验概述

介绍 ReTrace 的 episodic ledger、proposition graph、relation verifier、TMS gate、authorized current basis；然后说明评测 STALE、Memora、可选 MemoryAgentBench，以及一个只做诊断的 BoundaryAudit slice。

---

# 4. 方法 Pipeline：你真正要实现什么

## 4.1 总体 pipeline

```text
Incoming session x_t
      │
      ▼
Episode Ledger: 保存原始或最小压缩的 session evidence，带 timestamp/source_id
      │
      ▼
Belief Extractor: 提取开放文本 proposition，不映射到固定 domain slot
      │
      ▼
Candidate Relation Retrieval: 为新 evidence 找少量可能被影响的旧 beliefs
      │
      ▼
Learned Relation Verifier: 判断 SUPPORT / SUPERSEDE / BLOCK / CONDITION / NONE / UNCERTAIN
      │
      ▼
TMS-inspired Revision Gate: 只有存在被证据支持的 defeat path 才改变 authorized belief view
      │
      ▼
Query-time Current Basis: 检索并组织当前可用 beliefs、历史证据、被阻止前提与未知项
      │
      ▼
Fixed Answer Model: 基于 authorized basis 生成答案
```

## 4.2 你需要保留的两层 memory，但不要写成 semantic consolidation

### Episodic Ledger

```text
EpisodicEvidence:
- id: session_007_span_02
- timestamp: t_7
- text: "I broke my leg yesterday and will be in a cast for six weeks."
- immutable: true
```

### Belief View

```text
Belief:
- id: belief_bike_commute
- proposition: "The user commonly commutes by bicycle."
- supported_by: [session_002_span_01]
- temporal_scope: historical / currently_authorized / currently_blocked / unresolved
- challenged_by: [session_007_span_02]
- justification_path: optional
```

这里的 `currently_blocked` 等字段不是预设生活 slot，而是**通用授权结果**。你不做 `health.current_mobility`、`transport.current_commute_mode` 那种固定 ontology。

## 4.3 Relation Verifier：第一篇唯一必要的 learned module

### 输入

```text
new evidence e_t
candidate prior belief b_i
optional candidate condition c_i
local temporal context
```

### 输出关系

```text
SUPPORT       新证据继续支持该 belief
SUPERSEDE     新证据直接替换该 belief
BLOCK         新证据破坏该 belief 所需条件
CONDITION     belief 本身未失效，但执行需额外条件
NONE          没有合法影响
UNCERTAIN     信息不足，不允许旧 belief 继续成为默认前提
```

### 关键范围控制

第一篇**不做任意长 multi-hop graph reasoning**。只允许两种 revision proof：

1. 直接替换路径：

```text
new evidence --SUPERSEDE--> prior belief
```

2. 最长两跳 prerequisite 路径：

```text
new evidence --BLOCK--> condition <--REQUIRED_BY-- prior belief
```

这一步非常重要：它避免你的 Paper 1 膨胀为开放世界因果图学习；同时足以覆盖 STALE 的大量 Type I / Type II 情形。

## 4.4 TMS-inspired Gate：你与 LLM-as-judge 的区别

LLM 或 verifier 只能提出局部关系，不能直接任意删除 belief。系统规则是：

```text
- Episode 永不删除；
- 没有 accepted defeat path，不允许旧 belief 被阻止；
- CONDITION 不等于 INVALIDATE：保留偏好或计划，只限制其当前执行；
- UNCERTAIN 时禁止将旧 belief 当作当前默认值，但不能捏造替代状态；
- 新 evidence 解除 blocker 或时间约束到期后，belief 可重新进入授权候选。
```

例如：

```text
旧 belief：用户经常骑车上学。
新 evidence：用户腿骨折并需要六周恢复。
proof：fracture BLOCKS cycling_ability；bike_commute REQUIRED_BY cycling_ability。
结果：骑车通勤在六周内不能作为当前建议依据；历史 belief 仍保留。
```

而：

```text
旧 belief：用户喜欢泰国菜。
新 evidence：用户腿骨折。
proof：不存在 food_preference 依赖 cycling_ability 的合法路径。
结果：不允许 revision。
```

## 4.5 训练策略：有学习，但不做 RL

### Phase A：Prompt-only feasibility

先让强模型输出 relation 与 evidence path，在少量 STALE/Memora 样例和你手写的 20 个 cases 上测试。目的不是投稿，而是快速判断这个 interface 是否有效。

### Phase B：生成训练数据，但不把它包装成 benchmark

构造 `RevisionPairs-Train`：

- 自动生成新事件、旧 belief、必要条件与 relation label；
- 用 LLM 进行语言改写，保持 label 不变；
- 训练集与评测 benchmark topic/persona 分离；
- 人工只抽查一小部分质量，不做完整 benchmark annotation。

### Phase C：LoRA/SFT Relation Verifier

训练一个小型 open-weight instruct model 或 classification-style verifier，学习 relation label 与 evidence span extraction。第一篇不使用 PPO/GRPO；这样你能把精力放在可复现和实验对比上。

### Phase D：固定 answer model，比较 memory policy

主实验中固定 answer backbone、retrieval top-k、最大 memory tokens 与调用预算，只更换 memory authorization mechanism，避免 reviewer 质疑提升来自更多 API 调用。

---

# 5. 数学形式：第一篇只写够支撑方法的公式

## 5.1 Memory 表示

在时间步 \(t\)，系统维护：

\[
\mathcal{M}_t = (\mathcal{E}_t, \mathcal{B}_t, \mathcal{R}_t)
\]

其中：

- \(\mathcal{E}_t\)：不可破坏的 episodic evidence 集合；
- \(\mathcal{B}_t\)：从证据中提取的 belief propositions；
- \(\mathcal{R}_t\)：支持、替换、阻止、条件化与时间关系。

与 CUPMem 的差别：\(\mathcal{B}_t\) 不是固定 slot 中的值，而是开放文本 propositions；与 consolidation 方法的差别：\(\mathcal{E}_t\) 不被 semantic summary 替代。

## 5.2 Learned relation verifier

对于新证据 \(e_t \in \mathcal{E}_t\) 与候选 belief \(b \in \mathcal{B}_{t-1}\)，模型预测：

\[
p_\theta(r \mid e_t, b, h_t), \quad
r \in \{\texttt{SUPPORT},\texttt{SUPERSEDE},\texttt{BLOCK},\texttt{CONDITION},\texttt{NONE},\texttt{UNCERTAIN}\}
\]

其中 \(h_t\) 是最小必要时间上下文或候选 prerequisite 描述。

SFT 损失先保持简单：

\[
\mathcal{L}_{\text{rel}}
= - \sum_{i=1}^{N}
\log p_\theta(r_i^* \mid e_i, b_i, h_i)
\]

可以再加 evidence span 监督：

\[
\mathcal{L}
= \mathcal{L}_{\text{rel}} + \lambda \mathcal{L}_{\text{span}}
\]

其中 \(\mathcal{L}_{\text{span}}\) 要求模型指出支持 relation 的原句片段，增强可审计性。

## 5.3 Revision authorization gate

定义一个 belief 是否在时间 \(t\) 可以支配当前回答：

\[
A_t(b) = \mathbb{1}\left[
\neg \exists e,c:\; D_t(e,c,b)=1
\right]
\]

其中 defeat path：

\[
D_t(e,c,b)=
\mathbb{1}[\operatorname{BLOCK}(e,c)]
\cdot
\mathbb{1}[\operatorname{REQUIRES}(b,c)]
\cdot
\mathbb{1}[\operatorname{ValidAt}(e,t)]
\]

此外，直接更新可以写为：

\[
D_t^{\text{direct}}(e,b)=
\mathbb{1}[\operatorname{SUPERSEDE}(e,b)]
\cdot
\mathbb{1}[\operatorname{ValidAt}(e,t)]
\]

论文表述不要说这是“真实因果影响的完整建模”；只说它是**conservative revision authorization rule**：不存在证据支持的 defeat path，就不允许 revision 扩散。

## 5.4 Query-time authorized basis

给定查询 \(q\)，先检索候选 beliefs 与 episodes：

\[
\mathcal{C}_q = \operatorname{Retrieve}(q, \mathcal{E}_t \cup \mathcal{B}_t)
\]

然后构造当前可用 basis：

\[
\mathcal{G}_q =
\{b \in \mathcal{C}_q : A_t(b)=1\}
\cup
\{e : e \text{ supports or defeats a retrieved belief}\}
\]

最终回答：

\[
\hat{y} = F_\phi(q, \mathcal{G}_q)
\]

其中 \(F_\phi\) 在公平实验中固定不变。

## 5.5 你需要报告的诊断量

主 benchmark 使用官方指标；你自己的诊断不做新 benchmark headline：

\[
\operatorname{UnsupportedRevisionRate}
=
\frac{\#\{\text{suppressed beliefs without valid audit path}\}}
{\#\{\text{suppressed beliefs}\}}
\]

\[
\operatorname{ProtectedBeliefPreservation}
=
\frac{\#\{\text{protected beliefs left authorized}\}}
{\#\{\text{protected beliefs}\}}
\]

它们只在 `BoundaryAudit` 小型人工分析集上报告，用来说明你的方法不会为了提高 STALE 分数而粗暴屏蔽大量无关记忆。

---

# 6. 实验方案、代码落地、Figure/Table 与投稿节奏

## 6.1 Benchmark 最终选择

| 优先级 | Benchmark | 角色 | 必须报告的指标 | 代码 |
|---:|---|---|---|---|
| 1 | **STALE** | 主实验 A：implicit invalidation 与 stale premise | Overall；SR / PR / IPA；Type I / Type II | [icedreamc/STALE](https://github.com/icedreamc/STALE) |
| 1 | **Memora** | 主实验 B：weeks-to-months repeated mutation | FAMA；weekly/monthly/quarterly；remembering/reasoning/recommending | [geniesinc/Memora](https://github.com/geniesinc/Memora) |
| 2 | **MemoryAgentBench / FactConsolidation** | 补充 conflict-resolution 与 incremental multi-turn test | 官方 accuracy / subset 结果 | [HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) |
| 3 | **LongMemEval** | non-regression bridge，证明不因保守 revision 损害普通 memory | 官方各类任务结果 | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) |
| 分析集 | **BoundaryAudit，40–80 cases** | 只做 over-revision / preservation case study，不声称新 benchmark | URR / PBP + 人工案例 | 你自己构造，放 supplement |

### 为什么这套 benchmark 足够？

- STALE 直接挑战 CUPMem 与 implicit conflict；
- Memora 已经覆盖高 mutation pressure：weekly、monthly、quarterly 三档；
- MemoryAgentBench 已有 FactConsolidation，可补 conflict-resolution；
- BoundaryAudit 只补你方法最想说明的“不会误伤”，而不耗费你半年做完整 benchmark。

## 6.2 Baseline 最终选择

### 必须跑

| Baseline | 用途 | 实现策略 |
|---|---|---|
| Full History / Long Context | 无 memory architecture 参照 | 使用 benchmark 官方设置 |
| Retrieval-only | 证明问题不只是检索 | BM25 / embedding 两种至少一种 |
| Mem0 或 A-MEM | 工程型 memory baseline | 优先复用 Memora 已有集成；必要时自己跑 |
| **CUPMem** | STALE 上最接近对手 | 直接使用官方 repo |
| LLM Direct Revision | 证明不是多加一次强模型判断就行 | 与你的 verifier 使用同一 backbone/call budget |
| ReTrace w/o TMS Gate | 消融：只有 relation model、无 gate | 你的实现 |
| **ReTrace Full** | 主方法 | 你的实现 |

### 条件允许再跑

| Baseline | 加入价值 | 是否值得第一阶段投入 |
|---|---|---:|
| NEMORI | 开源、与 episodic/semantic memory 最接近 | 高，Memora/LongMemEval 结果可比较 |
| Graphiti-only | 检验 temporal KG 是否足够 | 中；集成较耗时，先不阻塞主实验 |
| TriMem | 新的三层 memory/provenance 对照 | 中；LoCoMo 接口可能要改接 benchmark |
| A-MAC | admission-control 对照 | 中低；不是 revision 主对手 |
| AgeMem | RL memory manager | 低；代码重、训练重，正文 related work 足够 |
| Auto-Dreamer | consolidation 对手 | 不可作为必须 baseline，目前未定位专属官方代码 |
| Useful Memories Become Faulty | motivation / diagnosis | 不可作为必须 baseline，目前未定位官方代码 |

## 6.3 公平实验控制

你必须固定并报告：

| 控制项 | 规则 |
|---|---|
| Answer model | 所有 memory methods 尽可能共享相同 answer backbone |
| Retrieval top-k | 保持一致或分别报告检索预算 |
| Input token budget | 上限一致 |
| Stored memory budget | 报告 episodic 与 belief view 的 token/entry 数量 |
| LLM/API calls | 统计 construction-time 与 query-time 调用次数 |
| Latency / cost | 至少报告平均 tokens 与 calls；若使用 API，再报告估算成本 |
| Random seeds / retry | 保持一致并记录失败重试规则 |

## 6.4 必须做的 ablations

| Ablation | 你要回答的问题 |
|---|---|
| 去掉 immutable episode ledger | evidence preservation 是否真的重要 |
| 去掉 TMS gate，仅用 verifier/LLM verdict | 结构化停止规则是否减少误修订 |
| 去掉 temporal validity | 暂时性 constraint 是否会被错误永久化 |
| 去掉 evidence span / provenance | 可回溯证据是否提升可靠性 |
| direct LLM judge vs trained verifier | 你的学习模块是否比黑箱临场判断稳定 |
| 等成本比较 | 提升是否只是来自更多 token 或 calls |

## 6.5 Figure 设计

### Figure 1：Motivation + failure case，放 Introduction

一张横向图，三列：

```text
Old episode                 New evidence                  Query / failure
骑车通勤                    腿骨折六周                    仍推荐骑车装备
```

下面两条 system path：

```text
CUPMem / unconstrained revision: typed slot 或 LLM adjudication
ReTrace: evidence → defeat path → currently block bike commute; preserve historical episode
```

目的：让 reviewer 30 秒理解你不是做 retrieval，也不是做 consolidation。

### Figure 2：ReTrace pipeline，放 Method 开头

画五个盒子：

```text
Episode Ledger → Proposition Graph → Relation Verifier → TMS Gate → Authorized Basis → Answer
```

突出：

- episode immutable；
- relation edge 可见；
- blocked belief 仍留在 history；
- rollback arrow 从 later evidence 回到 authorization view。

### Figure 3：STALE / Memora 主结果可视化

不要塞太多图。建议一个柱状图：

- 左：STALE PR / IPA；
- 右：Memora FAMA weekly/monthly/quarterly；
- methods 只显示 Retrieval、CUPMem/NEMORI、LLM judge、ReTrace。

### Figure 4：BoundaryAudit error decomposition，放 Analysis

堆叠条形图或矩阵：

```text
Correct revision / missed revision / unsupported revision / over-blocking / unresolved-safe
```

展示 ReTrace 为什么有效，而不只展示最终分数。

## 6.6 Table 设计

| 表格 | 内容 | 目的 |
|---|---|---|
| **Table 1：Related Work Positioning** | CUPMem、NEMORI、Auto-Dreamer、AgeMem、ReTrace 五行；比较 schema、episode preservation、revision proof、learned consolidation、benchmarks | 明确不撞题 |
| **Table 2：Main Results on STALE** | SR / PR / IPA，Type I / II，Overall | 最关键主表 |
| **Table 3：Main Results on Memora** | FAMA by weekly/monthly/quarterly 与 task type | 证明连续 mutation 有效 |
| **Table 4：Ablation + Cost** | 去掉各组件后的结果、tokens、calls、latency | 防守工程公平性质疑 |
| **Table 5：Qualitative Cases** | 3–5 个事件的 old belief/new evidence/path/authorized result | 展示解释性 |

## 6.7 代码实现顺序：让 Opus 帮你写代码时按这个拆任务

### Week 1：只跑官方仓库，不写自己系统

1. Clone 并跑 [STALE/CUPMem](https://github.com/icedreamc/STALE)。
2. Clone 并跑 [Memora](https://github.com/geniesinc/Memora) 的最小 persona/evaluation。
3. 跑 [NEMORI](https://github.com/nemori-ai/nemori) 的已有 benchmark command，理解 memory 输出格式。
4. 阅读 [Graphiti](https://github.com/getzep/graphiti) 的 episode / entity / temporal relation 数据结构。

### Week 2：建立统一 evaluation harness

你自己的 repo 先只建：

```text
retracemem/
  adapters/
    stale_adapter.py
    memora_adapter.py
    memoryagentbench_adapter.py
  backends/
    retrieval_baseline.py
    cupmem_wrapper.py
    nemori_wrapper.py
  evaluation/
    stale_metrics.py
    memora_fama.py
    cost_tracker.py
  outputs/
```

目标：所有方法输出统一 JSONL：

```json
{
  "query_id": "...",
  "retrieved_evidence": [],
  "authorized_basis": [],
  "answer": "...",
  "tokens": {},
  "calls": {}
}
```

### Week 3–4：实现 ReTrace 最小系统

```text
retracemem/
  memory/
    episode_ledger.py
    belief_store.py
    relation_types.py
    temporal_validity.py
  verifier/
    prompt_verifier.py
    relation_sft_data.py
    relation_model.py
  tms/
    gate.py
    authorization.py
    rollback.py
  generation/
    basis_builder.py
    answerer.py
```

### July：从 prompt 版切到 learned verifier

- Prompt verifier 只用来验证方向；
- 生成 RevisionPairs-Train；
- LoRA/SFT 一个关系判定器；
- 与同 backbone direct-judge 做公平比较。

### August：只做实验、写作和复现清理

- 主表；
- 消融；
- cost；
- BoundaryAudit 案例；
- README、运行脚本、配置文件；
- 不再扩展 latent 或新 benchmark。

## 6.8 时间与 Go / No-Go

ICLR 2027 官方 deadline 目前仍需以未来正式 CFP 为准。以 ICLR 2026 的 9 月下旬投稿节奏作为内部危险线，你应这样执行：

| 时间点 | 必须完成 | 不满足时的决定 |
|---|---|---|
| 6 月中旬 | STALE/CUPMem 与 Memora 最小评测跑通；完成 20 条 BoundaryAudit cases | 若仍未跑通，停止加新想法，只修 pipeline |
| 6 月底 | Prompt ReTrace 在少量样例上显出明显 revision quality 优势 | 若无优势，不进入训练，重新审题 |
| 7 月中旬 | Learned verifier 初版；STALE 主指标与 Memora small run | 若只在单一小指标提升，不宣称主会 ready |
| 8 月上旬 | 完整主表、消融、cost、人工案例分析 | 若 STALE 与 Memora 无一致提升，考虑延后投稿 |
| 8 月下旬 | 初稿 + 代码清理 + 外部反馈 | 停止新增系统模块 |
| 9 月 | 定稿与投稿 | 只修证据链与呈现 |

---

# 7. 后续论文只略定方向：不要影响 Paper 1 执行

## 7.1 未来 benchmark paper：先不立项为主线

你可以在 Paper 1 中保留一个 `BoundaryAudit` 小型分析集，规模 40–80 cases，用来测：

- 应阻止的旧 belief 是否被阻止；
- 不应改变的 preference 是否被误伤；
- CONDITION 是否被粗暴判成 invalidation；
- justification path 是否合理。

只有当 Paper 1 跑出明确结果：现有系统的 over-revision / unsupported suppression 确实严重，且你的标注协议稳定，才把它扩展成真正独立 benchmark paper。否则不要提前投入数百小时人工构建。

## 7.2 Paper 2：真正的 latent learning / reversible consolidation

**建议优先于完整 benchmark paper推进。** 题目暂定：

> **Learning Justification-Carrying Latent Belief Memory under Delayed Utility**

它继承 Paper 1 的：

- episodic provenance；
- defeat path；
- revision gate；
- obsolete misuse failure；
- BoundaryAudit 与现有 benchmark pipeline。

然后再增加：

- episodic → semantic / latent consolidation；
- future task utility；
- rollback；
- compact memory budget；
- 与 Auto-Dreamer、MEM1、AgeMem、NEMORI 比较。

## 7.3 最后一句执行原则

从现在起，你只围绕这一句话做 Paper 1：

> **ReTrace does not try to learn all of memory. It learns and constrains when a previously supported belief may cease to govern the present, while keeping the evidence necessary to revise that decision later.**

第一篇要赢在：问题窄、方法清楚、代码可复现、实验正面对齐 STALE 与 Memora、与 CUPMem 区别无法被一句话抹平。你真正想做的 latent memory，不消失，但不再拿来拖垮第一篇。

