# ICLR 2027 Paper 1 蓝图 v3：ReTrace 的证据保留式可逆授权

**目标会议：** ICLR 2027 Main Conference
**方法名：** ReTrace
**候选标题：** *ReTrace: Evidence-Preserving Reversible Authorization for Evolving Agent Memory*
**本文档定位：** 当前科学主张、实验顺序与 go/no-go 边界的中文权威草案。
**审计基线：** `integration/retrace-v1-complete` @ `5e8d6e2d1a494d572d6d0fa929595bb198154390`。

---

## 0. 这次更新为什么必要

论文想法本身没有被推翻。需要更新文档，是因为代码已从 AB-1B 的 offline controlled harness 推进到 v1 integration scaffold，但新代码尚未产生支持论文结论的真实比较结果，并且部分工程文案把“接口打通/模拟运行”写成了“正式阶段完成”。

本版本锁定三条事实：

1. **方法主线不变。** Paper 1 研究的是 evidence-preserving reversible authorization，不是 latent memory、RL memory policy 或 consolidation learning。
2. **工程状态必须降到事实层。** 当前已有 provider、end-to-end runner、官方 frozen STALE adapter / offline non-leaking wiring、Memora negative-pilot artifact 与 mock validation；尚未产生 official STALE model result。
3. **实验顺序必须前置可证伪测试。** 官方 STALE 400-case 数据已经公开；正式评测前必须先确认 non-leaking wiring、官方输出 schema、Stage A/B 公平持久状态语义与 evaluator 边界。

---

## 1. 论文的核心问题

长期 agent memory 不仅要记住过去，还必须知道过去的信息何时不再适合支配当前回答。

Paper 1 研究：

> 当 later evidence 改变用户当前状态时，一个曾被证据支持的 earlier belief，凭什么仍然有资格进入当前回答依据，或者凭什么应被暂时禁止使用？这个资格变化能否在保留原始证据的前提下，具有明确、可审计、可撤销的理由路径？

一句话方法：

```text
ReTrace = immutable evidence ledger
        + typed belief / condition / evidence-edge graph
        + deterministic Defeat-Path Authorization Algorithm (DPA)
```

核心观念：

```text
memory revision 不等于删除或覆盖历史证据；
memory revision = 在新证据下重新计算旧 belief 的当前使用资格。
```

---

## 2. Paper 1 明确不做什么

本文不把下列方向作为第一篇的贡献：

- latent memory / memory-token learning；
- RL 或 GRPO memory-action policy；
- learned episodic-to-semantic consolidation；
- 完整新 benchmark 论文；
- 把 external benchmark 的 official scored examples 用于调 prompt 或训练；
- 把 heuristic/manual fixture 包装成主方法；
- 在 A/B 结构价值未成立前训练 Stage C；
- 将 Stage A live outputs 自行宣布为 Stage C 的 gold labels。

Stage C 若未来启动，学习的只是 local typed-edge prediction，而非 latent memory representation；其训练数据必须来自 development-safe、经人工审核且与 official evaluation 隔离的数据。

---

## 3. ReTrace 的四层框架

### 第一层：Immutable Evidence Ledger

系统以 append-only 方式保存原始事件及其时间、来源和 provenance。例如：

```text
e_old:     “我平时骑自行车去学校。”
e_injury:  “我腿骨折了，六周内不能骑车。”
e_recover: “医生允许我重新骑车。”
```

此层解决：历史证据不能因系统更新而被无声覆盖。

### 第二层：Belief / Condition Construction

从 evidence 提取可用于推理的 open-text beliefs，并诱导其当前可用所依赖的条件：

```text
b_bike:     用户通常骑自行车通勤。
c_mobility: 用户当前具备骑车通勤所需的行动能力。
b_bike --REQUIRES--> c_mobility
```

完整 end-to-end 系统中，这一步可以由 prompt-based semantic components 完成；但在 primary controlled attribution 中，这些输入被固定，以避免 extraction/retrieval 错误污染核心机制比较。

### 第三层：Evidence-Preserving Authorization - 本文核心

Later evidence 只提出局部 typed effects：

```text
e_injury  --BLOCKS-->  c_mobility
e_recover --RELEASES--> c_mobility
e_newjob  --SUPERSEDES--> b_old_job
```

经过 RevisionGate 的结构合法性检查后，由 DPA 对 belief 输出：

```text
AUTHORIZED / BLOCKED / SUPERSEDED / UNRESOLVED
```

关键限制是：

> 一个 belief 只有在存在显式 typed defeat path 时才能被限制。没有路径支持的相关联想或全局范围扩张不得直接改变其授权资格。

这不是承诺 LLM 永不犯错；它是将错误影响限制在局部、可见、可审计的路径中，并让误伤能够被专门评测。

### 第四层：Authorized Basis -> Answer -> Evaluation

当前被授权的 beliefs 构成 answer generator 的依据：

```text
query + authorized basis -> same answer generator -> final answer
```

这一层本身不是核心算法创新；它是验证授权机制是否真正改善最终回答的必要出口。Stage A 与 Stage B 在 end-to-end 比较中必须共享 preprocessing、retrieval 和 answer generation，唯一被比较的差异应是 authorization mechanism。

---

## 4. Canonical typed graph 与 DPA

### 4.1 节点

- `EvidenceNode`: 不可变的原始或最小证据单元，保存 timestamp/source/provenance。
- `BeliefNode`: 可进入回答依据的开放文本命题，保存 source evidence grounding。
- `ConditionNode`: belief 当前可使用所依赖的 scope-specific 条件。

### 4.2 边

```text
DependencyEdge(REQUIRES): belief -> condition
EvidenceEdge(BLOCKS):     evidence -> condition
EvidenceEdge(RELEASES):   evidence -> condition
EvidenceEdge(SUPERSEDES): evidence -> prior belief + grounded replacement belief
EvidenceEdge(REAFFIRMS):  evidence -> belief
EvidenceEdge(UNCERTAIN):  evidence -> belief
```

### 4.3 DPA 的作用

DPA 不调用语义模型；它只在已 admitted 的 typed graph 上执行确定性授权计算：

- 若新证据 `SUPERSEDES` 一个旧 belief，且 replacement 被当前证据 grounded，则旧 belief 为 `SUPERSEDED`；
- 若新证据 `BLOCKS` 某 condition，且 belief 明确 `REQUIRES` 该 condition，则 belief 为 `BLOCKED`；
- 若有效 blocker 被 later `RELEASES` 解除，且没有 supersession，则 belief 可恢复 eligibility；
- 若关系不足以确定修改，则输出 `UNRESOLVED`，而非武断 suppression。

---

## 5. 误伤、范围扩张与 UNCERTAIN：论文最重要的切入点

### 5.1 最强的研究假设

本文不以“模型能不能找到明显冲突”作为唯一卖点，而测试更困难的问题：

> 与 direct adjudication 相比，局部 typed-edge proposal + deterministic DPA 能否在阻止 stale belief 的同时，更少误伤无关、历史型或仅暂时受限的 beliefs，并在证据不足时更愿意 abstain？

### 5.2 为什么这比简单 stale blocking 更关键

若新证据是“腿骨折六周不能骑车”，系统应当限制：

```text
当前骑车通勤建议
```

但不得自动限制：

```text
喜欢骑自行车的长期偏好
曾经完成骑行比赛的历史事实
喜欢泰国菜的偏好
```

因此本文需要同时测：

- **Obsolete-Memory Misuse**: 该挡的旧 belief 是否仍进入答案；
- **Protected-Belief Preservation (PBP)**: 不该挡的 belief 是否被保留；
- **Abstention Accuracy**: 模糊证据下是否输出 `UNCERTAIN/UNRESOLVED`；
- **Unsupported Confident Revision**: 本应不确定时是否武断 `BLOCKED/SUPERSEDED/NOT_USABLE`。

### 5.3 DPA 能保证什么、不能保证什么

DPA 能结构性禁止：

- 没有 `REQUIRES` / `SUPERSEDES` 路径支持的任意 suppression；
- 结构非法或未 grounded 的 replacement revision。

DPA 不能单独保证：

- LLM 提出的 condition 一定足够原子；
- dependency edge 一定正确；
- candidate retrieval 一定既不漏检也不过宽。

因此，ReTrace 的贡献不是“消灭语义错误”，而是“把黑箱全局错误压缩为可审计的局部结构错误，并专门评测范围控制与弃权行为”。

---

## 6. Stage A / Stage B / Stage C

### Stage A - `ReTrace-LLM`：主方法

Primary controlled track：

```text
fixed SharedCandidateView
-> local typed-edge predictions
-> RevisionGate
-> deterministic DPA
-> authorization result + provenance
```

Stage A 仍使用 LLM 做局部语义判断；其贡献来自限制模型的最终裁决权限与影响传播范围，而不是声称没有 LLM judge。

### Stage B - `DirectJudge-LLM`：结构归因 baseline

```text
same fixed semantic SharedCandidateView
-> direct USABLE / NOT_USABLE / UNCERTAIN verdicts
```

Stage B 不是普通小模块消融，而是 architectural/counterfactual baseline：如果同样输入直接让模型给最终判决已经足够好，则 Stage A 的额外结构没有证明价值。

### Stage C - `ReTrace-Local`：后置训练版本

Stage C 只在下列条件满足后才能讨论：

- Stage A/B 的真实模型比较显示结构有价值；
- training data 来自 development-safe、人工审核的 typed-edge annotations；
- 与 official STALE/Memora scored examples 完全隔离；
- 训练目标是 local edge classification/generation，而不是 latent memory。

---

## 7. 实验结构：先可证伪，再正式评测

### 7.1 Primary Controlled Attribution

输入固定为 `SharedCandidateView`，排除 extraction/retrieval/answer-generation confounds。比较：

```text
Stage A: typed edge + DPA
Stage B: direct usability verdict
```

允许声称：same fixed semantic view、same model family/revision when configured、observed cost reported。
禁止声称：strict matched call budget、identical prompts、已完成 budget-normalized analysis。

### 7.2 P0：Ambiguity-and-Scope Feasibility Gate - 已完成并保留

内部诊断 split（又称 AB-1B 内部评估器）已完成并保留作为可行性诊断。它覆盖：

- clear supersession；
- clear prerequisite blocking；
- protected unrelated belief；
- temporary ability constraint vs persistent preference；
- current change vs historical fact；
- tentative intention / possible future change；
- insufficient evidence requiring `UNCERTAIN/UNRESOLVED`；
- multi-belief scope-expansion traps；
- stateful release 仅在 prior blocker 真正进入执行图时纳入。

每个 case 标注：evidence、candidate beliefs、conditions、REQUIRES anchors、expected status、protected belief ids、ambiguity flag、rationale、provenance source。

P0 的 go/no-go：

| 观察 | 解释与动作 |
|---|---|
| A 保持 stale blocking 且 PBP/abstention 优于 B | 结构方向 promising；继续外部评测 |
| A/B 都近乎满分 | split 不具区分度；先增强 challenge，不作优越性结论 |
| A/B 都差 | 诊断模型能力、prompt 或输入构造，不急于正式实验 |
| B 明显优于 A | 科学阻断项；暂停 Stage C 与大规模正式主表，分析 decomposition 风险 |

### 7.3 Official frozen STALE pilot 与正式使用

STALE 是本文 primary external benchmark。官方 frozen 400-case 数据集已公开发布在 `STALEproj/STALE`，文件为 `T1_T2_400_FULL.json`，license 为 CC BY 4.0。不得再声称 frozen STALE 数据不可获得、需要向作者索取，或生成 substitute split。

可见字段边界：

- 方法可见：`uid`、有序 `haystack_session`、对应 `timestamps`、每次一个 `probing_queries`。
- evaluator/provenance-only：`M_old`、`M_new`、`explanation`、`relevant_session_index`、`type`。
- `type` 只能用于 post-run stratification。
- 若 `M_old` / `M_new` 文本独立出现在真实 haystack sessions 中，该 session text 仍然方法可见；禁止的是直接注入独立 gold fields 或 relevant-session indices。

STALE 的评价维度包括 State Resolution、Premise Resistance 与 Implicit Policy Adaptation，且官方结果显示该任务尚不饱和。

正确顺序：

1. 使用官方 frozen data 的小规模 non-leaking offline wiring demo 做 pipeline pilot；
2. 固定 prompts、models、retrieval、cost policy 与 manifests；
3. 再在 frozen official setting 上运行 Stage A/B 最终回答比较。

不得把 official scored inputs 或 evaluator judgments 变成 Stage C 训练标签。

### 7.4 Memora 的角色

Memora oracle-conditioned 30-question pilot 当前保留为 internal rejected-pilot / adapter-misalignment artifact。它说明直接把 answer-surface `forgetting_absence` rubric 映射为 DPA authorization target 会产生目标错配；不得继续扩大 Memora、不得把该 pilot 称为 official Memora/FAMA result，也不得据此调 Stage A。

---

## 8. 截至 commit `5e8d6e2` 的工程事实审计

### 8.1 已真实建立的工程骨架

- validated controlled Stage A/B execution 与 AB-1B evaluator 基础；
- `HTTPLLMProvider`、manifest 和 runner 入口代码；
- end-to-end internal runner 的 mock/replay 骨架；
- 官方 frozen STALE adapter 与 offline non-leaking wiring demo；
- Memora oracle-conditioned negative-pilot artifacts；
- Stage C defer/no-go 文档入口。

### 8.2 尚不能作为论文结果或正式完成项的部分

1. **Provider 尚未被真实验证为可用路径。** 当前 Gemini OpenAI-compatible endpoint 代码缺少官方所要求的 `/openai/` 路径段；在修复并进行 dev-only capped live smoke 前，不能称 real provider operational。
2. **End-to-end mock runner 不是方法验证结果。** Mock 模式使用 manual extraction/induction/edges/retrieval；live 模式使用 token-overlap retrieval。后者最多是 v1 engineering baseline，不应称作 paper-facing retrieval method，也没有验证 scope control。
3. **STALE runner 当前是 offline schema/wiring demo。** 它验证官方 frozen dataset 的 non-leaking adapter、Stage A/B 持久状态 wiring 与 official answer export schema；尚未运行 official judge，也不是 official model result。
4. **Memora runner 当前是 rejected-pilot artifact。** 30-question oracle-conditioned diagnostic 已显示 adapter/objective mismatch，应保留为负例而不是扩大或调参。
5. **Stage C report 的 training-data 说法必须纠正。** Stage A live output traces 不是 gold labels；official evaluation runs 更不得用于生成 Stage C 监督数据。
6. **快照中的文档不一致已被列为修复对象。** 在审计快照中，`AGENTS.md` 与 implementation status 曾宣称 v1/AB-3 fully operational，而旧中文 blueprint / execution contract 仍停在 AB-1B/AB-1C 前状态；本 v3 边界要求主动文档同步到事实层。

---

## 9. 近期工程顺序：小心测试优先

### P0 - 科学对齐与安全修复 (当前 packet)

- 修复 STALE runner 的运行模式，实现 replay、dev-live 和 official-eval 的严格隔离。
- 强制引入 `--i-confirm-official-evaluation` 确认 flag，禁止在非评测模式下或未确认时调用 official evaluator。
- 纠正 STALE 摄入 chunk_size 默认值为 1，将大于 1 的 chunking 标记为非 canonical（approximate）。
- 编写 focused 单元测试验证这些模式安全 and chunk 规则。
- 同步所有 canonical docs。

### P1 - 外部 benchmark pathway validation

- STALE：以官方 frozen 400-case dataset 为 primary external benchmark；先完成 non-leaking offline wiring，再进入授权的 official judge / live-provider run；
- Memora：不继续扩大；仅保留 negative-pilot artifact；
- 保留 upstream SHA、license、input checksum、prompt/model/config hashes 与官方 evaluator raw outputs。

### P3 - Stage C 决策

仅在 Stage A 相对 B 产生稳定、可解释的结构优势后，另立任务设计 human-audited training data；不得从 official test runs 自蒸馏为 gold。

---

## 10. 本文最稳的 contribution 表述

1. **Problem formulation:** 将 evolving agent memory revision 定义为 evidence-preserving reversible authorization，而非 destructive rewriting 或 latent consolidation。
2. **Mechanism:** 以 typed local effects 与 deterministic DPA 要求 suppression 必须由显式 defeat path 支持，从而约束 unsupported scope expansion，并保留审计与恢复路径。
3. **Attribution protocol:** 以 Stage A 对 Stage B 的同视图比较，专门检验 structured authorization 是否在 protected-belief preservation 与 ambiguity abstention 上优于 direct adjudication。
4. **Evaluation pathway:** 以 official frozen STALE 检验这种授权差异是否转化为最终回答中更少 obsolete-memory misuse；Memora pilot 仅作为 adapter-misalignment 负例保留。所有 official claims 必须等待授权的 official STALE run。

---

## 11. 当前禁止写入论文的结论

在真实可证伪测试和 frozen official evaluation 完成前，不得声称：

- ReTrace 优于 DirectJudge 或 CUPMem；
- Stage A 已解决范围误伤；
- official STALE / Memora 已跑出结果；
- real-provider pipeline 已通过真实调用验证；
- Stage C 已具备 gold training data；
- v1 mock/dry-run 等价于论文实验完成。

---

## 12. Related-work 与外部评测使用原则

- **STALE/CUPMem**：最近邻问题与方法对照；官方 frozen STALE 400-case dataset 是 primary external benchmark，CUPMem 需基于官方代码后再作强比较。
- **Memora**：当前仅保留 oracle-conditioned negative pilot；不提供 ReTrace typed-edge gold，也不作为下一阶段主评测。
- **LongMemEval / MemoryAgentBench**：可作为 update/abstention/selective-forgetting 补充，不阻塞主实验。
- **Nemori / Auto-Dreamer / A-MEM / consolidation 工作**：解释为何本文不做 memory construction/consolidation。
- **AgeMem / Memory-R1 / MEM1**：解释为何本文不做 RL/latent memory control。

---

## 13. 结论

ReTrace 的论文方向仍然成立，但当前最重要的不是继续把 mock infrastructure 包装成完成，而是尽快进行能失败的实验：

> 在同一真实基础模型与固定 semantic view 下，Stage A 是否能在不牺牲 stale blocking 的情况下，比 Stage B 更少误伤 protected beliefs、更愿意在 ambiguous evidence 下输出不确定？

如果答案是肯定的，后续 official frozen STALE evaluation 与 Stage C 讨论才有坚实理由；如果答案是否定的，及时停止扩张并分析失败同样比在错误叙事上继续堆工程更有价值。
