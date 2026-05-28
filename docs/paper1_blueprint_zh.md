# ICLR 2027 Paper 1 蓝图：ReTrace 的证据保留式可逆授权方法

**目标会议：** ICLR 2027 Main Conference
**方法名：** ReTrace
**标题候选：** *ReTrace: Evidence-Preserving Reversible Authorization for Evolving Agent Memory*
**当前工程状态：** `method/retrace-llm-directjudge` 已完成 AB-1A.5。AB-1B 尚未开始。

本文档是当前仓库中的中文科学蓝图。旧 blueprint 和旧 source-material
文件不再是活跃研究决策来源；Git history 已经保留其历史内容。

## 1. 论文身份

Paper 1 不是 latent learning paper，也不是完整新 benchmark paper。它研究：

> 当新证据改变用户当前状态时，曾经由证据支持的旧 belief 是否还可以支配当前回答？
> 系统能否在不删除历史证据的前提下，通过可审计、可撤销的理由路径完成这个授权决定？

一句话方法定义：

```text
ReTrace = immutable evidence ledger
        + typed belief / condition / evidence-edge graph
        + deterministic Defeat-Path Authorization Algorithm (DPA)
```

原始 evidence 永久保留；belief 是否进入当前回答依据，由 later evidence
是否形成合法 typed defeat path 决定。

## 2. 不做什么

Paper 1 不做：

- latent memory 或 memory-token learning；
- RL / GRPO memory-action policy；
- learned episodic-to-semantic consolidation；
- 完整新 benchmark contribution；
- CUPMem 固定 slot ontology 的重命名版本；
- 纯 LLM judge 直接重写 memory；
- 手写 heuristic scaffold 作为可发表主方法；
- Stage C 的本地 verifier 训练。

Stage C 不是 latent learning。它只是 A/B 结构价值成立后，可能训练的本地
typed-edge verifier，仍复用同一个 DPA core。

## 3. 准确主张

不能写：

> ReTrace eliminates LLM judgment from memory revision.

准确写法是：

> ReTrace restricts semantic-model judgment to local evidence-edge proposals
> and delegates final belief authorization to an auditable deterministic
> defeat-path procedure.

中文含义：

> ReTrace 仍依赖语义模型理解 later evidence 与 candidate belief/condition 的局部关系；
> 但不允许模型一句话决定旧 belief 的最终命运。最终授权必须由 typed graph 和 DPA
> 确定性计算。

## 4. 三个阶段

### Stage A：`ReTrace-LLM`

主方法路径。在 primary controlled track 中：

```text
SharedCandidateView
→ isolated typed graph
→ gate fixed REQUIRES anchors
→ PromptEvidenceEdgeVerifier per candidate belief
→ gate evidence proposals
→ deterministic DPA
→ authorization result
```

Stage A 预测局部 typed evidence edges，不直接输出 final usability verdict。

### Stage B：`DirectJudge-LLM`

shared-view-controlled direct-adjudication baseline。它消费同一份
`SharedCandidateView`，直接输出每个 candidate belief 的：

```text
USABLE / NOT_USABLE / UNCERTAIN
```

它不是 `EvidenceEdgeVerifier`，不生成 edge，不经过 DPA。它也不能被描述为
strict call-budget matched baseline。

### Stage C：`ReTrace-Local`

延期阶段。它是 learned local typed-edge verifier + 同一 DPA core，只有当 Stage
A/B 结果说明结构化授权分解有价值后才有理由启动。它不是 latent-memory
representation learning。

## 5. Canonical typed vocabulary

### 节点

- `EvidenceNode`：append-only evidence，含 source 与 timestamp provenance。
- `BeliefNode`：由 evidence 支持的 open-text proposition。
- `ConditionNode`：带 `scope_id` 的 prerequisite condition。

### 边

Dependency edge 只有：

```text
DependencyEdge(REQUIRES): belief -> condition
```

Evidence edges 只有：

```text
BLOCKS: evidence -> condition
RELEASES: evidence -> condition
SUPERSEDES: evidence -> prior belief, with grounded replacement_belief_id
REAFFIRMS: evidence -> belief
UNCERTAIN: evidence -> belief
```

旧的 flat relation vocabulary 已被替换：`SUPPORT`、`CONDITION`、`REQUIRED_BY`
不是当前 Stage A runtime scheme。

## 6. DPA 授权语义

DPA 对每个 belief 输出：

```text
AUTHORIZED
BLOCKED
SUPERSEDED
UNRESOLVED
```

优先级：

```text
SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED
```

### Supersession

`SUPERSEDES(e, b_old)` 必须有真实 replacement belief。DPA 返回
`SUPERSEDED`，并保留 old belief、replacement id 与 evidence edge provenance。

### Prerequisite block

`BLOCKS(e, c)` 只有在 DPA 为某个 belief 找到 `REQUIRES(b, c)` 时，才影响该
belief。仅仅存在 condition-level `BLOCKS` edge 不会任意屏蔽所有 belief。

### Release

`RELEASES(e, c)` 解除 active blocker，但不证明 belief 当前为真。若存在更高优先级
`SUPERSEDES`，仍由 supersession 决定。

### Uncertainty / Reaffirmation

`UNCERTAIN(e, b)` 使 belief 当前使用资格变为 `UNRESOLVED`。更晚的
`REAFFIRMS(e, b)` 可清除 uncertainty，但不能覆盖 active block 或 supersession。

## 7. 最小例子

历史 evidence：

```text
e_old: “我平时骑自行车去学校。”
```

提取：

```text
b_bike: “用户通常骑自行车通勤。”
c_mobility: “用户当前具备骑车所需行动能力。”
b_bike --REQUIRES--> c_mobility
```

新 evidence：

```text
e_injury: “我昨天腿骨折了，六周内要打石膏。”
```

Stage A 可提出：

```text
e_injury --BLOCKS--> c_mobility
```

DPA 得到：

```text
b_bike = BLOCKED
```

历史事实未删除；只是当前回答中不能继续以骑车作为安全通勤建议依据。

若后续 evidence 为：

```text
e_recovery: “医生已经允许我重新骑自行车。”
```

Stage A 可提出：

```text
e_recovery --RELEASES--> c_mobility
```

若没有有效 `SUPERSEDES`，`b_bike` 可重新获得授权资格。

无关 belief：

```text
b_food: “用户喜欢泰国菜。”
```

腿骨折 evidence 不存在合法 defeat path，因此不得误屏蔽。

## 8. Primary controlled attribution

两种方法共享固定输入：

```text
SharedCandidateView
- query
- ordered evidence_context
- current new_evidence
- candidate_beliefs
- candidate_replacement_beliefs
- candidate_conditions
- fixed DependencyEdge(REQUIRES) anchors
- deterministic view_fingerprint
```

该轨道首先评价 authorization，而不是最终回答文案。

Stage A 的 fine-grained status 映射为 Stage B 可比状态：

| Stage A DPA status | Comparable status |
|---|---|
| `AUTHORIZED` | `USABLE` |
| `BLOCKED` | `NOT_USABLE` |
| `SUPERSEDED` | `NOT_USABLE` |
| `UNRESOLVED` | `UNCERTAIN` |

允许声称：

- same fixed semantic candidate view；
- 正式配置时可使用相同 model family/model id/provider/model revision；
- per-instance calls/tokens/cache/latency 会被报告；
- 比较的是 structured authorization 与 direct adjudication。

禁止声称：

- strict matched call budget；
- identical prompt exposure；
- equal number of calls；
- 已完成 budget-normalized analysis；
- 已完成官方 benchmark evaluation；
- 已经证明优于 DirectJudge、CUPMem、STALE baselines 或 Memora systems。

当前调用数必须写清：

- Stage A：每个 candidate belief 一次 semantic edge-verifier call，共 N 次。
- Stage B：对完整 `SharedCandidateView` 一次 direct-adjudication call。

## 9. Secondary end-to-end pipeline

Secondary track 仍是计划或未来工作，尚未作为官方结果实现。它将包含：

```text
Incoming session
→ immutable EvidenceNode ledger
→ generic belief extraction
→ generic requirement/condition induction
→ impact candidate retrieval
→ Stage A or Stage B authorization mechanism
→ query-conditioned basis
→ fixed answer model
→ benchmark scoring
```

该轨道必须单独报告，因为 extraction 和 retrieval 会引入 primary controlled
track 中不存在的混杂因素。

## 10. 评价设计

Primary controlled metrics 是建议/计划指标，当前尚未形成官方结果：

- Authorization Accuracy；
- Obsolete-Memory Misuse Rate；
- Unsupported Revision Rate；
- Protected-Belief Preservation；
- Rollback Recovery；
- calls/tokens/latency/cache。

Secondary benchmark plans：

- STALE：implicit invalidation 与 stale premise；
- Memora：repeated mutation 与 obsolete reuse；
- BoundaryAudit：小型诊断集，不作为新 benchmark headline。

任何指标、表格、figure 或 benchmark 结果，除非由当前代码和冻结运行实际产生，
都只能标注为 proposed evaluation design 或 future work。

## 11. 当前工程状态

已完成：

- `main` 上 typed DPA execution spine；
- AB-0 offline method contracts、prompts、DirectJudge sibling path、mock/replay tests；
- AB-0.5 fairness and deterministic-grounding hardening；
- AB-1A offline controlled attribution harness；
- AB-1A.5 auditability and comparison-protocol lock。

AB-1A.5 已完成，不是下一步。

下一步只有在未来明确授权时才是 AB-1B：

- internal development cases；
- comparison evaluator；
- replay-only runner。

不在当前状态中：

- real provider adapter；
- live API call；
- official STALE/Memora evaluation；
- secondary end-to-end benchmark run；
- Stage C training。

## 12. Related work 定位

本文不能声称首次研究 stale memory 或 repeated memory mutation。STALE/CUPMem 和
Memora 已经覆盖了重要外部问题设置。

ReTrace 的定位是：

> unlike direct adjudication methods that let a model decide memory usability in
> one step, ReTrace preserves episodic evidence and revises only the authorized
> current-belief view through typed defeat paths resolved by deterministic DPA.

CUPMem、Memora、Nemori、Graphiti、TriMem、MemoryAgentBench、LongMemEval、Mem0、
A-MEM、A-MAC、AgeMem、MEM1 等工作的具体代码/集成角色见
`docs/upstream_integration.md`。

## 13. Go / No-Go 原则

- AB-1B 前不接真实 provider。
- AB-1C 前不做 live API dev calls。
- config/prompt freeze 前不跑正式 STALE/Memora evaluation。
- A/B 结构价值未建立前不训练 Stage C。
- 没有结果前不宣称 ReTrace 优于 DirectJudge、CUPMem 或任何外部方法。

最终执行原则：

> ReTrace 不试图学习“全部记忆”。它保留历史证据，并预测 later evidence 与
> earlier belief/condition 之间的局部 typed relations；一个旧 belief 是否还能支配
> 当前回答，最终只能由可审计、可撤销的 defeat-path authorization 决定。
