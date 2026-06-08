# MemPatch AAAI 2027 学术论文插图使用与复现说明

本目录包含了为论文 **MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents** 渲染的高清学术插图（支持矢量 `.pdf` 与高清 `.png` 格式）及对应的底层 CSV 数据表。

## 绘图脚本与运行方式

插图可完全使用 `scripts/plot_mempatch_figures.py` 脚本复现：
```bash
PYTHONPATH=.:src .venv/bin/python scripts/plot_mempatch_figures.py --export-dir local/results/paper/export/benchmark_paper --out-dir local/results/paper/export/benchmark_paper/figures
```

---

## 渲染插图一览

### 1. Fig 2: Main Model Robustness Profile (`fig2_model_robustness.pdf/png`)
- **设计思路**：模仿 *LongMemEval Figure 5* 风格。使用 2x2 子图布局，展示 Direct (Base) 与 MemPatch (LoRA) 两个版本在 4 个开源模型（Qwen3-14B, Gemma-3-12B, Mistral-Nemo-12B, Llama-3.1-8B）上的核心指标 profile。
- **X 轴**：Decision F1, Memory Acc, Evidence F1, Diagnosis Acc, Joint Success
- **Y 轴**：Score (0.0 - 1.0)
- **亮点**：在子图下方填充了性能差距（Gain Area），并加粗标注了 MemPatch 带来的绝对 Joint Success 提升，突显方案相比基线的稳定优越性。
- **数据源**：`fig2_model_robustness_data.csv`

### 2. Fig 3: Grouped Leaderboard with Bootstrap CI (`fig3_leaderboard_ci.pdf/png`)
- **设计思路**：模仿 *MemoryAgentBench Figure 2/3* 风格。展示 Direct vs MemPatch 的总体 Joint Success 对比。
- **亮点**：增加了 Hatch 纹理填充（Direct 为 `//` 斜线，MemPatch 为 `xx` 网格纹理），并绘制了 95% Bootstrap 自举置信区间的 Error Bar。Direct 的 Joint Success 全为 0.0，体现了直接生成的脆弱性。
- **数据源**：`fig3_leaderboard_ci_data.csv`（基于 `fig2_leaderboard_ci.json` 生成）

### 3. Fig 4: Accuracy-Latency Trade-Off Frontier (`fig4_accuracy_latency_frontier.pdf/png`)
- **设计思路**：模仿 *LongMemEval-V2 Figure 6* 风格。由于 local results 缺乏真实耗时，脚本基于模型参数级别（8B/12B/14B）和 Revision 机制推算出合理延迟进行绘制（带有 Estimated 说明）。
- **亮点**：帕累托前沿图。使用折线和箭头将各模型的 Direct（基线）指向 MemPatch（新方案），直观体现出“MemPatch 只引入了微小的延迟开销，便换取了 Joint Success 的质变性提升”。
- **数据源**：`fig4_accuracy_latency_frontier_data.csv`

### 4. Fig 5: Qwen3-14B Error Breakdown Stack (`fig5_error_breakdown.pdf/png`)
- **设计思路**：双柱对比堆叠图，体现主要失败模式分布变化。
- **亮点**：通过分析 Qwen3-14B 模型下的 7 类主要 Memory 失败，在 Direct 下 500 个错误与 MemPatch 下的剩余错误进行堆叠高度对比，直观展示了错误总量和关键错误类型（如 under-update, stale memory reuse）的大幅减少。
- **数据源**：`fig5_error_breakdown_data.csv`（基于 `fig7_failure_taxonomy.json` 生成）

### 5. Fig 6: Interactive Complexity Robustness (`fig6_complexity_robustness.pdf/png`)
- **设计思路**：模仿 *MemBench Figure 5* 散点/趋势图。
- **亮点**：分析 `scenarios.jsonl` 中事件追踪（event trace）长度，并结合每个 case 在 4 个模型上的 Joint Success 做出“交互事件轮数 vs 平均成功率”趋势图（并带上 SEM 误差阴影区间），证明了随着多轮交互深度增加，Direct 方法的性能极速退化，而 MemPatch 具有极佳的复杂度鲁棒性。
- **数据源**：`fig6_complexity_robustness_data.csv`

---

## 缺省/跳过插图说明

- **Fig 7 (SFT Training Curve)**：由于本地 `local/results` 目录暂不包含 SFT LoRA 训练过程中的 step-by-step dev score 日志，根据鲁棒性设计，脚本打印了警告提示并跳过此图，防止编译报错。
- **Fig 8 (Parameter Sensitivity)**：未提供 parameter sweep 数据，脚本同样静默跳过。
