# Live Stage A vs Stage B — DeepSeek-V3, paper1_balanced420

## Run Identity

- **dataset**: `paper1_balanced`
- **episodes**: 420
- **provider**: `siliconflow`
- **model**: `deepseek-ai/DeepSeek-V3`
- **temperature**: 0.0
- **constrained**: `true`
- **stage_a_variant**: `conflict_aware`
- **run mode**: live API
- **local run commit**: `deadb6495a243a37089aeccf6b71873bb821260c`
- **docs branch base**: latest main, expected `1000ab9ae4c0f43d877fdefc06b2d922d5b993d9`
- **generator_version**: `paper1_balanced_v1`
- **domains**: `software_engineering` 210, `research_workflow` 210
- **failure types**: 14 types × 30 cases each
- **raw artifacts**: `outputs/runs/paper1_balanced420_deepseek_v3_conflict_aware_live_20260531_174458` (not committed)

## Headline Metrics

### Stage A
- **final_status_accuracy**: 0.991358024691358
- **dpa_final_status_accuracy**: 0.991358024691358
- **macro_final_status_accuracy**: 0.9873015873015872
- **episode_exact_match_rate**: 0.9833333333333333
- **over_update_rate**: 0.0
- **stale_propagation_rate**: 0.0
- **under_update_rate**: 0.014285714285714285
- **uncertainty_error_rate**: 0.007407407407407408
- **grounding_error_rate**: 0.0
- **valid_output_rate**: 1.0
- **valid_json**: 1.0
- **target_grounding**: 1.0
- **parser_error_rate**: 0.0
- **first_pass_valid_json_rate**: 1.0
- **first_pass_parser_error_rate**: 0.0
- **multi_action_recall**: 0.9958333333333333

### Stage B
- **final_status_accuracy**: 0.3
- **strict_final_status_accuracy**: 0.3
- **canonicalized_final_status_accuracy**: 0.3
- **canonicalization_rate**: 0.0
- **fuzzy_canonicalization_rate**: 0.0
- **over_update_rate**: 0.5555555555555556
- **stale_propagation_rate**: 0.5555555555555556
- **under_update_rate**: 0.5642857142857143
- **uncertainty_error_rate**: 0.4444444444444444
- **grounding_error_rate**: 0.0
- **valid_output_rate**: 0.8333333333333334

## Failure-Type Breakdown

| failure_type | cases | A_acc | B_acc | A_parse | B_parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| ambiguous_update | 30 | 1.0000 | 0.0000 | 0 | 0 |
| blocks_uncertain | 30 | 0.9833 | 0.5000 | 0 | 0 |
| cross_agent_conflict | 30 | 1.0000 | 0.5000 | 0 | 0 |
| direct_supersession | 30 | 1.0000 | 0.0000 | 0 | 30 |
| duplicate_evidence | 30 | 1.0000 | 0.8000 | 0 | 0 |
| evidence_conflict | 30 | 0.9833 | 0.5000 | 0 | 0 |
| multi_action_supersedes_blocks | 30 | 1.0000 | 0.0000 | 0 | 30 |
| multi_action_supersedes_releases | 30 | 0.9889 | 0.0000 | 0 | 30 |
| no_revision | 30 | 1.0000 | 1.0000 | 0 | 0 |
| reaffirms_only | 30 | 1.0000 | 1.0000 | 0 | 0 |
| scope_expansion | 30 | 1.0000 | 0.5000 | 0 | 0 |
| stale_propagation | 30 | 1.0000 | 0.0000 | 0 | 30 |
| target_ambiguity | 30 | 1.0000 | 0.3333 | 0 | 30 |
| temporary_blocker_recovery | 30 | 0.8667 | 0.3000 | 0 | 0 |

## Interpretation

- **性能对比**：在内部 balanced 验证集上，Stage A (Decomposition + DPA) 极大地超越了 DirectJudge Stage B。
- **错误分析**：Stage A 实现了 0.0 的 stale propagation 发生率、0.0 的 over-update 发生率、0.0 的 grounding 错误率以及 0.0 的 parser 错误率。而 Stage B 极易受到 stale propagation、under-update 和 over-update 的影响，并且 valid output rate 较低。
- **弱项/痛点**：对于 Stage A 来说，最难的失败类型是 `temporary_blocker_recovery`（准确率仅为 0.8667），这需要在最终确定论文性能前予以深入分析。
- **对照组**：`no_revision` 和 `reaffirms_only` 属于容易的对照组，两个方法均拿到了 1.0 的满分。

## Caveats

- **评估性质**：这仅是内部 balanced 验证，并非外部 benchmark。
- **非最终结论**：该 run 在 manifest 中被标记为 `development_live_api_run / not_final_paper_result`。
- **Git 提交不一致**：实验在 local commit `deadb64` 下完成，而此文档在同步至最新 `main` 后被添加。
- **工件隔离**：原始工件保留在 `outputs/runs` 下，不被提交至仓库。
- **与 dev70 数据隔离**：勿与 dev70 的结果混淆，它们的 dataset 规模和细节存在差异。
- **精细指标解读**：动作层级的精确匹配 (exact-match) / 证据接地 (evidence-grounding) 指标偏低应被谨慎解读。因为最终的 belief 状态是由 parser + RevisionGate + DPA 计算得出的，而 action-level 匹配更为严格，可能对等效或提供额外辅助 edge 的 actions 进行惩罚。
