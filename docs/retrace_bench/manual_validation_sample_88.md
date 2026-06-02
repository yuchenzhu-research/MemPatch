# Manual Validation Sample (88 cells) — Report Template

Status: **template — not yet reviewed by a human.** All reviewer checkboxes
below are intentionally unchecked. This file enumerates a script-assisted
stratified sample of the canonical paper-facing split
`data/retrace_bench/test_800_templateheldout_en/`, one scenario per
`(domain, failure_mode)` cell. All 8 domains × 11 failure modes (88 cells) are
populated.

See [`manual_validation_protocol.md`](manual_validation_protocol.md) for the
procedure and the meaning of each check column. Do **not** mark this as a
completed validation study until the table is filled in by a human reviewer.

Selection: deterministic (lowest scenario_id per cell) over 800 scenarios.
`cell size` is the number of scenarios available in that `(domain, failure_mode)`
cell. Checkbox columns: ☐ = not reviewed, ✓ = pass, ✗ = fail (add a note).

## Sample

| # | domain | failure_mode | scenario_id | cell size | traces solvable from visible evidence? | hidden labels match revision logic? | no hidden-label leak in public text? | non-answer justified? | reviewer notes |
| ---: | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| 1 | calendar_task_workflow | conflict_collapse | `rt-templateheldout-test-000004` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 2 | calendar_task_workflow | failure_to_forget | `rt-templateheldout-test-000076` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 3 | calendar_task_workflow | failure_to_release_or_restore | `rt-templateheldout-test-000044` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 4 | calendar_task_workflow | memory_hallucination | `rt-templateheldout-test-000052` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 5 | calendar_task_workflow | over_update | `rt-templateheldout-test-000036` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 6 | calendar_task_workflow | policy_violation | `rt-templateheldout-test-000028` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 7 | calendar_task_workflow | scope_leakage | `rt-templateheldout-test-000060` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 8 | calendar_task_workflow | stale_memory_reuse | `rt-templateheldout-test-000012` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 9 | calendar_task_workflow | under_update | `rt-templateheldout-test-000068` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 10 | calendar_task_workflow | unnecessary_memory_write | `rt-templateheldout-test-000020` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 11 | calendar_task_workflow | wrong_source_attribution | `rt-templateheldout-test-000084` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 12 | customer_support_crm | conflict_collapse | `rt-templateheldout-test-000059` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 13 | customer_support_crm | failure_to_forget | `rt-templateheldout-test-000043` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 14 | customer_support_crm | failure_to_release_or_restore | `rt-templateheldout-test-000011` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 15 | customer_support_crm | memory_hallucination | `rt-templateheldout-test-000019` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 16 | customer_support_crm | over_update | `rt-templateheldout-test-000003` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 17 | customer_support_crm | policy_violation | `rt-templateheldout-test-000083` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 18 | customer_support_crm | scope_leakage | `rt-templateheldout-test-000027` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 19 | customer_support_crm | stale_memory_reuse | `rt-templateheldout-test-000067` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 20 | customer_support_crm | under_update | `rt-templateheldout-test-000035` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 21 | customer_support_crm | unnecessary_memory_write | `rt-templateheldout-test-000075` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 22 | customer_support_crm | wrong_source_attribution | `rt-templateheldout-test-000051` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 23 | data_analysis_bi | conflict_collapse | `rt-templateheldout-test-000048` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 24 | data_analysis_bi | failure_to_forget | `rt-templateheldout-test-000032` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 25 | data_analysis_bi | failure_to_release_or_restore | `rt-templateheldout-test-000088` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 26 | data_analysis_bi | memory_hallucination | `rt-templateheldout-test-000008` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 27 | data_analysis_bi | over_update | `rt-templateheldout-test-000080` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 28 | data_analysis_bi | policy_violation | `rt-templateheldout-test-000072` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 29 | data_analysis_bi | scope_leakage | `rt-templateheldout-test-000016` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 30 | data_analysis_bi | stale_memory_reuse | `rt-templateheldout-test-000056` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 31 | data_analysis_bi | under_update | `rt-templateheldout-test-000024` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 32 | data_analysis_bi | unnecessary_memory_write | `rt-templateheldout-test-000064` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 33 | data_analysis_bi | wrong_source_attribution | `rt-templateheldout-test-000040` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 34 | ecommerce_recommendation | conflict_collapse | `rt-templateheldout-test-000015` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 35 | ecommerce_recommendation | failure_to_forget | `rt-templateheldout-test-000087` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 36 | ecommerce_recommendation | failure_to_release_or_restore | `rt-templateheldout-test-000055` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 37 | ecommerce_recommendation | memory_hallucination | `rt-templateheldout-test-000063` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 38 | ecommerce_recommendation | over_update | `rt-templateheldout-test-000047` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 39 | ecommerce_recommendation | policy_violation | `rt-templateheldout-test-000039` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 40 | ecommerce_recommendation | scope_leakage | `rt-templateheldout-test-000071` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 41 | ecommerce_recommendation | stale_memory_reuse | `rt-templateheldout-test-000023` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 42 | ecommerce_recommendation | under_update | `rt-templateheldout-test-000079` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 43 | ecommerce_recommendation | unnecessary_memory_write | `rt-templateheldout-test-000031` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 44 | ecommerce_recommendation | wrong_source_attribution | `rt-templateheldout-test-000007` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 45 | enterprise_multi_tool_workflow | conflict_collapse | `rt-templateheldout-test-000026` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 46 | enterprise_multi_tool_workflow | failure_to_forget | `rt-templateheldout-test-000010` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 47 | enterprise_multi_tool_workflow | failure_to_release_or_restore | `rt-templateheldout-test-000066` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 48 | enterprise_multi_tool_workflow | memory_hallucination | `rt-templateheldout-test-000074` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 49 | enterprise_multi_tool_workflow | over_update | `rt-templateheldout-test-000058` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 50 | enterprise_multi_tool_workflow | policy_violation | `rt-templateheldout-test-000050` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 51 | enterprise_multi_tool_workflow | scope_leakage | `rt-templateheldout-test-000082` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 52 | enterprise_multi_tool_workflow | stale_memory_reuse | `rt-templateheldout-test-000034` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 53 | enterprise_multi_tool_workflow | under_update | `rt-templateheldout-test-000002` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 54 | enterprise_multi_tool_workflow | unnecessary_memory_write | `rt-templateheldout-test-000042` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 55 | enterprise_multi_tool_workflow | wrong_source_attribution | `rt-templateheldout-test-000018` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 56 | personal_assistant_preference | conflict_collapse | `rt-templateheldout-test-000070` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 57 | personal_assistant_preference | failure_to_forget | `rt-templateheldout-test-000054` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 58 | personal_assistant_preference | failure_to_release_or_restore | `rt-templateheldout-test-000022` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 59 | personal_assistant_preference | memory_hallucination | `rt-templateheldout-test-000030` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 60 | personal_assistant_preference | over_update | `rt-templateheldout-test-000014` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 61 | personal_assistant_preference | policy_violation | `rt-templateheldout-test-000006` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 62 | personal_assistant_preference | scope_leakage | `rt-templateheldout-test-000038` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 63 | personal_assistant_preference | stale_memory_reuse | `rt-templateheldout-test-000078` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 64 | personal_assistant_preference | under_update | `rt-templateheldout-test-000046` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 65 | personal_assistant_preference | unnecessary_memory_write | `rt-templateheldout-test-000086` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 66 | personal_assistant_preference | wrong_source_attribution | `rt-templateheldout-test-000062` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 67 | research_knowledge_work | conflict_collapse | `rt-templateheldout-test-000037` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 68 | research_knowledge_work | failure_to_forget | `rt-templateheldout-test-000021` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 69 | research_knowledge_work | failure_to_release_or_restore | `rt-templateheldout-test-000077` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 70 | research_knowledge_work | memory_hallucination | `rt-templateheldout-test-000085` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 71 | research_knowledge_work | over_update | `rt-templateheldout-test-000069` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 72 | research_knowledge_work | policy_violation | `rt-templateheldout-test-000061` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 73 | research_knowledge_work | scope_leakage | `rt-templateheldout-test-000005` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 74 | research_knowledge_work | stale_memory_reuse | `rt-templateheldout-test-000045` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 75 | research_knowledge_work | under_update | `rt-templateheldout-test-000013` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 76 | research_knowledge_work | unnecessary_memory_write | `rt-templateheldout-test-000053` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 77 | research_knowledge_work | wrong_source_attribution | `rt-templateheldout-test-000029` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 78 | software_engineering_agent | conflict_collapse | `rt-templateheldout-test-000081` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 79 | software_engineering_agent | failure_to_forget | `rt-templateheldout-test-000065` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 80 | software_engineering_agent | failure_to_release_or_restore | `rt-templateheldout-test-000033` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 81 | software_engineering_agent | memory_hallucination | `rt-templateheldout-test-000041` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 82 | software_engineering_agent | over_update | `rt-templateheldout-test-000025` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 83 | software_engineering_agent | policy_violation | `rt-templateheldout-test-000017` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 84 | software_engineering_agent | scope_leakage | `rt-templateheldout-test-000049` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 85 | software_engineering_agent | stale_memory_reuse | `rt-templateheldout-test-000001` | 10 | ☐ | ☐ | ☐ | ☐ |  |
| 86 | software_engineering_agent | under_update | `rt-templateheldout-test-000057` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 87 | software_engineering_agent | unnecessary_memory_write | `rt-templateheldout-test-000009` | 9 | ☐ | ☐ | ☐ | ☐ |  |
| 88 | software_engineering_agent | wrong_source_attribution | `rt-templateheldout-test-000073` | 9 | ☐ | ☐ | ☐ | ☐ |  |
