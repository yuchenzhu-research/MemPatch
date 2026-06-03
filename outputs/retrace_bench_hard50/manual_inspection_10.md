# Hard50 Manual Inspection (10 cases)

## rt-hard-000002

- **Selection reason:** all three API models fail joint_revision_success
- **Pattern / failure / difficulty:** authority_conflict / under_update / L4
- **Expected decision:** `mark_unresolved`
- **Expected evidence:** `['e-rt-hard-000002-1', 'e-rt-hard-000002-2', 'e-rt-hard-000002-3']`

### Event trace summary
- `e-rt-hard-000002-1` (user): Issue #200 reports SSO timeout error on CASE-300001.
- `e-rt-hard-000002-2` (maintainer): Maintainer closed Issue #200 as duplicate of master Issue #98.
- `e-rt-hard-000002-3` (maintainer): Master Issue #98 remains open with labels: needs-investigation and priority-high.
- `e-rt-hard-000002-distractor-ver` (user): User notes that on nightly build in workspace-beta, the role-based permission update has been applied to CASE-300001 ...
- `e-rt-hard-000002-distractor-auth` (user): An unverified user posted on the chat channel that the role-based permission conflict for CASE-300001 was resolved an...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000002-1', 'e-rt-hard-000002-2', 'e-rt-hard-000002-3', 'e-rt-hard-000002-distractor-ver', 'e-rt-hard-000002-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000002-1', 'e-rt-hard-000002-2', 'e-rt-hard-000002-3']` diagnosis=`scope_leakage`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000002-1', 'e-rt-hard-000002-2', 'e-rt-hard-000002-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000002-1', 'e-rt-hard-000002-2', 'e-rt-hard-000002-3', 'e-rt-hard-000002-distractor-ver', 'e-rt-hard-000002-distractor-rollback', 'e-rt-hard-000002-distractor-ci']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000004

- **Selection reason:** all three API models fail joint_revision_success
- **Pattern / failure / difficulty:** version_or_release_chain / conflict_collapse / L4
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3']`

### Event trace summary
- `e-rt-hard-000004-1` (release_note): Release v2.0.0 ships with strict schema validation active.
- `e-rt-hard-000004-2` (maintainer): Revert PR #401 merged: reverts strict validation due to regressions.
- `e-rt-hard-000004-3` (release_note): Release v2.0.1 reverts strict validation and returns default config to opt-in.
- `e-rt-hard-000004-distractor-ver` (user): User notes that on nightly build in workspace-beta, the room booking update has been applied to CASE-300003 and the n...
- `e-rt-hard-000004-distractor-auth` (user): An unverified user posted on the chat channel that the room booking conflict for CASE-300003 was resolved and the old...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3', 'e-rt-hard-000004-distractor-ver', 'e-rt-hard-000004-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000004-2', 'e-rt-hard-000004-3']` diagnosis=`stale_memory_reuse`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3']` diagnosis=`stale_memory_reuse`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3', 'e-rt-hard-000004-distractor-ver', 'e-rt-hard-000004-distractor-rollback', 'e-rt-hard-000004-distractor-ci']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000007

- **Selection reason:** all three API models fail joint_revision_success
- **Pattern / failure / difficulty:** authority_conflict / wrong_source_attribution / L3
- **Expected decision:** `mark_unresolved`
- **Expected evidence:** `['e-rt-hard-000007-2']`

### Event trace summary
- `e-rt-hard-000007-1` (user): User claims CVE-999 vulnerability has been fixed in latest patch.
- `e-rt-hard-000007-2` (security): Security auditor confirms CVE-999 remains active and unpatched in current builds.
- `e-rt-hard-000007-distractor-ver` (user): User notes that on nightly build in workspace-beta, the stock availability update has been applied to CASE-300006 and...
- `e-rt-hard-000007-distractor-auth` (user): An unverified user posted on the chat channel that the stock availability conflict for CASE-300006 was resolved and t...
- `e-rt-hard-000007-bg-1` (bot): The operations log record shows database connection pool state is active.
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000007-2', 'e-rt-hard-000007-distractor-ver', 'e-rt-hard-000007-distractor-auth']` diagnosis=`wrong_source_attribution`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000007-2', 'e-rt-hard-000007-distractor-ver']` diagnosis=`scope_leakage`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000007-2']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000007-2', 'e-rt-hard-000007-distractor-ver', 'e-rt-hard-000007-bg-1', 'e-rt-hard-000007-bg-2', 'e-rt-hard-000007-bg-3']` overcitation=0.8

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000001

- **Selection reason:** one API model succeeds while others fail joint_revision_success
- **Pattern / failure / difficulty:** version_or_release_chain / stale_memory_reuse / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000001-1', 'e-rt-hard-000001-3']`

### Event trace summary
- `e-rt-hard-000001-1` (user): Issue #100 reports CASE-300000 lacks YAML support.
- `e-rt-hard-000001-2` (reviewer): PR #101 implementing YAML support for CASE-300000 was merged to branch main/dev.
- `e-rt-hard-000001-3` (release_note): Release v1.4.0 notes list only hotfixes and do not include the YAML feature for CASE-300000.
- `e-rt-hard-000001-distractor-ver` (user): User notes that on nightly build in workspace-beta, the PR review update has been applied to CASE-300000 and the new ...
- `e-rt-hard-000001-distractor-auth` (user): An unverified user posted on the chat channel that the PR review conflict for CASE-300000 was resolved and the old se...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000001-1', 'e-rt-hard-000001-2', 'e-rt-hard-000001-3', 'e-rt-hard-000001-distractor-ver', 'e-rt-hard-000001-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`None` evidence=`[]` diagnosis=`None`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000001-1', 'e-rt-hard-000001-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000001-1', 'e-rt-hard-000001-2', 'e-rt-hard-000001-3', 'e-rt-hard-000001-distractor-ver', 'e-rt-hard-000001-bg-1', 'e-rt-hard-000001-bg-2']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000003

- **Selection reason:** one API model succeeds while others fail joint_revision_success
- **Pattern / failure / difficulty:** authority_conflict / over_update / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000003-1', 'e-rt-hard-000003-2', 'e-rt-hard-000003-3']`

### Event trace summary
- `e-rt-hard-000003-1` (docs): README docs updated: CASE-300002 now supports batch deletes.
- `e-rt-hard-000003-2` (reviewer): Code PR #300 implementing batch delete remains unmerged and tests are failing.
- `e-rt-hard-000003-3` (maintainer): Maintainer notes docs were merged ahead of code implementation by mistake.
- `e-rt-hard-000003-distractor-ver` (user): User notes that on nightly build in workspace-beta, the refund policy update has been applied to CASE-300002 and the ...
- `e-rt-hard-000003-distractor-auth` (user): An unverified user posted on the chat channel that the refund policy conflict for CASE-300002 was resolved and the ol...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000003-1', 'e-rt-hard-000003-2', 'e-rt-hard-000003-3', 'e-rt-hard-000003-distractor-ver', 'e-rt-hard-000003-distractor-auth']` diagnosis=`stale_memory_reuse`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000003-1', 'e-rt-hard-000003-2', 'e-rt-hard-000003-3']` diagnosis=`stale_memory_reuse`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000003-1', 'e-rt-hard-000003-2', 'e-rt-hard-000003-3']` diagnosis=`stale_memory_reuse`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000003-1', 'e-rt-hard-000003-2', 'e-rt-hard-000003-3', 'e-rt-hard-000003-distractor-ver', 'e-rt-hard-000003-bg-1', 'e-rt-hard-000003-bg-2']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000005

- **Selection reason:** one API model succeeds while others fail joint_revision_success
- **Pattern / failure / difficulty:** scope_collision / scope_leakage / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000005-1', 'e-rt-hard-000005-3']`

### Event trace summary
- `e-rt-hard-000005-1` (user): XML exporter is required for production reports on v1.
- `e-rt-hard-000005-2` (reviewer): Developer removed XML exporter on v2 development branch.
- `e-rt-hard-000005-3` (maintainer): Maintainer confirms XML export will remain fully supported on v1 stable.
- `e-rt-hard-000005-distractor-ver` (user): User notes that on nightly build in workspace-beta, the paper claim update has been applied to CASE-300004 and the ne...
- `e-rt-hard-000005-distractor-auth` (user): An unverified user posted on the chat channel that the paper claim conflict for CASE-300004 was resolved and the old ...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000005-1', 'e-rt-hard-000005-3', 'e-rt-hard-000005-distractor-ver', 'e-rt-hard-000005-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000005-1', 'e-rt-hard-000005-3']` diagnosis=`scope_leakage`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000005-1', 'e-rt-hard-000005-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000005-1', 'e-rt-hard-000005-2', 'e-rt-hard-000005-3', 'e-rt-hard-000005-distractor-ver', 'e-rt-hard-000005-bg-1', 'e-rt-hard-000005-bg-2']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000008

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / memory_hallucination / L4
- **Expected decision:** `ask_clarification`
- **Expected evidence:** `['e-rt-hard-000008-1', 'e-rt-hard-000008-2']`

### Event trace summary
- `e-rt-hard-000008-1` (user): Developer states performance hotfix is ready and merged.
- `e-rt-hard-000008-2` (ci): CI pipeline check for performance hotfix failed during compilation.
- `e-rt-hard-000008-distractor-ver` (user): User notes that on nightly build in workspace-beta, the source-table lineage update has been applied to CASE-300007 a...
- `e-rt-hard-000008-distractor-auth` (user): An unverified user posted on the chat channel that the source-table lineage conflict for CASE-300007 was resolved and...
- `e-rt-hard-000008-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300007 to restore the previous metric definition configurations ...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000008-1', 'e-rt-hard-000008-2', 'e-rt-hard-000008-distractor-ver', 'e-rt-hard-000008-distractor-auth', 'e-rt-hard-000008-distractor-rollback', 'e-rt-hard-000008-distractor-ci']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`mark_unresolved` evidence=`['e-rt-hard-000008-1', 'e-rt-hard-000008-2']` diagnosis=`under_update`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000008-1', 'e-rt-hard-000008-2']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000008-1', 'e-rt-hard-000008-2', 'e-rt-hard-000008-distractor-ver', 'e-rt-hard-000008-distractor-rollback', 'e-rt-hard-000008-distractor-ci', 'e-rt-hard-000008-bg-1']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000011

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / failure_to_release_or_restore / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000011-2']`

### Event trace summary
- `e-rt-hard-000011-1` (user): User asserts database connection pool limit is now 100.
- `e-rt-hard-000011-2` (maintainer): Maintainer clarifies pool limit is kept at 20 to prevent server exhaustion.
- `e-rt-hard-000011-distractor-ver` (user): User notes that on nightly build in workspace-beta, the loyalty tier update has been applied to CASE-300010 and the n...
- `e-rt-hard-000011-distractor-auth` (user): An unverified user posted on the chat channel that the loyalty tier conflict for CASE-300010 was resolved and the old...
- `e-rt-hard-000011-bg-1` (bot): The operations log record shows database connection pool state is active.
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000011-1', 'e-rt-hard-000011-2', 'e-rt-hard-000011-distractor-ver', 'e-rt-hard-000011-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000011-2']` diagnosis=`scope_leakage`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000011-2']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000011-1', 'e-rt-hard-000011-2', 'e-rt-hard-000011-distractor-ver', 'e-rt-hard-000011-bg-1', 'e-rt-hard-000011-bg-2', 'e-rt-hard-000011-bg-3']` overcitation=0.8333333333333334

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000019

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** version_or_release_chain / memory_hallucination / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000019-1', 'e-rt-hard-000019-2', 'e-rt-hard-000019-3']`

### Event trace summary
- `e-rt-hard-000019-1` (release_note): Release v2.0.0 ships with strict schema validation active.
- `e-rt-hard-000019-2` (maintainer): Revert PR #401 merged: reverts strict validation due to regressions.
- `e-rt-hard-000019-3` (release_note): Release v2.0.1 reverts strict validation and returns default config to opt-in.
- `e-rt-hard-000019-distractor-ver` (user): User notes that on nightly build in workspace-beta, the fraud flag update has been applied to CASE-300018 and the new...
- `e-rt-hard-000019-distractor-auth` (user): An unverified user posted on the chat channel that the fraud flag conflict for CASE-300018 was resolved and the old s...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000019-1', 'e-rt-hard-000019-2', 'e-rt-hard-000019-3', 'e-rt-hard-000019-distractor-ver', 'e-rt-hard-000019-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`use_current_memory` evidence=`['e-rt-hard-000019-1', 'e-rt-hard-000019-2', 'e-rt-hard-000019-3']` diagnosis=`stale_memory_reuse`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000019-1', 'e-rt-hard-000019-2', 'e-rt-hard-000019-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000019-1', 'e-rt-hard-000019-2', 'e-rt-hard-000019-3', 'e-rt-hard-000019-distractor-ver', 'e-rt-hard-000019-bg-1', 'e-rt-hard-000019-bg-2']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000047

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / over_update / L3
- **Expected decision:** `mark_unresolved`
- **Expected evidence:** `['e-rt-hard-000047-1', 'e-rt-hard-000047-2', 'e-rt-hard-000047-3']`

### Event trace summary
- `e-rt-hard-000047-1` (user): Issue #200 reports SSO timeout error on CASE-300046.
- `e-rt-hard-000047-2` (maintainer): Maintainer closed Issue #200 as duplicate of master Issue #98.
- `e-rt-hard-000047-3` (maintainer): Master Issue #98 remains open with labels: needs-investigation and priority-high.
- `e-rt-hard-000047-distractor-ver` (user): User notes that on nightly build in workspace-beta, the seller policy update has been applied to CASE-300046 and the ...
- `e-rt-hard-000047-distractor-auth` (user): An unverified user posted on the chat channel that the seller policy conflict for CASE-300046 was resolved and the ol...
- ... and 2 more events

### Model outputs
- **Kimi-K2.6:** decision=`use_current_memory` evidence=`['e-rt-hard-000047-1', 'e-rt-hard-000047-2', 'e-rt-hard-000047-3', 'e-rt-hard-000047-distractor-ver', 'e-rt-hard-000047-distractor-auth']` diagnosis=`scope_leakage`
- **GLM-5.1:** decision=`None` evidence=`[]` diagnosis=`None`
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000047-1', 'e-rt-hard-000047-2', 'e-rt-hard-000047-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000047-1', 'e-rt-hard-000047-2', 'e-rt-hard-000047-3', 'e-rt-hard-000047-distractor-ver', 'e-rt-hard-000047-bg-1', 'e-rt-hard-000047-bg-2']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

