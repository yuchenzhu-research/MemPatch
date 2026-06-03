# Hard50 Manual Inspection (10 cases)

## rt-hard-000001

- **Selection reason:** retrieve_all overcites evidence
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
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000001-1', 'e-rt-hard-000001-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000001-1', 'e-rt-hard-000001-2', 'e-rt-hard-000001-3', 'e-rt-hard-000001-distractor-ver', 'e-rt-hard-000001-bg-1', 'e-rt-hard-000001-bg-2']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000005

- **Selection reason:** retrieve_all overcites evidence
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
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000005-1', 'e-rt-hard-000005-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000005-1', 'e-rt-hard-000005-2', 'e-rt-hard-000005-3', 'e-rt-hard-000005-distractor-ver', 'e-rt-hard-000005-bg-1', 'e-rt-hard-000005-bg-2']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000035

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** scope_collision / scope_leakage / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000035-1', 'e-rt-hard-000035-3']`

### Event trace summary
- `e-rt-hard-000035-1` (user): XML exporter is required for production reports on v1.
- `e-rt-hard-000035-2` (reviewer): Developer removed XML exporter on v2 development branch.
- `e-rt-hard-000035-3` (maintainer): Maintainer confirms XML export will remain fully supported on v1 stable.
- `e-rt-hard-000035-distractor-ver` (user): User notes that on nightly build in workspace-beta, the refund policy update has been applied to CASE-300034 and the ...
- `e-rt-hard-000035-distractor-auth` (user): An unverified user posted on the chat channel that the refund policy conflict for CASE-300034 was resolved and the ol...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000035-1', 'e-rt-hard-000035-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000035-1', 'e-rt-hard-000035-2', 'e-rt-hard-000035-3', 'e-rt-hard-000035-distractor-ver', 'e-rt-hard-000035-bg-1', 'e-rt-hard-000035-bg-2']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000033

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / wrong_source_attribution / L3
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000033-1', 'e-rt-hard-000033-2', 'e-rt-hard-000033-3']`

### Event trace summary
- `e-rt-hard-000033-1` (docs): README docs updated: CASE-300032 now supports batch deletes.
- `e-rt-hard-000033-2` (reviewer): Code PR #300 implementing batch delete remains unmerged and tests are failing.
- `e-rt-hard-000033-3` (maintainer): Maintainer notes docs were merged ahead of code implementation by mistake.
- `e-rt-hard-000033-distractor-ver` (user): User notes that on nightly build in workspace-beta, the PR review update has been applied to CASE-300032 and the new ...
- `e-rt-hard-000033-distractor-auth` (user): An unverified user posted on the chat channel that the PR review conflict for CASE-300032 was resolved and the old se...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000033-1', 'e-rt-hard-000033-2', 'e-rt-hard-000033-3']` diagnosis=`stale_memory_reuse`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000033-1', 'e-rt-hard-000033-2', 'e-rt-hard-000033-3', 'e-rt-hard-000033-distractor-ver', 'e-rt-hard-000033-bg-1', 'e-rt-hard-000033-bg-2']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000030

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / stale_memory_reuse / L4
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000030-1', 'e-rt-hard-000030-3']`

### Event trace summary
- `e-rt-hard-000030-1` (user): Issue #500 reports SSL routing error.
- `e-rt-hard-000030-2` (user): Developer opens PR #501 to resolve SSL routing error.
- `e-rt-hard-000030-3` (reviewer): Reviewer states PR #501 is on hold and no merge action has been approved.
- `e-rt-hard-000030-distractor-ver` (user): User notes that on nightly build in workspace-beta, the notification style update has been applied to CASE-300029 and...
- `e-rt-hard-000030-distractor-auth` (user): An unverified user posted on the chat channel that the notification style conflict for CASE-300029 was resolved and t...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000030-1', 'e-rt-hard-000030-2', 'e-rt-hard-000030-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000030-1', 'e-rt-hard-000030-2', 'e-rt-hard-000030-3', 'e-rt-hard-000030-distractor-ver', 'e-rt-hard-000030-distractor-rollback', 'e-rt-hard-000030-distractor-ci']` overcitation=0.6666666666666666

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000036

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** scope_collision / scope_leakage / L4
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000036-2']`

### Event trace summary
- `e-rt-hard-000036-1` (reviewer): PR on feature branch feat-311 enables Python 3.11 runtimes.
- `e-rt-hard-000036-2` (maintainer): Main release branch targets only Python 3.10 and does not accept Python 3.11 commits.
- `e-rt-hard-000036-distractor-ver` (user): User notes that on nightly build in workspace-beta, the room booking update has been applied to CASE-300035 and the n...
- `e-rt-hard-000036-distractor-auth` (user): An unverified user posted on the chat channel that the room booking conflict for CASE-300035 was resolved and the old...
- `e-rt-hard-000036-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300035 to restore the previous schedule rule configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000036-2']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000036-1', 'e-rt-hard-000036-2', 'e-rt-hard-000036-distractor-ver', 'e-rt-hard-000036-distractor-rollback', 'e-rt-hard-000036-distractor-ci', 'e-rt-hard-000036-bg-1']` overcitation=0.8333333333333334

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000026

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / wrong_source_attribution / L4
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000026-2']`

### Event trace summary
- `e-rt-hard-000026-1` (user): User asserts database connection pool limit is now 100.
- `e-rt-hard-000026-2` (maintainer): Maintainer clarifies pool limit is kept at 20 to prevent server exhaustion.
- `e-rt-hard-000026-distractor-ver` (user): User notes that on nightly build in workspace-beta, the vendor intake update has been applied to CASE-300025 and the ...
- `e-rt-hard-000026-distractor-auth` (user): An unverified user posted on the chat channel that the vendor intake conflict for CASE-300025 was resolved and the ol...
- `e-rt-hard-000026-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300025 to restore the previous handoff route configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000026-1', 'e-rt-hard-000026-2']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000026-1', 'e-rt-hard-000026-2', 'e-rt-hard-000026-distractor-ver', 'e-rt-hard-000026-distractor-rollback', 'e-rt-hard-000026-distractor-ci', 'e-rt-hard-000026-bg-1']` overcitation=0.8333333333333334

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000047

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / stale_memory_reuse / L3
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
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000047-1', 'e-rt-hard-000047-2', 'e-rt-hard-000047-3']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000047-1', 'e-rt-hard-000047-2', 'e-rt-hard-000047-3', 'e-rt-hard-000047-distractor-ver', 'e-rt-hard-000047-bg-1', 'e-rt-hard-000047-bg-2']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000018

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** authority_conflict / over_update / L4
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000018-1', 'e-rt-hard-000018-2', 'e-rt-hard-000018-3']`

### Event trace summary
- `e-rt-hard-000018-1` (docs): README docs updated: CASE-300017 now supports batch deletes.
- `e-rt-hard-000018-2` (reviewer): Code PR #300 implementing batch delete remains unmerged and tests are failing.
- `e-rt-hard-000018-3` (maintainer): Maintainer notes docs were merged ahead of code implementation by mistake.
- `e-rt-hard-000018-distractor-ver` (user): User notes that on nightly build in workspace-beta, the admin approval update has been applied to CASE-300017 and the...
- `e-rt-hard-000018-distractor-auth` (user): An unverified user posted on the chat channel that the admin approval conflict for CASE-300017 was resolved and the o...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000018-1', 'e-rt-hard-000018-2', 'e-rt-hard-000018-3']` diagnosis=`stale_memory_reuse`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000018-1', 'e-rt-hard-000018-2', 'e-rt-hard-000018-3', 'e-rt-hard-000018-distractor-ver', 'e-rt-hard-000018-distractor-rollback', 'e-rt-hard-000018-distractor-ci']` overcitation=0.5

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

## rt-hard-000006

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** scope_collision / scope_leakage / L4
- **Expected decision:** `use_current_memory`
- **Expected evidence:** `['e-rt-hard-000006-2']`

### Event trace summary
- `e-rt-hard-000006-1` (reviewer): PR on feature branch feat-311 enables Python 3.11 runtimes.
- `e-rt-hard-000006-2` (maintainer): Main release branch targets only Python 3.10 and does not accept Python 3.11 commits.
- `e-rt-hard-000006-distractor-ver` (user): User notes that on nightly build in workspace-beta, the consent boundary update has been applied to CASE-300005 and t...
- `e-rt-hard-000006-distractor-auth` (user): An unverified user posted on the chat channel that the consent boundary conflict for CASE-300005 was resolved and the...
- `e-rt-hard-000006-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300005 to restore the previous preference rule configurations te...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000006-2']` diagnosis=`scope_leakage`
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000006-1', 'e-rt-hard-000006-2', 'e-rt-hard-000006-distractor-ver', 'e-rt-hard-000006-distractor-rollback', 'e-rt-hard-000006-distractor-ci', 'e-rt-hard-000006-bg-1']` overcitation=0.8333333333333334

**Why hard:** Requires multi-event reasoning over verified/trusted records; latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.

