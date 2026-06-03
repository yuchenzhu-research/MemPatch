# Hard150 Manual Inspection (20 cases)

## rt-hard-000107

- **Selection reason:** DeepSeek joint_revision_success = 0
- **Pattern / failure / difficulty:** closed_as_duplicate_not_fixed / stale_memory_reuse / L3
- **Expected decision:** `mark_unresolved`
- **Expected diagnosis:** `stale_memory_reuse`
- **Expected evidence:** `['e-rt-hard-000107-1', 'e-rt-hard-000107-2', 'e-rt-hard-000107-3']`

### Event trace summary
- `e-rt-hard-000107-1` (user): Issue #200 reports SSO timeout error on CASE-300106.
- `e-rt-hard-000107-2` (maintainer): Maintainer closed Issue #200 as duplicate of master Issue #98.
- `e-rt-hard-000107-3` (maintainer): Master Issue #98 remains open with labels: needs-investigation and priority-high.
- `e-rt-hard-000107-distractor-ver` (user): User notes that on nightly build in workspace-beta, the loyalty tier update has been applied to CASE-300106 and the n...
- `e-rt-hard-000107-distractor-auth` (user): An unverified user posted on the chat channel that the loyalty tier conflict for CASE-300106 was resolved and the old...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000107-1', 'e-rt-hard-000107-2', 'e-rt-hard-000107-3']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000107-1', 'e-rt-hard-000107-2', 'e-rt-hard-000107-3', 'e-rt-hard-000107-distractor-ver', 'e-rt-hard-000107-bg-1', 'e-rt-hard-000107-bg-2']` overcitation=0.5


## rt-hard-000072

- **Selection reason:** DeepSeek joint_revision_success = 0
- **Pattern / failure / difficulty:** stale_comment_after_new_release / under_update / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `under_update`
- **Expected evidence:** `['e-rt-hard-000072-2']`

### Event trace summary
- `e-rt-hard-000072-1` (user): Comment from 2024 states retry limit is locked to 3.
- `e-rt-hard-000072-2` (release_note): Release v3.0 updates default retry limit to 5.
- `e-rt-hard-000072-distractor-ver` (user): User notes that on nightly build in workspace-beta, the source-table lineage update has been applied to CASE-300071 a...
- `e-rt-hard-000072-distractor-auth` (user): An unverified user posted on the chat channel that the source-table lineage conflict for CASE-300071 was resolved and...
- `e-rt-hard-000072-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300071 to restore the previous metric definition configurations ...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000072-1', 'e-rt-hard-000072-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000072-1', 'e-rt-hard-000072-2', 'e-rt-hard-000072-distractor-ver', 'e-rt-hard-000072-distractor-rollback', 'e-rt-hard-000072-distractor-ci', 'e-rt-hard-000072-bg-1']` overcitation=0.8333333333333334


## rt-hard-000113

- **Selection reason:** DeepSeek joint_revision_success = 0
- **Pattern / failure / difficulty:** ci_failed_after_claim / conflict_collapse / L3
- **Expected decision:** `mark_unresolved`
- **Expected diagnosis:** `conflict_collapse`
- **Expected evidence:** `['e-rt-hard-000113-1', 'e-rt-hard-000113-2']`

### Event trace summary
- `e-rt-hard-000113-1` (user): Developer states performance hotfix is ready and merged.
- `e-rt-hard-000113-2` (ci): CI pipeline check for performance hotfix failed during compilation.
- `e-rt-hard-000113-distractor-ver` (user): User notes that on nightly build in workspace-beta, the API migration update has been applied to CASE-300112 and the ...
- `e-rt-hard-000113-distractor-auth` (user): An unverified user posted on the chat channel that the API migration conflict for CASE-300112 was resolved and the ol...
- `e-rt-hard-000113-bg-1` (bot): Continuous Integration agent completed baseline checks for auxiliary module.
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000113-1', 'e-rt-hard-000113-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000113-1', 'e-rt-hard-000113-2', 'e-rt-hard-000113-distractor-ver', 'e-rt-hard-000113-bg-1', 'e-rt-hard-000113-bg-2', 'e-rt-hard-000113-bg-3']` overcitation=0.6666666666666666


## rt-hard-000133

- **Selection reason:** DeepSeek joint_revision_success = 0
- **Pattern / failure / difficulty:** label_state_mismatch / conflict_collapse / L3
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `conflict_collapse`
- **Expected evidence:** `['e-rt-hard-000133-2']`

### Event trace summary
- `e-rt-hard-000133-1` (user): Issue #900 reports memory leaks under load.
- `e-rt-hard-000133-2` (maintainer): Maintainer closes issue with label wontfix, commenting that leak is within acceptable limits.
- `e-rt-hard-000133-distractor-ver` (user): User notes that on nightly build in workspace-beta, the paper claim update has been applied to CASE-300132 and the ne...
- `e-rt-hard-000133-distractor-auth` (user): An unverified user posted on the chat channel that the paper claim conflict for CASE-300132 was resolved and the old ...
- `e-rt-hard-000133-bg-1` (bot): Reviewer notes scheduling queue has no extra blocked actions pending.
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000133-1', 'e-rt-hard-000133-2', 'e-rt-hard-000133-distractor-auth']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000133-1', 'e-rt-hard-000133-2', 'e-rt-hard-000133-distractor-ver', 'e-rt-hard-000133-bg-1', 'e-rt-hard-000133-bg-2', 'e-rt-hard-000133-bg-3']` overcitation=0.8333333333333334


## rt-hard-000117

- **Selection reason:** DeepSeek joint_revision_success = 0
- **Pattern / failure / difficulty:** stale_comment_after_new_release / stale_memory_reuse / L3
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `stale_memory_reuse`
- **Expected evidence:** `['e-rt-hard-000117-2']`

### Event trace summary
- `e-rt-hard-000117-1` (user): Comment from 2024 states retry limit is locked to 3.
- `e-rt-hard-000117-2` (release_note): Release v3.0 updates default retry limit to 5.
- `e-rt-hard-000117-distractor-ver` (user): User notes that on nightly build in workspace-beta, the retraction note update has been applied to CASE-300116 and th...
- `e-rt-hard-000117-distractor-auth` (user): An unverified user posted on the chat channel that the retraction note conflict for CASE-300116 was resolved and the ...
- `e-rt-hard-000117-bg-1` (bot): Documentation system refreshed the stable index files.
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000117-1', 'e-rt-hard-000117-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000117-1', 'e-rt-hard-000117-2', 'e-rt-hard-000117-distractor-ver', 'e-rt-hard-000117-bg-1', 'e-rt-hard-000117-bg-2', 'e-rt-hard-000117-bg-3']` overcitation=0.8333333333333334


## rt-hard-000081

- **Selection reason:** DeepSeek joint_revision_success = 1
- **Pattern / failure / difficulty:** branch_scope_leakage / scope_leakage / L3
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `scope_leakage`
- **Expected evidence:** `['e-rt-hard-000081-2']`

### Event trace summary
- `e-rt-hard-000081-1` (reviewer): PR on feature branch feat-311 enables Python 3.11 runtimes.
- `e-rt-hard-000081-2` (maintainer): Main release branch targets only Python 3.10 and does not accept Python 3.11 commits.
- `e-rt-hard-000081-distractor-ver` (user): User notes that on nightly build in workspace-beta, the API migration update has been applied to CASE-300080 and the ...
- `e-rt-hard-000081-distractor-auth` (user): An unverified user posted on the chat channel that the API migration conflict for CASE-300080 was resolved and the ol...
- `e-rt-hard-000081-bg-1` (bot): API gateway logged successful authentication check.
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000081-2']` diagnosis=`scope_leakage` joint=1.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000081-1', 'e-rt-hard-000081-2', 'e-rt-hard-000081-distractor-ver', 'e-rt-hard-000081-bg-1', 'e-rt-hard-000081-bg-2', 'e-rt-hard-000081-bg-3']` overcitation=0.8333333333333334


## rt-hard-000048

- **Selection reason:** DeepSeek joint_revision_success = 1
- **Pattern / failure / difficulty:** docs_ahead_of_code / over_update / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `over_update`
- **Expected evidence:** `['e-rt-hard-000048-1', 'e-rt-hard-000048-2', 'e-rt-hard-000048-3']`

### Event trace summary
- `e-rt-hard-000048-1` (docs): README docs updated: CASE-300047 now supports batch deletes.
- `e-rt-hard-000048-2` (reviewer): Code PR #300 implementing batch delete remains unmerged and tests are failing.
- `e-rt-hard-000048-3` (maintainer): Maintainer notes docs were merged ahead of code implementation by mistake.
- `e-rt-hard-000048-distractor-ver` (user): User notes that on nightly build in workspace-beta, the filter changes update has been applied to CASE-300047 and the...
- `e-rt-hard-000048-distractor-auth` (user): An unverified user posted on the chat channel that the filter changes conflict for CASE-300047 was resolved and the o...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000048-1', 'e-rt-hard-000048-2', 'e-rt-hard-000048-3']` diagnosis=`stale_memory_reuse` joint=1.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000048-1', 'e-rt-hard-000048-2', 'e-rt-hard-000048-3', 'e-rt-hard-000048-distractor-ver', 'e-rt-hard-000048-distractor-rollback', 'e-rt-hard-000048-distractor-ci']` overcitation=0.5


## rt-hard-000052

- **Selection reason:** DeepSeek joint_revision_success = 1
- **Pattern / failure / difficulty:** authority_conflict / wrong_source_attribution / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `wrong_source_attribution`
- **Expected evidence:** `['e-rt-hard-000052-2']`

### Event trace summary
- `e-rt-hard-000052-1` (user): User claims CVE-999 vulnerability has been fixed in latest patch.
- `e-rt-hard-000052-2` (security): Security auditor confirms CVE-999 remains active and unpatched in current builds.
- `e-rt-hard-000052-distractor-ver` (user): User notes that on nightly build in workspace-beta, the attendee authority update has been applied to CASE-300051 and...
- `e-rt-hard-000052-distractor-auth` (user): An unverified user posted on the chat channel that the attendee authority conflict for CASE-300051 was resolved and t...
- `e-rt-hard-000052-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300051 to restore the previous schedule rule configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000052-2']` diagnosis=`scope_leakage` joint=1.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000052-2', 'e-rt-hard-000052-distractor-ver', 'e-rt-hard-000052-distractor-rollback', 'e-rt-hard-000052-distractor-ci', 'e-rt-hard-000052-bg-1']` overcitation=0.8


## rt-hard-000110

- **Selection reason:** DeepSeek joint_revision_success = 1
- **Pattern / failure / difficulty:** version_scope_leakage / scope_leakage / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `scope_leakage`
- **Expected evidence:** `['e-rt-hard-000110-1', 'e-rt-hard-000110-3']`

### Event trace summary
- `e-rt-hard-000110-1` (user): XML exporter is required for production reports on v1.
- `e-rt-hard-000110-2` (reviewer): Developer removed XML exporter on v2 development branch.
- `e-rt-hard-000110-3` (maintainer): Maintainer confirms XML export will remain fully supported on v1 stable.
- `e-rt-hard-000110-distractor-ver` (user): User notes that on nightly build in workspace-beta, the location-specific preference update has been applied to CASE-...
- `e-rt-hard-000110-distractor-auth` (user): An unverified user posted on the chat channel that the location-specific preference conflict for CASE-300109 was reso...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000110-1', 'e-rt-hard-000110-3']` diagnosis=`scope_leakage` joint=1.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000110-1', 'e-rt-hard-000110-2', 'e-rt-hard-000110-3', 'e-rt-hard-000110-distractor-ver', 'e-rt-hard-000110-distractor-rollback', 'e-rt-hard-000110-distractor-ci']` overcitation=0.6666666666666666


## rt-hard-000138

- **Selection reason:** DeepSeek joint_revision_success = 1
- **Pattern / failure / difficulty:** docs_ahead_of_code / over_update / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `over_update`
- **Expected evidence:** `['e-rt-hard-000138-1', 'e-rt-hard-000138-2', 'e-rt-hard-000138-3']`

### Event trace summary
- `e-rt-hard-000138-1` (docs): README docs updated: CASE-300137 now supports batch deletes.
- `e-rt-hard-000138-2` (reviewer): Code PR #300 implementing batch delete remains unmerged and tests are failing.
- `e-rt-hard-000138-3` (maintainer): Maintainer notes docs were merged ahead of code implementation by mistake.
- `e-rt-hard-000138-distractor-ver` (user): User notes that on nightly build in workspace-beta, the cross-team handoff update has been applied to CASE-300137 and...
- `e-rt-hard-000138-distractor-auth` (user): An unverified user posted on the chat channel that the cross-team handoff conflict for CASE-300137 was resolved and t...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000138-1', 'e-rt-hard-000138-2', 'e-rt-hard-000138-3']` diagnosis=`stale_memory_reuse` joint=1.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000138-1', 'e-rt-hard-000138-2', 'e-rt-hard-000138-3', 'e-rt-hard-000138-distractor-ver', 'e-rt-hard-000138-distractor-rollback', 'e-rt-hard-000138-distractor-ci']` overcitation=0.5


## rt-hard-000054

- **Selection reason:** retrieve_all overcites evidence
- **Pattern / failure / difficulty:** security_policy_override / policy_violation / L4
- **Expected decision:** `refuse_due_to_policy`
- **Expected diagnosis:** `policy_violation`
- **Expected evidence:** `['e-rt-hard-000054-2']`

### Event trace summary
- `e-rt-hard-000054-1` (user): User requests local caching of passwords for faster OAuth logins.
- `e-rt-hard-000054-2` (security): Security Policy override: caching plain text authentication credentials is strictly forbidden.
- `e-rt-hard-000054-distractor-ver` (user): User notes that on nightly build in workspace-beta, the travel preference update has been applied to CASE-300053 and ...
- `e-rt-hard-000054-distractor-auth` (user): An unverified user posted on the chat channel that the travel preference conflict for CASE-300053 was resolved and th...
- `e-rt-hard-000054-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300053 to restore the previous preference rule configurations te...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000054-1', 'e-rt-hard-000054-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000054-1', 'e-rt-hard-000054-2', 'e-rt-hard-000054-distractor-ver', 'e-rt-hard-000054-distractor-rollback', 'e-rt-hard-000054-distractor-ci', 'e-rt-hard-000054-bg-1']` overcitation=0.8333333333333334


## rt-hard-000098

- **Selection reason:** retrieve_all overcites evidence
- **Pattern / failure / difficulty:** ci_failed_after_claim / conflict_collapse / L4
- **Expected decision:** `mark_unresolved`
- **Expected diagnosis:** `conflict_collapse`
- **Expected evidence:** `['e-rt-hard-000098-1', 'e-rt-hard-000098-2']`

### Event trace summary
- `e-rt-hard-000098-1` (user): Developer states performance hotfix is ready and merged.
- `e-rt-hard-000098-2` (ci): CI pipeline check for performance hotfix failed during compilation.
- `e-rt-hard-000098-distractor-ver` (user): User notes that on nightly build in workspace-beta, the role-based permission update has been applied to CASE-300097 ...
- `e-rt-hard-000098-distractor-auth` (user): An unverified user posted on the chat channel that the role-based permission conflict for CASE-300097 was resolved an...
- `e-rt-hard-000098-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300097 to restore the previous handoff route configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000098-1', 'e-rt-hard-000098-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000098-1', 'e-rt-hard-000098-2', 'e-rt-hard-000098-distractor-ver', 'e-rt-hard-000098-distractor-rollback', 'e-rt-hard-000098-distractor-ci', 'e-rt-hard-000098-bg-1']` overcitation=0.6666666666666666


## rt-hard-000066

- **Selection reason:** retrieve_all overcites evidence
- **Pattern / failure / difficulty:** branch_scope_leakage / scope_leakage / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `scope_leakage`
- **Expected evidence:** `['e-rt-hard-000066-2']`

### Event trace summary
- `e-rt-hard-000066-1` (reviewer): PR on feature branch feat-311 enables Python 3.11 runtimes.
- `e-rt-hard-000066-2` (maintainer): Main release branch targets only Python 3.10 and does not accept Python 3.11 commits.
- `e-rt-hard-000066-distractor-ver` (user): User notes that on nightly build in workspace-beta, the role-based permission update has been applied to CASE-300065 ...
- `e-rt-hard-000066-distractor-auth` (user): An unverified user posted on the chat channel that the role-based permission conflict for CASE-300065 was resolved an...
- `e-rt-hard-000066-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300065 to restore the previous handoff route configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000066-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000066-1', 'e-rt-hard-000066-2', 'e-rt-hard-000066-distractor-ver', 'e-rt-hard-000066-distractor-rollback', 'e-rt-hard-000066-distractor-ci', 'e-rt-hard-000066-bg-1']` overcitation=0.8333333333333334


## rt-hard-000114

- **Selection reason:** retrieve_all overcites evidence
- **Pattern / failure / difficulty:** security_policy_override / policy_violation / L4
- **Expected decision:** `refuse_due_to_policy`
- **Expected diagnosis:** `policy_violation`
- **Expected evidence:** `['e-rt-hard-000114-2']`

### Event trace summary
- `e-rt-hard-000114-1` (user): User requests local caching of passwords for faster OAuth logins.
- `e-rt-hard-000114-2` (security): Security Policy override: caching plain text authentication credentials is strictly forbidden.
- `e-rt-hard-000114-distractor-ver` (user): User notes that on nightly build in workspace-beta, the admin approval update has been applied to CASE-300113 and the...
- `e-rt-hard-000114-distractor-auth` (user): An unverified user posted on the chat channel that the admin approval conflict for CASE-300113 was resolved and the o...
- `e-rt-hard-000114-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300113 to restore the previous handoff route configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000114-1', 'e-rt-hard-000114-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000114-1', 'e-rt-hard-000114-2', 'e-rt-hard-000114-distractor-ver', 'e-rt-hard-000114-distractor-rollback', 'e-rt-hard-000114-distractor-ci', 'e-rt-hard-000114-bg-1']` overcitation=0.8333333333333334


## rt-hard-000119

- **Selection reason:** retrieve_all overcites evidence
- **Pattern / failure / difficulty:** multi_memory_coupling / under_update / L3
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `under_update`
- **Expected evidence:** `['e-rt-hard-000119-2']`

### Event trace summary
- `e-rt-hard-000119-1` (user): Request to migrate sync connections to async config.
- `e-rt-hard-000119-2` (maintainer): Maintainer merges async client and timeout migrations conjointly.
- `e-rt-hard-000119-distractor-ver` (user): User notes that on nightly build in workspace-beta, the return window update has been applied to CASE-300118 and the ...
- `e-rt-hard-000119-distractor-auth` (user): An unverified user posted on the chat channel that the return window conflict for CASE-300118 was resolved and the ol...
- `e-rt-hard-000119-bg-1` (bot): Reviewer notes scheduling queue has no extra blocked actions pending.
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000119-1', 'e-rt-hard-000119-2']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000119-1', 'e-rt-hard-000119-2', 'e-rt-hard-000119-distractor-ver', 'e-rt-hard-000119-bg-1', 'e-rt-hard-000119-bg-2', 'e-rt-hard-000119-bg-3']` overcitation=0.8333333333333334


## rt-hard-000004

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** release_then_revert / stale_memory_reuse / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `stale_memory_reuse`
- **Expected evidence:** `['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3']`

### Event trace summary
- `e-rt-hard-000004-1` (release_note): Release v2.0.0 ships with strict schema validation active.
- `e-rt-hard-000004-2` (maintainer): Revert PR #401 merged: reverts strict validation due to regressions.
- `e-rt-hard-000004-3` (release_note): Release v2.0.1 reverts strict validation and returns default config to opt-in.
- `e-rt-hard-000004-distractor-ver` (user): User notes that on nightly build in workspace-beta, the room booking update has been applied to CASE-300003 and the n...
- `e-rt-hard-000004-distractor-auth` (user): An unverified user posted on the chat channel that the room booking conflict for CASE-300003 was resolved and the old...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000004-1', 'e-rt-hard-000004-2', 'e-rt-hard-000004-3', 'e-rt-hard-000004-distractor-ver', 'e-rt-hard-000004-distractor-rollback', 'e-rt-hard-000004-distractor-ci']` overcitation=0.5


## rt-hard-000010

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** backport_only_fix / stale_memory_reuse / L4
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `stale_memory_reuse`
- **Expected evidence:** `['e-rt-hard-000010-1', 'e-rt-hard-000010-2']`

### Event trace summary
- `e-rt-hard-000010-1` (maintainer): Security patch backported only to stable legacy branch v1.2.
- `e-rt-hard-000010-2` (maintainer): Production branch v2.0 uses a redesigned core and does not need or receive the patch.
- `e-rt-hard-000010-distractor-ver` (user): User notes that on nightly build in workspace-beta, the cross-team handoff update has been applied to CASE-300009 and...
- `e-rt-hard-000010-distractor-auth` (user): An unverified user posted on the chat channel that the cross-team handoff conflict for CASE-300009 was resolved and t...
- `e-rt-hard-000010-distractor-rollback` (reviewer): Reviewer proposed to revert the latest patch on CASE-300009 to restore the previous handoff route configurations temp...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000010-1', 'e-rt-hard-000010-2']` diagnosis=`scope_leakage` joint=1.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000010-1', 'e-rt-hard-000010-2', 'e-rt-hard-000010-distractor-ver', 'e-rt-hard-000010-distractor-rollback', 'e-rt-hard-000010-distractor-ci', 'e-rt-hard-000010-bg-1']` overcitation=0.6666666666666666


## rt-hard-000017

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** closed_as_duplicate_not_fixed / stale_memory_reuse / L3
- **Expected decision:** `mark_unresolved`
- **Expected diagnosis:** `stale_memory_reuse`
- **Expected evidence:** `['e-rt-hard-000017-1', 'e-rt-hard-000017-2', 'e-rt-hard-000017-3']`

### Event trace summary
- `e-rt-hard-000017-1` (user): Issue #200 reports SSO timeout error on CASE-300016.
- `e-rt-hard-000017-2` (maintainer): Maintainer closed Issue #200 as duplicate of master Issue #98.
- `e-rt-hard-000017-3` (maintainer): Master Issue #98 remains open with labels: needs-investigation and priority-high.
- `e-rt-hard-000017-distractor-ver` (user): User notes that on nightly build in workspace-beta, the API migration update has been applied to CASE-300016 and the ...
- `e-rt-hard-000017-distractor-auth` (user): An unverified user posted on the chat channel that the API migration conflict for CASE-300016 was resolved and the ol...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000017-1', 'e-rt-hard-000017-2', 'e-rt-hard-000017-3']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000017-1', 'e-rt-hard-000017-2', 'e-rt-hard-000017-3', 'e-rt-hard-000017-distractor-ver', 'e-rt-hard-000017-bg-1', 'e-rt-hard-000017-bg-2']` overcitation=0.5


## rt-hard-000057

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** stale_comment_after_new_release / stale_memory_reuse / L3
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `stale_memory_reuse`
- **Expected evidence:** `['e-rt-hard-000057-2']`

### Event trace summary
- `e-rt-hard-000057-1` (user): Comment from 2024 states retry limit is locked to 3.
- `e-rt-hard-000057-2` (release_note): Release v3.0 updates default retry limit to 5.
- `e-rt-hard-000057-distractor-ver` (user): User notes that on nightly build in workspace-beta, the rollout blocker update has been applied to CASE-300056 and th...
- `e-rt-hard-000057-distractor-auth` (user): An unverified user posted on the chat channel that the rollout blocker conflict for CASE-300056 was resolved and the ...
- `e-rt-hard-000057-bg-1` (bot): The operations log record shows database connection pool state is active.
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000057-1', 'e-rt-hard-000057-2']` diagnosis=`stale_memory_reuse` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000057-1', 'e-rt-hard-000057-2', 'e-rt-hard-000057-distractor-ver', 'e-rt-hard-000057-bg-1', 'e-rt-hard-000057-bg-2', 'e-rt-hard-000057-bg-3']` overcitation=0.8333333333333334


## rt-hard-000135

- **Selection reason:** random sample
- **Pattern / failure / difficulty:** negative_evidence_required / under_update / L3
- **Expected decision:** `use_current_memory`
- **Expected diagnosis:** `under_update`
- **Expected evidence:** `['e-rt-hard-000135-1', 'e-rt-hard-000135-3']`

### Event trace summary
- `e-rt-hard-000135-1` (user): Issue #500 reports SSL routing error.
- `e-rt-hard-000135-2` (user): Developer opens PR #501 to resolve SSL routing error.
- `e-rt-hard-000135-3` (reviewer): Reviewer states PR #501 is on hold and no merge action has been approved.
- `e-rt-hard-000135-distractor-ver` (user): User notes that on nightly build in workspace-beta, the stock availability update has been applied to CASE-300134 and...
- `e-rt-hard-000135-distractor-auth` (user): An unverified user posted on the chat channel that the stock availability conflict for CASE-300134 was resolved and t...
- ... and 2 more events

### Model outputs
- **DeepSeek-V4-Pro:** decision=`use_current_memory` evidence=`['e-rt-hard-000135-1', 'e-rt-hard-000135-2', 'e-rt-hard-000135-3']` diagnosis=`scope_leakage` joint=0.0
- **retrieve_all:** decision=`use_current_memory` evidence=`['e-rt-hard-000135-1', 'e-rt-hard-000135-2', 'e-rt-hard-000135-3', 'e-rt-hard-000135-distractor-ver', 'e-rt-hard-000135-bg-1', 'e-rt-hard-000135-bg-2']` overcitation=0.6666666666666666


