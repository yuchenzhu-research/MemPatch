# Industrial Domains

ReTrace-Bench v2 evaluates memory reliability across 8 realistic industrial domains.

---

## 1. software_engineering_agent
- **Description**: An autonomous coding agent debugging issues, running tests, and updating files based on pull requests and issue tracker tickets.
- **Typical Event Sources**: GitHub issue updates, CI/CD run logs, git branch checkouts, compiler error messages.
- **Typical Memory Entries**: Ticket state, file dependencies, past compiler error resolutions, API tokens.
- **Major Failure Modes**: Stale code-context reuse, under-update after new branch pushes, failure to forget deprecated packages.
- **Example Black-box Task**: "Fix the failing test in `utils.py` using the updated library version description in Issue #42."
- **Example Memory-state Task**: "Identify if the memory entry regarding the deprecated `fetch_data` function is active or superseded."
- **Example Audit/Source Task**: "Localize the compiler error log that justifies the decision to replace `numpy.int` with `int`."

---

## 2. enterprise_multi_tool_workflow
- **Description**: A cross-platform assistant coordinating actions between Slack, Jira, linear task trackers, and Git.
- **Typical Event Sources**: Slack threads, Jira ticket description changes, GitHub webhook notifications.
- **Typical Memory Entries**: Task progress status, assigned owners, cross-system resource mappings.
- **Major Failure Modes**: Conflict collapse between Slack agreement and Jira ticket status, policy violations when sharing database credentials.
- **Example Black-box Task**: "Update the database schema changes on Jira after Slack consensus is reached."
- **Example Memory-state Task**: "Evaluate whether the Jira assignee memory entry is currently blocked by a prerequisite onboarding task."
- **Example Audit/Source Task**: "Which Slack message confirmed the approval of the schema change?"

---

## 3. customer_support_crm
- **Description**: A customer support CRM agent answering inquiries, updating user profiles, and tracking ticket resolutions.
- **Typical Event Sources**: Customer support emails, live chat messages, CRM account logs, refund tool triggers.
- **Typical Memory Entries**: User contact information, subscription tier, refund status, active support ticket IDs.
- **Major Failure Modes**: Stale memory reuse (using old email addresses), over-updating preferences based on sarcastic messages.
- **Example Black-box Task**: "Determine the correct email address to send the password reset link based on the user's latest updates."
- **Example Memory-state Task**: "Check the status of the user's refund status memory (is it authorized or superseded by a newer cancellation request)?"
- **Example Audit/Source Task**: "Find the event ID where the user requested to change their primary email."

---

## 4. calendar_task_workflow
- **Description**: A scheduling agent managing calendar meetings, task checklists, and personal availability alerts.
- **Typical Event Sources**: Email invites, calendar cancel notifications, daily planner logs.
- **Typical Memory Entries**: Meeting times, attendee responses, task deadlines, vacation schedules.
- **Major Failure Modes**: Failure to release or restore a meeting time slot after a cancel notification, stale schedule conflicts.
- **Example Black-box Task**: "Reschedule the weekly sync to Thursday afternoon based on the latest email replies."
- **Example Memory-state Task**: "Is the sync meeting memory entry currently authorized or blocked due to conflict?"
- **Example Audit/Source Task**: "Identify the email that canceled the Wednesday slot."

---

## 5. research_knowledge_work
- **Description**: A research assistant extracting facts, compiling literature reviews, and building citation maps.
- **Typical Event Sources**: Paper PDFs, database queries, reviewer feedback emails.
- **Typical Memory Entries**: Extracted claims, publication years, author lists, citation status.
- **Major Failure Modes**: Memory hallucination of non-existent papers, under-update of findings after errata releases.
- **Example Black-box Task**: "Compile a summary of research gaps in memory reliability as of 2026."
- **Example Memory-state Task**: "Decide if the extracted claim about LoCoMo is superseded by a newer erratum."
- **Example Audit/Source Task**: "Cite the exact PDF page where the claim was found."

---

## 6. personal_assistant_preference
- **Description**: A personal home assistant tracking user dietary requirements, work schedules, and home automation rules.
- **Typical Event Sources**: User voice instructions, smart home sensor logs, grocery store delivery notices.
- **Typical Memory Entries**: User allergies, favorite foods, thermostat settings, alarm times.
- **Major Failure Modes**: Scope leakage (exposing allergen information to public delivery services), under-updating preferences.
- **Example Black-box Task**: "Suggest a dinner menu excluding items the user is allergic to."
- **Example Memory-state Task**: "Verify if the favorite food memory entry is active or has been updated."
- **Example Audit/Source Task**: "What command did the user say to update their peanut allergy state?"

---

## 7. ecommerce_recommendation
- **Description**: An agent suggesting products and tracking order states across shopping carts.
- **Typical Event Sources**: Click logs, shopping cart checkout webhooks, order return notifications.
- **Typical Memory Entries**: Cart items, recent search keywords, discount codes.
- **Major Failure Modes**: Stale cart reuse, over-updating recommendation categories based on accidental clicks.
- **Example Black-box Task**: "Apply the valid discount code to the cart checkout."
- **Example Memory-state Task**: "Is the discount code memory entry authorized or superseded by a newer promotion?"
- **Example Audit/Source Task**: "Which user search term originally triggered this recommendation?"

---

## 8. data_analysis_bi
- **Description**: A data analyst agent querying SQL databases and generating reports for business intelligence.
- **Typical Event Sources**: Database schema updates, SQL query execution trace files, user presentation slide drafts.
- **Typical Memory Entries**: Table names, row counts, query performance logs, chart templates.
- **Major Failure Modes**: Scope leakage, under-updating calculations when source data refreshes.
- **Example Black-box Task**: "Report quarterly revenue growth based on the latest database transactions."
- **Example Memory-state Task**: "Identify if the database connection state memory entry is authorized or blocked by system maintenance."
- **Example Audit/Source Task**: "Point to the SQL query trace that returned the incorrect row count."
