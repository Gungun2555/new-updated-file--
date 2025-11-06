# ================================================================
# 📘 schema_context.py
# Complete Descriptive SQL Context for List Management Agent
# Includes detailed table, view, and reasoning guidance
# ================================================================

SCHEMA_CONTEXT = """
================================================================
🧠 SYSTEM OVERVIEW
================================================================
This database powers a **List Management and Evolution Tracking System** used to
capture, track, and analyze lists of healthcare professionals (HCPs),
their versions, business requests, and related activities.

Each list evolves through multiple versions, reflecting updates,
additions, or removals of doctors (HCPs) and is connected to requests,
sales activities, competitor tracking, and audit logs.

The database allows answering:
- Who requested a list and for what reason?
- How has a list evolved across versions?
- Who made each change and why?
- What is the current state of the list?
- Which sales reps or competitors interacted with the HCPs?

================================================================
🏗️ CORE TABLES AND THEIR PURPOSES
================================================================

----------------------------------------------------------------
1️⃣ domains
----------------------------------------------------------------
Purpose:
Defines the high-level **business domains** (e.g., "Cardiology", "Oncology").

When to use:
- To categorize or filter requests based on domain.
- To retrieve all subdomains under a specific domain.

Example query:
SELECT * FROM domains WHERE domain_name ILIKE '%Oncology%';


----------------------------------------------------------------
2️⃣ subdomains
----------------------------------------------------------------
Purpose:
Defines subdivisions under each domain (e.g., "Cardiology → Interventional").

Relationships:
Each subdomain belongs to one domain.

When to use:
- When you need to organize requests or lists by finer business categories.

Example:
SELECT * FROM subdomains WHERE domain_id = 1;


----------------------------------------------------------------
3️⃣ list_requests
----------------------------------------------------------------
Purpose:
Represents a **business request** for creating or updating a list of HCPs.

Columns:
- requester_name → Who requested it.
- request_purpose → Why it was requested.
- status → Current progress ("Requested", "In Progress", etc.).
- assigned_to → Who is responsible.
- created_at → When the request was made.

When to use:
- To answer "Who requested what and why?"
- To filter by request status or creation date.

Example:
SELECT requester_name, request_purpose, status FROM list_requests;


----------------------------------------------------------------
4️⃣ list_versions
----------------------------------------------------------------
Purpose:
Tracks the **evolution** of each request via numbered versions.

Columns:
- request_id → Link to the parent request.
- version_number → The version index (1, 2, 3…).
- change_type → Type of update ("Additions", "Removals", "Edits").
- change_rationale → Reason for the change.
- created_by → Who made this version.
- is_current → Marks if it's the latest version.

When to use:
- To see how a list evolved over time.
- To identify who made specific changes.

Example:
SELECT * FROM list_versions WHERE request_id = 10 ORDER BY version_number;


----------------------------------------------------------------
5️⃣ target_list_entries
----------------------------------------------------------------
Purpose:
Stores **HCPs (doctors)** in each list version — this is the *core data* of lists.

Columns:
- hcp_id, hcp_name → HCP identifiers.
- specialty, territory → HCP details.
- tier → Categorization (A/B/C/D tiers for priority).

When to use:
- To retrieve the current or previous HCPs in a target list.
- To compare HCPs between two versions.

Example:
SELECT hcp_name, specialty, tier FROM target_list_entries WHERE version_id = 20;


----------------------------------------------------------------
6️⃣ call_list_entries
----------------------------------------------------------------
Purpose:
Represents **planned or completed calls** made by sales reps to HCPs.

Columns:
- hcp_id, hcp_name
- call_date, sales_rep, status ("Planned", "Completed", etc.)

When to use:
- To track field interactions with doctors.
- To generate call performance reports.

Example:
SELECT hcp_name, call_date, status FROM call_list_entries WHERE sales_rep = 'Alex';


----------------------------------------------------------------
7️⃣ competitor_target_entries
----------------------------------------------------------------
Purpose:
Tracks **competitor engagements** with HCPs.

Columns:
- competitor_product, conversion_potential (High/Medium/Low), assigned_rep

When to use:
- To analyze competitor activity per HCP or territory.
- To assess conversion potential or overlaps with internal targets.

Example:
SELECT hcp_name, competitor_product, conversion_potential FROM competitor_target_entries;


----------------------------------------------------------------
8️⃣ digital_engagement_entries
----------------------------------------------------------------
Purpose:
Captures **digital outreach activities** such as email communications.

Columns:
- contact_name, email, specialty, opt_in (boolean)

When to use:
- To see which contacts are engaged digitally.
- To respect opt-in preferences in campaigns.

Example:
SELECT * FROM digital_engagement_entries WHERE opt_in = true;


----------------------------------------------------------------
9️⃣ formulary_decision_maker_entries
----------------------------------------------------------------
Purpose:
Stores contacts who make formulary or approval decisions for drugs/products.

Columns:
- contact_name, organization, influence_level (High/Medium/Low)

When to use:
- To find decision-makers relevant to a domain or region.

Example:
SELECT contact_name, organization FROM formulary_decision_maker_entries WHERE influence_level = 'High';


----------------------------------------------------------------
🔟 high_value_prescriber_entries
----------------------------------------------------------------
Purpose:
Captures HCPs generating **high prescription or revenue volume**.

Columns:
- total_prescriptions, revenue, value_tier

When to use:
- To identify high-value targets.
- To segment HCPs by revenue or prescription volume.

Example:
SELECT hcp_name, revenue FROM high_value_prescriber_entries ORDER BY revenue DESC;


----------------------------------------------------------------
1️⃣1️⃣ idn_health_system_entries
----------------------------------------------------------------
Purpose:
Tracks **health systems or hospital networks** and their importance.

Columns:
- system_name, contact_name, importance (Tier 1/2/3)

When to use:
- To identify key healthcare organizations and points of contact.

Example:
SELECT system_name, importance FROM idn_health_system_entries WHERE importance = 'Tier 1';


----------------------------------------------------------------
1️⃣2️⃣ work_logs
----------------------------------------------------------------
Purpose:
Maintains an **audit trail** for all requests, versions, and activities.

Columns:
- worker_name, activity_description, decisions_made, activity_date

When to use:
- To find who performed which action and when.
- To trace accountability and collaboration history.

Example:
SELECT * FROM work_logs WHERE worker_name = 'Priya';


================================================================
👁️ DATABASE VIEWS (Aggregated Insights)
================================================================

----------------------------------------------------------------
view_request_context
----------------------------------------------------------------
Purpose:
Provides a **complete context** of requests, versions, domains, and work logs.

Use When:
- You need an all-in-one overview of the request lifecycle.

Example:
SELECT * FROM view_request_context WHERE requester_name ILIKE '%Rohan%';


----------------------------------------------------------------
view_target_list_full
----------------------------------------------------------------
Purpose:
Joins list_versions with target_list_entries — shows HCPs with version details.

Use When:
- You want to see HCP details along with who created the version and why.

Example:
SELECT * FROM view_target_list_full WHERE version_number = 3;


----------------------------------------------------------------
view_list_evolution
----------------------------------------------------------------
Purpose:
Summarizes how a list changed over time, showing rationale and author.

Use When:
- You want historical analysis of list versions.

Example:
SELECT * FROM view_list_evolution WHERE change_type = 'Addition';


----------------------------------------------------------------
v_current_state_target_list
----------------------------------------------------------------
Purpose:
Compares original and current HCP lists, showing what changed (Added/Removed/Modified).

Use When:
- You want to understand differences between the first and latest versions.

Example:
SELECT * FROM v_current_state_target_list WHERE change_status = 'Removed';


----------------------------------------------------------------
view_work_attribution
----------------------------------------------------------------
Purpose:
Shows who contributed to which request or version (by name, action, and date).

Use When:
- You want to track workload and contributions.

Example:
SELECT * FROM view_work_attribution WHERE domain_name = 'Oncology';


================================================================
🔗 RELATIONSHIP SUMMARY
================================================================
- domains (1) ──< subdomains
- subdomains (1) ──< list_requests
- list_requests (1) ──< list_versions
- list_versions (1) ──< target_list_entries / call_list_entries / competitor_target_entries / others
- list_requests (1) ──< work_logs

So:
→ domains organize data hierarchically  
→ list_requests define the *why*  
→ list_versions define the *how it changed*  
→ entries define the *what data*  
→ work_logs define the *who did what*


================================================================
🧩 INTELLIGENT QUERY REASONING HINTS
================================================================
If user asks for:
- **"Current list", "latest version", "HCPs"** → use `view_target_list_full`
- **"Changes", "differences", "what changed"** → use `v_current_state_target_list` or `view_list_evolution`
- **"Who requested", "purpose", "assigned person"** → use `list_requests` or `view_request_context`
- **"Who made updates", "actions taken", "history"** → use `work_logs` or `view_work_attribution`
- **"Competitor", "market", "conversion"** → use `competitor_target_entries`
- **"Sales call", "rep performance"** → use `call_list_entries`
- **"Decision makers", "formulary"** → use `formulary_decision_maker_entries`
- **"High value", "top doctors"** → use `high_value_prescriber_entries`
- **"Hospitals", "systems", "network"** → use `idn_health_system_entries`

================================================================
💡 BEST PRACTICES FOR SQL GENERATION
================================================================
1. Always use JOINs through foreign keys — never guess implicit relations.
2. Prefer VIEWS for business queries (they already have joins done).
3. Use filters with ILIKE for user search terms (case-insensitive).
4. When uncertain, prioritize returning *informative summaries* over raw IDs.

================================================================
✅ END OF CONTEXT
================================================================
"""
