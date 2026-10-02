# /incident-runbook

Generate a structured incident runbook or post-mortem document from investigation findings.

## When to use

Invoke after an incident has been triaged or resolved to:
- Produce a formal post-mortem document
- Create a reusable runbook for a recurring incident pattern
- Hand off an active incident to another engineer
- Document root cause and prevention steps for stakeholders

## Instructions

When this skill is invoked:

1. Collect context (from conversation history or ask):
   - Incident title / summary
   - Severity (P1/P2/P3)
   - Affected services and regions
   - Start time and resolution time (if resolved)
   - Timeline of key events
   - Root cause (confirmed or suspected)
   - Fix that was applied (or is proposed)
   - Follow-up actions

2. If the incident is still active (not yet resolved):
   - Generate a **live runbook** with current findings and next investigation steps
   - Mark unresolved sections clearly

3. If the incident is resolved:
   - Generate a **post-mortem document** ready for stakeholder review

4. Write the document to `incidents/YYYY-MM-DD-<slug>.md` (e.g. `incidents/2026-09-30-TICKET-12345-cloud-sql-quota.md`).

5. After writing the runbook, record the incident to long-term memory:

   ```bash
   python3 scripts/memory_manager.py add-incident --stdin << 'EOF'
   {
     "id": "<jira-key>",
     "title": "<short title>",
     "severity": "P1|P2|P3",
     "platform": "gcp|k8s|aws|terraform",
     "affected_services": ["<service>"],
     "symptoms": ["<observed symptom>"],
     "root_cause": "<precise explanation>",
     "root_cause_category": "<pattern-category>",
     "resolution": "<steps taken>",
     "environment": "prod|test|uat",
     "duration_minutes": <N>,
     "detection": "alert|user report|monitoring",
     "runbook_file": "incidents/<filename>.md",
     "lessons": "<prevention action>"
   }
   EOF
   python3 scripts/memory_manager.py add-pattern --category <root_cause_category> --stdin << 'EOF'
   {"seen_count": 1, "symptoms": ["<any new symptom>"]}
   EOF
   ```

## Output Template

```markdown
## Incident: <title>

**Date**: <YYYY-MM-DD>
**Severity**: P1 / P2 / P3
**Status**: Resolved / Active
**Duration**: <start> → <end> (<total duration>)
**Affected**: <services, namespaces, GCP resources>
**Detected via**: <alert / user report / monitoring>

---

### Summary

<2-3 sentence plain-language summary suitable for stakeholders>

---

### Timeline

| Time (UTC) | Event |
|---|---|
| HH:MM | Symptom first observed |
| HH:MM | Alert fired / reported |
| HH:MM | Investigation started |
| HH:MM | Root cause identified |
| HH:MM | Fix applied |
| HH:MM | Service recovered / verified |

---

### Root Cause

<Precise technical explanation of what failed and why>

### Contributing Factors

- <factor>
- <factor>

---

### Resolution

Steps taken to resolve:

```bash
# Exact commands used
```

### Verification

<How recovery was confirmed — metrics recovered, error rate dropped, smoke test passed>

---

### Follow-up Actions

| Action | Owner | Due |
|---|---|---|
| <add alert for X> | <team> | <date> |
| <update resource limits> | <team> | <date> |
| <write/update runbook> | <team> | <date> |

---

### Prevention

<What change in configuration, alerting, or process would prevent recurrence>

---

### Lessons Learned

- <what went well>
- <what could be improved>
- <detection gap>
```
