# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# DevOps — GCP & Kubernetes

## Project Purpose

This project provides AI-assisted incident response for Google Cloud Platform (GCP), AWS and Kubernetes environments. Use it to diagnose cluster health issues, investigate failing workloads, database issues and trace infrastructure anomalies to resolution.

---

## Core Skills

| Slash Command | What it does |
|---|---|
| `/gcp-incident` | GCP-focused investigation: quotas, networking, IAM, Cloud SQL, GKE control plane |
| `/k8s-incident` | Kubernetes workload deep-dive: pods, deployments, HPA, events, OOMKill, CrashLoops |
| `/rds-replica-lag` | RDS read replica lag diagnosis: queries, write spikes, checkpoints, WAL replay issues |
| `/incident-runbook` | Generate a structured runbook from symptoms → root cause → resolution |
| `/read-jira` | Fetch Jira ticket details, search with JQL, post comments — uses macOS Keychain for auth |
| `/log-investigator` | Investigate Cloud Logging URLs — expands shortened links, fetches logs, analyzes errors and patterns |
| `/security-scan` | Scan `.claude/` directory for security vulnerabilities, misconfigurations, and injection risks |

---

## Available Agents

| Agent | Role |
|---|---|
| `GCP Investigator` | Queries GCP APIs, reads logs/metrics, checks quotas and IAM |
| `K8s Investigator` | Inspects cluster state via kubectl, finds failing workloads |
| `Incident Commander` | Orchestrates both agents, synthesizes findings, produces incident report |
| `AWS Investigator` | Diagnoses AWS issues: RDS, IAM, VPC/SGs/NACLs, ALB, Route53, CloudWatch, Lambda, S3, KMS, WAF, EC2/EBS, CloudTrail |
| `Terraform Inspector` | Reads Terraform code in any repo under `~/ACME/Bitbucket/` to map intended infrastructure and detect misconfigurations — code only, no live cloud queries |
| `security-reviewer` | Scans code for secrets, SSRF, injection, and OWASP Top 10 vulnerabilities |

### Automatic Hooks

Three hooks run automatically on every session:

1. **AWS Read-Only Guard** (`PreToolUse`) — blocks mutating `aws` CLI commands (create, delete, modify, put, attach, etc.) and returns an error with `[AWS Read-Only Guard]` prefix. To run a mutating AWS command, ask the user for explicit confirmation first.

2. **Bash Logger** (`PostToolUse`) — logs every Bash command to `logs/YYYY-MM-DD.md` for audit and debugging.

3. **Session Memory Hook** (`Stop`) — at session end, scans for completed runbooks with a `### Root Cause` section and auto-records them to `incidents.jsonl` (episodic memory). No manual `add-incident` call needed.

---

## Project Structure

```
.
├── .claude/
│   ├── agents/          # Agent definitions (GCP, K8s, AWS, Terraform, Incident Commander)
│   ├── skills/          # Skill definitions (/gcp-incident, /k8s-incident, /read-jira, etc.)
│   ├── commands/        # Legacy commands (security-scan)
│   ├── memory/          # Long-term memory (episodic + semantic)
│   └── settings.json    # Hooks, permissions, allowed commands
├── incidents/           # Generated runbooks (YYYY-MM-DD-<slug>.md format)
├── knowledge/           # Repository registry (Terraform resource → repo mapping)
├── logs/                # Auto-generated Bash command logs (YYYY-MM-DD.md, one per day)
└── scripts/             # Python utilities (memory_manager, hooks, ec2_tags_from_ips)
```

---

## Long-Term Memory System

The project maintains two memory types under `.claude/memory/` that persist across sessions:

| Type | File | What it stores |
|------|------|----------------|
| **Episodic** | `.claude/memory/episodic/incidents.jsonl` | One record per resolved incident: symptoms, root cause, resolution, lessons |
| **Semantic** | `.claude/memory/semantic/incident_patterns.json` | Known failure pattern templates: symptom list, first checks, resolution, seen count |

All memory operations go through `scripts/memory_manager.py`:

```bash
# At start of investigation — recall similar past incidents
python3 scripts/memory_manager.py search-incidents --keywords "OOMKill,service-worker-1,ssl"

# Get first-checks for a known failure pattern
python3 scripts/memory_manager.py get-patterns --symptom "CrashLoopBackOff"
python3 scripts/memory_manager.py get-patterns --symptom "certificate expired"

# After resolution — record the incident (auto-handled by Stop hook, manual only if needed)
python3 scripts/memory_manager.py add-incident --stdin < incident.json

# After resolution — update or add a failure pattern
python3 scripts/memory_manager.py add-pattern --category oom-memory-leak --stdin < patch.json
```

> **Note:** `memory_manager.py` also has legacy provisioning commands (`find-repo`, `scan-root`, `get-workflow`) for Terraform resource lookup — still functional but not actively used in incident response.

**When to write:** The `Stop` hook (`session_memory_hook.py`) automatically records completed runbooks to `incidents.jsonl` at the end of each turn — no manual `add-incident` call needed. You must still update semantic patterns manually with `add-pattern` after each incident.

**Repository registry** — `knowledge/repository-registry.json` maps Terraform repos under `~/ACME/Bitbucket/` to their resource types. Refresh with:

```bash
python3 scripts/memory_manager.py refresh-registry
```

**`incident_patterns.json` categories** — each tagged with a `platform` field (`gcp`, `k8s`, `aws`, `terraform`):

| Platform | Categories |
|---|---|
| `gcp` | `ssl-cert-expiry`, `cloud-sql-connectivity`, `quota-exhaustion`, `gcp-restricted-universe-auth` |
| `k8s` | `oom-memory-leak`, `crashloopbackoff-config`, `imagepullbackoff`, `node-notready`, `hpa-not-scaling` |
| `aws` | `aws-iam-access-denied`, `aws-alb-unhealthy-targets`, `aws-route53-dns-failure`, `aws-lambda-error`, `aws-s3-access-denied`, `aws-kms-key-issue`, `rds-replica-lag-queries`, `aws-missing-egress-ip-whitelist` |
| `terraform` | `terraform-state-drift`, `terraform-state-lock`, `atlantis-plan-failure` |

---

## Tooling & Access Assumptions

- **gcloud** CLI is authenticated (`gcloud auth list`)
- **kubectl** is configured with the target cluster context
- **GCP project** is set via `CLOUDSDK_CORE_PROJECT` or `gcloud config set project <project>`
- **aws** CLI is configured (`aws sts get-caller-identity`)
- Log access via **Cloud Logging** (`gcloud logging read`)
- Metrics via **Cloud Monitoring** (`gcloud monitoring` or direct API)
- **Jira API token** stored in macOS Keychain under service name `JiraAPIToken`

Check prerequisites:

```bash
gcloud auth list
kubectl config current-context
gcloud config get-value project
aws sts get-caller-identity
security find-generic-password -s "JiraAPIToken" -w > /dev/null && echo "Jira token found"
```

### Useful Utilities

```bash
# Look up EC2 instance names/tags from IP addresses (useful for AWS incident triage)
python3 scripts/ec2_tags_from_ips.py <IP1> <IP2> ...

# Search repository registry for Terraform repos containing a resource type or keyword
python3 scripts/memory_manager.py find-repo --resource-type google_sql_database_instance
python3 scripts/memory_manager.py find-repo --keywords "postgres"
```

---

## Incident Workflow

```
0. Intake    → read the Jira ticket or a provided .md file to understand the reported issue
1. Plan      → write a concise investigation plan (which platform, which agents, what to check)
              → present the plan to the user and wait for explicit Go / No-Go before proceeding
2. Recall    → search memory for similar past incidents and matching patterns
3. Triage    → understand symptoms, affected service, blast radius
4. Diagnose  → run the relevant investigators in parallel based on platform
5. Correlate → Incident Commander merges findings, identifies root cause
6. Fix       → apply remediation (with confirmation for destructive actions)
7. Verify    → confirm recovery, check dependent services
8. Runbook   → document root cause and resolution; write to incidents/<YYYY-MM-DD>-<slug>.md
9. Remember  → record the incident and update patterns in long-term memory
```

### Step 0 — Intake: read the issue description

Before anything else, gather the full problem statement from one of these sources:

- **Jira ticket** — invoke the `/read-jira` skill with the ticket key:
  ```
  /read-jira <TICKET-KEY>
  ```
  The skill resolves credentials from the macOS keychain automatically, smoke-tests auth, fetches the full ticket, and extracts incident intake fields (service, environment, symptoms, timeline). If auth fails (401) it prompts the user to refresh `JiraAPIToken` in the keychain.

- **Markdown file** — read the file the user points to (e.g. `incidents/draft-2026-09-30-oom.md`).

Extract: affected service(s), environment, symptoms reported, any timeline already known.

### Step 1 — Plan: draft and confirm before starting

After reading the issue, write the investigation plan to a markdown file at:

```
incidents/plan-<TICKET-ID>-<slug>.md
```

Use this structure:

```markdown
## Investigation Plan: <TICKET-ID> — <issue summary>

**Date**: YYYY-MM-DD
**Issue summary**: <one-line from Jira/MD>
**Platform**: GCP | K8s | AWS | Terraform | Full-stack
**Environment**: prod | test | uat
**Agents to run**: <list>

### Steps
1. ...
2. ...

### Open questions
- ...
```

Then tell the user the file path and ask: **"Ready to start? (Go / No-Go)"**

Only proceed with Step 2 after the user replies **Go** (or equivalent confirmation).

### Step 2 — Recall from memory

Before starting diagnosis, check whether a similar incident has been seen before:

```bash
# Search past incidents by symptom or service name
python3 scripts/memory_manager.py search-incidents --keywords "OOMKill,service-worker-1"

# Get known failure patterns matching the current symptom
python3 scripts/memory_manager.py get-patterns --symptom "CrashLoopBackOff"
python3 scripts/memory_manager.py get-patterns --symptom "certificate expired TLS"
```

Pattern results include `first_checks` — run those immediately to accelerate diagnosis.

### Step 4 — Diagnose: choose investigators by platform

Select agents based on the platform where the symptom is observed. Run independent investigators in parallel.

#### GCP infrastructure issue (quotas, networking, IAM, Cloud SQL, GKE control plane)

```
Agent: GCP Investigator
Skill: /gcp-incident
```

Investigation checklist (in order):
1. Baseline — `gcloud config get-value project && gcloud auth list`
2. GKE cluster health — `gcloud container clusters list`
3. Quotas — `gcloud compute project-info describe --format="yaml(quotas)"`
4. IAM / Workload Identity — `gcloud projects get-iam-policy <PROJECT>`
5. Cloud Logging errors — `gcloud logging read 'severity>=ERROR' --freshness=1h --limit=50`
6. VPC / firewall / NAT — `gcloud compute firewall-rules list`
7. Cloud SQL — `gcloud sql instances list && gcloud sql instances describe <INSTANCE>`
8. Recent GCP operations — `gcloud container operations list --limit=20`

#### Kubernetes workload issue (pods, deployments, HPA, nodes, PVCs)

```
Agent: K8s Investigator
Skill: /k8s-incident
```

Investigation checklist (in order):
1. Confirm context — `kubectl config current-context`
2. Node health — `kubectl get nodes -o wide && kubectl top nodes`
3. Pod status — `kubectl get pods -n $NS -o wide --sort-by='.status.startTime'`
4. Events — `kubectl get events -n $NS --sort-by='.lastTimestamp' | tail -30`
5. Failing pod — `kubectl describe pod $POD -n $NS && kubectl logs $POD -n $NS --previous --tail=100`
6. Deployment / rollout — `kubectl rollout status deployment/$DEPLOY -n $NS`
7. HPA — `kubectl get hpa -n $NS && kubectl describe hpa $HPA -n $NS`
8. Services / endpoints — `kubectl get endpoints $SVC -n $NS`

#### AWS issue (IAM, VPC/SGs, ALB, Route53, Lambda, S3, KMS, WAF, EC2, RDS)

```
Agent: AWS Investigator
Skill: /rds-replica-lag (for RDS-specific issues)
```

Investigation checklist (in order):
1. Baseline — `aws sts get-caller-identity && aws configure get region`
2. IAM — `aws iam simulate-principal-policy` for access denied; `aws cloudtrail lookup-events` for audit
3. VPC / SGs / NACLs — `aws ec2 describe-security-groups && aws ec2 describe-network-acls`
4. ALB — `aws elbv2 describe-target-health --target-group-arn <TG_ARN>`
5. Route53 — `aws route53 list-resource-record-sets --hosted-zone-id <ZONE_ID>`
6. CloudWatch alarms — `aws cloudwatch describe-alarms --state-value ALARM`
7. Lambda — `aws lambda get-function --function-name <NAME>` + CloudWatch Insights query
8. S3 / KMS — bucket policy, public access block, key state
9. RDS — `aws rds describe-db-instances`, replica lag via CloudWatch, Performance Insights for query analysis
10. CloudTrail audit — `aws cloudtrail lookup-events --lookup-attributes AttributeKey=ResourceName,AttributeValue=<ID>`

> **AWS Read-Only Guard:** the `PreToolUse` hook blocks all mutating `aws` commands automatically. Confirm with the user before any write operation.

**For RDS replica lag specifically**, invoke `/rds-replica-lag` skill for comprehensive diagnosis:
- CloudWatch metrics (ReplicaLag, CPUUtilization, IOPS)
- Performance Insights (DB load, top SQL queries)
- PostgreSQL logs analysis (long queries, checkpoints, conflicts)
- Root cause diagnosis (long queries, write spikes, slow checkpoints, VACUUM conflicts)

#### Terraform / infrastructure drift issue

```
Agent: Terraform Inspector
Repo root: ~/ACME/Bitbucket/<repo>
```

Investigation checklist (in order):
1. Orient — `ls <REPO_ROOT>/stacks/` and detect provider (`grep -r "provider " --include="*.tf"`)
2. Read code — map resource topology with `grep -rn 'resource "' --include="*.tf"`
3. Security scan — check for open `0.0.0.0/0`, unencrypted storage, hardcoded secrets
4. Verify live state — compare `gcloud`/`aws describe` output against `.tf` definitions
5. Detect drift — resources in code missing from cloud, or cloud resources absent from state
6. Atlantis — `cat atlantis.yaml` + `git log --oneline -10` for recent changes

#### Full-stack incident (GCP + K8s symptoms together)

```
Agent: Incident Commander  ← orchestrates GCP Investigator + K8s Investigator in parallel
```

Use the Incident Commander when symptoms span both layers (e.g. pods failing because of a GCP quota breach, or GKE control plane unhealthy causing workload disruption). The Commander runs both investigators concurrently, merges findings, and produces the final incident report.

### Step 9 — Record to memory

The `Stop` hook automatically records the runbook to `incidents.jsonl` once `### Root Cause` is present in the file — no manual command needed.

You must still update the **semantic pattern** manually so the system learns for next time:

```bash
# Increment seen_count and add any new symptoms to the matching pattern
python3 scripts/memory_manager.py add-pattern --category oom-memory-leak --stdin << 'EOF'
{
  "seen_count": 1,
  "symptoms": ["any new symptom not already in the pattern"]
}
EOF
```

If the incident reveals a **new** pattern not yet in `incident_patterns.json`, create it with `add-pattern --category new-category-name --stdin < full_pattern.json`.

---

## Key Conventions

- **Always confirm** before mutating cluster state (delete pod, drain node, scale to 0).
- **Prefer read-only** commands first: `kubectl get`, `kubectl describe`, `kubectl logs`, `gcloud ... list/describe`.
- When running `kubectl exec` or `gcloud sql connect`, state intent to the user first.
- Surface raw log lines and metric values — do not paraphrase evidence.
- Tag every finding with the source command so it can be reproduced.

---

## Common Incident Patterns

### GCP
- **Quota exhaustion**: check `gcloud compute project-info describe --format="yaml(quotas)"`
- **Networking**: VPC firewall rules, Cloud NAT, Private Service Connect
- **IAM/auth failures**: check service account bindings, Workload Identity
- **GKE control plane**: check cluster status, master version, upgrade state
- **Restricted Universe auth**: use `gcloud iam service-accounts get-access-token` for restricted-universe.example domains

### Kubernetes
- **CrashLoopBackOff**: logs → `kubectl logs <pod> --previous`, check exit codes
- **OOMKilled**: `kubectl describe pod` → Last State, check resource limits
- **ImagePullBackOff**: registry auth, image tag existence
- **Pending pods**: node capacity, taints/tolerations, PVC binding, resource requests
- **HPA not scaling**: check metrics-server, custom metrics, min/max bounds
- **Node NotReady**: kubelet logs, disk pressure, network plugin

### AWS
- **RDS replica lag**: check CloudWatch ReplicaLag, Performance Insights DB load, long-running queries in PostgreSQL logs
- **IAM access denied**: use `aws iam simulate-principal-policy` to test permissions
- **ALB unhealthy targets**: check security groups, health check config, target instance health
- **Lambda timeout/throttling**: check timeout settings, concurrency limits, VPC egress
- **S3 access denied**: check bucket policy, KMS key policy, public access block settings

---

## Environment Variables

```bash
# Set before starting an investigation session
export GCP_PROJECT=<your-project-id>
export K8S_CONTEXT=<your-kubectl-context>
export K8S_NAMESPACE=<target-namespace>   # defaults to 'default'
```

Jira credentials are stored in macOS Keychain — no environment variable needed. The `/read-jira` skill fetches the token inline via:

```bash
security find-generic-password -s "JiraAPIToken" -w
```

---

## Runbook Output Format

After generating the runbook, write it to `incidents/<YYYY-MM-DD>-<slug>.md` (e.g. `incidents/2026-09-29-cloud-sql-quota.md`). Existing examples are in the `incidents/` directory.

Generated runbooks follow this structure:

```
## Incident: <title>
**Severity**: P1/P2/P3
**Affected**: <services/components>
**Detection**: <how it was found>

### Timeline
- HH:MM — <event>

### Root Cause
<concise explanation>

### Resolution Steps
1. <step with exact commands>

### Prevention
<follow-up actions / alerts to add>
```
