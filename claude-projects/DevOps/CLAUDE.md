# DevOps Incident Troubleshooting — GCP & Kubernetes

## Project Purpose

This project provides AI-assisted incident response for Google Cloud Platform (GCP) and Kubernetes environments. Use it to diagnose cluster health issues, investigate failing workloads, trace infrastructure anomalies, and drive incidents to resolution.

---

## Core Skills

| Slash Command | What it does |
|---|---|
| `/gcp-incident` | GCP-focused investigation: quotas, networking, IAM, Cloud SQL, GKE control plane |
| `/k8s-incident` | Kubernetes workload deep-dive: pods, deployments, HPA, events, OOMKill, CrashLoops |
| `/incident-runbook` | Generate a structured runbook from symptoms → root cause → resolution |

---

## Available Agents

| Agent | Role |
|---|---|
| `GCP Investigator` | Queries GCP APIs, reads logs/metrics, checks quotas and IAM |
| `K8s Investigator` | Inspects cluster state via kubectl, finds failing workloads |
| `Incident Commander` | Orchestrates both agents, synthesizes findings, produces incident report |

**Parallel sub-agent limit: 5.** Never spawn more than 5 sub-agents concurrently. Queue additional agents and start them as slots free up.

---

## Tooling & Access Assumptions

- **gcloud** CLI is authenticated (`gcloud auth list`)
- **kubectl** is configured with the target cluster context
- **GCP project** is set via `CLOUDSDK_CORE_PROJECT` or `gcloud config set project <project>`
- Log access via **Cloud Logging** (`gcloud logging read`)
- Metrics via **Cloud Monitoring** (`gcloud monitoring` or direct API)

Check prerequisites:

```bash
gcloud auth list
kubectl config current-context
gcloud config get-value project
```

---

## Incident Workflow

```
1. Triage    → understand symptoms, affected service, blast radius
2. Diagnose  → GCP Investigator + K8s Investigator run in parallel
3. Correlate → Incident Commander merges findings
4. Fix       → apply remediation (with confirmation for destructive actions)
5. Verify    → confirm recovery, check dependent services
6. Runbook   → document root cause and resolution for post-mortem
```

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

### Kubernetes
- **CrashLoopBackOff**: logs → `kubectl logs <pod> --previous`, check exit codes
- **OOMKilled**: `kubectl describe pod` → Last State, check resource limits
- **ImagePullBackOff**: registry auth, image tag existence
- **Pending pods**: node capacity, taints/tolerations, PVC binding, resource requests
- **HPA not scaling**: check metrics-server, custom metrics, min/max bounds
- **Node NotReady**: kubelet logs, disk pressure, network plugin

---

## Environment Variables

```bash
# Set before starting an investigation session
export GCP_PROJECT=<your-project-id>
export K8S_CONTEXT=<your-kubectl-context>
export K8S_NAMESPACE=<target-namespace>   # defaults to 'default'
```

---

## Runbook Output Format

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
