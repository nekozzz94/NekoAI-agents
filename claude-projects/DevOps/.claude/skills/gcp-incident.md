# /gcp-incident

Trigger a focused GCP-layer incident investigation.

## When to use

Invoke when symptoms suggest a GCP infrastructure problem:
- GKE cluster unhealthy or control plane unreachable
- Quota errors, resource exhaustion
- IAM / Workload Identity auth failures
- Cloud SQL connectivity or performance issues
- VPC / firewall / Cloud NAT / DNS resolution failures
- Unexpected billing spike or resource usage

## Instructions

When this skill is invoked:

1. If the user hasn't described symptoms, ask:
   - What error or symptom are they seeing?
   - Which GCP project and region?
   - Approximate start time?

2. Run the GCP Investigator agent with the symptom context.

3. Work through the GCP investigation checklist:
   - Confirm gcloud project/auth
   - Check quotas (compute + regional)
   - Review Cloud Logging for errors in the relevant time window
   - Inspect GKE cluster status if Kubernetes is involved
   - Check IAM bindings and service account health
   - Review recent GCP operations that may have caused the issue

4. Present findings ranked by severity (HIGH / MED / LOW) with:
   - The exact command that surfaced the finding
   - Raw evidence (log lines, quota values, status fields)
   - Confidence level in root cause hypothesis

5. Propose next steps — distinguish safe read-only steps from actions requiring confirmation.

## Key commands quick reference

```bash
# Auth / project
gcloud auth list && gcloud config list

# Cluster health
gcloud container clusters list
gcloud container clusters describe $CLUSTER --region=$REGION

# Quotas
gcloud compute project-info describe --format="yaml(quotas)"

# Recent errors
gcloud logging read 'severity>=ERROR' --freshness=1h --limit=50

# IAM
gcloud projects get-iam-policy $PROJECT --flatten="bindings[].members" \
  --format="table(bindings.role,bindings.members)"

# Operations
gcloud container operations list --limit=20
```
