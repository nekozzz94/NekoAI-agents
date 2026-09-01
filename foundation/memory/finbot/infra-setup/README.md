# FinBot Infra Setup

Terraform configuration that provisions GCP infrastructure for the personal financial chatbot.

## Resources provisioned

| Resource | Details |
|---|---|
| Firestore database | Native mode, `(default)` database — stores episodic and semantic memory |
| `finbot-sa` service account | Least-privilege SA for the chatbot application |
| IAM binding | `roles/datastore.user` on `finbot-sa` |
| APIs enabled | `firestore`, `aiplatform`, `iam`, `iamcredentials` |

---

## Terraform service account

Terraform runs as **`neko-agent-tf-sa@%changeme%.iam.gserviceaccount.com`** via short-lived access token impersonation (`google_service_account_access_token`).

### Required roles on `%changeme%` project

The following project-level IAM roles must be granted to `neko-agent-tf-sa` before running `terraform apply`:

| Role | Why it is needed |
|---|---|
| `roles/serviceusage.serviceUsageAdmin` | Enable / disable GCP APIs (`google_project_service`) |
| `roles/datastore.owner` or `roles/firebase.admin` | Create and manage the Firestore database (`google_firestore_database`) |
| `roles/iam.serviceAccountAdmin` | Create and manage the `finbot-sa` service account (`google_service_account`) |
| `roles/iam.serviceAccountKeyAdmin` | Create SA keys when `generate_sa_key = true` (`google_service_account_key`) |
| `roles/resourcemanager.projectIamAdmin` | Grant IAM roles to `finbot-sa` on the project (`google_project_iam_member`) |

### Required role for impersonation

The caller (human or CI identity) that runs `terraform` locally must be allowed to impersonate `neko-agent-tf-sa`:

| Role | Granted on |
|---|---|
| `roles/iam.serviceAccountTokenCreator` | `neko-agent-tf-sa` service account (resource-level, not project-level) |

Grant it with:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  neko-agent-tf-sa@%changeme%.iam.gserviceaccount.com \
  --member="user:YOUR_EMAIL@example.com" \
  --role="roles/iam.serviceAccountTokenCreator"

gcloud iam service-accounts add-iam-policy-binding \
    finbot-sa@%changeme%.iam.gserviceaccount.com \
    --member="user:%changeme%" \
    --role="roles/iam.serviceAccountTokenCreator"
```

---

## Usage

```bash
# Authenticate as yourself (impersonation happens inside Terraform)
gcloud auth application-default login

# Init and plan
terraform init
terraform plan -var-file=dev.tfvars -out=dev.plan

# Apply
terraform apply dev.plan
```

### Variables (`dev.tfvars`)

| Variable | Description | Default |
|---|---|---|
| `project_id` | GCP project ID | *(required)* |
| `region` | GCP region | `us-east1` |
| `firestore_location` | Firestore location | `us-east1` |
| `service_account_name` | Name of the chatbot SA | `finbot-sa` |
| `generate_sa_key` | Export a JSON key for local dev | `false` |
| `environment` | `dev` / `staging` / `prod` | `dev` |

### Export SA key (local dev only)

```bash
# Set generate_sa_key = true in your tfvars, then:
terraform output -raw service_account_key_base64 | base64 -d > finbot-sa-key.json
export GOOGLE_APPLICATION_CREDENTIALS="./finbot-sa-key.json"
```

Prefer Workload Identity over key files for Cloud Run / GKE deployments.
