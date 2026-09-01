# Service account for the FinBot chatbot application
resource "google_service_account" "finbot" {
  project      = var.project_id
  account_id   = var.service_account_name
  display_name = "FinBot Service Account"
  description  = "Least-privilege SA for the personal financial chatbot (Firestore read/write)"

  depends_on = [google_project_service.apis["iam.googleapis.com"]]
}

locals {
  sa_member = "serviceAccount:${google_service_account.finbot.email}"

  # Roles granted to the service account
  sa_roles = [
    "roles/datastore.user",  # Firestore read/write (episodic + semantic memory)
    # "roles/aiplatform.user" — only needed if switching from GEMINI_API_KEY to Vertex AI
  ]
}

resource "google_project_iam_member" "finbot_roles" {
  for_each = toset(local.sa_roles)

  project = var.project_id
  role    = each.value
  member  = local.sa_member
}

# Optional: export a JSON key for local development
# Prefer Workload Identity (below) for production / Cloud Run / GKE deployments
resource "google_service_account_key" "finbot_key" {
  count = var.generate_sa_key ? 1 : 0

  service_account_id = google_service_account.finbot.name
  keepers = {
    # Rotate by changing this date
    rotation_date = "2026-01-01"
  }
}
