output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "firestore_database" {
  description = "Firestore database name"
  value       = google_firestore_database.finbot.name
}

output "firestore_location" {
  description = "Firestore database location"
  value       = google_firestore_database.finbot.location_id
}

output "service_account_email" {
  description = "FinBot service account email — set as GOOGLE_APPLICATION_CREDENTIALS target"
  value       = google_service_account.finbot.email
}

output "service_account_key_base64" {
  description = "Base64-encoded SA key JSON (only set when generate_sa_key = true)"
  value       = var.generate_sa_key ? google_service_account_key.finbot_key[0].private_key : null
  sensitive   = true
}

output "env_file_snippet" {
  description = "Paste this into your .env file"
  value       = <<-EOT
    GCP_PROJECT_ID="${var.project_id}"
    # GOOGLE_APPLICATION_CREDENTIALS="./finbot-sa-key.json"
    # Run: terraform output -raw service_account_key_base64 | base64 -d > finbot-sa-key.json
  EOT
}
