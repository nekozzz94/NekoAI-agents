# Firestore Native-mode database
# Stores episodic memory (session summaries) and semantic memory (user financial profile)
resource "google_firestore_database" "finbot" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.firestore_location
  type        = "FIRESTORE_NATIVE"

  # Prevent accidental deletion of the database (and all user memory data)
  deletion_policy = "DELETE"

  depends_on = [google_project_service.apis["firestore.googleapis.com"]]
}
