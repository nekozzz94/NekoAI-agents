variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-east1"
}

variable "firestore_location" {
  description = "Firestore multi-region or region location ID (must match Firestore's supported locations)"
  type        = string
  default     = "us-east1"
}

variable "service_account_name" {
  description = "Name of the service account created for the chatbot"
  type        = string
  default     = "finbot-sa"
}

variable "generate_sa_key" {
  description = "Whether to generate and export a service account JSON key. Prefer false and use Workload Identity in production."
  type        = bool
  default     = false
}

variable "environment" {
  description = "Deployment environment label (dev / staging / prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}
