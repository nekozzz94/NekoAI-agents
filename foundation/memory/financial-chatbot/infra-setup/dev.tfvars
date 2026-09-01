# Copy this file to terraform.tfvars and fill in your values
# terraform.tfvars is gitignored — never commit it

project_id           = "%changeme%"
region               = "asia-southeast1"
firestore_location   = "asia-southeast1"
service_account_name = "finbot-sa"
environment          = "dev"

# Set to true only for local dev when you need a key file
# Prefer Workload Identity / ADC in Cloud environments
generate_sa_key = false
